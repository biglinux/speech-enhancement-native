//! Shape-specialized layers. All tensors use `[time, frequency, channel]` internally.
//! No transposes, tensor concatenations or heap operations in forward methods.
use crate::{
    kernels::{GruWeights, LayerNorm},
    weights::*,
};
use serde_json::Value;
#[path = "conv_fast.rs"]
mod conv_fast;

#[derive(Clone, Copy)]
pub enum Activation {
    None,
    Relu,
    Sigmoid,
    Tanh,
}
impl Activation {
    fn load(v: &Value) -> Result<Self> {
        match string(v, "activation")? {
            "none" => Ok(Self::None),
            "relu" => Ok(Self::Relu),
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            _ => Err("unsupported activation".into()),
        }
    }
    #[inline]
    pub fn scalar(self, x: f32) -> f32 {
        match self {
            Self::None => x,
            Self::Relu => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
        }
    }
    /// [`Self::scalar`] over a slice, `bias` added first when given. The match
    /// runs once per slice: per element it compiled to a jump table that held
    /// ~90% of `Conv::apply`'s samples, and the ReLU/None loops now vectorize.
    /// Same per-element operations, so bit-identical to calling `scalar`.
    #[inline]
    pub fn apply_slice(self, y: &mut [f32], bias: Option<&[f32]>) {
        macro_rules! each {
            ($f:expr) => {
                match bias {
                    Some(b) => y.iter_mut().zip(b).for_each(|(v, &b)| *v = $f(*v + b)),
                    None => y.iter_mut().for_each(|v| *v = $f(*v)),
                }
            };
        }
        match self {
            Self::None => each!(|x: f32| x),
            Self::Relu => each!(|x: f32| x.max(0.0)),
            Self::Sigmoid => each!(|x: f32| 1.0 / (1.0 + (-x).exp())),
            Self::Tanh => each!(|x: f32| x.tanh()),
        }
    }
}

pub struct Linear {
    pub input: usize,
    pub output: usize,
    groups: usize,
    w: F32s,
    b: F32s,
    act: Activation,
}
impl Linear {
    pub fn load(b: &Bundle, v: &Value) -> Result<Self> {
        let input = num(v, "input")?;
        let output = num(v, "output")?;
        let groups = num(v, "groups")?;
        require(
            input > 0
                && input <= 4096
                && output > 0
                && output <= 4096
                && groups > 0
                && input % groups == 0
                && output % groups == 0,
            "invalid grouped-linear shape",
        )?;
        let w = b.f32s(&v["weight"], input * (output / groups))?;
        let bias = b.f32s(&v["bias"], output)?;
        Ok(Self {
            input,
            output,
            groups,
            w,
            b: bias,
            act: Activation::load(v)?,
        })
    }
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.input);
        debug_assert_eq!(y.len(), self.output);
        let ip = self.input / self.groups;
        let op = self.output / self.groups;
        #[cfg(not(feature = "scalar-reference"))]
        dfn_ops::grouped_linear(y, x, &self.w, self.groups, ip, op);
        #[cfg(feature = "scalar-reference")]
        for g in 0..self.groups {
            for o in 0..op {
                let mut s = 0.0;
                for i in 0..ip {
                    s += x[g * ip + i] * self.w[(g * ip + i) * op + o];
                }
                y[g * op + o] = s;
            }
        }
        self.act.apply_slice(y, Some(&self.b));
    }
}

pub struct History {
    data: Vec<f32>,
    frames: usize,
    frame_len: usize,
    next: usize,
}
impl History {
    pub fn new(frames: usize, frame_len: usize) -> Self {
        assert!(frames > 0);
        Self {
            data: vec![0.0; frames * frame_len],
            frames,
            frame_len,
            next: 0,
        }
    }
    pub fn push(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.frame_len);
        self.data[self.next * self.frame_len..(self.next + 1) * self.frame_len].copy_from_slice(x);
        self.next = (self.next + 1) % self.frames;
    }
    pub fn frame(&self, chronological: usize) -> &[f32] {
        let p = (self.next + chronological) % self.frames;
        &self.data[p * self.frame_len..(p + 1) * self.frame_len]
    }
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.next = 0;
    }
    pub fn view(&self, f: usize, c: usize) -> View<'_> {
        debug_assert_eq!(f * c, self.frame_len);
        View {
            data: &self.data,
            t: self.frames,
            f,
            c,
            start: self.next,
        }
    }
}
#[derive(Clone, Copy)]
pub struct View<'a> {
    pub data: &'a [f32],
    pub t: usize,
    pub f: usize,
    pub c: usize,
    pub start: usize,
}
impl<'a> View<'a> {
    pub fn frame(data: &'a [f32], f: usize, c: usize) -> Self {
        Self {
            data,
            t: 1,
            f,
            c,
            start: 0,
        }
    }
    #[inline]
    fn row(&self, t: usize, f: usize) -> &[f32] {
        let i = (((self.start + t) % self.t) * self.f + f) * self.c;
        &self.data[i..i + self.c]
    }
}

/// Weights [kt,kf,group,input_per_group,output_per_group]. BN is folded offline.
struct Conv {
    ci: usize,
    co: usize,
    kt: usize,
    kf: usize,
    stride: usize,
    pad: usize,
    groups: usize,
    w: F32s,
    b: F32s,
    act: Activation,
    fast: Option<conv_fast::Kernel>,
    df5_padded: Option<F32s>,
}
impl Conv {
    fn load(bundle: &Bundle, v: &Value) -> Result<Self> {
        let ci = num(v, "ci")?;
        let co = num(v, "co")?;
        let kt = num(v, "kt")?;
        let kf = num(v, "kf")?;
        let stride = num(v, "stride")?;
        let pad = num(v, "pad")?;
        let groups = num(v, "groups")?;
        require(
            ci > 0
                && ci <= 64
                && co > 0
                && co <= 64
                && kt > 0
                && kt <= 5
                && kf > 0
                && kf <= 3
                && stride > 0
                && stride <= 3,
            "unsupported convolution dimensions",
        )?;
        require(
            groups > 0 && ci % groups == 0 && co % groups == 0 && pad <= 1,
            "invalid convolution grouping/padding",
        )?;
        let mut conv = Self {
            ci,
            co,
            kt,
            kf,
            stride,
            pad,
            groups,
            w: bundle.f32s(&v["weight"], kt * kf * ci * (co / groups))?,
            b: bundle.f32s(&v["bias"], co)?,
            act: Activation::load(v)?,
            fast: None,
            df5_padded: None,
        };
        conv.df5_padded = conv_fast::prepare_df5(&conv);
        conv.fast = conv_fast::select(&conv);
        Ok(conv)
    }
    fn out_f(&self, f: usize) -> Result<usize> {
        require(f + 2 * self.pad >= self.kf, "convolution input too short")?;
        Ok((f + 2 * self.pad - self.kf) / self.stride + 1)
    }
    /// Writes subpixel phases directly into interleaved output; no phase tensor copies.
    fn apply(&self, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
        debug_assert_eq!(x.c, self.ci);
        debug_assert_eq!(x.t, self.kt);
        #[cfg(not(feature = "scalar-reference"))]
        if let Some(kernel) = self.fast {
            kernel(self, x, out, of, spacing, offset);
            return;
        }
        self.apply_legacy(x, out, of, spacing, offset);
    }
    /// The generic convolution: shapes without a specialized kernel, scalar-reference
    /// builds, and the reference the specialized kernels are tested against.
    fn apply_legacy(&self, x: View<'_>, out: &mut [f32], of: usize, spacing: usize, offset: usize) {
        let ip = self.ci / self.groups;
        let op = self.co / self.groups;
        #[cfg(not(feature = "scalar-reference"))]
        if self.kt == 1 && self.kf == 1 && self.pad == 0 && self.stride == 1 && self.groups == 1 {
            // Resolve SIMD once per output position, rather than once per input channel.
            for f in 0..of {
                let y = &mut out[f * spacing + offset..f * spacing + offset + self.co];
                dfn_ops::matvec_t(y, &self.w, x.row(0, f), self.ci, self.co);
                self.act.apply_slice(y, Some(&self.b));
            }
            return;
        }
        #[cfg(not(feature = "scalar-reference"))]
        if self.kt == 1 && self.kf == 3 && self.groups == self.ci && self.ci == self.co {
            // Depthwise 3-tap over frequency: hold each channel-tile in registers across
            // the three taps (one store per position, not three read-modify-writes).
            dfn_ops::depthwise_1x3(
                out,
                x.data,
                &self.w,
                &self.b,
                self.co,
                of,
                x.f,
                self.stride,
                self.pad,
                spacing,
                offset,
            );
            for f in 0..of {
                let y = &mut out[f * spacing + offset..f * spacing + offset + self.co];
                self.act.apply_slice(y, None);
            }
            return;
        }
        for f in 0..of {
            let y = &mut out[f * spacing + offset..f * spacing + offset + self.co];
            y.copy_from_slice(&self.b);
            for t in 0..self.kt {
                for k in 0..self.kf {
                    let xi = f * self.stride + k;
                    if xi < self.pad || xi - self.pad >= x.f {
                        continue;
                    }
                    let xr = x.row(t, xi - self.pad);
                    let base = (t * self.kf + k) * self.ci * op;
                    if self.groups == self.ci && self.ci == self.co {
                        // Depthwise: channels and weights contiguous, suitable for autovectorization.
                        for c in 0..self.co {
                            y[c] += xr[c] * self.w[base + c];
                        }
                    } else if op == 1 {
                        // Final spectral mask: one SIMD dot per tap, not ci tiny AXPY calls.
                        for (g, yg) in y.iter_mut().enumerate().take(self.groups) {
                            let start = g * ip;
                            *yg += crate::kernels::dot_f32(
                                &self.w[base + start..base + start + ip],
                                &xr[start..start + ip],
                            );
                        }
                    } else {
                        for g in 0..self.groups {
                            for i in 0..ip {
                                let woff = base + (g * ip + i) * op;
                                let yg = &mut y[g * op..(g + 1) * op];
                                let wg = &self.w[woff..woff + op];
                                #[cfg(not(feature = "scalar-reference"))]
                                if op <= 8 {
                                    // E.g. the 5-channel groups in the complex-filter pathway.
                                    // Avoid a CPUID-dispatch call for every very short row.
                                    let a = xr[g * ip + i];
                                    for o in 0..op {
                                        yg[o] += wg[o] * a;
                                    }
                                } else {
                                    dfn_ops::axpy_f32(yg, wg, xr[g * ip + i]);
                                }
                                #[cfg(feature = "scalar-reference")]
                                for o in 0..op {
                                    yg[o] += wg[o] * xr[g * ip + i];
                                }
                            }
                        }
                    }
                }
            }
            self.act.apply_slice(y, None);
        }
    }
}
enum Op {
    Conv(Conv),
    Subpixel(Vec<Conv>),
}
struct Stage {
    op: Op,
    out: Vec<f32>,
    f: usize,
    c: usize,
}
pub struct Pipeline {
    stages: Vec<Stage>,
    pub in_f: usize,
    pub in_c: usize,
    pub in_t: usize,
    pub out_f: usize,
    pub out_c: usize,
}
impl Pipeline {
    pub fn load(b: &Bundle, v: &Value) -> Result<Self> {
        let in_f = num(v, "in_f")?;
        let in_c = num(v, "in_c")?;
        let in_t = num(v, "in_t")?;
        require(
            in_f > 0 && in_f <= 481 && in_c > 0 && in_c <= 64 && in_t > 0 && in_t <= 5,
            "invalid pipeline input",
        )?;
        let (mut f, mut c, mut t) = (in_f, in_c, in_t);
        let mut stages = Vec::new();
        let ops = array(v, "ops")?;
        require(!ops.is_empty() && ops.len() <= 4, "invalid pipeline length")?;
        for ov in ops {
            let op = match string(ov, "kind")? {
                "conv" => Op::Conv(Conv::load(b, ov)?),
                "subpixel" => {
                    let branches = array(ov, "branches")?;
                    require(
                        branches.len() == 2 || branches.len() == 3,
                        "invalid subpixel factor",
                    )?;
                    Op::Subpixel(
                        branches
                            .iter()
                            .map(|z| Conv::load(b, z))
                            .collect::<Result<_>>()?,
                    )
                }
                _ => return Err("unknown convolution operation".into()),
            };
            let (base, factor) = match &op {
                Op::Conv(z) => (z, 1),
                Op::Subpixel(z) => (&z[0], z.len()),
            };
            require(
                base.ci == c && base.kt == t,
                "convolution chain shape mismatch",
            )?;
            let next_f = base.out_f(f)?;
            if let Op::Subpixel(z) = &op {
                for zz in z {
                    require(
                        zz.ci == c && zz.co == base.co && zz.kt == t && zz.out_f(f)? == next_f,
                        "subpixel branch mismatch",
                    )?;
                }
            }
            f = next_f * factor;
            c = base.co;
            t = 1;
            require(f <= 481, "convolution output too large")?;
            stages.push(Stage {
                op,
                out: vec![0.0; f * c],
                f,
                c,
            });
        }
        Ok(Self {
            stages,
            in_f,
            in_c,
            in_t,
            out_f: f,
            out_c: c,
        })
    }
    pub fn forward(&mut self, x: View<'_>) -> &[f32] {
        // Validate the public View once before any private unchecked SIMD access.
        assert_eq!((x.f, x.c, x.t), (self.in_f, self.in_c, self.in_t));
        assert!(x.start < x.t);
        assert_eq!(x.data.len(), x.f * x.c * x.t);
        for idx in 0..self.stages.len() {
            let (past, future) = self.stages.split_at_mut(idx);
            let input = if idx == 0 {
                x
            } else {
                let s = &past[idx - 1];
                View::frame(&s.out, s.f, s.c)
            };
            let s = &mut future[0];
            match &s.op {
                Op::Conv(op) => op.apply(input, &mut s.out, s.f, s.c, 0),
                Op::Subpixel(ops) => {
                    let factor = ops.len();
                    for (p, op) in ops.iter().enumerate() {
                        op.apply(input, &mut s.out, s.f / factor, s.c * factor, p * s.c);
                    }
                }
            }
        }
        self.output()
    }
    pub fn output(&self) -> &[f32] {
        &self.stages[self.stages.len() - 1].out
    }
}

pub struct Squeezed {
    lin: Linear,
    lout: Option<Linear>,
    grus: Vec<GruWeights>,
    states: Vec<Vec<f32>>,
    x: Vec<f32>,
    wx: Vec<f32>,
    rh: Vec<f32>,
    q: Vec<i16>,
    out: Vec<f32>,
    pub input: usize,
    pub output: usize,
}
impl Squeezed {
    pub fn load(b: &Bundle, v: &Value) -> Result<Self> {
        let lin = Linear::load(b, &v["linear_in"])?;
        require(lin.output == 256, "squeezed hidden must be 256")?;
        let lout = if v["linear_out"].is_null() {
            None
        } else {
            Some(Linear::load(b, &v["linear_out"])?)
        };
        let grus = array(v, "grus")?
            .iter()
            .map(|z| GruWeights::load(b, z))
            .collect::<Result<Vec<_>>>()?;
        require(
            !grus.is_empty() && grus.len() <= 2,
            "invalid squeezed depth",
        )?;
        for g in &grus {
            require(
                g.hidden == 256 && g.input.cols == 256,
                "squeezed GRU mismatch",
            )?;
        }
        if let Some(l) = &lout {
            require(l.input == 256, "squeezed output projection mismatch")?;
        }
        let input = lin.input;
        let output = lout.as_ref().map_or(256, |l| l.output);
        let states = vec![vec![0.0; 256]; grus.len()];
        Ok(Self {
            lin,
            lout,
            grus,
            states,
            x: vec![0.0; 256],
            wx: vec![0.0; 768],
            rh: vec![0.0; 768],
            q: vec![0; 256],
            out: vec![0.0; output],
            input,
            output,
        })
    }
    pub fn forward(&mut self, input: &[f32]) -> &[f32] {
        self.lin.apply(input, &mut self.x);
        for (g, h) in self.grus.iter().zip(self.states.iter_mut()) {
            g.input.apply(&self.x, &mut self.wx, &mut self.q);
            g.recurrent.apply(h, &mut self.rh, &mut self.q);
            g.update(&self.wx, &self.rh, h);
            self.x.copy_from_slice(h);
        }
        if let Some(l) = &self.lout {
            l.apply(&self.x, &mut self.out);
        } else {
            self.out.copy_from_slice(&self.x);
        }
        &self.out
    }
    pub fn reset(&mut self) {
        for h in &mut self.states {
            h.fill(0.0);
        }
    }
}

pub struct DprnnBlock {
    f: usize,
    c: usize,
    forward: GruWeights,
    backward: GruWeights,
    temporal: GruWeights,
    fc_intra: Linear,
    fc_inter: Linear,
    ln_intra: LayerNorm,
    ln_inter: LayerNorm,
    proj: Vec<f32>,
    rh: Vec<f32>,
    q: Vec<i16>,
    shared_q: Vec<i16>,
    shared_scales: Vec<f32>,
    hf: Vec<f32>,
    hb: Vec<f32>,
    ht: Vec<f32>,
    intra: Vec<f32>,
    inter: Vec<f32>,
    out: Vec<f32>,
}
impl DprnnBlock {
    pub fn load(b: &Bundle, v: &Value, f: usize, c: usize) -> Result<Self> {
        let mut forward = GruWeights::load(b, &v["forward"])?;
        let mut backward = GruWeights::load(b, &v["backward"])?;
        let temporal = GruWeights::load(b, &v["temporal"])?;
        // Cache only the two small, frequently reused spectral recurrent matrices.
        // Do not expand input/temporal/large GRU weights or change the disk bundle.
        forward.recurrent.enable_recurrent_cache(b)?;
        backward.recurrent.enable_recurrent_cache(b)?;
        for g in [&forward, &backward, &temporal] {
            require(
                g.hidden == c && g.input.cols == c,
                "DPRNN GRU width mismatch",
            )?;
        }
        let fc_intra = Linear::load(b, &v["fc_intra"])?;
        let fc_inter = Linear::load(b, &v["fc_inter"])?;
        require(
            fc_intra.input == 2 * c
                && fc_intra.output == c
                && fc_inter.input == c
                && fc_inter.output == c,
            "DPRNN FC mismatch",
        )?;
        Ok(Self {
            f,
            c,
            forward,
            backward,
            temporal,
            fc_intra,
            fc_inter,
            ln_intra: LayerNorm::load(b, &v["ln_intra"], c)?,
            ln_inter: LayerNorm::load(b, &v["ln_inter"], c)?,
            proj: vec![0.0; f * 3 * c],
            rh: vec![0.0; 4 * 3 * c],
            q: vec![0; 4 * c],
            shared_q: vec![0; f * c],
            shared_scales: vec![0.0; f],
            hf: vec![0.0; c],
            hb: vec![0.0; c],
            ht: vec![0.0; f * c],
            intra: vec![0.0; f * 2 * c],
            inter: vec![0.0; f * c],
            out: vec![0.0; f * c],
        })
    }
    pub(crate) fn forward(&mut self, x: &[f32]) -> &[f32] {
        if cfg!(not(feature = "scalar-reference")) {
            self.forward_exact(x)
        } else {
            self.forward_legacy(x)
        }
    }
    /// The plain schedule: scalar-reference builds, and the reference in tests.
    fn forward_legacy(&mut self, x: &[f32]) -> &[f32] {
        let c = self.c;
        let gates = 3 * c;
        // Intra-frequency recurrence starts from zero on EVERY frame, in BOTH directions.
        self.hf.fill(0.0);
        self.hb.fill(0.0);
        self.forward
            .input
            .batch(x, &mut self.proj, self.f, &mut self.q);
        for f in 0..self.f {
            self.forward
                .recurrent
                .apply(&self.hf, &mut self.rh[..gates], &mut self.q);
            self.forward.update(
                &self.proj[f * gates..(f + 1) * gates],
                &self.rh[..gates],
                &mut self.hf,
            );
            self.intra[f * 2 * c..f * 2 * c + c].copy_from_slice(&self.hf);
        }
        self.backward
            .input
            .batch(x, &mut self.proj, self.f, &mut self.q);
        for f in (0..self.f).rev() {
            self.backward
                .recurrent
                .apply(&self.hb, &mut self.rh[..gates], &mut self.q);
            self.backward.update(
                &self.proj[f * gates..(f + 1) * gates],
                &self.rh[..gates],
                &mut self.hb,
            );
            self.intra[f * 2 * c + c..(f + 1) * 2 * c].copy_from_slice(&self.hb);
        }
        for f in 0..self.f {
            let y = &mut self.inter[f * c..(f + 1) * c];
            self.fc_intra
                .apply(&self.intra[f * 2 * c..(f + 1) * 2 * c], y);
            self.ln_intra.apply_add(y, &x[f * c..(f + 1) * c]);
        }
        self.temporal
            .input
            .batch(&self.inter, &mut self.proj, self.f, &mut self.q);
        // Temporal states are independent across F. Batch four recurrent products too.
        for start in (0..self.f).step_by(4) {
            let count = (self.f - start).min(4);
            self.temporal.recurrent.batch(
                &self.ht[start * c..(start + count) * c],
                &mut self.rh[..count * gates],
                count,
                &mut self.q,
            );
            for k in 0..count {
                let f = start + k;
                let h = &mut self.ht[f * c..(f + 1) * c];
                self.temporal.update(
                    &self.proj[f * gates..(f + 1) * gates],
                    &self.rh[k * gates..(k + 1) * gates],
                    h,
                );
                let y = &mut self.out[f * c..(f + 1) * c];
                self.fc_inter.apply(h, y);
                self.ln_inter.apply_add(y, &self.inter[f * c..(f + 1) * c]);
            }
        }
        &self.out
    }
    /// Same states/operations, less repeated preprocessing and smaller active
    /// temporal working set. Spectral recurrence remains fully sequential in
    /// each direction. There is no frame/silence skipping and no extra delay.
    fn forward_exact(&mut self, x: &[f32]) -> &[f32] {
        let c = self.c;
        let gates = 3 * c;
        self.hf.fill(0.0);
        self.hb.fill(0.0);
        let shared = self.forward.input.is_quantized() && self.backward.input.is_quantized();
        if shared {
            for f in 0..self.f {
                self.shared_scales[f] = dfn_ops::quantize_i16(
                    &x[f * c..(f + 1) * c],
                    &mut self.shared_q[f * c..(f + 1) * c],
                );
            }
            self.forward.input.batch_prequantized(
                &self.shared_q,
                &self.shared_scales,
                &mut self.proj,
                self.f,
            );
        } else {
            self.forward
                .input
                .batch(x, &mut self.proj, self.f, &mut self.q);
        }
        for f in 0..self.f {
            if f == 0 && self.forward.recurrent.is_quantized() {
                // Integer dot of the known reset state is exactly +0. Preserve
                // all six bias vectors and run the complete gate update below.
                self.rh[..gates].fill(0.0);
            } else {
                self.forward
                    .recurrent
                    .apply(&self.hf, &mut self.rh[..gates], &mut self.q);
            }
            self.forward.update(
                &self.proj[f * gates..(f + 1) * gates],
                &self.rh[..gates],
                &mut self.hf,
            );
            self.intra[f * 2 * c..f * 2 * c + c].copy_from_slice(&self.hf);
        }
        if shared {
            self.backward.input.batch_prequantized(
                &self.shared_q,
                &self.shared_scales,
                &mut self.proj,
                self.f,
            );
        } else {
            self.backward
                .input
                .batch(x, &mut self.proj, self.f, &mut self.q);
        }
        for f in (0..self.f).rev() {
            if f == self.f - 1 && self.backward.recurrent.is_quantized() {
                self.rh[..gates].fill(0.0);
            } else {
                self.backward
                    .recurrent
                    .apply(&self.hb, &mut self.rh[..gates], &mut self.q);
            }
            self.backward.update(
                &self.proj[f * gates..(f + 1) * gates],
                &self.rh[..gates],
                &mut self.hb,
            );
            self.intra[f * 2 * c + c..(f + 1) * 2 * c].copy_from_slice(&self.hb);
        }
        // After both spectral directions, time states for distinct frequencies
        // are independent. Intermediates are consumed in tiles of four instead
        // of materializing full inter/Wx planes. No LN reduction is reassociated.
        for start in (0..self.f).step_by(4) {
            let count = (self.f - start).min(4);
            for k in 0..count {
                let f = start + k;
                let y = &mut self.inter[k * c..(k + 1) * c];
                self.fc_intra
                    .apply(&self.intra[f * 2 * c..(f + 1) * 2 * c], y);
                self.ln_intra.apply_add(y, &x[f * c..(f + 1) * c]);
            }
            self.temporal.input.batch(
                &self.inter[..count * c],
                &mut self.proj[..count * gates],
                count,
                &mut self.q,
            );
            self.temporal.recurrent.batch(
                &self.ht[start * c..(start + count) * c],
                &mut self.rh[..count * gates],
                count,
                &mut self.q,
            );
            for k in 0..count {
                let f = start + k;
                let h = &mut self.ht[f * c..(f + 1) * c];
                self.temporal.update(
                    &self.proj[k * gates..(k + 1) * gates],
                    &self.rh[k * gates..(k + 1) * gates],
                    h,
                );
                let y = &mut self.out[f * c..(f + 1) * c];
                self.fc_inter.apply(h, y);
                self.ln_inter.apply_add(y, &self.inter[k * c..(k + 1) * c]);
            }
        }
        &self.out
    }
    fn reset(&mut self) {
        self.ht.fill(0.0);
        self.hf.fill(0.0);
        self.hb.fill(0.0);
    }
}
pub struct Dprnn {
    blocks: Vec<DprnnBlock>,
    pub f: usize,
    pub c: usize,
}
impl Dprnn {
    pub fn load(b: &Bundle, v: &Value, expected_f: usize, depth: usize) -> Result<Self> {
        require(
            num(v, "frequencies")? == expected_f && num(v, "channels")? == 64,
            "DPRNN shape mismatch",
        )?;
        let values = array(v, "blocks")?;
        require(values.len() == depth, "DPRNN depth mismatch")?;
        let blocks = values
            .iter()
            .map(|z| DprnnBlock::load(b, z, expected_f, 64))
            .collect::<Result<_>>()?;
        Ok(Self {
            blocks,
            f: expected_f,
            c: 64,
        })
    }
    pub fn forward(&mut self, x: &[f32]) -> &[f32] {
        debug_assert_eq!(x.len(), self.f * self.c);
        for i in 0..self.blocks.len() {
            let (before, after) = self.blocks.split_at_mut(i);
            let inp = if i == 0 { x } else { &before[i - 1].out };
            after[0].forward(inp);
        }
        &self.blocks[self.blocks.len() - 1].out
    }
    pub fn reset(&mut self) {
        for b in &mut self.blocks {
            b.reset();
        }
    }
    pub(crate) fn into_blocks(self) -> Vec<DprnnBlock> {
        self.blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_has_chronological_order() {
        let mut h = History::new(3, 1);
        for n in 1i32..10 {
            h.push(&[n as f32]);
            assert_eq!(h.frame(2), &[n as f32]);
            assert_eq!(h.frame(0), &[(n - 2).max(0) as f32]);
        }
        h.reset();
        assert_eq!(h.frame(2), &[0.0]);
    }
}

impl Squeezed {
    pub fn output(&self) -> &[f32] {
        &self.out
    }
}
impl Dprnn {
    pub fn output(&self) -> &[f32] {
        &self.blocks[self.blocks.len() - 1].out
    }
}

#[cfg(test)]
mod conv_tests {
    use super::*;
    use crate::test_support::{assert_bits, Writer};
    #[test]
    fn specialized_convolutions_match_the_generic_path() {
        // Includes both input boundaries, all circular-buffer start positions,
        // interleaved output spacing and activations after the complete sum.
        for (ci, co, kt, kf, groups) in [
            (1, 64, 3, 3, 1),
            (2, 64, 3, 3, 2),
            (64, 10, 5, 1, 2),
            (64, 64, 1, 1, 64),
            (64, 1, 1, 3, 1),
        ] {
            for act in ["none", "relu", "sigmoid"] {
                let mut w = Writer::new();
                let weight = w.random_floats(kt * kf * ci * (co / groups));
                let bias = w.random_floats(co);
                let bundle = w.finish();
                let v = serde_json::json!({"ci":ci,"co":co,"kt":kt,"kf":kf,"groups":groups,
                    "stride":1,"pad":if kf==3 {1} else {0},"weight":weight,"bias":bias,"activation":act});
                let c = Conv::load(&bundle, &v).unwrap();
                let f = 17;
                let of = c.out_f(f).unwrap();
                let data: Vec<f32> = (0..kt * f * ci)
                    .map(|i| ((i * 173 % 1021) as f32 - 510.0) * 0.003)
                    .collect();
                for start in 0..kt {
                    for factor in [1, 2, 3] {
                        let spacing = co * factor;
                        let offset = (factor - 1) * co;
                        let mut out = vec![91.75; of * spacing + 16];
                        let mut want = out.clone();
                        let view = View {
                            data: &data,
                            t: kt,
                            f,
                            c: ci,
                            start,
                        };
                        c.apply_legacy(view, &mut want, of, spacing, offset);
                        c.apply(view, &mut out, of, spacing, offset);
                        assert_bits(&out, &want);
                    }
                }
            }
        }
    }
    #[test]
    fn dprnn_shared_quantization_zero_shortcuts_and_tiling_are_exact() {
        for quant in [true, false] {
            for freq in [5, 40, 48] {
                let mut w = Writer::new();
                let v = w.block(quant);
                let b = w.finish();
                let mut candidate = DprnnBlock::load(&b, &v, freq, 64).unwrap();
                let mut previous = DprnnBlock::load(&b, &v, freq, 64).unwrap();
                for frame in 0..6 {
                    let x: Vec<f32> = (0..freq * 64)
                        .map(|i| {
                            if frame == 0 {
                                0.0
                            } else {
                                ((i * 131 + frame * 71) % 337) as f32 * 0.003 - 0.5
                            }
                        })
                        .collect();
                    let want = previous.forward_legacy(&x).to_vec();
                    assert_bits(candidate.forward_exact(&x), &want);
                    assert_bits(&candidate.ht, &previous.ht);
                    if frame == 3 {
                        candidate.reset();
                        previous.reset();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod conv_shape_tests {
    use super::*;
    use crate::test_support::{assert_bits, Writer};
    #[test]
    fn specialized_shapes_match_the_generic_path_for_strides_and_boundaries() {
        for (ci, co, kt, kf, groups) in [(1, 64, 3, 3, 1), (2, 64, 3, 3, 2), (64, 64, 1, 1, 64)] {
            for nf in [3usize, 4, 7, 17, 48, 480] {
                for stride in [1usize, 2, 3] {
                    for pad in [0usize, 1] {
                        for act in ["none", "relu", "sigmoid"] {
                            let mut writer = Writer::new();
                            let weight = writer.random_floats(kt * kf * ci * (co / groups));
                            let bias = writer.random_floats(co);
                            let b = writer.finish();
                            let desc = serde_json::json!({"ci":ci,"co":co,"kt":kt,"kf":kf,"stride":stride,"pad":pad,"groups":groups,"weight":weight,"bias":bias,"activation":act});
                            let c = Conv::load(&b, &desc).unwrap();
                            let of = c.out_f(nf).unwrap();
                            let data: Vec<f32> = (0..kt * nf * ci)
                                .map(|i| ((i * 719 % 401) as f32 - 200.0) * 0.003)
                                .collect();
                            for start in 0..kt {
                                let spacing = co * 3;
                                let offset = co;
                                let mut out = vec![111.25; of * spacing + 2];
                                let mut want = out.clone();
                                let x = View {
                                    data: &data,
                                    t: kt,
                                    f: nf,
                                    c: ci,
                                    start,
                                };
                                c.apply(x, &mut out[1..], of, spacing, offset);
                                c.apply_legacy(x, &mut want[1..], of, spacing, offset);
                                assert_bits(&out, &want);
                            }
                        }
                    }
                }
            }
        }
    }
}
