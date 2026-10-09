//! Fixed DPDFNet2/8-48k-HR topology, matched to Ceva commit 9bd9844a.
use crate::{layers::*, weights::*};
pub const FFT: usize = 960;
pub const HOP: usize = 480;
pub const BINS: usize = 481;
pub const DF_BINS: usize = 96;
pub const MODEL_DELAY: usize = 4; // mask delay 2 + middle of DF history delay 2
const C: usize = 64;
const COEFS: usize = DF_BINS * 5 * 2;

pub(crate) struct DeepFilter {
    raw: History,
    masked: History,
    coefs: History,
    temp: Vec<f32>,
}
impl DeepFilter {
    fn new() -> Self {
        Self {
            raw: History::new(5, BINS * 2),
            masked: History::new(5, BINS * 2),
            coefs: History::new(3, COEFS),
            temp: vec![0.0; BINS * 2],
        }
    }
    pub fn apply(
        &mut self,
        spec: &[f32],
        mask: &[f32],
        coefs: &[f32],
        out: &mut [f32],
        dry_mix: f32,
    ) {
        self.raw.push(spec);
        // Upstream before_df: mask(t) multiplies spectrum(t-2), not the current one.
        for ((dst, src), &m) in self
            .temp
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(self.raw.frame(2).as_chunks::<2>().0)
            .zip(mask)
        {
            dst[0] = src[0] * m;
            dst[1] = src[1] * m;
        }
        self.masked.push(&self.temp);
        self.coefs.push(coefs); // upstream coefs_buffer delay_frames=2
        out.copy_from_slice(self.masked.frame(2));
        let co = self.coefs.frame(0);
        for f in 0..DF_BINS {
            let (mut re, mut im) = (0.0, 0.0);
            for tap in 0..5 {
                let s = self.masked.frame(tap);
                let j = f * 10 + tap * 2;
                re += s[2 * f] * co[j] - s[2 * f + 1] * co[j + 1];
                im += s[2 * f] * co[j + 1] + s[2 * f + 1] * co[j];
            }
            out[2 * f] = re;
            out[2 * f + 1] = im;
        }
        let dry = self.raw.frame(0);
        for (y, &x) in out.iter_mut().zip(dry) {
            *y = (1.0 - dry_mix) * (*y) + dry_mix * x;
        }
    }
    pub fn reset(&mut self) {
        self.raw.reset();
        self.masked.reset();
        self.coefs.reset();
        self.temp.fill(0.0);
    }
}

/// Spectral features and the convolutions that open both encoder branches.
pub(crate) struct Front {
    wnorm: f32,
    alpha: f32,
    beta: f32,
    eps: f32,
    mu0: F32s,
    s0: F32s,
    mu: Vec<f32>,
    s: Vec<f32>,
    mag: Vec<f32>,
    complex: Vec<f32>,
    scaled: Vec<f32>,
    mag_hist: History,
    complex_hist: History,
    erb0: Pipeline,
    erb1: Pipeline,
    erb2: Pipeline,
    erb3: Pipeline,
    df0: Pipeline,
    df1: Pipeline,
}
impl Front {
    fn run(&mut self, spec: &[f32]) {
        for (y, &x) in self.scaled.iter_mut().zip(spec) {
            // Keep finite external values within a wide, non-audio-clipping numerical safety bound.
            *y = if x.is_finite() {
                x.clamp(-1e10, 1e10) * self.wnorm
            } else {
                0.0
            };
        }
        for f in 0..BINS {
            let re = self.scaled[2 * f];
            let im = self.scaled[2 * f + 1];
            let mag = (re * re + im * im).sqrt();
            // Upstream deliberately uses 10*log10(magnitude + 1e-10), not 20*log10.
            let db = 10.0 * (mag + 1e-10).log10();
            self.mu[f] = self.alpha * self.mu[f] + self.beta * db;
            self.mag[f] = (db - self.mu[f]) / (40.0 + self.eps);
            if f < DF_BINS {
                self.s[f] = self.alpha * self.s[f] + self.beta * mag;
                let denom = (self.s[f] + self.eps).sqrt();
                self.complex[2 * f] = re / denom;
                self.complex[2 * f + 1] = im / denom;
            }
        }
        self.mag_hist.push(&self.mag[..480]);
        self.complex_hist.push(&self.complex);
        self.erb0.forward(self.mag_hist.view(480, 1));
        self.erb1.forward(View::frame(self.erb0.output(), 480, C));
        self.erb2.forward(View::frame(self.erb1.output(), 160, C));
        self.erb3.forward(View::frame(self.erb2.output(), 80, C));
        self.df0.forward(self.complex_hist.view(DF_BINS, 2));
        self.df1.forward(View::frame(self.df0.output(), DF_BINS, C));
    }
    fn reset(&mut self) {
        self.mu.copy_from_slice(&self.mu0);
        self.s.copy_from_slice(&self.s0);
        self.mag_hist.reset();
        self.complex_hist.reset();
    }
}

/// Mask decoder from the encoder state up to 160 bins.
pub(crate) struct MaskHigh {
    erb_gru: Squeezed,
    erb_fc: Linear,
    expanded: Vec<f32>,
    skip3: Pipeline,
    up3: Pipeline,
    skip2: Pipeline,
    up2: Pipeline,
    add: Vec<f32>,
}
impl MaskHigh {
    fn run(&mut self, enc: &[f32], e3: &[f32], e2: &[f32]) -> &[f32] {
        self.erb_fc
            .apply(self.erb_gru.forward(enc), &mut self.expanded);
        self.skip3.forward(View::frame(e3, 40, C));
        add(&mut self.add[..40 * C], self.skip3.output(), &self.expanded);
        self.up3.forward(View::frame(&self.add[..40 * C], 40, C));
        self.skip2.forward(View::frame(e2, 80, C));
        add(
            &mut self.add[..80 * C],
            self.skip2.output(),
            self.up3.output(),
        );
        self.up2.forward(View::frame(&self.add[..80 * C], 80, C))
    }
}

/// Mask decoder from 160 bins to the 481-bin mask.
pub(crate) struct MaskLow {
    skip1: Pipeline,
    up1: Pipeline,
    skip0: Pipeline,
    mask_out: Pipeline,
    add: Vec<f32>,
    mask: Vec<f32>,
}
impl MaskLow {
    fn run(&mut self, up2: &[f32], e1: &[f32], e0: &[f32]) -> &[f32] {
        self.skip1.forward(View::frame(e1, 160, C));
        add(&mut self.add[..160 * C], self.skip1.output(), up2);
        self.up1.forward(View::frame(&self.add[..160 * C], 160, C));
        self.skip0.forward(View::frame(e0, 480, C));
        add(&mut self.add, self.skip0.output(), self.up1.output());
        self.mask[..480].copy_from_slice(self.mask_out.forward(View::frame(&self.add, 480, C)));
        self.mask[480] = self.mask[478]; // torch pad(..., (0,1), mode='reflect'), not replication
        &self.mask
    }
}

/// Deep-filter coefficients from the encoder state and the first complex convolution.
pub(crate) struct DfDecoder {
    df_gru: Squeezed,
    df_skip: Linear,
    df_out: Linear,
    path_hist: History,
    df_path: Pipeline,
    df_embed: Vec<f32>,
    df_skip_buf: Vec<f32>,
    coefs: Vec<f32>,
}
impl DfDecoder {
    fn run(&mut self, enc: &[f32], d0: &[f32]) -> &[f32] {
        self.df_embed.copy_from_slice(self.df_gru.forward(enc));
        self.df_skip.apply(enc, &mut self.df_skip_buf);
        for (y, &v) in self.df_embed.iter_mut().zip(&self.df_skip_buf) {
            *y += v;
        }
        self.df_out.apply(&self.df_embed, &mut self.coefs);
        self.path_hist.push(d0);
        let path = self.df_path.forward(self.path_hist.view(DF_BINS, C));
        // tanh belongs to df_out, before the pathway sum; upstream does not clamp the sum.
        for (y, &v) in self.coefs.iter_mut().zip(path) {
            *y += v;
        }
        &self.coefs
    }
    fn reset(&mut self) {
        self.df_gru.reset();
        self.path_hist.reset();
    }
}

/// Mask, deep filter and dry mix applied to the unnormalized spectrum.
pub(crate) struct Output {
    wnorm: f32,
    filter: DeepFilter,
    out: Vec<f32>,
}
impl Output {
    /// `dry_mix` is in [0, 1]: every caller derives it from `atten_lim_from_db`.
    fn run(&mut self, scaled: &[f32], mask: &[f32], coefs: &[f32], dry_mix: f32) -> &[f32] {
        self.filter
            .apply(scaled, mask, coefs, &mut self.out, dry_mix);
        for v in &mut self.out {
            *v /= self.wnorm;
        }
        &self.out
    }
    fn reset(&mut self) {
        self.filter.reset();
        self.out.fill(0.0);
    }
}

pub struct Model {
    pub(crate) window: F32s,
    front: Front,
    erb_dual: Dprnn,
    df_dual: Dprnn,
    enc_erb_fc: Linear,
    enc_df_fc: Linear,
    enc_gru: Squeezed,
    joined: Vec<f32>,
    mask_high: MaskHigh,
    mask_low: MaskLow,
    df_decoder: DfDecoder,
    output: Output,
}
impl Model {
    pub fn new(b: &Bundle) -> Result<Self> {
        let parsed = b.manifest()?;
        let m = &parsed;
        require(
            num(m, "sample_rate")? == 48000
                && num(m, "fft")? == FFT
                && num(m, "hop")? == HOP
                && num(m, "df_bins")? == DF_BINS,
            "only 48k HR geometry is implemented",
        )?;
        let depth = num(m, "depth")?;
        require(
            depth == 2 || depth == 8,
            "only trained depths 2 and 8 supported",
        )?;
        require(
            string(m, "mask_method")? == "before_df" && num(m, "lookahead")? == 2,
            "mask/lookahead contract mismatch",
        )?;
        let mu0 = b.f32s(&m["norm"]["mu0"], BINS)?;
        let s0 = b.f32s(&m["norm"]["s0"], DF_BINS)?;
        require(
            s0.iter().all(|&x| x >= 0.0),
            "negative initial spectral normalization",
        )?;
        let alpha = float(&m["norm"], "alpha")?;
        let eps = float(&m["norm"], "eps")?;
        let beta = float(&m["norm"], "one_minus_alpha")?;
        require(
            alpha > 0.0
                && alpha < 1.0
                && beta > 0.0
                && (alpha + beta - 1.0).abs() < 1e-6
                && eps > 0.0,
            "normalization parameters out of bounds",
        )?;
        require(
            float(&m["norm"], "std")? == 40.0,
            "only fixed variance normalization supported",
        )?;
        let window = b.f32s(&m["window"], FFT)?;
        let wnorm = float(m, "wnorm")?;
        require(wnorm > 0.0, "invalid FFT normalization")?;
        // COLA/NOLA: this executor assumes 50% overlap with a power-complementary window.
        for i in 0..HOP {
            require(
                ((window[i] * window[i] + window[i + HOP] * window[i + HOP]) - 1.0).abs() < 2e-5,
                "window is not power-complementary at 50% overlap",
            )?;
        }
        let e = &m["encoder"];
        let d = &m["mask_decoder"];
        let df = &m["df_decoder"];
        fn pipe(
            b: &Bundle,
            v: &serde_json::Value,
            shape: (usize, usize, usize),
            out: (usize, usize),
        ) -> Result<Pipeline> {
            let p = Pipeline::load(b, v)?;
            require(
                (p.in_t, p.in_f, p.in_c) == shape && (p.out_f, p.out_c) == out,
                "network convolution shape mismatch",
            )?;
            Ok(p)
        }
        fn lin(b: &Bundle, v: &serde_json::Value, i: usize, o: usize) -> Result<Linear> {
            let l = Linear::load(b, v)?;
            require(l.input == i && l.output == o, "network projection mismatch")?;
            Ok(l)
        }
        fn sq(b: &Bundle, v: &serde_json::Value, i: usize, o: usize) -> Result<Squeezed> {
            let l = Squeezed::load(b, v)?;
            require(l.input == i && l.output == o, "network squeezed mismatch")?;
            Ok(l)
        }
        let mut model = Self {
            window,
            front: Front {
                wnorm,
                alpha,
                beta,
                eps,
                mu: mu0.to_vec(),
                s: s0.to_vec(),
                mu0,
                s0,
                mag: vec![0.0; BINS],
                complex: vec![0.0; DF_BINS * 2],
                scaled: vec![0.0; BINS * 2],
                mag_hist: History::new(3, 480),
                complex_hist: History::new(3, DF_BINS * 2),
                erb0: pipe(b, &e["erb0"], (3, 480, 1), (480, C))?,
                erb1: pipe(b, &e["erb1"], (1, 480, C), (160, C))?,
                erb2: pipe(b, &e["erb2"], (1, 160, C), (80, C))?,
                erb3: pipe(b, &e["erb3"], (1, 80, C), (40, C))?,
                df0: pipe(b, &e["df0"], (3, DF_BINS, 2), (DF_BINS, C))?,
                df1: pipe(b, &e["df1"], (1, DF_BINS, C), (48, C))?,
            },
            erb_dual: Dprnn::load(b, &e["erb_dual"], 40, depth)?,
            df_dual: Dprnn::load(b, &e["df_dual"], 48, depth)?,
            enc_erb_fc: lin(b, &e["erb_fc"], 40 * C, 512)?,
            enc_df_fc: lin(b, &e["df_fc"], 48 * C, 512)?,
            enc_gru: sq(b, &e["gru"], 1024, 512)?,
            joined: vec![0.0; 1024],
            mask_high: MaskHigh {
                erb_gru: sq(b, &d["gru"], 512, 512)?,
                erb_fc: lin(b, &d["fc"], 512, 40 * C)?,
                expanded: vec![0.0; 40 * C],
                skip3: pipe(b, &d["skip3"], (1, 40, C), (40, C))?,
                up3: pipe(b, &d["up3"], (1, 40, C), (80, C))?,
                skip2: pipe(b, &d["skip2"], (1, 80, C), (80, C))?,
                up2: pipe(b, &d["up2"], (1, 80, C), (160, C))?,
                add: vec![0.0; 80 * C],
            },
            mask_low: MaskLow {
                skip1: pipe(b, &d["skip1"], (1, 160, C), (160, C))?,
                up1: pipe(b, &d["up1"], (1, 160, C), (480, C))?,
                skip0: pipe(b, &d["skip0"], (1, 480, C), (480, C))?,
                mask_out: pipe(b, &d["out"], (1, 480, C), (480, 1))?,
                add: vec![0.0; 480 * C],
                mask: vec![0.0; BINS],
            },
            df_decoder: DfDecoder {
                df_gru: sq(b, &df["gru"], 512, 256)?,
                df_skip: lin(b, &df["skip"], 512, 256)?,
                df_out: lin(b, &df["out"], 256, COEFS)?,
                path_hist: History::new(5, DF_BINS * C),
                df_path: pipe(b, &df["path"], (5, DF_BINS, C), (DF_BINS, 10))?,
                df_embed: vec![0.0; 256],
                df_skip_buf: vec![0.0; 256],
                coefs: vec![0.0; COEFS],
            },
            output: Output {
                wnorm,
                filter: DeepFilter::new(),
                out: vec![0.0; BINS * 2],
            },
        };
        // Touches every buffer before the first real frame.
        let silence = [0.0f32; BINS * 2];
        require(
            model
                .process_spectrum(&silence, 0.0)
                .iter()
                .all(|v| v.is_finite()),
            "model produced non-finite output during initialization",
        )?;
        model.reset();
        Ok(model)
    }
    /// Input and output: unnormalized FFT, [F, real/imag]. Same convention as official ONNX wrapper.
    /// Every frame is processed, including silence and 0 dB attenuation settings.
    pub fn process_spectrum(&mut self, spec: &[f32], dry_mix: f32) -> &[f32] {
        assert_eq!(spec.len(), BINS * 2);
        let front = &mut self.front;
        front.run(spec);
        self.enc_erb_fc.apply(
            self.erb_dual.forward(front.erb3.output()),
            &mut self.joined[..512],
        );
        self.enc_df_fc.apply(
            self.df_dual.forward(front.df1.output()),
            &mut self.joined[512..],
        );
        self.enc_gru.forward(&self.joined);
        let up2 = self.mask_high.run(
            self.enc_gru.output(),
            front.erb3.output(),
            front.erb2.output(),
        );
        let mask = self
            .mask_low
            .run(up2, front.erb1.output(), front.erb0.output());
        let coefs = self
            .df_decoder
            .run(self.enc_gru.output(), front.df0.output());
        self.output.run(&front.scaled, mask, coefs, dry_mix)
    }
    pub fn reset(&mut self) {
        self.front.reset();
        self.erb_dual.reset();
        self.df_dual.reset();
        self.enc_gru.reset();
        self.mask_high.erb_gru.reset();
        self.df_decoder.reset();
        self.output.reset();
    }
    /// The last frame's intermediate values, numbered as in include/dpdfnet_native.h.
    /// Read from the processing thread.
    pub fn trace(&self, id: usize) -> Option<&[f32]> {
        let front = &self.front;
        match id {
            0 => Some(&front.mag),
            1 => Some(&front.complex),
            2 => Some(front.erb0.output()),
            3 => Some(front.erb1.output()),
            4 => Some(front.erb2.output()),
            5 => Some(front.erb3.output()),
            6 => Some(front.df0.output()),
            7 => Some(front.df1.output()),
            8 => Some(self.erb_dual.output()),
            9 => Some(self.df_dual.output()),
            10 => Some(self.enc_gru.output()),
            11 => Some(&self.mask_low.mask),
            12 => Some(&self.df_decoder.coefs),
            13 => Some(&self.output.out),
            _ => None,
        }
    }
}
/// One hop on its way through the stages of [`Model::into_stages`]: every
/// value a later stage reads from an earlier one.
pub(crate) struct Frame {
    /// The analysis spectrum on the way in, the enhanced spectrum on the way out.
    pub(crate) spec: Vec<f32>,
    pub(crate) dry_mix: f32,
    scaled: Vec<f32>,
    e: [Vec<f32>; 4],
    d0: Vec<f32>,
    d1: Vec<f32>,
    erb_x: Vec<f32>,
    df_x: Vec<f32>,
    joined: Vec<f32>,
    enc: Vec<f32>,
    coefs: Vec<f32>,
    up2: Vec<f32>,
}
impl Default for Frame {
    fn default() -> Self {
        Self {
            spec: vec![0.0; BINS * 2],
            dry_mix: 0.0,
            scaled: vec![0.0; BINS * 2],
            e: [
                vec![0.0; 480 * C],
                vec![0.0; 160 * C],
                vec![0.0; 80 * C],
                vec![0.0; 40 * C],
            ],
            d0: vec![0.0; DF_BINS * C],
            d1: vec![0.0; 48 * C],
            erb_x: vec![0.0; 40 * C],
            df_x: vec![0.0; 48 * C],
            joined: vec![0.0; 1024],
            enc: vec![0.0; 512],
            coefs: vec![0.0; COEFS],
            up2: vec![0.0; 160 * C],
        }
    }
}

/// A part of the network whose state depends only on its own previous hops,
/// so consecutive hops can be in different stages at the same time. Running
/// the stages in order on one frame computes exactly `process_spectrum`.
pub(crate) enum Stage {
    Front(Box<Front>),
    Block {
        block: Box<DprnnBlock>,
        erb: bool,
        first: bool,
        /// The branch projection, on the branch's last block.
        fc: Option<Linear>,
    },
    Decoder {
        enc_gru: Box<Squeezed>,
        df: Box<DfDecoder>,
    },
    MaskHigh(Box<MaskHigh>),
    MaskLow(Box<MaskLow>, Box<Output>),
}
impl Stage {
    pub(crate) fn run(&mut self, f: &mut Frame) {
        match self {
            Stage::Front(front) => {
                front.run(&f.spec);
                f.scaled.copy_from_slice(&front.scaled);
                let convs = [&front.erb0, &front.erb1, &front.erb2, &front.erb3];
                for (dst, conv) in f.e.iter_mut().zip(convs) {
                    dst.copy_from_slice(conv.output());
                }
                f.d0.copy_from_slice(front.df0.output());
                f.d1.copy_from_slice(front.df1.output());
            }
            Stage::Block {
                block,
                erb,
                first,
                fc,
            } => {
                let input = match (*erb, *first) {
                    (true, true) => &f.e[3],
                    (true, false) => &f.erb_x,
                    (false, true) => &f.d1,
                    (false, false) => &f.df_x,
                };
                let out = block.forward(input);
                match (fc, *erb) {
                    (Some(fc), true) => fc.apply(out, &mut f.joined[..512]),
                    (Some(fc), false) => fc.apply(out, &mut f.joined[512..]),
                    (None, true) => f.erb_x.copy_from_slice(out),
                    (None, false) => f.df_x.copy_from_slice(out),
                }
            }
            Stage::Decoder { enc_gru, df } => {
                f.enc.copy_from_slice(enc_gru.forward(&f.joined));
                f.coefs.copy_from_slice(df.run(&f.enc, &f.d0));
            }
            Stage::MaskHigh(mask) => {
                f.up2.copy_from_slice(mask.run(&f.enc, &f.e[3], &f.e[2]));
            }
            Stage::MaskLow(mask, output) => {
                let mask = mask.run(&f.up2, &f.e[1], &f.e[0]);
                f.spec
                    .copy_from_slice(output.run(&f.scaled, mask, &f.coefs, f.dry_mix));
            }
        }
    }
    /// Microseconds per hop on an i5-13400 P-core, for grouping stages onto threads.
    pub(crate) fn cost(&self) -> u32 {
        match self {
            Stage::Front(_) => 74,
            Stage::Block { erb: true, .. } => 85,
            Stage::Block { erb: false, .. } => 100,
            Stage::Decoder { .. } => 60,
            Stage::MaskHigh(_) => 54,
            Stage::MaskLow(..) => 75,
        }
    }
}
impl Model {
    /// The network as stages in data order, each owning its layers and state.
    pub(crate) fn into_stages(self) -> Vec<Stage> {
        let mut stages = vec![Stage::Front(Box::new(self.front))];
        for (dual, erb, fc) in [
            (self.erb_dual, true, self.enc_erb_fc),
            (self.df_dual, false, self.enc_df_fc),
        ] {
            let blocks = dual.into_blocks();
            let last = blocks.len() - 1;
            let mut fc = Some(fc);
            for (i, block) in blocks.into_iter().enumerate() {
                stages.push(Stage::Block {
                    block: Box::new(block),
                    erb,
                    first: i == 0,
                    fc: if i == last { fc.take() } else { None },
                });
            }
        }
        stages.push(Stage::Decoder {
            enc_gru: Box::new(self.enc_gru),
            df: Box::new(self.df_decoder),
        });
        stages.push(Stage::MaskHigh(Box::new(self.mask_high)));
        stages.push(Stage::MaskLow(
            Box::new(self.mask_low),
            Box::new(self.output),
        ));
        stages
    }
}

fn add(y: &mut [f32], a: &[f32], b: &[f32]) {
    for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
        *y = a + b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn before_df_alignment_identity() {
        let mut d = DeepFilter::new();
        let m = vec![1.0; BINS];
        let mut c = vec![0.0; COEFS];
        for f in 0..DF_BINS {
            c[f * 10 + 4] = 1.0;
        } // center tap; coefficients start zero for 2 frames
        let mut out = vec![0.0; BINS * 2];
        for n in 1i32..20 {
            let x = vec![n as f32; BINS * 2];
            d.apply(&x, &m, &c, &mut out, 0.0);
            if n >= 5 {
                for y in &out {
                    assert_eq!(*y, (n - 4) as f32);
                }
            }
        }
    }
    #[test]
    fn dry_reference_delayed_four_hops() {
        let mut d = DeepFilter::new();
        let m = vec![0.0; BINS];
        let c = vec![0.0; COEFS];
        let mut y = vec![0.0; BINS * 2];
        for n in 1i32..20 {
            d.apply(&vec![n as f32; BINS * 2], &m, &c, &mut y, 1.0);
            assert_eq!(y[20], (n - 4).max(0) as f32);
        }
    }
}
