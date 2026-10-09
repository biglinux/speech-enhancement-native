//! GTCRN-AEC core forward, a scalar Rust port of LocalVQE `gtcrn.cpp`. Input: two spectra `spec_e` (near/error) and `spec_y`
//! (far reference), each `(257, T, 2)` row-major; output: masked `spec_e`.
//!
//! GGUF stores tensor dims fastest-first, i.e. reversed from PyTorch/numpy, but
//! the same byte layout — so a weight slice is indexed with numpy strides while
//! its numpy shape is the GGUF dims reversed (see [`Tensor`]).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // numeric kernels index by design

use crate::gguf::Gguf;
use ops::sigmoid;

const FB: usize = 257;

/// Scratch belongs to one StreamCore, not to the thread running it. The
/// streaming graph has fixed shapes and data-independent temporary lifetimes,
/// so `StreamCore::prepared` records the exact inventory of one frame, resets
/// the recurrent state and seals the workspace; a sealed workspace never
/// allocates. The batch forward uses an unsealed local workspace.
#[derive(Default)]
struct Workspace {
    free: std::cell::RefCell<std::collections::BTreeMap<usize, Vec<Vec<f32>>>>,
    sealed: std::cell::Cell<bool>,
}

impl Workspace {
    fn take(&self, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let mut free = self.free.borrow_mut();
        if let Some(v) = free.get_mut(&n).and_then(Vec::pop) {
            // The shape bucket stores fully sized vectors; no resize in RT.
            return v;
        }
        assert!(
            !self.sealed.get(),
            "unprepared GTCRN workspace shape/lifetime"
        );
        free.entry(n).or_default();
        vec![0.0; n]
    }

    fn put(&self, v: Vec<f32>) {
        if v.is_empty() {
            return;
        }
        let mut free = self.free.borrow_mut();
        let bucket = free.get_mut(&v.len()).expect("workspace-owned buffer");
        // At seal every temporary has returned. Subsequent frames have the
        // same liveness graph, so even the outer Vec never needs to grow.
        assert!(!self.sealed.get() || bucket.len() < bucket.capacity());
        bucket.push(v);
    }

    fn seal(&self) {
        self.sealed.set(true);
    }
}

/// Dense (C,T,F), row-major. Temporaries cannot outlive their instance's
/// workspace.
pub struct GTensor<'a> {
    pub c: usize,
    pub t: usize,
    pub f: usize,
    pub d: Vec<f32>,
    arena: &'a Workspace,
}
impl Drop for GTensor<'_> {
    fn drop(&mut self) {
        self.arena.put(std::mem::take(&mut self.d));
    }
}

struct Scratch<'a> {
    data: Vec<f32>,
    arena: &'a Workspace,
}
impl<'a> Scratch<'a> {
    fn zeros(arena: &'a Workspace, n: usize) -> Self {
        let mut data = arena.take(n);
        data.fill(0.0);
        Self { data, arena }
    }
}
impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.arena.put(std::mem::take(&mut self.data));
    }
}
impl std::ops::Deref for Scratch<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.data
    }
}
impl std::ops::DerefMut for Scratch<'_> {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl<'a> GTensor<'a> {
    fn new(arena: &'a Workspace, c: usize, t: usize, f: usize) -> Self {
        let mut d = arena.take(c * t * f);
        d.fill(0.0);
        Self { c, t, f, d, arena }
    }
    #[inline]
    fn at(&self, c: usize, t: usize, f: usize) -> f32 {
        self.d[(c * self.t + t) * self.f + f]
    }
    #[inline]
    fn set(&mut self, c: usize, t: usize, f: usize, v: f32) {
        self.d[(c * self.t + t) * self.f + f] = v;
    }
}

/// One weight tensor; `dims` in numpy order (GGUF order reversed).
struct Tensor {
    data: Box<[f32]>,
    dims: Vec<usize>,
}

/// Convolution weights; `prelu` is empty for the last decoder block (tanh).
struct Conv {
    w: Tensor,
    b: Box<[f32]>,
    prelu: Box<[f32]>,
}

/// GRU weights, PyTorch layout, gate order [r,z,n].
struct Gru {
    wih: Box<[f32]>,
    whh: Box<[f32]>,
    bih: Box<[f32]>,
    bhh: Box<[f32]>,
    hidden: usize,
}

struct Linear {
    w: Box<[f32]>,
    b: Box<[f32]>,
    out: usize,
}

struct Norm {
    w: Box<[f32]>,
    b: Box<[f32]>,
}

/// GTConv block: pointwise, depthwise, pointwise, temporal GRU attention.
struct GtBlock {
    pc1: Conv,
    dw: Conv,
    pc2: Conv,
    gru: Gru,
    fc: Linear,
}

/// Grouped RNN: two GRUs over the halves of the input, `rev` when bidirectional.
struct Grnn {
    fwd: [Gru; 2],
    rev: Option<[Gru; 2]>,
}

struct DpGrnn {
    intra: Grnn,
    intra_fc: Linear,
    intra_ln: Norm,
    inter: Grnn,
    inter_fc: Linear,
    inter_ln: Norm,
}

/// The network's weights, moved out of the GGUF at load so no frame looks a
/// tensor up by name.
pub struct Net {
    erb_bm: Box<[f32]>,
    erb_bs: Box<[f32]>,
    en0: Conv,
    en1: Conv,
    /// Encoder blocks 2–4, then decoder blocks 0–2, dilated by [`GT_DILATIONS`].
    gt: [GtBlock; 6],
    dp: [DpGrnn; 2],
    de3: Conv,
    de4: Conv,
}

const GT_DILATIONS: [usize; 6] = [1, 2, 5, 5, 2, 1];

impl Net {
    /// Moves the network tensors out of `g`, which `Model::load` has checked
    /// against the shipped model's schema.
    pub(crate) fn take(g: &mut Gguf) -> Result<Self, String> {
        Ok(Self {
            erb_bm: take_data(g, "erb.bm".into())?,
            erb_bs: take_data(g, "erb.bs".into())?,
            en0: conv_weights(g, "encoder.en_convs.0", true)?,
            en1: conv_weights(g, "encoder.en_convs.1", true)?,
            gt: [
                gt_block_weights(g, "encoder.en_convs.2")?,
                gt_block_weights(g, "encoder.en_convs.3")?,
                gt_block_weights(g, "encoder.en_convs.4")?,
                gt_block_weights(g, "decoder.de_convs.0")?,
                gt_block_weights(g, "decoder.de_convs.1")?,
                gt_block_weights(g, "decoder.de_convs.2")?,
            ],
            dp: [dpgrnn_weights(g, "dpgrnn1")?, dpgrnn_weights(g, "dpgrnn2")?],
            de3: conv_weights(g, "decoder.de_convs.3", true)?,
            de4: conv_weights(g, "decoder.de_convs.4", false)?,
        })
    }
}

fn take_tensor(g: &mut Gguf, name: String) -> Result<Tensor, String> {
    let (data, mut dims) = g
        .take(&name)
        .ok_or_else(|| format!("model missing tensor {name}"))?;
    dims.reverse();
    Ok(Tensor { data, dims })
}

fn take_data(g: &mut Gguf, name: String) -> Result<Box<[f32]>, String> {
    Ok(take_tensor(g, name)?.data)
}

fn conv_weights(g: &mut Gguf, p: &str, prelu: bool) -> Result<Conv, String> {
    Ok(Conv {
        w: take_tensor(g, format!("{p}.w"))?,
        b: take_data(g, format!("{p}.b"))?,
        prelu: if prelu {
            take_data(g, format!("{p}.prelu"))?
        } else {
            Box::default()
        },
    })
}

fn gru_weights(g: &mut Gguf, p: &str, suffix: &str) -> Result<Gru, String> {
    let whh = take_tensor(g, format!("{p}.weight_hh_l0{suffix}"))?;
    Ok(Gru {
        hidden: whh.dims[1], // numpy (3H, H)
        whh: whh.data,
        wih: take_data(g, format!("{p}.weight_ih_l0{suffix}"))?,
        bih: take_data(g, format!("{p}.bias_ih_l0{suffix}"))?,
        bhh: take_data(g, format!("{p}.bias_hh_l0{suffix}"))?,
    })
}

fn linear_weights(g: &mut Gguf, p: &str) -> Result<Linear, String> {
    let w = take_tensor(g, format!("{p}.w"))?;
    Ok(Linear {
        out: w.dims[0], // numpy (O, I)
        w: w.data,
        b: take_data(g, format!("{p}.b"))?,
    })
}

fn norm_weights(g: &mut Gguf, p: &str) -> Result<Norm, String> {
    Ok(Norm {
        w: take_data(g, format!("{p}.w"))?,
        b: take_data(g, format!("{p}.b"))?,
    })
}

fn gt_block_weights(g: &mut Gguf, p: &str) -> Result<GtBlock, String> {
    Ok(GtBlock {
        pc1: conv_weights(g, &format!("{p}.pc1"), true)?,
        dw: conv_weights(g, &format!("{p}.dw"), true)?,
        pc2: conv_weights(g, &format!("{p}.pc2"), false)?,
        gru: gru_weights(g, &format!("{p}.tra.gru"), "")?,
        fc: linear_weights(g, &format!("{p}.tra.fc"))?,
    })
}

fn grnn_weights(g: &mut Gguf, p: &str, bidir: bool) -> Result<Grnn, String> {
    let (r1, r2) = (format!("{p}.rnn1"), format!("{p}.rnn2"));
    Ok(Grnn {
        fwd: [gru_weights(g, &r1, "")?, gru_weights(g, &r2, "")?],
        rev: if bidir {
            Some([
                gru_weights(g, &r1, "_reverse")?,
                gru_weights(g, &r2, "_reverse")?,
            ])
        } else {
            None
        },
    })
}

fn dpgrnn_weights(g: &mut Gguf, p: &str) -> Result<DpGrnn, String> {
    Ok(DpGrnn {
        intra: grnn_weights(g, &format!("{p}.intra_rnn"), true)?,
        intra_fc: linear_weights(g, &format!("{p}.intra_fc"))?,
        intra_ln: norm_weights(g, &format!("{p}.intra_ln"))?,
        inter: grnn_weights(g, &format!("{p}.inter_rnn"), false)?,
        inter_fc: linear_weights(g, &format!("{p}.inter_fc"))?,
        inter_ln: norm_weights(g, &format!("{p}.inter_ln"))?,
    })
}

#[allow(clippy::too_many_arguments)]
fn conv2d<'a>(
    arena: &'a Workspace,
    x: &GTensor,
    w: &[f32],
    wd: &[usize],
    b: &[f32],
    groups: usize,
    stride_f: usize,
    pad_f: i64,
    pad_t_top: i64,
    pad_t_bot: i64,
    dil_t: i64,
    dil_f: i64,
) -> GTensor<'a> {
    let (oc, ing, kt, kf) = (wd[0], wd[1], wd[2], wd[3]); // numpy OC,ING,KT,KF
    let (ic, tin, fin) = (x.c, x.t as i64, x.f as i64);
    let tout = (tin + pad_t_top + pad_t_bot - dil_t * (kt as i64 - 1)) as usize;
    let fout = ((fin + 2 * pad_f - dil_f * (kf as i64 - 1) - 1) / stride_f as i64 + 1) as usize;
    let (opg, ipg) = (oc / groups, ic / groups);
    let mut y = GTensor::new(arena, oc, tout, fout);
    // Pointwise 1x1 (pc1/pc2): out[co] = bias + Σ_ci w[co,ci]·in[ci], over the whole
    // (t,f) plane in one kernel call.
    if groups == 1
        && kt == 1
        && kf == 1
        && pad_f == 0
        && pad_t_top == 0
        && pad_t_bot == 0
        && stride_f == 1
        && !b.is_empty()
    {
        ops::pointwise_conv2d(&mut y.d, &x.d, w, b, ic, oc, x.t * x.f);
        return y;
    }
    let (xt, xf) = (x.t, x.f);
    // Same accumulation order as the scalar reference — for a fixed output (co,t,f)
    // the contributions are still summed in (ci, k, kfi) order — but with the freq
    // index f as the innermost, contiguous loop so it autovectorizes and the pad /
    // bounds tests are hoisted out of the hot path. The stride_f==1 && dil_f==1 case
    // (every gt_block / pointwise / depthwise call) becomes a slice multiply-add.
    let fast = stride_f == 1 && dil_f == 1;
    // Strided/dilated (enc0/enc1, stride 2): the valid f-range of each tap and its
    // gathered input run depend on (cin, ti, kfi), not on the output channel, so
    // gather each run once into a contiguous scratch row shared by every co. The
    // accumulation below is then a contiguous multiply-add that vectorizes.
    let s = stride_f as i64;
    let tap_range = |kfi: usize| {
        let off = kfi as i64 * dil_f - pad_f;
        let f0 = if off >= 0 {
            0
        } else {
            ((-off + s - 1) / s) as usize
        };
        let hi = fin - off;
        let f1 = if hi <= 0 {
            0
        } else {
            (((hi - 1) / s + 1) as usize).min(fout)
        };
        (off, f0, f1)
    };
    let mut gathered = GTensor::new(arena, if fast { 0 } else { ic * kf }, xt, fout);
    if !fast {
        for cin in 0..ic {
            for ti in 0..xt {
                let xrow = &x.d[(cin * xt + ti) * xf..(cin * xt + ti) * xf + xf];
                for kfi in 0..kf {
                    let (off, f0, f1) = tap_range(kfi);
                    let row = ((cin * kf + kfi) * xt + ti) * fout;
                    for f in f0..f1 {
                        gathered.d[row + f] = xrow[(f as i64 * s + off) as usize];
                    }
                }
            }
        }
    }
    for g in 0..groups {
        for o in 0..opg {
            let co = g * opg + o;
            let bias = if b.is_empty() { 0.0 } else { b[co] };
            for t in 0..tout {
                let orow = &mut y.d[(co * tout + t) * fout..(co * tout + t) * fout + fout];
                orow.fill(bias);
                for ci in 0..ing {
                    let cin = g * ipg + ci;
                    for k in 0..kt {
                        let ti = t as i64 + k as i64 * dil_t - pad_t_top;
                        if ti < 0 || ti >= tin {
                            continue;
                        }
                        let xrow =
                            &x.d[(cin * xt + ti as usize) * xf..(cin * xt + ti as usize) * xf + xf];
                        for kfi in 0..kf {
                            let wt = w[((co * ing + ci) * kt + k) * kf + kfi];
                            if fast {
                                // fi = f + (kfi - pad_f): the overlapping run is one
                                // contiguous AXPY, with no per-element bounds check.
                                let shift = kfi as i64 - pad_f;
                                let f0 = (-shift).max(0) as usize;
                                let f1 = (fin - shift).min(fout as i64).max(0) as usize;
                                if f1 > f0 {
                                    let xs = (f0 as i64 + shift) as usize;
                                    ops::axpy_f32(&mut orow[f0..f1], &xrow[xs..xs + (f1 - f0)], wt);
                                }
                            } else {
                                // strided/dilated: contiguous run gathered above,
                                // same per-element multiply then add, same order.
                                let (_, f0, f1) = tap_range(kfi);
                                let row = ((cin * kf + kfi) * xt + ti as usize) * fout;
                                for (o, &v) in
                                    orow[f0..f1].iter_mut().zip(&gathered.d[row + f0..row + f1])
                                {
                                    *o += wt * v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    y
}

fn prelu_(x: &mut GTensor, slope: &[f32]) {
    let scalar = slope.len() == 1;
    for c in 0..x.c {
        let a = if scalar { slope[0] } else { slope[c] };
        for t in 0..x.t {
            for f in 0..x.f {
                let v = x.at(c, t, f);
                if v < 0.0 {
                    x.set(c, t, f, v * a);
                }
            }
        }
    }
}

fn tanh_inplace(x: &mut GTensor) {
    for v in &mut x.d {
        *v = v.tanh();
    }
}

/// 3-tap freq unfold: out channel `c*3+{0,1,2}` = {f-1, f, f+1} (zero pad). Each
/// output channel is the input freq-row shifted by one bin: three shifted copies.
fn sfe<'a>(arena: &'a Workspace, x: &GTensor) -> GTensor<'a> {
    let f = x.f;
    let mut y = GTensor::new(arena, x.c * 3, x.t, f);
    let yt = y.t;
    for c in 0..x.c {
        for t in 0..x.t {
            let xbase = (c * x.t + t) * f;
            let xrow = &x.d[xbase..xbase + f];
            let b0 = ((c * 3) * yt + t) * f;
            let b1 = ((c * 3 + 1) * yt + t) * f;
            let b2 = ((c * 3 + 2) * yt + t) * f;
            // c*3   : shifted right (out[fi] = x[fi-1]), x[-1]=0
            y.d[b0 + 1..b0 + f].copy_from_slice(&xrow[..f - 1]);
            // c*3+1 : identity
            y.d[b1..b1 + f].copy_from_slice(xrow);
            // c*3+2 : shifted left (out[fi] = x[fi+1]), x[f]=0
            y.d[b2..b2 + f - 1].copy_from_slice(&xrow[1..]);
        }
    }
    y
}

fn erb_bm<'a>(arena: &'a Workspace, x: &GTensor, bmw: &[f32]) -> GTensor<'a> {
    use crate::erb::{BANDED, FULL, HIGH_ERB, HIGH_FULL, LOW};
    let mut y = GTensor::new(arena, x.c, x.t, BANDED);
    for c in 0..x.c {
        for t in 0..x.t {
            let xbase = (c * x.t + t) * x.f;
            let xrow = &x.d[xbase..xbase + FULL];
            let ybase = (c * y.t + t) * BANDED;
            y.d[ybase..ybase + LOW].copy_from_slice(&xrow[..LOW]);
            let xhi = &xrow[LOW..LOW + HIGH_FULL];
            for j in 0..HIGH_ERB {
                let row = &bmw[j * HIGH_FULL..(j + 1) * HIGH_FULL];
                y.d[ybase + LOW + j] = ops::vdot_f32(xhi, row);
            }
        }
    }
    y
}

fn erb_bs<'a>(arena: &'a Workspace, x: &GTensor, bsw: &[f32]) -> GTensor<'a> {
    use crate::erb::{BANDED, FULL, HIGH_ERB, HIGH_FULL, LOW};
    let mut y = GTensor::new(arena, x.c, x.t, FULL);
    for c in 0..x.c {
        for t in 0..x.t {
            let xbase = (c * x.t + t) * x.f;
            let xrow = &x.d[xbase..xbase + BANDED];
            let ybase = (c * y.t + t) * FULL;
            y.d[ybase..ybase + LOW].copy_from_slice(&xrow[..LOW]);
            let xhi = &xrow[LOW..LOW + HIGH_ERB];
            for j in 0..HIGH_FULL {
                let row = &bsw[j * HIGH_ERB..(j + 1) * HIGH_ERB];
                y.d[ybase + LOW + j] = ops::vdot_f32(xhi, row);
            }
        }
    }
    y
}

/// One GRU step over `x`, carrying `h` in place (PyTorch layout, gate order
/// [r,z,n]). Shared with the DAF controllers.
pub(crate) fn gru_cell(
    x: &[f32],
    h: &mut [f32],
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    let (nin, nh) = (x.len(), h.len());
    // The shipped schema bounds every GRU hidden dimension by 16.
    assert!(nh <= 16, "GRU hidden size exceeds the fixed scratch");
    let mut gi = [0.0f32; 3 * 16];
    let mut gh = [0.0f32; 3 * 16];
    for g in 0..3 * nh {
        let mut s = bih[g];
        let w = &wih[g * nin..g * nin + nin];
        for i in 0..nin {
            s += w[i] * x[i];
        }
        gi[g] = s;
        let mut t = bhh[g];
        let v = &whh[g * nh..g * nh + nh];
        for i in 0..nh {
            t += v[i] * h[i];
        }
        gh[g] = t;
    }
    for i in 0..nh {
        let r = sigmoid(gi[i] + gh[i]);
        let z = sigmoid(gi[nh + i] + gh[nh + i]);
        let n = (gi[2 * nh + i] + r * gh[2 * nh + i]).tanh();
        h[i] = (1.0 - z) * n + z * h[i];
    }
}

/// GRU over a length-`l` sequence of `i_dim` features, from a zero state.
fn gru_seq<'a>(
    arena: &'a Workspace,
    x: &[f32],
    l: usize,
    i_dim: usize,
    g: &Gru,
    reverse: bool,
) -> Scratch<'a> {
    let h_dim = g.hidden;
    let mut out = Scratch::zeros(arena, l * h_dim);
    let mut h_buf = [0.0f32; 16];
    let h = &mut h_buf[..h_dim];
    for s in 0..l {
        let ti = if reverse { l - 1 - s } else { s };
        let xt = &x[ti * i_dim..ti * i_dim + i_dim];
        gru_cell(xt, h, &g.wih, &g.whh, &g.bih, &g.bhh);
        out[ti * h_dim..ti * h_dim + h_dim].copy_from_slice(h);
    }
    out
}

/// Grouped RNN over `n` sequences of `seq` steps: the `i_dim` features are
/// split in half for the two GRUs, each optionally bidirectional.
fn grnn<'a>(
    arena: &'a Workspace,
    x: &[f32],
    n: usize,
    seq: usize,
    i_dim: usize,
    r: &Grnn,
) -> Scratch<'a> {
    let half = i_dim / 2;
    let dirs = if r.rev.is_some() { 2 } else { 1 };
    let (h1, h2) = (r.fwd[0].hidden, r.fwd[1].hidden);
    let iout = dirs * (h1 + h2);
    let mut y = Scratch::zeros(arena, n * seq * iout);
    let mut sub = Scratch::zeros(arena, seq * half);
    for ni in 0..n {
        for rnn in 0..2 {
            let feat_off = rnn * half;
            let out_off = rnn * dirs * h1;
            let hh = r.fwd[rnn].hidden;
            for s in 0..seq {
                for i in 0..half {
                    sub[s * half + i] = x[(ni * seq + s) * i_dim + feat_off + i];
                }
            }
            let yf = gru_seq(arena, &sub, seq, half, &r.fwd[rnn], false);
            let yr = match &r.rev {
                Some(rev) => gru_seq(arena, &sub, seq, half, &rev[rnn], true),
                None => Scratch::zeros(arena, 0),
            };
            for s in 0..seq {
                let dst = (ni * seq + s) * iout + out_off;
                y[dst..dst + hh].copy_from_slice(&yf[s * hh..s * hh + hh]);
                if !yr.is_empty() {
                    y[dst + hh..dst + 2 * hh].copy_from_slice(&yr[s * hh..s * hh + hh]);
                }
            }
        }
    }
    y
}

/// Linear over last dim: x (M,I) -> (M,O), w numpy (O,I).
fn linear<'a>(arena: &'a Workspace, x: &[f32], m: usize, l: &Linear) -> Scratch<'a> {
    let (i_dim, o_dim) = (l.w.len() / l.out, l.out);
    let mut y = Scratch::zeros(arena, m * o_dim);
    for mi in 0..m {
        let xr = &x[mi * i_dim..mi * i_dim + i_dim];
        for o in 0..o_dim {
            let wr = &l.w[o * i_dim..o * i_dim + i_dim];
            y[mi * o_dim + o] = l.b[o] + ops::vdot_f32(wr, xr);
        }
    }
    y
}

/// LayerNorm over the joint (F,C) tail per time frame.
fn layernorm_last2(x: &mut [f32], t: usize, fd: usize, c: usize, ln: &Norm) {
    let n = fd * c;
    let eps = 1e-8f32;
    for ti in 0..t {
        let xt = &mut x[ti * n..ti * n + n];
        let mu = xt.iter().sum::<f32>() / n as f32;
        let var = xt.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            xt[i] = (xt[i] - mu) * inv * ln.w[i] + ln.b[i];
        }
    }
}

/// Temporal recurrent attention gate.
fn tra<'a>(arena: &'a Workspace, x: &GTensor, g: &GtBlock) -> GTensor<'a> {
    let (c, t, fd) = (x.c, x.t, x.f);
    let mut seq = vec![0.0f32; t * c];
    for ti in 0..t {
        for ci in 0..c {
            let mut s = 0.0;
            for f in 0..fd {
                let v = x.at(ci, ti, f);
                s += v * v;
            }
            seq[ti * c + ci] = s / fd as f32;
        }
    }
    let y = gru_seq(arena, &seq, t, c, &g.gru, false);
    let at = linear(arena, &y, t, &g.fc);
    let mut out = GTensor::new(arena, x.c, x.t, x.f);
    out.d.copy_from_slice(&x.d);
    let ot = out.t;
    for ti in 0..t {
        for ci in 0..c {
            let g = sigmoid(at[ti * c + ci]);
            let base = (ci * ot + ti) * fd;
            for v in &mut out.d[base..base + fd] {
                *v *= g;
            }
        }
    }
    out
}

fn shuffle2<'a>(arena: &'a Workspace, h: &GTensor, x2: &GTensor) -> GTensor<'a> {
    let f = h.f;
    let mut y = GTensor::new(arena, h.c * 2, h.t, f);
    let yt = y.t;
    for c in 0..h.c {
        for t in 0..h.t {
            let hb = (c * h.t + t) * f;
            let xb = (c * x2.t + t) * f;
            let yb0 = ((2 * c) * yt + t) * f;
            let yb1 = ((2 * c + 1) * yt + t) * f;
            y.d[yb0..yb0 + f].copy_from_slice(&h.d[hb..hb + f]);
            y.d[yb1..yb1 + f].copy_from_slice(&x2.d[xb..xb + f]);
        }
    }
    y
}

fn chunk<'a>(arena: &'a Workspace, x: &GTensor, second: bool) -> GTensor<'a> {
    let f = x.f;
    let mut y = GTensor::new(arena, x.c / 2, x.t, f);
    let off = if second { x.c / 2 } else { 0 };
    let yt = y.t;
    for c in 0..y.c {
        for t in 0..x.t {
            let xb = ((off + c) * x.t + t) * f;
            let yb = (c * yt + t) * f;
            y.d[yb..yb + f].copy_from_slice(&x.d[xb..xb + f]);
        }
    }
    y
}

fn upsample_zero_f<'a>(arena: &'a Workspace, x: &GTensor, factor: usize) -> GTensor<'a> {
    let fout = (x.f - 1) * factor + 1;
    let mut y = GTensor::new(arena, x.c, x.t, fout);
    for c in 0..x.c {
        for t in 0..x.t {
            for f in 0..x.f {
                y.set(c, t, f * factor, x.at(c, t, f));
            }
        }
    }
    y
}

fn add<'a>(arena: &'a Workspace, a: &GTensor, b: &GTensor) -> GTensor<'a> {
    // From the workspace, so a skip connection never allocates.
    let mut y = GTensor::new(arena, a.c, a.t, a.f);
    for ((yi, ai), bi) in y.d.iter_mut().zip(&a.d).zip(&b.d) {
        *yi = ai + bi;
    }
    y
}

/// `conv2d` with the weights and bias of `c`.
fn conv<'a>(
    arena: &'a Workspace,
    x: &GTensor,
    c: &Conv,
    groups: usize,
    stride_f: usize,
    pad_f: i64,
    pad_t_top: i64,
    dil_t: i64,
) -> GTensor<'a> {
    conv2d(
        arena, x, &c.w.data, &c.w.dims, &c.b, groups, stride_f, pad_f, pad_t_top, 0, dil_t, 1,
    )
}

fn gt_block<'a>(arena: &'a Workspace, x: &GTensor, g: &GtBlock, dil: i64) -> GTensor<'a> {
    let x1 = chunk(arena, x, false);
    let x2 = chunk(arena, x, true);
    let s = sfe(arena, &x1);
    let mut h = conv(arena, &s, &g.pc1, 1, 1, 0, 0, 1);
    prelu_(&mut h, &g.pc1.prelu);
    let hidden = h.c;
    h = conv(arena, &h, &g.dw, hidden, 1, 1, 2 * dil, dil);
    prelu_(&mut h, &g.dw.prelu);
    h = conv(arena, &h, &g.pc2, 1, 1, 0, 0, 1);
    let h = tra(arena, &h, g);
    shuffle2(arena, &h, &x2)
}

fn conv_block<'a>(
    arena: &'a Workspace,
    x: &GTensor,
    c: &Conv,
    groups: usize,
    stride_f: usize,
    deconv: bool,
    is_last: bool,
) -> GTensor<'a> {
    let mut y = if !deconv {
        conv(arena, x, c, groups, stride_f, 2, 0, 1)
    } else {
        let xu = upsample_zero_f(arena, x, stride_f);
        conv(arena, &xu, c, groups, 1, 2, 0, 1)
    };
    if is_last {
        tanh_inplace(&mut y);
    } else {
        prelu_(&mut y, &c.prelu);
    }
    y
}

fn dpgrnn<'a>(arena: &'a Workspace, x: &GTensor, d: &DpGrnn) -> GTensor<'a> {
    let (c, t, fd) = (x.c, x.t, x.f);
    let mut xp = vec![0.0f32; t * fd * c];
    for ti in 0..t {
        for f in 0..fd {
            for ci in 0..c {
                xp[(ti * fd + f) * c + ci] = x.at(ci, ti, f);
            }
        }
    }
    let mut intra = grnn(arena, &xp, t, fd, c, &d.intra);
    intra = linear(arena, &intra, t * fd, &d.intra_fc);
    layernorm_last2(&mut intra, t, fd, c, &d.intra_ln);
    for i in 0..intra.len() {
        intra[i] += xp[i];
    }

    let mut xq = vec![0.0f32; fd * t * c];
    for f in 0..fd {
        for ti in 0..t {
            for ci in 0..c {
                xq[(f * t + ti) * c + ci] = intra[(ti * fd + f) * c + ci];
            }
        }
    }
    let mut inter = grnn(arena, &xq, fd, t, c, &d.inter);
    inter = linear(arena, &inter, fd * t, &d.inter_fc);
    let mut interp = vec![0.0f32; t * fd * c];
    for f in 0..fd {
        for ti in 0..t {
            for ci in 0..c {
                interp[(ti * fd + f) * c + ci] = inter[(f * t + ti) * c + ci];
            }
        }
    }
    layernorm_last2(&mut interp, t, fd, c, &d.inter_ln);
    for i in 0..interp.len() {
        interp[i] += intra[i];
    }

    let mut out = GTensor::new(arena, c, t, fd);
    for ti in 0..t {
        for f in 0..fd {
            for ci in 0..c {
                out.set(ci, ti, f, interp[(ti * fd + f) * c + ci]);
            }
        }
    }
    out
}

/// One spectrum `(257,T,2)` -> feature `(9,T,129)`: [mag,re,im] -> ERB merge -> SFE.
fn feat<'a>(arena: &'a Workspace, spec: &[f32], t: usize, erb_bm_w: &[f32]) -> GTensor<'a> {
    let mut f3 = GTensor::new(arena, 3, t, FB);
    for ti in 0..t {
        for fb in 0..FB {
            let re = spec[(fb * t + ti) * 2];
            let im = spec[(fb * t + ti) * 2 + 1];
            f3.set(1, ti, fb, re);
            f3.set(2, ti, fb, im);
            f3.set(0, ti, fb, (re * re + im * im + 1e-12).sqrt());
        }
    }
    let banded = erb_bm(arena, &f3, erb_bm_w);
    sfe(arena, &banded)
}

/// Full core forward. `spec_e`/`spec_y` are `(257,T,2)` row-major (freq-major).
/// Returns masked `(257,T,2)`.
#[must_use]
pub fn forward(net: &Net, spec_e: &[f32], spec_y: &[f32], t: usize) -> Vec<f32> {
    forward_stages(net, spec_e, spec_y, t, |_, _| {})
}

/// The named stage tensors of [`forward`], to compare with a reference dump.
#[cfg(test)]
pub(crate) fn forward_capture(
    net: &Net,
    spec_e: &[f32],
    spec_y: &[f32],
    t: usize,
) -> Vec<(&'static str, Vec<f32>)> {
    let mut cap = Vec::new();
    forward_stages(net, spec_e, spec_y, t, |name, v| {
        cap.push((name, v.to_vec()))
    });
    cap
}

fn forward_stages(
    net: &Net,
    spec_e: &[f32],
    spec_y: &[f32],
    t: usize,
    mut stage: impl FnMut(&'static str, &[f32]),
) -> Vec<f32> {
    let storage = Workspace::default();
    let arena = &storage;
    let fe = feat(arena, spec_e, t, &net.erb_bm);
    let fy = feat(arena, spec_y, t, &net.erb_bm);
    let mut ft = GTensor::new(arena, 18, t, 129);
    for c in 0..9 {
        for ti in 0..t {
            for f in 0..129 {
                ft.set(c, ti, f, fe.at(c, ti, f));
                ft.set(9 + c, ti, f, fy.at(c, ti, f));
            }
        }
    }
    stage("feat", &ft.d);
    let en0 = conv_block(arena, &ft, &net.en0, 1, 2, false, false);
    stage("enc0", &en0.d);
    let en1 = conv_block(arena, &en0, &net.en1, 2, 2, false, false);
    stage("enc1", &en1.d);
    let d = GT_DILATIONS.map(|d| d as i64);
    let en2 = gt_block(arena, &en1, &net.gt[0], d[0]);
    stage("enc2", &en2.d);
    let en3 = gt_block(arena, &en2, &net.gt[1], d[1]);
    stage("enc3", &en3.d);
    let en4 = gt_block(arena, &en3, &net.gt[2], d[2]);
    stage("enc4", &en4.d);
    let d1 = dpgrnn(arena, &en4, &net.dp[0]);
    stage("dpgrnn1", &d1.d);
    let d2 = dpgrnn(arena, &d1, &net.dp[1]);
    stage("dpgrnn2", &d2.d);
    let mut x = gt_block(arena, &add(arena, &d2, &en4), &net.gt[3], d[3]);
    stage("dec0", &x.d);
    x = gt_block(arena, &add(arena, &x, &en3), &net.gt[4], d[4]);
    stage("dec1", &x.d);
    x = gt_block(arena, &add(arena, &x, &en2), &net.gt[5], d[5]);
    stage("dec2", &x.d);
    x = conv_block(arena, &add(arena, &x, &en1), &net.de3, 2, 2, true, false);
    stage("dec3", &x.d);
    x = conv_block(arena, &add(arena, &x, &en0), &net.de4, 1, 2, true, true);
    stage("dec4", &x.d);
    let m = erb_bs(arena, &x, &net.erb_bs);
    stage("mask", &m.d);
    let mut out = vec![0.0f32; FB * t * 2];
    for fb in 0..FB {
        for ti in 0..t {
            let er = spec_e[(fb * t + ti) * 2];
            let ei = spec_e[(fb * t + ti) * 2 + 1];
            let mr = m.at(0, ti, fb);
            let mi = m.at(1, ti, fb);
            out[(fb * t + ti) * 2] = er * mr - ei * mi;
            out[(fb * t + ti) * 2 + 1] = ei * mr + er * mi;
        }
    }
    out
}

// Streaming: one frame per call, recurrent state carried across calls.
// `streaming_core_matches_batch` checks it against `forward`.

struct GtRing {
    cap: usize,
    frames: Vec<f32>, // cap * hidden * f, time-major (oldest first)
    tra_h: Vec<f32>,
}

impl GtRing {
    fn new(dil: usize) -> Self {
        Self {
            cap: 2 * dil + 1,
            frames: Vec::new(),
            tra_h: vec![0.0; 16],
        }
    }
}

/// Per-frame streaming core. `process_frame` takes one `(257,1,2)` spec_e/spec_y
/// frame (freq-major `[fb*2 + {re,im}]`) and returns the masked frame, carrying
/// all time-recurrent state (gt-block dw rings + temporal-attention GRUs, and the
/// dual-path inter GRUs) across calls.
pub struct StreamCore {
    state: CoreState,
    arena: Workspace,
    out_frame: Vec<f32>,
}

struct CoreState {
    gt: Vec<GtRing>,
    inter_h: Vec<Vec<f32>>,
}

impl StreamCore {
    fn new() -> Self {
        Self {
            state: CoreState {
                gt: GT_DILATIONS.iter().map(|&d| GtRing::new(d)).collect(),
                inter_h: vec![Vec::new(), Vec::new()],
            },
            arena: Workspace::default(),
            out_frame: vec![0.0; FB * 2],
        }
    }

    /// Build every shape/state before streaming, then retain and zero it.
    /// Every operation of `process_frame` runs on every frame whatever the samples,
    /// and the rings start at their full dilation span, so one traversal records
    /// every buffer shape.
    #[must_use]
    pub fn prepared(net: &Net) -> Self {
        let mut core = Self::new();
        core.process_frame(net, &[0.0; FB * 2], &[0.0; FB * 2]);
        core.reset();
        core.arena.seal();
        core
    }

    pub fn reset(&mut self) {
        for r in &mut self.state.gt {
            r.frames.fill(0.0);
            r.tra_h.fill(0.0);
        }
        for h in &mut self.state.inter_h {
            h.fill(0.0);
        }
        self.out_frame.fill(0.0);
    }
}

impl CoreState {
    fn gt_step<'a>(
        &mut self,
        arena: &'a Workspace,
        x: &GTensor,
        g: &GtBlock,
        idx: usize,
    ) -> GTensor<'a> {
        let x1 = chunk(arena, x, false);
        let x2 = chunk(arena, x, true);
        let s = sfe(arena, &x1);
        let mut h = conv(arena, &s, &g.pc1, 1, 1, 0, 0, 1);
        prelu_(&mut h, &g.pc1.prelu); // (hidden,1,F)
        let hidden = h.c;
        let fdim = h.f;
        // dw time ring (oldest first, length cap)
        let r = &mut self.gt[idx];
        if r.frames.is_empty() {
            r.frames = vec![0.0; r.cap * hidden * fdim];
        }
        // shift down one frame, append current at the newest slot
        let fsz = hidden * fdim;
        r.frames.copy_within(fsz.., 0);
        r.frames[(r.cap - 1) * fsz..].copy_from_slice(&h.d);
        let mut ring = GTensor::new(arena, hidden, r.cap, fdim);
        // ring as (hidden, cap, F): ring[(c*cap+t)*F+f] = frames[t*fsz + c*F + f].
        // For fixed (c,t) the F-run is contiguous in both, so this is a copy.
        for t in 0..r.cap {
            for c in 0..hidden {
                let src = t * fsz + c * fdim;
                let dst = (c * r.cap + t) * fdim;
                ring.d[dst..dst + fdim].copy_from_slice(&r.frames[src..src + fdim]);
            }
        }
        // pad_t 0: the ring already holds the context
        let mut dw = conv(
            arena,
            &ring,
            &g.dw,
            hidden,
            1,
            1,
            0,
            GT_DILATIONS[idx] as i64,
        );
        prelu_(&mut dw, &g.dw.prelu); // (hidden,1,F)
        let mut pc2 = conv(arena, &dw, &g.pc2, 1, 1, 0, 0, 1);
        // temporal attention (streaming GRU)
        let c = pc2.c;
        let mut seq = Scratch::zeros(arena, c);
        for ci in 0..c {
            let row = &pc2.d[ci * fdim..ci * fdim + fdim];
            let sm: f32 = row.iter().map(|&v| v * v).sum();
            seq[ci] = sm / fdim as f32;
        }
        let h = &mut self.gt[idx].tra_h[..g.gru.hidden];
        gru_cell(&seq, h, &g.gru.wih, &g.gru.whh, &g.gru.bih, &g.gru.bhh);
        let at = linear(arena, h, 1, &g.fc);
        for ci in 0..c {
            let g = sigmoid(at[ci]);
            for v in &mut pc2.d[ci * fdim..ci * fdim + fdim] {
                *v *= g;
            }
        }
        shuffle2(arena, &pc2, &x2)
    }

    fn dpgrnn_step<'a>(
        &mut self,
        arena: &'a Workspace,
        x: &GTensor,
        d: &DpGrnn,
        idx: usize,
    ) -> GTensor<'a> {
        let (c, fd) = (x.c, x.f);
        let mut xp = Scratch::zeros(arena, fd * c);
        for f in 0..fd {
            for ci in 0..c {
                xp[f * c + ci] = x.at(ci, 0, f);
            }
        }
        // intra: bidirectional grnn over freq (per-frame, no carried state)
        let mut intra = grnn(arena, &xp, 1, fd, c, &d.intra);
        intra = linear(arena, &intra, fd, &d.intra_fc);
        layernorm_last2(&mut intra, 1, fd, c, &d.intra_ln);
        for i in 0..intra.len() {
            intra[i] += xp[i];
        }
        // inter: per-freq unidirectional GRU carrying state across frames
        if self.inter_h[idx].is_empty() {
            self.inter_h[idx] = vec![0.0; fd * c];
        }
        let half = c / 2;
        // Per-freq GRUs are independent across `f` and share weights, so batch them
        // 8 freqs per SIMD lane (feature-major), same trick as the DAF controllers.
        let mut inter = Scratch::zeros(arena, fd * c);
        let ih = &mut self.inter_h[idx];
        for (rnn, g) in d.inter.fwd.iter().enumerate() {
            let off = rnn * half;
            let h1 = g.hidden;
            let mut f0 = 0;
            while f0 < fd {
                let lanes = (fd - f0).min(8);
                let mut x8 = [0.0f32; 16 * 8];
                let mut h8 = [0.0f32; 16 * 8];
                for lane in 0..lanes {
                    let f = f0 + lane;
                    for i in 0..half {
                        x8[i * 8 + lane] = intra[f * c + off + i];
                    }
                    for j in 0..h1 {
                        h8[j * 8 + lane] = ih[f * c + off + j];
                    }
                }
                ops::gru8(&x8, half, &mut h8, h1, &g.wih, &g.whh, &g.bih, &g.bhh);
                for lane in 0..lanes {
                    let f = f0 + lane;
                    for j in 0..h1 {
                        let v = h8[j * 8 + lane];
                        ih[f * c + off + j] = v;
                        inter[f * c + off + j] = v;
                    }
                }
                f0 += 8;
            }
        }
        inter = linear(arena, &inter, fd, &d.inter_fc);
        layernorm_last2(&mut inter, 1, fd, c, &d.inter_ln);
        for i in 0..inter.len() {
            inter[i] += intra[i];
        }
        let mut out = GTensor::new(arena, c, 1, fd);
        for f in 0..fd {
            for ci in 0..c {
                out.set(ci, 0, f, inter[f * c + ci]);
            }
        }
        out
    }
}

impl StreamCore {
    /// Process one spec frame: `spec_e`/`spec_y` are `[fb*2 + {re,im}]` (257 bins).
    pub fn process_frame(&mut self, net: &Net, spec_e: &[f32], spec_y: &[f32]) -> &[f32] {
        let arena = &self.arena;
        let fe = feat(arena, spec_e, 1, &net.erb_bm);
        let fy = feat(arena, spec_y, 1, &net.erb_bm);
        let mut ft = GTensor::new(arena, 18, 1, 129);
        for c in 0..9 {
            for f in 0..129 {
                ft.set(c, 0, f, fe.at(c, 0, f));
                ft.set(9 + c, 0, f, fy.at(c, 0, f));
            }
        }
        let st = &mut self.state;
        let en0 = conv_block(arena, &ft, &net.en0, 1, 2, false, false);
        let en1 = conv_block(arena, &en0, &net.en1, 2, 2, false, false);
        let en2 = st.gt_step(arena, &en1, &net.gt[0], 0);
        let en3 = st.gt_step(arena, &en2, &net.gt[1], 1);
        let en4 = st.gt_step(arena, &en3, &net.gt[2], 2);
        let d1 = st.dpgrnn_step(arena, &en4, &net.dp[0], 0);
        let d2 = st.dpgrnn_step(arena, &d1, &net.dp[1], 1);
        let mut x = st.gt_step(arena, &add(arena, &d2, &en4), &net.gt[3], 3);
        x = st.gt_step(arena, &add(arena, &x, &en3), &net.gt[4], 4);
        x = st.gt_step(arena, &add(arena, &x, &en2), &net.gt[5], 5);
        x = conv_block(arena, &add(arena, &x, &en1), &net.de3, 2, 2, true, false);
        x = conv_block(arena, &add(arena, &x, &en0), &net.de4, 1, 2, true, true);
        let m = erb_bs(arena, &x, &net.erb_bs);
        // The output is written into the reused frame buffer and returned as a view.
        for fb in 0..FB {
            let er = spec_e[fb * 2];
            let ei = spec_e[fb * 2 + 1];
            let mr = m.at(0, 0, fb);
            let mi = m.at(1, 0, fb);
            self.out_frame[fb * 2] = er * mr - ei * mi;
            self.out_frame[fb * 2 + 1] = ei * mr + er * mi;
        }
        &self.out_frame
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    #[test]
    fn prepared_storage_can_move_threads_and_reuse_exact_shapes() {
        let arena = Workspace::default();
        {
            let _a = Scratch::zeros(&arena, 33);
            let _b = Scratch::zeros(&arena, 33);
            let _c = Scratch::zeros(&arena, 257);
        }
        arena.seal();
        std::thread::spawn(move || {
            let a = Scratch::zeros(&arena, 33);
            let b = Scratch::zeros(&arena, 33);
            let c = Scratch::zeros(&arena, 257);
            assert_eq!((a.len(), b.len(), c.len()), (33, 33, 257));
            assert!(a.iter().all(|&v| v == 0.0));
        })
        .join()
        .unwrap();
    }
    #[test]
    fn erb_merge_passes_low_bins() {
        let m = crate::Model::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../model/localvqe-pi-aec-v1-49k-f32.gguf"
        ))
        .unwrap();
        let arena = Workspace::default();
        let mut frame = GTensor::new(&arena, 1, 1, crate::erb::FULL);
        for (i, v) in frame.d.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let banded = erb_bm(&arena, &frame, &m.net.erb_bm);
        assert_eq!(banded.d.len(), crate::erb::BANDED);
        assert_eq!(banded.d[..crate::erb::LOW], frame.d[..crate::erb::LOW]);
        assert!(banded.d.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "unprepared GTCRN workspace")]
    fn sealed_workspace_has_no_allocating_fallback() {
        let arena = Workspace::default();
        arena.seal();
        let _unplanned = Scratch::zeros(&arena, 1);
    }
}
