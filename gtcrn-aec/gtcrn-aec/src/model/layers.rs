//! The core's layers over arena tensors: convolutions, activations, the ERB
//! bands, GRUs, the gt blocks and the dual-path blocks.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn conv2d<'a>(
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
    // Same accumulation order as the scalar reference: for a fixed output (co,t,f)
    // the contributions are still summed in (ci, k, kfi) order, but with the freq
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

pub(super) fn prelu_(x: &mut GTensor, slope: &[f32]) {
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

pub(super) fn tanh_inplace(x: &mut GTensor) {
    for v in &mut x.d {
        *v = v.tanh();
    }
}

/// 3-tap freq unfold: out channel `c*3+{0,1,2}` = {f-1, f, f+1} (zero pad). Each
/// output channel is the input freq-row shifted by one bin: three shifted copies.
pub(super) fn sfe<'a>(arena: &'a Workspace, x: &GTensor) -> GTensor<'a> {
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

pub(super) fn erb_bm<'a>(arena: &'a Workspace, x: &GTensor, bmw: &[f32]) -> GTensor<'a> {
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

pub(super) fn erb_bs<'a>(arena: &'a Workspace, x: &GTensor, bsw: &[f32]) -> GTensor<'a> {
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
pub(super) fn gru_seq<'a>(
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
pub(super) fn grnn<'a>(
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
pub(super) fn linear<'a>(arena: &'a Workspace, x: &[f32], m: usize, l: &Linear) -> Scratch<'a> {
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
pub(super) fn layernorm_last2(x: &mut [f32], t: usize, fd: usize, c: usize, ln: &Norm) {
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
pub(super) fn tra<'a>(arena: &'a Workspace, x: &GTensor, g: &GtBlock) -> GTensor<'a> {
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

pub(super) fn shuffle2<'a>(arena: &'a Workspace, h: &GTensor, x2: &GTensor) -> GTensor<'a> {
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

pub(super) fn chunk<'a>(arena: &'a Workspace, x: &GTensor, second: bool) -> GTensor<'a> {
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

pub(super) fn upsample_zero_f<'a>(arena: &'a Workspace, x: &GTensor, factor: usize) -> GTensor<'a> {
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

pub(super) fn add<'a>(arena: &'a Workspace, a: &GTensor, b: &GTensor) -> GTensor<'a> {
    // From the workspace, so a skip connection never allocates.
    let mut y = GTensor::new(arena, a.c, a.t, a.f);
    for ((yi, ai), bi) in y.d.iter_mut().zip(&a.d).zip(&b.d) {
        *yi = ai + bi;
    }
    y
}

/// `conv2d` with the weights and bias of `c`.
pub(super) fn conv<'a>(
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

pub(super) fn gt_block<'a>(
    arena: &'a Workspace,
    x: &GTensor,
    g: &GtBlock,
    dil: i64,
) -> GTensor<'a> {
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

pub(super) fn conv_block<'a>(
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

pub(super) fn dpgrnn<'a>(arena: &'a Workspace, x: &GTensor, d: &DpGrnn) -> GTensor<'a> {
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
pub(super) fn feat<'a>(
    arena: &'a Workspace,
    spec: &[f32],
    t: usize,
    erb_bm_w: &[f32],
) -> GTensor<'a> {
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
