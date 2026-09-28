//! GTCRN-AEC core forward — scalar Rust port of LocalVQE `gtcrn.cpp`, the
//! validated reference. Input: two spectra `spec_e` (near/error) and `spec_y`
//! (far reference), each `(257, T, 2)` row-major; output: masked `spec_e`.
//!
//! GGUF stores tensor dims fastest-first, i.e. reversed from PyTorch/numpy, but
//! the same byte layout — so a weight slice is indexed with numpy strides while
//! its numpy shape is the GGUF dims reversed (see `npdims`).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // numeric kernels index by design

use crate::gguf::Gguf;
use dfn_ops::{sigmoid, tanh_f};

const FB: usize = 257;

/// Scratch belongs to one StreamCore, not to the thread running it. The
/// streaming graph has fixed shapes and data-independent temporary lifetimes.
/// `StreamCore::prepared` records its exact high-water inventory during init,
/// resets recurrent state, then seals this workspace. No allocation fallback is
/// permitted after sealing. Batch/debug forward uses an unsealed local workspace.
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
/// workspace. No Rc/Arc, TLS, global state, unsafe pool, or cross-thread Drop.
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
impl<'a> Clone for GTensor<'a> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.arena, self.c, self.t, self.f);
        out.d.copy_from_slice(&self.d);
        out
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

/// Tensor dims in numpy order (GGUF order reversed), inline on the stack so the
/// per-hop weight lookups never allocate. GGUF rank is 1..=4 (validated at load).
#[derive(Clone, Copy)]
struct Dims {
    d: [usize; 4],
    n: usize,
}
impl std::ops::Deref for Dims {
    type Target = [usize];
    fn deref(&self) -> &[usize] {
        &self.d[..self.n]
    }
}

/// Weight accessor: numpy-order dims (GGUF dims reversed) + the data slice.
struct W<'a>(&'a Gguf);
impl<'a> W<'a> {
    fn get(&self, name: &str) -> (&'a [f32], Dims) {
        let (d, dims) = self
            .0
            .tensor(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        let mut np = [0usize; 4];
        for (i, &v) in dims.iter().rev().enumerate() {
            np[i] = v;
        }
        (
            d,
            Dims {
                d: np,
                n: dims.len(),
            },
        )
    }
    fn data(&self, name: &str) -> &'a [f32] {
        self.0
            .tensor(name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .0
    }
    fn get2(&self, prefix: &str, suffix: &str) -> (&'a [f32], Dims) {
        let key = WeightKey::new(prefix, suffix);
        self.get(key.as_str())
    }
    fn data2(&self, prefix: &str, suffix: &str) -> &'a [f32] {
        let key = WeightKey::new(prefix, suffix);
        self.data(key.as_str())
    }
}

struct WeightKey {
    bytes: [u8; 64],
    len: usize,
}
impl WeightKey {
    fn new(prefix: &str, suffix: &str) -> Self {
        let len = prefix.len() + suffix.len();
        assert!(len <= 64, "tensor name exceeds validated GGUF contract");
        let mut key = Self {
            bytes: [0; 64],
            len,
        };
        key.bytes[..prefix.len()].copy_from_slice(prefix.as_bytes());
        key.bytes[prefix.len()..len].copy_from_slice(suffix.as_bytes());
        key
    }
    fn as_str(&self) -> &str {
        // Concatenating complete UTF-8 strings preserves UTF-8 boundaries.
        std::str::from_utf8(&self.bytes[..self.len]).expect("UTF-8 tensor name")
    }
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
    // (t,f) plane. One SIMD kernel call for the whole conv, no per-(co,ci,t) scaffold.
    if groups == 1
        && kt == 1
        && kf == 1
        && pad_f == 0
        && pad_t_top == 0
        && pad_t_bot == 0
        && stride_f == 1
        && !b.is_empty()
    {
        dfn_ops::pointwise_f32(&mut y.d, &x.d, w, b, ic, oc, x.t * x.f);
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
    // accumulation below is then a contiguous multiply-add that vectorizes; the
    // strided scalar loop it replaces was ~15% of the streaming profile.
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
                                // fi = f + (kfi - pad_f); accumulate the overlapping run
                                // as one AXPY (contiguous, element-wise → SIMD, no per-
                                // element bounds check / i64 cast).
                                let shift = kfi as i64 - pad_f;
                                let f0 = (-shift).max(0) as usize;
                                let f1 = (fin - shift).min(fout as i64).max(0) as usize;
                                if f1 > f0 {
                                    let xs = (f0 as i64 + shift) as usize;
                                    dfn_ops::axpy_f32(
                                        &mut orow[f0..f1],
                                        &xrow[xs..xs + (f1 - f0)],
                                        wt,
                                    );
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
        *v = tanh_f(*v);
    }
}

/// 3-tap freq unfold: out channel `c*3+{0,1,2}` = {f-1, f, f+1} (zero pad). Each
/// output channel is the input freq-row shifted by one bin, so it is three
/// contiguous copies (memcpy), not a bounds-checked per-element gather.
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
                y.d[ybase + LOW + j] = dfn_ops::vdot_f32(xhi, row);
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
                y.d[ybase + LOW + j] = dfn_ops::vdot_f32(xhi, row);
            }
        }
    }
    y
}

/// GRU over a length-L sequence, PyTorch layout, gate order [r,z,n].
fn gru_seq<'a>(
    arena: &'a Workspace,
    x: &[f32],
    l: usize,
    i_dim: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
    h_dim: usize,
    reverse: bool,
) -> Scratch<'a> {
    let mut out = Scratch::zeros(arena, l * h_dim);
    // The shipped schema bounds every GRU hidden dimension by 16.
    assert!(h_dim <= 16);
    let mut h_buf = [0.0f32; 16];
    let mut gi_buf = [0.0f32; 48];
    let mut gh_buf = [0.0f32; 48];
    let h = &mut h_buf[..h_dim];
    let gi = &mut gi_buf[..3 * h_dim];
    let gh = &mut gh_buf[..3 * h_dim];
    for s in 0..l {
        let ti = if reverse { l - 1 - s } else { s };
        let xt = &x[ti * i_dim..ti * i_dim + i_dim];
        for gidx in 0..3 * h_dim {
            let mut a = bih[gidx];
            let wr = &wih[gidx * i_dim..gidx * i_dim + i_dim];
            for k in 0..i_dim {
                a += wr[k] * xt[k];
            }
            gi[gidx] = a;
            let mut cc = bhh[gidx];
            let hr = &whh[gidx * h_dim..gidx * h_dim + h_dim];
            for j in 0..h_dim {
                cc += hr[j] * h[j];
            }
            gh[gidx] = cc;
        }
        for k in 0..h_dim {
            let r = sigmoid(gi[k] + gh[k]);
            let z = sigmoid(gi[h_dim + k] + gh[h_dim + k]);
            let n = tanh_f(gi[2 * h_dim + k] + r * gh[2 * h_dim + k]);
            h[k] = (1.0 - z) * n + z * h[k];
        }
        out[ti * h_dim..ti * h_dim + h_dim].copy_from_slice(h);
    }
    out
}

/// One GRU step carrying `h` in place (PyTorch layout, gate order [r,z,n]).
fn gru_step(
    x: &[f32],
    nin: usize,
    h: &mut [f32],
    nh: usize,
    wih: &[f32],
    whh: &[f32],
    bih: &[f32],
    bhh: &[f32],
) {
    assert!(nh <= 16);
    let mut gi_buf = [0.0f32; 48];
    let mut gh_buf = [0.0f32; 48];
    let gi = &mut gi_buf[..3 * nh];
    let gh = &mut gh_buf[..3 * nh];
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
        let n = tanh_f(gi[2 * nh + i] + r * gh[2 * nh + i]);
        h[i] = (1.0 - z) * n + z * h[i];
    }
}

/// Grouped RNN: split I in half → rnn1/rnn2, each optionally bidirectional.
fn grnn<'a>(
    arena: &'a Workspace,
    x: &[f32],
    n: usize,
    seq: usize,
    i_dim: usize,
    w: &W,
    prefix: &str,
    bidir: bool,
) -> Scratch<'a> {
    let half = i_dim / 2;
    let (_, whh1) = w.get2(prefix, ".rnn1.weight_hh_l0");
    let h1 = whh1[1]; // numpy (3H,H) -> dim1 = H
    let out1 = if bidir { 2 * h1 } else { h1 };
    let (_, whh2) = w.get2(prefix, ".rnn2.weight_hh_l0");
    let h2 = whh2[1];
    let out2 = if bidir { 2 * h2 } else { h2 };
    let iout = out1 + out2;
    let mut y = Scratch::zeros(arena, n * seq * iout);
    let mut sub = Scratch::zeros(arena, seq * half);
    // Resolve every rnn weight slice once (was one format!+lookup per (ni,rnn));
    // the inter GRU runs this n=fd times per frame, so the lookups dominated.
    let rnn_w: [[&[f32]; 8]; 2] = std::array::from_fn(|rnn| {
        [
            w.data2(
                prefix,
                if rnn == 0 {
                    ".rnn1.weight_ih_l0"
                } else {
                    ".rnn2.weight_ih_l0"
                },
            ),
            w.data2(
                prefix,
                if rnn == 0 {
                    ".rnn1.weight_hh_l0"
                } else {
                    ".rnn2.weight_hh_l0"
                },
            ),
            w.data2(
                prefix,
                if rnn == 0 {
                    ".rnn1.bias_ih_l0"
                } else {
                    ".rnn2.bias_ih_l0"
                },
            ),
            w.data2(
                prefix,
                if rnn == 0 {
                    ".rnn1.bias_hh_l0"
                } else {
                    ".rnn2.bias_hh_l0"
                },
            ),
            if bidir {
                w.data2(
                    prefix,
                    if rnn == 0 {
                        ".rnn1.weight_ih_l0_reverse"
                    } else {
                        ".rnn2.weight_ih_l0_reverse"
                    },
                )
            } else {
                &[]
            },
            if bidir {
                w.data2(
                    prefix,
                    if rnn == 0 {
                        ".rnn1.weight_hh_l0_reverse"
                    } else {
                        ".rnn2.weight_hh_l0_reverse"
                    },
                )
            } else {
                &[]
            },
            if bidir {
                w.data2(
                    prefix,
                    if rnn == 0 {
                        ".rnn1.bias_ih_l0_reverse"
                    } else {
                        ".rnn2.bias_ih_l0_reverse"
                    },
                )
            } else {
                &[]
            },
            if bidir {
                w.data2(
                    prefix,
                    if rnn == 0 {
                        ".rnn1.bias_hh_l0_reverse"
                    } else {
                        ".rnn2.bias_hh_l0_reverse"
                    },
                )
            } else {
                &[]
            },
        ]
    });
    for ni in 0..n {
        for rnn in 0..2 {
            let ww = &rnn_w[rnn];
            let feat_off = rnn * half;
            let out_off = if rnn == 0 { 0 } else { out1 };
            let hh = if rnn == 0 { h1 } else { h2 };
            for s in 0..seq {
                for i in 0..half {
                    sub[s * half + i] = x[(ni * seq + s) * i_dim + feat_off + i];
                }
            }
            let yf = gru_seq(
                arena, &sub, seq, half, ww[0], ww[1], ww[2], ww[3], hh, false,
            );
            let yr = if bidir {
                gru_seq(arena, &sub, seq, half, ww[4], ww[5], ww[6], ww[7], hh, true)
            } else {
                Scratch::zeros(arena, 0)
            };
            for s in 0..seq {
                let dst = (ni * seq + s) * iout + out_off;
                for k in 0..hh {
                    y[dst + k] = yf[s * hh + k];
                }
                if bidir {
                    for k in 0..hh {
                        y[dst + hh + k] = yr[s * hh + k];
                    }
                }
            }
        }
    }
    y
}

/// Linear over last dim: x (M,I) -> (M,O), w numpy (O,I).
fn linear<'a>(
    arena: &'a Workspace,
    x: &[f32],
    m: usize,
    i_dim: usize,
    w: &[f32],
    b: &[f32],
    o_dim: usize,
) -> Scratch<'a> {
    let mut y = Scratch::zeros(arena, m * o_dim);
    for mi in 0..m {
        let xr = &x[mi * i_dim..mi * i_dim + i_dim];
        for o in 0..o_dim {
            let wr = &w[o * i_dim..o * i_dim + i_dim];
            y[mi * o_dim + o] = b[o] + dfn_ops::vdot_f32(wr, xr);
        }
    }
    y
}

/// LayerNorm over the joint (F,C) tail per time frame.
fn layernorm_last2(x: &mut [f32], t: usize, fd: usize, c: usize, w: &[f32], b: &[f32]) {
    let n = fd * c;
    let eps = 1e-8f32;
    for ti in 0..t {
        let xt = &mut x[ti * n..ti * n + n];
        let mu = xt.iter().sum::<f32>() / n as f32;
        let var = xt.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..n {
            xt[i] = (xt[i] - mu) * inv * w[i] + b[i];
        }
    }
}

/// Temporal recurrent attention gate.
fn tra<'a>(arena: &'a Workspace, x: &GTensor, w: &W, prefix: &str) -> GTensor<'a> {
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
    let (_, whh) = w.get2(prefix, ".tra.gru.weight_hh_l0");
    let hh = whh[1];
    let y = gru_seq(
        arena,
        &seq,
        t,
        c,
        w.data2(prefix, ".tra.gru.weight_ih_l0"),
        w.data2(prefix, ".tra.gru.weight_hh_l0"),
        w.data2(prefix, ".tra.gru.bias_ih_l0"),
        w.data2(prefix, ".tra.gru.bias_hh_l0"),
        hh,
        false,
    );
    let (fcw, fcd) = w.get2(prefix, ".tra.fc.w");
    let at = linear(arena, &y, t, hh, fcw, w.data2(prefix, ".tra.fc.b"), fcd[0]);
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
    // Pooled allocation instead of `a.clone()` (Vec::clone bypasses the pool and
    // hits the allocator on every decoder skip connection each hop).
    let mut y = GTensor::new(arena, a.c, a.t, a.f);
    for ((yi, ai), bi) in y.d.iter_mut().zip(&a.d).zip(&b.d) {
        *yi = ai + bi;
    }
    y
}

fn gt_block<'a>(arena: &'a Workspace, x: &GTensor, w: &W, p: &str, dil: i64) -> GTensor<'a> {
    let x1 = chunk(arena, x, false);
    let x2 = chunk(arena, x, true);
    let s = sfe(arena, &x1);
    let (w1, d1) = w.get2(p, ".pc1.w");
    let mut h = conv2d(
        arena,
        &s,
        w1,
        &d1,
        w.data2(p, ".pc1.b"),
        1,
        1,
        0,
        0,
        0,
        1,
        1,
    );
    prelu_(&mut h, w.data2(p, ".pc1.prelu"));
    let hidden = h.c;
    let (wd, dd) = w.get2(p, ".dw.w");
    h = conv2d(
        arena,
        &h,
        wd,
        &dd,
        w.data2(p, ".dw.b"),
        hidden,
        1,
        1,
        2 * dil,
        0,
        dil,
        1,
    );
    prelu_(&mut h, w.data2(p, ".dw.prelu"));
    let (w2, d2) = w.get2(p, ".pc2.w");
    h = conv2d(
        arena,
        &h,
        w2,
        &d2,
        w.data2(p, ".pc2.b"),
        1,
        1,
        0,
        0,
        0,
        1,
        1,
    );
    let h = tra(arena, &h, w, p);
    shuffle2(arena, &h, &x2)
}

fn conv_block<'a>(
    arena: &'a Workspace,
    x: &GTensor,
    w: &W,
    p: &str,
    groups: usize,
    stride_f: usize,
    deconv: bool,
    is_last: bool,
) -> GTensor<'a> {
    let (ww, wd) = w.get2(p, ".w");
    let mut y = if !deconv {
        conv2d(
            arena,
            x,
            ww,
            &wd,
            w.data2(p, ".b"),
            groups,
            stride_f,
            2,
            0,
            0,
            1,
            1,
        )
    } else {
        let xu = upsample_zero_f(arena, x, stride_f);
        conv2d(
            arena,
            &xu,
            ww,
            &wd,
            w.data2(p, ".b"),
            groups,
            1,
            2,
            0,
            0,
            1,
            1,
        )
    };
    if is_last {
        tanh_inplace(&mut y);
    } else {
        prelu_(&mut y, w.data2(p, ".prelu"));
    }
    y
}

fn dpgrnn<'a>(arena: &'a Workspace, x: &GTensor, w: &W, d: &str) -> GTensor<'a> {
    let (c, t, fd) = (x.c, x.t, x.f);
    let mut xp = vec![0.0f32; t * fd * c];
    for ti in 0..t {
        for f in 0..fd {
            for ci in 0..c {
                xp[(ti * fd + f) * c + ci] = x.at(ci, ti, f);
            }
        }
    }
    let mut intra = grnn(arena, &xp, t, fd, c, w, &format!("{d}.intra_rnn"), true);
    let (ifw, ifd) = w.get2(d, ".intra_fc.w");
    intra = linear(
        arena,
        &intra,
        t * fd,
        ifw.len() / ifd[0],
        ifw,
        w.data2(d, ".intra_fc.b"),
        ifd[0],
    );
    layernorm_last2(
        &mut intra,
        t,
        fd,
        c,
        w.data2(d, ".intra_ln.w"),
        w.data2(d, ".intra_ln.b"),
    );
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
    let mut inter = grnn(arena, &xq, fd, t, c, w, &format!("{d}.inter_rnn"), false);
    let (efw, efd) = w.get2(d, ".inter_fc.w");
    inter = linear(
        arena,
        &inter,
        fd * t,
        efw.len() / efd[0],
        efw,
        w.data2(d, ".inter_fc.b"),
        efd[0],
    );
    let mut interp = vec![0.0f32; t * fd * c];
    for f in 0..fd {
        for ti in 0..t {
            for ci in 0..c {
                interp[(ti * fd + f) * c + ci] = inter[(f * t + ti) * c + ci];
            }
        }
    }
    layernorm_last2(
        &mut interp,
        t,
        fd,
        c,
        w.data2(d, ".inter_ln.w"),
        w.data2(d, ".inter_ln.b"),
    );
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
fn feat<'a>(arena: &'a Workspace, spec: &[f32], t: usize, w: &W) -> GTensor<'a> {
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
    let banded = erb_bm(arena, &f3, w.data("erb.bm"));
    sfe(arena, &banded)
}

/// Full core forward. `spec_e`/`spec_y` are `(257,T,2)` row-major (freq-major).
/// Returns masked `(257,T,2)`.
#[must_use]
pub fn forward(gg: &Gguf, spec_e: &[f32], spec_y: &[f32], t: usize) -> Vec<f32> {
    forward_capture(gg, spec_e, spec_y, t, &mut Vec::new())
}

/// Forward that also records each named stage tensor (for parity debugging).
pub fn forward_capture(
    gg: &Gguf,
    spec_e: &[f32],
    spec_y: &[f32],
    t: usize,
    cap: &mut Vec<(&'static str, Vec<f32>)>,
) -> Vec<f32> {
    let storage = Workspace::default();
    let arena = &storage;
    let w = W(gg);
    let fe = feat(arena, spec_e, t, &w);
    let fy = feat(arena, spec_y, t, &w);
    let mut ft = GTensor::new(arena, 18, t, 129);
    for c in 0..9 {
        for ti in 0..t {
            for f in 0..129 {
                ft.set(c, ti, f, fe.at(c, ti, f));
                ft.set(9 + c, ti, f, fy.at(c, ti, f));
            }
        }
    }
    cap.push(("feat", ft.d.clone()));
    let en0 = conv_block(arena, &ft, &w, "encoder.en_convs.0", 1, 2, false, false);
    cap.push(("enc0", en0.d.clone()));
    let en1 = conv_block(arena, &en0, &w, "encoder.en_convs.1", 2, 2, false, false);
    cap.push(("enc1", en1.d.clone()));
    let en2 = gt_block(arena, &en1, &w, "encoder.en_convs.2", 1);
    cap.push(("enc2", en2.d.clone()));
    let en3 = gt_block(arena, &en2, &w, "encoder.en_convs.3", 2);
    cap.push(("enc3", en3.d.clone()));
    let en4 = gt_block(arena, &en3, &w, "encoder.en_convs.4", 5);
    cap.push(("enc4", en4.d.clone()));
    let d1 = dpgrnn(arena, &en4, &w, "dpgrnn1");
    cap.push(("dpgrnn1", d1.d.clone()));
    let d2 = dpgrnn(arena, &d1, &w, "dpgrnn2");
    cap.push(("dpgrnn2", d2.d.clone()));
    let mut x = gt_block(arena, &add(arena, &d2, &en4), &w, "decoder.de_convs.0", 5);
    cap.push(("dec0", x.d.clone()));
    x = gt_block(arena, &add(arena, &x, &en3), &w, "decoder.de_convs.1", 2);
    cap.push(("dec1", x.d.clone()));
    x = gt_block(arena, &add(arena, &x, &en2), &w, "decoder.de_convs.2", 1);
    cap.push(("dec2", x.d.clone()));
    x = conv_block(
        arena,
        &add(arena, &x, &en1),
        &w,
        "decoder.de_convs.3",
        2,
        2,
        true,
        false,
    );
    cap.push(("dec3", x.d.clone()));
    x = conv_block(
        arena,
        &add(arena, &x, &en0),
        &w,
        "decoder.de_convs.4",
        1,
        2,
        true,
        true,
    );
    cap.push(("dec4", x.d.clone()));
    let m = erb_bs(arena, &x, w.data("erb.bs"));
    cap.push(("mask", m.d.clone()));
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

// ── Streaming (per-frame, carried state). Parity must be revalidated against
// batch/reference after changes; preparing scratch must not warm recurrent state. ──

struct GtRing {
    cap: usize,
    hidden: usize,
    f: usize,
    frames: Vec<f32>, // cap * hidden * f, time-major (oldest first)
    tra_h: Vec<f32>,
}

impl GtRing {
    fn new(dil: usize) -> Self {
        Self {
            cap: 2 * dil + 1,
            hidden: 0,
            f: 0,
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

impl Default for StreamCore {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamCore {
    /// Unprepared core for offline/debug use. RT callers use `prepared`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: CoreState {
                gt: [1usize, 2, 5, 5, 2, 1]
                    .iter()
                    .map(|&d| GtRing::new(d))
                    .collect(),
                inter_h: vec![Vec::new(), Vec::new()],
            },
            arena: Workspace::default(),
            out_frame: vec![0.0; FB * 2],
        }
    }

    /// Build every shape/state before streaming, then retain and zero it.
    /// Every process_frame operation executes for every frame regardless of
    /// sample values; rings immediately have their full dilation span. Thus one
    /// traversal establishes all shape/liveness buckets, not a sampled guess.
    #[must_use]
    pub fn prepared(gg: &Gguf) -> Self {
        let mut core = Self::new();
        core.process_frame(gg, &[0.0; FB * 2], &[0.0; FB * 2]);
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
        w: &W,
        p: &str,
        dil: usize,
        idx: usize,
    ) -> GTensor<'a> {
        let x1 = chunk(arena, x, false);
        let x2 = chunk(arena, x, true);
        let s = sfe(arena, &x1);
        let (w1, d1) = w.get2(p, ".pc1.w");
        let mut h = conv2d(
            arena,
            &s,
            w1,
            &d1,
            w.data2(p, ".pc1.b"),
            1,
            1,
            0,
            0,
            0,
            1,
            1,
        );
        prelu_(&mut h, w.data2(p, ".pc1.prelu")); // (hidden,1,F)
        let hidden = h.c;
        let fdim = h.f;
        // dw time ring (oldest first, length cap)
        let r = &mut self.gt[idx];
        if r.frames.is_empty() {
            r.hidden = hidden;
            r.f = fdim;
            r.frames = vec![0.0; r.cap * hidden * fdim];
        }
        // shift down one frame, append current at the newest slot
        let fsz = hidden * fdim;
        r.frames.copy_within(fsz.., 0);
        r.frames[(r.cap - 1) * fsz..].copy_from_slice(&h.d);
        let mut ring = GTensor::new(arena, hidden, r.cap, fdim);
        // ring as (hidden, cap, F): ring[(c*cap+t)*F+f] = frames[t*fsz + c*F + f].
        // For fixed (c,t) the F-run is contiguous in both → a copy, not a gather.
        for t in 0..r.cap {
            for c in 0..hidden {
                let src = t * fsz + c * fdim;
                let dst = (c * r.cap + t) * fdim;
                ring.d[dst..dst + fdim].copy_from_slice(&r.frames[src..src + fdim]);
            }
        }
        let (wd, dd) = w.get2(p, ".dw.w");
        let mut dw = conv2d(
            arena,
            &ring,
            wd,
            &dd,
            w.data2(p, ".dw.b"),
            hidden,
            1,
            1,
            0,
            0,
            dil as i64,
            1,
        ); // pad_t 0: ring already holds the context
        prelu_(&mut dw, w.data2(p, ".dw.prelu")); // (hidden,1,F)
        let (w2, d2) = w.get2(p, ".pc2.w");
        let mut pc2 = conv2d(
            arena,
            &dw,
            w2,
            &d2,
            w.data2(p, ".pc2.b"),
            1,
            1,
            0,
            0,
            0,
            1,
            1,
        );
        // temporal attention (streaming GRU)
        let c = pc2.c;
        let mut seq = Scratch::zeros(arena, c);
        for ci in 0..c {
            let row = &pc2.d[ci * fdim..ci * fdim + fdim];
            let sm: f32 = row.iter().map(|&v| v * v).sum();
            seq[ci] = sm / fdim as f32;
        }
        let (_, whh) = w.get2(p, ".tra.gru.weight_hh_l0");
        let hh = whh[1];
        gru_step(
            &seq,
            c,
            &mut self.gt[idx].tra_h,
            hh,
            w.data2(p, ".tra.gru.weight_ih_l0"),
            w.data2(p, ".tra.gru.weight_hh_l0"),
            w.data2(p, ".tra.gru.bias_ih_l0"),
            w.data2(p, ".tra.gru.bias_hh_l0"),
        );
        let (fcw, fcd) = w.get2(p, ".tra.fc.w");
        let at = linear(
            arena,
            &self.gt[idx].tra_h,
            1,
            hh,
            fcw,
            w.data2(p, ".tra.fc.b"),
            fcd[0],
        );
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
        w: &W,
        d: &str,
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
        let prefix = WeightKey::new(d, ".intra_rnn");
        let mut intra = grnn(arena, &xp, 1, fd, c, w, prefix.as_str(), true);
        let (ifw, ifd) = w.get2(d, ".intra_fc.w");
        intra = linear(
            arena,
            &intra,
            fd,
            ifw.len() / ifd[0],
            ifw,
            w.data2(d, ".intra_fc.b"),
            ifd[0],
        );
        layernorm_last2(
            &mut intra,
            1,
            fd,
            c,
            w.data2(d, ".intra_ln.w"),
            w.data2(d, ".intra_ln.b"),
        );
        for i in 0..intra.len() {
            intra[i] += xp[i];
        }
        // inter: per-freq unidirectional GRU carrying state across frames
        if self.inter_h[idx].is_empty() {
            self.inter_h[idx] = vec![0.0; fd * c];
        }
        let half = c / 2;
        let (_, whh1) = w.get2(d, ".inter_rnn.rnn1.weight_hh_l0");
        let h1 = whh1[1];
        // Resolve the two inter GRUs' weights once; the f-loop runs fd times per
        // frame and used to do 8 format!+lookups each.
        let rw: [[&[f32]; 4]; 2] = [
            [
                w.data2(d, ".inter_rnn.rnn1.weight_ih_l0"),
                w.data2(d, ".inter_rnn.rnn1.weight_hh_l0"),
                w.data2(d, ".inter_rnn.rnn1.bias_ih_l0"),
                w.data2(d, ".inter_rnn.rnn1.bias_hh_l0"),
            ],
            [
                w.data2(d, ".inter_rnn.rnn2.weight_ih_l0"),
                w.data2(d, ".inter_rnn.rnn2.weight_hh_l0"),
                w.data2(d, ".inter_rnn.rnn2.bias_ih_l0"),
                w.data2(d, ".inter_rnn.rnn2.bias_hh_l0"),
            ],
        ];
        // Per-freq GRUs are independent across `f` and share weights, so batch them
        // 8 freqs per SIMD lane (feature-major), same trick as the DAF controllers.
        let mut inter = Scratch::zeros(arena, fd * c);
        let ih = &mut self.inter_h[idx];
        for rnn in 0..2 {
            let off = rnn * half;
            let (wih, whh, bih, bhh) = (rw[rnn][0], rw[rnn][1], rw[rnn][2], rw[rnn][3]);
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
                dfn_ops::gru8(&x8, half, &mut h8, h1, wih, whh, bih, bhh);
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
        let (efw, efd) = w.get2(d, ".inter_fc.w");
        inter = linear(
            arena,
            &inter,
            fd,
            efw.len() / efd[0],
            efw,
            w.data2(d, ".inter_fc.b"),
            efd[0],
        );
        layernorm_last2(
            &mut inter,
            1,
            fd,
            c,
            w.data2(d, ".inter_ln.w"),
            w.data2(d, ".inter_ln.b"),
        );
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
    pub fn process_frame(&mut self, gg: &Gguf, spec_e: &[f32], spec_y: &[f32]) -> &[f32] {
        let arena = &self.arena;
        let w = W(gg);
        let fe = feat(arena, spec_e, 1, &w);
        let fy = feat(arena, spec_y, 1, &w);
        let mut ft = GTensor::new(arena, 18, 1, 129);
        for c in 0..9 {
            for f in 0..129 {
                ft.set(c, 0, f, fe.at(c, 0, f));
                ft.set(9 + c, 0, f, fy.at(c, 0, f));
            }
        }
        let en0 = conv_block(arena, &ft, &w, "encoder.en_convs.0", 1, 2, false, false);
        let en1 = conv_block(arena, &en0, &w, "encoder.en_convs.1", 2, 2, false, false);
        let en2 = self
            .state
            .gt_step(arena, &en1, &w, "encoder.en_convs.2", 1, 0);
        let en3 = self
            .state
            .gt_step(arena, &en2, &w, "encoder.en_convs.3", 2, 1);
        let en4 = self
            .state
            .gt_step(arena, &en3, &w, "encoder.en_convs.4", 5, 2);
        let d1 = self.state.dpgrnn_step(arena, &en4, &w, "dpgrnn1", 0);
        let d2 = self.state.dpgrnn_step(arena, &d1, &w, "dpgrnn2", 1);
        let mut x = self.state.gt_step(
            arena,
            &add(arena, &d2, &en4),
            &w,
            "decoder.de_convs.0",
            5,
            3,
        );
        x = self
            .state
            .gt_step(arena, &add(arena, &x, &en3), &w, "decoder.de_convs.1", 2, 4);
        x = self
            .state
            .gt_step(arena, &add(arena, &x, &en2), &w, "decoder.de_convs.2", 1, 5);
        x = conv_block(
            arena,
            &add(arena, &x, &en1),
            &w,
            "decoder.de_convs.3",
            2,
            2,
            true,
            false,
        );
        x = conv_block(
            arena,
            &add(arena, &x, &en0),
            &w,
            "decoder.de_convs.4",
            1,
            2,
            true,
            true,
        );
        let m = erb_bs(arena, &x, w.data("erb.bs"));
        // Write into the reused frame buffer and hand back a view: no per-hop
        // allocation for the output either.
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
    #[should_panic(expected = "unprepared GTCRN workspace")]
    fn sealed_workspace_has_no_allocating_fallback() {
        let arena = Workspace::default();
        arena.seal();
        let _unplanned = Scratch::zeros(&arena, 1);
    }
}
