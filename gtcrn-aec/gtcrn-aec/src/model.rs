//! GTCRN-AEC core forward, a scalar Rust port of LocalVQE `gtcrn.cpp`. Input: two spectra `spec_e` (near/error) and `spec_y`
//! (far reference), each `(257, T, 2)` row-major; output: masked `spec_e`.
//!
//! GGUF stores tensor dims fastest-first, reversed from PyTorch/numpy, over the
//! same byte layout: a weight slice is indexed with numpy strides, and its numpy
//! shape is the GGUF dims reversed (see [`Tensor`]).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)] // numeric kernels index by design

use crate::gguf::Gguf;
use ops::sigmoid;

mod layers;
mod stream;

pub(crate) use layers::gru_cell;
use layers::*;
pub use stream::StreamCore;

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

#[cfg(test)]
mod workspace_tests {
    use super::*;
    #[test]
    fn prepared_storage_moves_between_threads() {
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
