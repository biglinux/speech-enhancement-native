//! DAF (delay-and-filter) AEC front end, a scalar Rust port of LocalVQE
//! `daf_frontend.cpp`. A frequency-domain block Kalman filter (N = 128
//! partitions of M = 128 samples) steered by three small GRU controllers, after
//! a GCC-PHAT coarse delay estimate. It produces the error `e` and the echo
//! estimate `yhat` that feed the GTCRN core.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod block;
mod gcc;

use crate::gguf::Gguf;
use block::Scratch;
use gcc::GccEstimator;

const M: usize = 128;
const N: usize = 128;
const NFFT: usize = 256;
const NB: usize = 129;
const NBP: usize = 136; // NB padded to a multiple of 8 for the batched per-bin GRU
const K_ITER: usize = 2;
const A_DECAY: f32 = 0.999;
const P_INIT: f32 = 1e3;
const G_WIN: usize = 16384;
const G_HOP: usize = 8000;
const G_NFFT: usize = 32768;
const G_MAXLAG: usize = 16384;
const G_CONF_THR: f32 = 8.0;
const LOCK_AFTER: i64 = 56000;
/// GCC-PHAT steps per DAF block while an observation runs, about 60 µs.
const GCC_STEPS_PER_BLOCK: usize = 4;
/// Samples between the block that starts an observation and the block that
/// applies its result (18 blocks, 144 ms). Windows close this much earlier
/// than the delay schedule, so estimates and the lock take effect on the same
/// samples as a synchronous estimator would.
const GCC_LEAD: i64 = ((gcc::STEPS.div_ceil(GCC_STEPS_PER_BLOCK) - 1) * M) as i64;
// An observation must finish before the next window closes.
const _: () = assert!(gcc::STEPS.div_ceil(GCC_STEPS_PER_BLOCK) < G_HOP / M);

/// DAF front-end state + weights.
pub struct Daf {
    // weights
    g_ln_w: Vec<f32>,
    g_ln_b: Vec<f32>,
    g_wih: Vec<f32>,
    g_whh: Vec<f32>,
    g_bih: Vec<f32>,
    g_bhh: Vec<f32>,
    b_ln_w: Vec<f32>,
    b_ln_b: Vec<f32>,
    b_wih: Vec<f32>,
    b_whh: Vec<f32>,
    b_bih: Vec<f32>,
    b_bhh: Vec<f32>,
    p_ln_w: Vec<f32>,
    p_ln_b: Vec<f32>,
    p_wih: Vec<f32>,
    p_whh: Vec<f32>,
    p_bih: Vec<f32>,
    p_bhh: Vec<f32>,
    head_w: Vec<f32>,
    head_b: Vec<f32>,
    p_head_w: Vec<f32>,
    p_head_b: Vec<f32>,
    // adaptive filter state
    hr: Vec<f32>,
    hi: Vec<f32>,
    xr_r: Vec<f32>,
    xr_i: Vec<f32>,
    p: Vec<f32>,
    x_old: Vec<f32>,
    p_idx: usize,
    hg: Vec<f32>,
    hb: Vec<f32>,
    hp: Vec<f32>,
    // coarse delay
    estimator: GccEstimator,
    mic_ring: Vec<f32>,
    ref_ring: Vec<f32>,
    gcc_conf: f32,
    cur_shift: i64,
    gcc_locked: bool,
    n_seen: i64,
    ref_dline: Vec<f32>,
    ref_dpos: usize,
    sc: Scratch,
}

impl Daf {
    pub fn new(gg: &Gguf) -> Result<Self, String> {
        let g = |n: &str| {
            gg.tensor(n)
                .map(|(d, _)| d.to_vec())
                .ok_or_else(|| format!("model lacks DAF tensor {n}"))
        };
        Ok(Self {
            g_ln_w: g("daf.glob.norm.ln.weight")?,
            g_ln_b: g("daf.glob.norm.ln.bias")?,
            g_wih: g("daf.glob.gru.weight_ih")?,
            g_whh: g("daf.glob.gru.weight_hh")?,
            g_bih: g("daf.glob.gru.bias_ih")?,
            g_bhh: g("daf.glob.gru.bias_hh")?,
            b_ln_w: g("daf.bins.norm.ln.weight")?,
            b_ln_b: g("daf.bins.norm.ln.bias")?,
            b_wih: g("daf.bins.gru.weight_ih")?,
            b_whh: g("daf.bins.gru.weight_hh")?,
            b_bih: g("daf.bins.gru.bias_ih")?,
            b_bhh: g("daf.bins.gru.bias_hh")?,
            p_ln_w: g("daf.part.ln.weight")?,
            p_ln_b: g("daf.part.ln.bias")?,
            p_wih: g("daf.part.gru.weight_ih")?,
            p_whh: g("daf.part.gru.weight_hh")?,
            p_bih: g("daf.part.gru.bias_ih")?,
            p_bhh: g("daf.part.gru.bias_hh")?,
            head_w: g("daf.head.weight")?,
            head_b: g("daf.head.bias")?,
            p_head_w: g("daf.part.head.weight")?,
            p_head_b: g("daf.part.head.bias")?,
            hr: vec![0.0; N * NB],
            hi: vec![0.0; N * NB],
            xr_r: vec![0.0; N * NB],
            xr_i: vec![0.0; N * NB],
            p: vec![P_INIT; N * NB],
            x_old: vec![0.0; M],
            p_idx: 0,
            hg: vec![0.0; 8],
            hb: vec![0.0; NBP * 16],
            hp: vec![0.0; N * 8],
            estimator: GccEstimator::new(),
            mic_ring: vec![0.0; G_WIN],
            ref_ring: vec![0.0; G_WIN],
            gcc_conf: 0.0,
            cur_shift: 0,
            gcc_locked: false,
            n_seen: 0,
            ref_dline: vec![0.0; G_MAXLAG + M],
            ref_dpos: 0,
            sc: Scratch::new(),
        })
    }

    /// Return to the freshly built state without allocating.
    pub fn reset(&mut self) {
        self.hr.fill(0.0);
        self.hi.fill(0.0);
        self.xr_r.fill(0.0);
        self.xr_i.fill(0.0);
        self.p.fill(P_INIT);
        self.x_old.fill(0.0);
        self.p_idx = 0;
        self.hg.fill(0.0);
        self.hb.fill(0.0);
        self.hp.fill(0.0);
        self.estimator.reset();
        self.mic_ring.fill(0.0);
        self.ref_ring.fill(0.0);
        self.gcc_conf = 0.0;
        self.cur_shift = 0;
        self.gcc_locked = false;
        self.n_seen = 0;
        self.ref_dline.fill(0.0);
        self.ref_dpos = 0;
    }

    fn apply_estimate(&mut self, e: gcc::DelayEstimate) {
        self.cur_shift = e.shift;
        self.gcc_conf = e.confidence;
        self.gcc_locked = e.locked;
    }

    /// Whether an observation window closes in this block, on a schedule
    /// advanced by `lead` samples.
    fn window_closes(&self, lead: i64) -> bool {
        let n = self.n_seen + lead;
        n >= G_WIN as i64 && ((n - G_WIN as i64) % G_HOP as i64) < M as i64
    }

    /// Process `hop` samples (a multiple of M): mic+ref -> e (echo removed) + yhat.
    pub fn process(
        &mut self,
        mic: &[f32],
        r#ref: &[f32],
        hop: usize,
        e_out: &mut [f32],
        yhat_out: &mut [f32],
    ) {
        let dl = self.ref_dline.len();
        let mut o = 0;
        while o < hop {
            // The estimator reads these histories only until the delay locks.
            if !self.gcc_locked {
                self.mic_ring.copy_within(M.., 0);
                self.mic_ring[G_WIN - M..].copy_from_slice(&mic[o..o + M]);
                self.ref_ring.copy_within(M.., 0);
                self.ref_ring[G_WIN - M..].copy_from_slice(&r#ref[o..o + M]);
            }
            for i in 0..M {
                self.ref_dline[(self.ref_dpos + i) % dl] = r#ref[o + i];
            }
            self.n_seen += M as i64;
            if !self.gcc_locked && self.window_closes(GCC_LEAD) {
                self.estimator
                    .start(&self.mic_ring, &self.ref_ring, self.n_seen + GCC_LEAD);
            }
            if let Some(e) = self.estimator.advance(GCC_STEPS_PER_BLOCK) {
                self.apply_estimate(e);
            }
            let mut xs = [0.0f32; M];
            for i in 0..M {
                let idx = self.ref_dpos as i64 + i as i64 - self.cur_shift;
                xs[i] = if self.n_seen - M as i64 + i as i64 >= self.cur_shift {
                    self.ref_dline[(idx.rem_euclid(dl as i64)) as usize]
                } else {
                    0.0
                };
            }
            self.ref_dpos = (self.ref_dpos + M) % dl;
            self.block(&mic[o..o + M], &xs, &mut e_out[o..o + M]);
            o += M;
        }
        for i in 0..hop {
            yhat_out[i] = mic[i] - e_out[i];
        }
    }

    /// Whole-clip GCC to freeze the bulk delay before filtering (removes onset).
    pub fn prime_delay(&mut self, mic: &[f32], r#ref: &[f32], n: usize) {
        self.reset();
        let (mut best_shift, mut best_conf) = (0i64, 0.0f32);
        let mut o = 0;
        while o + M <= n {
            self.mic_ring.copy_within(M.., 0);
            self.mic_ring[G_WIN - M..].copy_from_slice(&mic[o..o + M]);
            self.ref_ring.copy_within(M.., 0);
            self.ref_ring[G_WIN - M..].copy_from_slice(&r#ref[o..o + M]);
            self.n_seen += M as i64;
            if self.window_closes(0) {
                // Every observation counts here, so reopen the lock first.
                self.estimator.estimate.locked = false;
                if self
                    .estimator
                    .start(&self.mic_ring, &self.ref_ring, self.n_seen)
                {
                    let e = self.estimator.advance(gcc::STEPS);
                    self.apply_estimate(e.expect("an observation completes in STEPS"));
                }
                self.gcc_locked = false;
                if self.gcc_conf > best_conf {
                    best_conf = self.gcc_conf;
                    best_shift = self.cur_shift;
                }
            }
            o += M;
        }
        self.reset();
        if best_conf > G_CONF_THR {
            self.cur_shift = best_shift;
            self.gcc_locked = true;
        }
    }
}
