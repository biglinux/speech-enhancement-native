//! GCC-PHAT coarse delay estimate between the microphone and the reference.
//!
//! One observation costs three 32768-point real FFTs, about 1 ms. To keep that
//! out of a single audio callback, an observation is a job of `STEPS` steps of
//! similar cost (a butterfly stage, or a quarter of a linear pass), which the
//! stream advances a fixed number of steps per DAF block. Its result lands a
//! fixed number of blocks after the window closed, whatever the host block size.

use std::ops::Range;

use super::{G_CONF_THR, G_MAXLAG, G_NFFT, G_WIN, LOCK_AFTER, M};
use crate::fft::{
    bit_reverse_part, butterflies, fill_twiddles, irfft_split, irfft_unpack, rfft_combine,
    rfft_pack, scale_inverse,
};

const G_GATE_DB: f32 = 26.0;
/// Butterfly stages of the packed `G_NFFT / 2`-point complex FFT.
const STAGES: usize = (G_NFFT / 2).trailing_zeros() as usize;
/// Linear passes run in this many steps.
const PARTS: usize = 4;
// Step layout of one observation: forward FFT of the reference (bit reversal,
// butterfly stages, combine), the same for the microphone, the PHAT cross
// spectrum, the inverse FFT (split, bit reversal, stages, scale, unpack), and
// the peak search.
const REF_REV: usize = 0;
const REF_STAGES: usize = REF_REV + PARTS;
const REF_COMBINE: usize = REF_STAGES + STAGES;
const MIC_REV: usize = REF_COMBINE + PARTS;
const MIC_STAGES: usize = MIC_REV + PARTS;
const MIC_COMBINE: usize = MIC_STAGES + STAGES;
const CROSS: usize = MIC_COMBINE + PARTS;
const SPLIT: usize = CROSS + PARTS;
const INV_REV: usize = SPLIT + PARTS;
const INV_STAGES: usize = INV_REV + PARTS;
const SCALE: usize = INV_STAGES + STAGES;
const UNPACK: usize = SCALE + 1;
const PEAK: usize = UNPACK + 1;
pub(super) const STEPS: usize = PEAK + 1;

/// Part `p` of `0..n`.
fn part(n: usize, p: usize) -> Range<usize> {
    n * p / PARTS..n * (p + 1) / PARTS
}

#[derive(Clone, Copy, Default)]
pub(super) struct DelayEstimate {
    pub shift: i64,
    pub confidence: f32,
    pub locked: bool,
}

/// Accumulates the phase-normalised cross spectrum over successive windows and
/// picks the correlation peak. All buffers are sized at construction.
pub(super) struct GccEstimator {
    gcc_sr: Vec<f32>,
    gcc_si: Vec<f32>,
    gcc_maxrms: f32,
    pub estimate: DelayEstimate,
    next_step: Option<usize>, // the running observation's next step
    seen: i64,                // samples streamed when its window closed
    // Packed complex points of the reference and microphone windows; the
    // reference pair is reused for the inverse transform.
    ref_re: Vec<f32>,
    ref_im: Vec<f32>,
    mic_re: Vec<f32>,
    mic_im: Vec<f32>,
    xr: Vec<f32>,
    xi: Vec<f32>,
    dr: Vec<f32>,
    di: Vec<f32>,
    corr: Vec<f32>,
    tw_cos: Vec<f32>,
    tw_sin: Vec<f32>,
    tw_h_cos: Vec<f32>,
    tw_h_sin: Vec<f32>,
}

impl GccEstimator {
    pub fn new() -> Self {
        let (n2, nh) = (G_NFFT / 2, G_NFFT / 2 + 1);
        let mut tw_cos = vec![0.0; G_NFFT / 2];
        let mut tw_sin = vec![0.0; G_NFFT / 2];
        fill_twiddles(G_NFFT, &mut tw_cos, &mut tw_sin);
        let mut tw_h_cos = vec![0.0; G_NFFT / 4];
        let mut tw_h_sin = vec![0.0; G_NFFT / 4];
        fill_twiddles(G_NFFT / 2, &mut tw_h_cos, &mut tw_h_sin);
        Self {
            gcc_sr: vec![0.0; nh],
            gcc_si: vec![0.0; nh],
            gcc_maxrms: 1e-12,
            estimate: DelayEstimate::default(),
            next_step: None,
            seen: 0,
            ref_re: vec![0.0; n2],
            ref_im: vec![0.0; n2],
            mic_re: vec![0.0; n2],
            mic_im: vec![0.0; n2],
            xr: vec![0.0; nh],
            xi: vec![0.0; nh],
            dr: vec![0.0; nh],
            di: vec![0.0; nh],
            corr: vec![0.0; G_NFFT],
            tw_cos,
            tw_sin,
            tw_h_cos,
            tw_h_sin,
        }
    }

    /// Forget all evidence and drop a running observation.
    pub fn reset(&mut self) {
        self.gcc_sr.fill(0.0);
        self.gcc_si.fill(0.0);
        self.gcc_maxrms = 1e-12;
        self.estimate = DelayEstimate::default();
        self.next_step = None;
    }

    pub fn busy(&self) -> bool {
        self.next_step.is_some()
    }

    /// Begin an observation of one `G_WIN` window of `mic`/`reference`. `seen`
    /// is the stream position at which its result takes effect; the estimate
    /// locks once that reaches `LOCK_AFTER`. Returns false, changing nothing
    /// but the level tracker, when the reference is too quiet to be evidence.
    pub fn start(&mut self, mic: &[f32], reference: &[f32], seen: i64) -> bool {
        debug_assert!(!self.busy());
        if self.estimate.locked {
            return false;
        }
        let w = G_WIN;
        let mut rms = 0.0f32;
        for i in 0..w {
            rms += reference[i] * reference[i];
        }
        rms = (rms / w as f32).sqrt();
        // A decaying peak lets the gate recover after a loud transient. The
        // time constant is per observation (~0.5 s), not per sample.
        self.gcc_maxrms = (0.98 * self.gcc_maxrms).max(rms);
        if !(rms > 1e-9 && rms > self.gcc_maxrms * 10f32.powf(-G_GATE_DB / 20.0)) {
            return false;
        }
        // The window fills the first half of the zero-padded transform.
        let packed = w / 2;
        rfft_pack(
            &reference[..w],
            &mut self.ref_re[..packed],
            &mut self.ref_im[..packed],
        );
        rfft_pack(
            &mic[..w],
            &mut self.mic_re[..packed],
            &mut self.mic_im[..packed],
        );
        self.ref_re[packed..].fill(0.0);
        self.ref_im[packed..].fill(0.0);
        self.mic_re[packed..].fill(0.0);
        self.mic_im[packed..].fill(0.0);
        self.seen = seen;
        self.next_step = Some(0);
        true
    }

    /// Run up to `steps` steps of the running observation; returns the new
    /// estimate when it completes.
    pub fn advance(&mut self, steps: usize) -> Option<DelayEstimate> {
        for _ in 0..steps {
            let step = self.next_step?;
            self.next_step = Some(step + 1);
            if step == PEAK {
                self.next_step = None;
                return Some(self.peak());
            }
            self.run(step);
        }
        None
    }

    fn run(&mut self, step: usize) {
        let n2 = G_NFFT / 2;
        let (twc_h, tws_h) = (&self.tw_h_cos, &self.tw_h_sin);
        let (twc, tws) = (&self.tw_cos, &self.tw_sin);
        let (re, im) = if (MIC_REV..CROSS).contains(&step) {
            (&mut self.mic_re, &mut self.mic_im)
        } else {
            (&mut self.ref_re, &mut self.ref_im)
        };
        match step {
            s if s < REF_STAGES => bit_reverse_part(re, im, part(n2, s - REF_REV)),
            s if s < REF_COMBINE => butterflies(re, im, 2 << (s - REF_STAGES), false, twc_h, tws_h),
            s if s < MIC_REV => {
                let p = part(n2 + 1, s - REF_COMBINE);
                rfft_combine(re, im, &mut self.xr, &mut self.xi, twc, tws, p);
            }
            s if s < MIC_STAGES => bit_reverse_part(re, im, part(n2, s - MIC_REV)),
            s if s < MIC_COMBINE => butterflies(re, im, 2 << (s - MIC_STAGES), false, twc_h, tws_h),
            s if s < CROSS => {
                let p = part(n2 + 1, s - MIC_COMBINE);
                rfft_combine(re, im, &mut self.dr, &mut self.di, twc, tws, p);
            }
            s if s < SPLIT => {
                for k in part(n2 + 1, s - CROSS) {
                    let sr = self.dr[k] * self.xr[k] + self.di[k] * self.xi[k];
                    let si = self.di[k] * self.xr[k] - self.dr[k] * self.xi[k];
                    let mag = (sr * sr + si * si).sqrt() + 1e-9;
                    self.gcc_sr[k] += sr / mag;
                    self.gcc_si[k] += si / mag;
                }
            }
            s if s < INV_REV => {
                let p = part(n2, s - SPLIT);
                irfft_split(&self.gcc_sr, &self.gcc_si, re, im, twc, tws, p);
            }
            s if s < INV_STAGES => bit_reverse_part(re, im, part(n2, s - INV_REV)),
            s if s < SCALE => butterflies(re, im, 2 << (s - INV_STAGES), true, twc_h, tws_h),
            SCALE => scale_inverse(re, im),
            _ => irfft_unpack(re, im, &mut self.corr),
        }
    }

    fn peak(&mut self) -> DelayEstimate {
        let (mut best, mut peak, mut asum) = (0usize, self.corr[0], 0.0f32);
        for l in 0..G_MAXLAG {
            if self.corr[l] > peak {
                peak = self.corr[l];
                best = l;
            }
            asum += self.corr[l].abs();
        }
        let conf = peak / (asum / G_MAXLAG as f32 + 1e-12);
        self.estimate.confidence = conf;
        // Round down to whole blocks, one block short of the peak, so the
        // filter keeps a causal margin.
        self.estimate.shift = if conf > G_CONF_THR {
            ((best as i64 - M as i64).max(0) / M as i64) * M as i64
        } else {
            0
        };
        if conf > G_CONF_THR && self.seen >= LOCK_AFTER {
            self.estimate.locked = true;
        }
        self.estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stepped_observation_finds_the_delay() {
        let mut seed = 3u32;
        let reference: Vec<f32> = (0..G_WIN + 1000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 9) as f32 / 8_388_608.0 - 1.0
            })
            .collect();
        let mic = &reference[..G_WIN];
        let reference = &reference[1000..];
        let mut gcc = GccEstimator::new();
        assert!(gcc.start(mic, reference, LOCK_AFTER));
        let steps = std::iter::repeat_with(|| gcc.advance(1))
            .position(|e| e.is_some())
            .unwrap();
        assert_eq!(steps + 1, STEPS);
        assert!(!gcc.busy());
        assert!(gcc.estimate.locked);
        assert_eq!(
            gcc.estimate.shift, 768,
            "1000 samples, one block short, whole blocks"
        );
    }
}
