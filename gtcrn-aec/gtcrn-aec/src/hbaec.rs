//! High-band linear echo canceller.
//!
//! The split-band plugin (`spa-aec-gtcrn`) runs the neural AEC at 16 kHz and
//! adds back the microphone's band above 8 kHz, which carries echo the network
//! never sees. This partitioned block frequency-domain adaptive filter
//! (overlap-save, constrained NLMS) cancels the linear echo in that band from
//! the loopback reference.
//!
//! The caller decides when the weights may adapt (`adapt`); its permission is a
//! heuristic, not a double-talk detector. Reference power statistics keep
//! following the signal while the weights are frozen, so resuming adaptation
//! never uses a stale normalizer. All scratch lives in the struct: after
//! construction `process_block` does not allocate.

use crate::fft::{fill_twiddles, irfft, rfft};

/// Weight step size of the normalized LMS update.
const MU: f32 = 0.3;
/// Per-block weight leakage; keeps the weights bounded.
const LEAK: f32 = 0.9995;

/// One partitioned block frequency-domain adaptive filter over the high band.
pub struct HbAec {
    b: usize,
    nfft: usize,
    bins: usize,
    p: usize,
    hr: Vec<f32>,
    hi: Vec<f32>,
    xr: Vec<f32>,
    xi: Vec<f32>,
    pw: Vec<f32>,
    norm_pw: Vec<f32>, // max(current partition power, smoothed power)
    prev_ref: Vec<f32>,
    /// Next partition the overlap-save constraint visits.
    p_idx: usize,
    twc: Vec<f32>,
    tws: Vec<f32>,
    twc_h: Vec<f32>,
    tws_h: Vec<f32>,
    // scratch (owned so process_block never allocates)
    sre: Vec<f32>,
    sim: Vec<f32>,
    xin: Vec<f32>,
    ytime: Vec<f32>,
    ebuf: Vec<f32>,
    htime: Vec<f32>,
    yr: Vec<f32>,
    yi: Vec<f32>,
    er: Vec<f32>,
    ei: Vec<f32>,
    hcr: Vec<f32>,
    hci: Vec<f32>,
}

impl HbAec {
    /// `block` samples per call, `partitions` covering the echo tail
    /// (`partitions*block` samples). At 48 kHz, block 256 / 64 partitions ≈
    /// 341 ms of coverage.
    pub fn new(block: usize, partitions: usize) -> Self {
        assert!(
            block.is_power_of_two() && block >= 2,
            "block must be a power of two of at least 2"
        );
        assert!(partitions > 0, "at least one partition is required");
        let nfft = block.checked_mul(2).expect("FFT size overflow");
        let bins = nfft / 2 + 1;
        let mut twc = vec![0.0; nfft / 2];
        let mut tws = vec![0.0; nfft / 2];
        fill_twiddles(nfft, &mut twc, &mut tws);
        let mut twc_h = vec![0.0; nfft / 4];
        let mut tws_h = vec![0.0; nfft / 4];
        fill_twiddles(nfft / 2, &mut twc_h, &mut tws_h);
        let pb = partitions
            .checked_mul(bins)
            .expect("partition size overflow");
        Self {
            b: block,
            nfft,
            bins,
            p: partitions,
            hr: vec![0.0; pb],
            hi: vec![0.0; pb],
            xr: vec![0.0; pb],
            xi: vec![0.0; pb],
            pw: vec![1e-6; bins],
            norm_pw: vec![1e-6; bins],
            prev_ref: vec![0.0; block],
            p_idx: 0,
            twc,
            tws,
            twc_h,
            tws_h,
            sre: vec![0.0; block],
            sim: vec![0.0; block],
            xin: vec![0.0; nfft],
            ytime: vec![0.0; nfft],
            ebuf: vec![0.0; nfft],
            htime: vec![0.0; nfft],
            yr: vec![0.0; bins],
            yi: vec![0.0; bins],
            er: vec![0.0; bins],
            ei: vec![0.0; bins],
            hcr: vec![0.0; bins],
            hci: vec![0.0; bins],
        }
    }

    /// Forget the echo path and the reference history without allocating.
    pub fn reset(&mut self) {
        self.hr.fill(0.0);
        self.hi.fill(0.0);
        self.xr.fill(0.0);
        self.xi.fill(0.0);
        self.pw.fill(1e-6);
        self.norm_pw.fill(1e-6);
        self.prev_ref.fill(0.0);
        self.p_idx = 0;
    }

    /// Cancel the high-band echo in one block: `mic`, `refb` and `out` hold
    /// `block` samples each. `adapt` permits weight updates. A non-finite result
    /// resets the filter and emits silence for the block.
    pub fn process_block(&mut self, mic: &[f32], refb: &[f32], adapt: bool, out: &mut [f32]) {
        let (b, bins, p, nfft) = (self.b, self.bins, self.p, self.nfft);
        assert!(mic.len() == b && refb.len() == b && out.len() == b);
        // Overlap-save reference block: [prev | current].
        self.xin[..b].copy_from_slice(&self.prev_ref);
        self.xin[b..].copy_from_slice(&refb[..b]);
        self.prev_ref.copy_from_slice(&refb[..b]);
        // Shift the partition ring down and insert the new spectrum at p=0.
        self.xr.copy_within(0..(p - 1) * bins, bins);
        self.xi.copy_within(0..(p - 1) * bins, bins);
        rfft(
            &self.xin,
            nfft,
            &mut self.xr[..bins],
            &mut self.xi[..bins],
            &mut self.sre,
            &mut self.sim,
            &self.twc,
            &self.tws,
            &self.twc_h,
            &self.tws_h,
        );
        // Predict echo spectrum Y = Σ_p H_p * X_p.
        self.yr.fill(0.0);
        self.yi.fill(0.0);
        for pi in 0..p {
            let base = pi * bins;
            for k in 0..bins {
                let (hr, hi) = (self.hr[base + k], self.hi[base + k]);
                let (xr, xi) = (self.xr[base + k], self.xi[base + k]);
                self.yr[k] += hr * xr - hi * xi;
                self.yi[k] += hr * xi + hi * xr;
            }
        }
        // echo(time) = last b samples of irfft(Y); err = mic - echo.
        irfft(
            &self.yr,
            &self.yi,
            nfft,
            &mut self.ytime,
            &mut self.sre,
            &mut self.sim,
            &self.twc,
            &self.tws,
            &self.twc_h,
            &self.tws_h,
        );
        for i in 0..b {
            out[i] = mic[i] - self.ytime[b + i];
        }
        if out.iter().any(|v| !v.is_finite()) {
            self.reset();
            out.fill(0.0);
            return;
        }
        // Adapt from the true prediction error even when that prediction is
        // rejected for playback. A stale echo path must not create sound from
        // a silent microphone or amplify a quieter microphone after a gain change.
        self.ebuf[b..].copy_from_slice(out);
        let energy = |x: &[f32]| x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        if energy(out) > 4.0 * energy(mic) {
            out.copy_from_slice(mic);
        }
        // Track render statistics even while coefficient updates are frozen.
        // On an onset, EMA alone is about 0.1*current power and can multiply
        // the intended normalized step by almost ten. Cap that amplification
        // without increasing the chosen mu or changing the FIR constraints.
        let mut mean_pw = 0.0f32;
        for k in 0..bins {
            let mut pw = 0.0f32;
            for pi in 0..p {
                let base = pi * bins;
                pw += self.xr[base + k] * self.xr[base + k] + self.xi[base + k] * self.xi[base + k];
            }
            self.pw[k] = 0.9 * self.pw[k] + 0.1 * pw;
            self.norm_pw[k] = pw.max(self.pw[k]);
            mean_pw += self.norm_pw[k];
        }
        mean_pw /= bins as f32;
        if !mean_pw.is_finite() || self.norm_pw.iter().any(|v| !v.is_finite()) {
            self.reset();
            out.fill(0.0);
            return;
        }
        if !adapt {
            return;
        }
        // E = rfft([0; err]).
        self.ebuf[..b].fill(0.0);
        rfft(
            &self.ebuf,
            nfft,
            &mut self.er,
            &mut self.ei,
            &mut self.sre,
            &mut self.sim,
            &self.twc,
            &self.tws,
            &self.twc_h,
            &self.tws_h,
        );
        // Regularise by a fraction of the mean reference power: the band above
        // 8 kHz carries little energy, and a fixed floor lets near-empty bins
        // take huge NLMS steps and diverge.
        let eps = (1e-2 * mean_pw).max(1e-6);
        // H_p = LEAK*H_p + MU * conj(X_p) * E / power.
        for pi in 0..p {
            let base = pi * bins;
            for k in 0..bins {
                let norm = MU / (self.norm_pw[k] + eps);
                let (xr, xi) = (self.xr[base + k], self.xi[base + k]);
                let gr = xr * self.er[k] + xi * self.ei[k];
                let gi = xr * self.ei[k] - xi * self.er[k];
                self.hr[base + k] = LEAK * self.hr[base + k] + norm * gr;
                self.hi[base + k] = LEAK * self.hi[base + k] + norm * gi;
            }
        }
        // Overlap-save gradient constraint (first b taps only), one partition
        // per block: constraining all of them costs two FFTs per partition and
        // the circular part left in the others stays small between visits.
        let base = self.p_idx * bins;
        self.hcr.copy_from_slice(&self.hr[base..base + bins]);
        self.hci.copy_from_slice(&self.hi[base..base + bins]);
        irfft(
            &self.hcr,
            &self.hci,
            nfft,
            &mut self.htime,
            &mut self.sre,
            &mut self.sim,
            &self.twc,
            &self.tws,
            &self.twc_h,
            &self.tws_h,
        );
        self.htime[b..].fill(0.0);
        rfft(
            &self.htime,
            nfft,
            &mut self.hr[base..base + bins],
            &mut self.hi[base..base + bins],
            &mut self.sre,
            &mut self.sim,
            &self.twc,
            &self.tws,
            &self.twc_h,
            &self.tws_h,
        );
        self.p_idx = (self.p_idx + 1) % p;
    }
}

#[cfg(test)]
mod tests {
    use super::HbAec;

    #[test]
    fn converges_on_a_linear_echo() {
        let (b, p) = (256usize, 24usize);
        let mut aec = HbAec::new(b, p);
        let sr = 48000usize;
        let delay = 300usize;
        let mut seed = 1u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / 8_388_608.0 - 1.0
        };
        let n = 6 * sr;
        let refsig: Vec<f32> = (0..n).map(|_| 0.3 * rng()).collect();
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                if i >= delay {
                    0.5 * refsig[i - delay]
                } else {
                    0.0
                }
            })
            .collect();
        let mut out = vec![0.0f32; b];
        let (mut ein, mut eout) = (0.0f64, 0.0f64);
        let mut o = 0;
        while o + b <= n {
            aec.process_block(&mic[o..o + b], &refsig[o..o + b], true, &mut out);
            if o >= 4 * sr {
                for i in 0..b {
                    ein += f64::from(mic[o + i]) * f64::from(mic[o + i]);
                    eout += f64::from(out[i]) * f64::from(out[i]);
                }
            }
            o += b;
        }
        let erle = 10.0 * (ein / eout.max(1e-12)).log10();
        eprintln!("HbAec linear-echo ERLE: {erle:.1} dB");
        assert!(
            erle > 20.0,
            "adaptive filter did not converge: {erle:.1} dB"
        );
    }

    #[test]
    fn frozen_filter_passes_near_end() {
        let (b, p) = (256usize, 8usize);
        let mut aec = HbAec::new(b, p);
        let near: Vec<f32> = (0..b).map(|i| 0.4 * (i as f32 * 0.3).sin()).collect();
        let zero = vec![0.0f32; b];
        let mut out = vec![0.0f32; b];
        aec.process_block(&near, &zero, false, &mut out);
        for i in 0..b {
            assert!(
                (out[i] - near[i]).abs() < 1e-6,
                "near-end altered while frozen"
            );
        }
    }
    #[test]
    fn frozen_filter_tracks_render_power_without_changing_weights() {
        let mut h = HbAec::new(256, 4);
        let mic = [0.0; 256];
        let render = [0.125; 256];
        let mut out = [0.0; 256];
        for _ in 0..8 {
            h.process_block(&mic, &render, false, &mut out);
        }
        assert!(h.pw.iter().any(|&p| p > 1e-3));
        assert!(h.hr.iter().chain(&h.hi).all(|&w| w == 0.0));
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn normalization_covers_current_partition_energy_at_onset() {
        let mut h = HbAec::new(256, 4);
        let mic = [0.0; 256];
        let mut render = [0.0; 256];
        render[0] = 0.5;
        let mut out = [0.0; 256];
        h.process_block(&mic, &render, true, &mut out);
        for k in 0..h.bins {
            let current: f32 = (0..h.p)
                .map(|p| {
                    let i = p * h.bins + k;
                    h.xr[i] * h.xr[i] + h.xi[i] * h.xi[i]
                })
                .sum();
            assert!(h.norm_pw[k] >= current);
            assert!(h.norm_pw[k] >= h.pw[k]);
        }
        assert!(h.hr.iter().chain(&h.hi).all(|x| x.is_finite()));
    }
}

#[cfg(test)]
mod fault_tests {
    use super::HbAec;
    #[test]
    fn numerical_fault_resets_and_emits_silence() {
        let mut aec = HbAec::new(256, 8);
        let mut out = [1.0; 256];
        aec.process_block(&[f32::NAN; 256], &[0.0; 256], true, &mut out);
        assert!(out.iter().all(|v| *v == 0.0));
        let near = [0.25; 256];
        aec.process_block(&near, &[0.0; 256], true, &mut out);
        assert_eq!(out, near, "the filter must recover after a fault");
    }
}
