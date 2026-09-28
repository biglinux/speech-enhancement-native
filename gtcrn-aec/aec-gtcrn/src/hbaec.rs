//! High-band linear echo canceller.
//!
//! The split-band core (`spa-aec-gtcrn`) runs the neural AEC at 16 kHz and
//! reinjects the mic's >8 kHz band, which carried echo the core never saw. This
//! is a **partitioned block frequency-domain adaptive filter** (PBFDAF,
//! overlap-save, constrained NLMS) that cancels the linear echo in that band
//! from the loopback reference — a classic linear AEC, no model, no training.
//!
//! Runs on the high-band signals (`mic - lpf(mic)`, `ref - lpf(ref)`). The
//! caller supplies an adaptation permission (`adapt`). A suppression ratio is
//! not a double-talk detector: a mistaken permission can still damage near-end
//! speech. Reference statistics must keep following the signal while weights
//! are frozen, otherwise resuming adaptation uses a stale denominator.
//!
//! All scratch lives in the struct and every FFT is an associated function over
//! disjoint fields, so `process_block` allocates nothing after construction —
//! the split-band Engine calls it on the real-time path. Self-contained radix-2
//! FFT so the bit-exact DAF core is untouched.

/// Radix-2 in-place complex FFT.
fn fft(re: &mut [f32], im: &mut [f32], inv: bool, twc: &[f32], tws: &[f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let h = len / 2;
        let stride = n / len;
        let mut i = 0;
        while i < n {
            for k in 0..h {
                let m = k * stride;
                let (cr, ci) = (twc[m], if inv { -tws[m] } else { tws[m] });
                let (ur, ui) = (re[i + k], im[i + k]);
                let (br, bi) = (re[i + k + h], im[i + k + h]);
                let (vr, vi) = (br * cr - bi * ci, br * ci + bi * cr);
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + h] = ur - vr;
                im[i + k + h] = ui - vi;
            }
            i += len;
        }
        len <<= 1;
    }
    if inv {
        let s = 1.0 / n as f32;
        for v in re.iter_mut() {
            *v *= s;
        }
        for v in im.iter_mut() {
            *v *= s;
        }
    }
}

fn fill_twiddles(n: usize, twc: &mut [f32], tws: &mut [f32]) {
    for m in 0..n / 2 {
        let th = -2.0 * std::f64::consts::PI * m as f64 / n as f64;
        twc[m] = th.cos() as f32;
        tws[m] = th.sin() as f32;
    }
}

/// Optional, conservative adaptation controller for A/B testing. The aligned
/// predicted echo and mic are compared; the low-band suppression ratio is NOT
/// used as a surrogate voice detector. Thresholds below are hypotheses, not
/// values inherited from WebRTC nor a claim of double-talk safety.
#[derive(Default)]
struct AdaptationGuard {
    mic_power: f64,
    echo_power: f64,
    cross: f64,
    render_power: f64,
    open: bool,
    good: usize,
    hold: usize,
    bootstrap: usize,
}
impl AdaptationGuard {
    fn step(&mut self, mic: &[f32], render: &[f32], echo: &[f32]) -> f32 {
        let n = mic.len() as f64;
        let power = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n;
        let mp = power(mic);
        let ep = power(echo);
        let rp = power(render);
        let cross = mic
            .iter()
            .zip(echo)
            .map(|(&d, &y)| f64::from(d) * f64::from(y))
            .sum::<f64>()
            / n;
        self.mic_power = 0.9 * self.mic_power + 0.1 * mp;
        self.echo_power = 0.9 * self.echo_power + 0.1 * ep;
        self.render_power = 0.9 * self.render_power + 0.1 * rp;
        self.cross = 0.9 * self.cross + 0.1 * cross;
        if rp < 1e-10 || self.render_power < 1e-10 {
            self.open = false;
            self.good = 0;
            return 0.0;
        }
        // Bootstrap is bounded to ~1 second for the supported 48k/256 case.
        // It is intentionally a much smaller step, not evidence of no near-end.
        if self.bootstrap < 188 && self.echo_power < 0.01 * self.mic_power {
            self.bootstrap += 1;
            return 0.1;
        }
        let coherence = (self.cross.max(0.0).powi(2) / (self.mic_power * self.echo_power + 1e-24))
            .clamp(0.0, 1.0);
        // A sudden incoherent onset closes immediately rather than waiting
        // for the smoothed statistic to absorb a new near-end syllable.
        let instant = cross.max(0.0).powi(2) / (mp * ep + 1e-24);
        if coherence < 0.55 || instant < 0.3 {
            self.open = false;
            self.good = 0;
            self.hold = 32;
        } else if self.hold > 0 {
            self.hold -= 1;
        } else if coherence > 0.8 {
            self.good += 1;
            if self.good >= 4 {
                self.open = true;
            }
        }
        if self.open {
            ((coherence - 0.55) / 0.45) as f32
        } else {
            0.0
        }
    }
}

/// One partitioned block frequency-domain adaptive filter over the high band.
pub struct HbAec {
    b: usize,
    nfft: usize,
    bins: usize,
    p: usize,
    mu: f32,
    guard: Option<AdaptationGuard>,
    faulted: bool,
    hr: Vec<f32>,
    hi: Vec<f32>,
    xr: Vec<f32>,
    xi: Vec<f32>,
    pw: Vec<f32>,
    norm_pw: Vec<f32>, // max(current partition power, smoothed power)
    prev_ref: Vec<f32>,
    twc: Vec<f32>,
    tws: Vec<f32>,
    // scratch (owned so process_block never allocates)
    re: Vec<f32>,
    im: Vec<f32>,
    xin: Vec<f32>,
    ytime: Vec<f32>,
    ebuf: Vec<f32>,
    htime: Vec<f32>,
    nxr: Vec<f32>,
    nxi: Vec<f32>,
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
            block.is_power_of_two(),
            "block must be a nonzero power of two"
        );
        assert!(partitions > 0, "at least one partition is required");
        let nfft = block.checked_mul(2).expect("FFT size overflow");
        let bins = nfft / 2 + 1;
        let mut twc = vec![0.0; nfft / 2];
        let mut tws = vec![0.0; nfft / 2];
        fill_twiddles(nfft, &mut twc, &mut tws);
        let pb = partitions
            .checked_mul(bins)
            .expect("partition size overflow");
        Self {
            b: block,
            nfft,
            bins,
            p: partitions,
            mu: 0.3,
            guard: None,
            faulted: false,
            hr: vec![0.0; pb],
            hi: vec![0.0; pb],
            xr: vec![0.0; pb],
            xi: vec![0.0; pb],
            pw: vec![1e-6; bins],
            norm_pw: vec![1e-6; bins],
            prev_ref: vec![0.0; block],
            twc,
            tws,
            re: vec![0.0; nfft],
            im: vec![0.0; nfft],
            xin: vec![0.0; nfft],
            ytime: vec![0.0; nfft],
            ebuf: vec![0.0; nfft],
            htime: vec![0.0; nfft],
            nxr: vec![0.0; bins],
            nxi: vec![0.0; bins],
            yr: vec![0.0; bins],
            yi: vec![0.0; bins],
            er: vec![0.0; bins],
            ei: vec![0.0; bins],
            hcr: vec![0.0; bins],
            hci: vec![0.0; bins],
        }
    }

    /// Initialization-only experimental opt-in, intended for the 48k/256 setup.
    pub fn enable_adaptation_guard(&mut self) {
        self.guard = Some(AdaptationGuard::default());
    }

    pub fn is_faulted(&self) -> bool {
        self.faulted
    }

    /// Real FFT of `x[0..nfft]` → half spectrum `outr/outi[0..bins]`, using
    /// `re/im` as scratch. Associated fn so callers pass disjoint fields.
    #[allow(clippy::too_many_arguments)]
    fn rfft(
        x: &[f32],
        nfft: usize,
        bins: usize,
        outr: &mut [f32],
        outi: &mut [f32],
        re: &mut [f32],
        im: &mut [f32],
        twc: &[f32],
        tws: &[f32],
    ) {
        re[..nfft].copy_from_slice(&x[..nfft]);
        for v in im[..nfft].iter_mut() {
            *v = 0.0;
        }
        fft(&mut re[..nfft], &mut im[..nfft], false, twc, tws);
        outr[..bins].copy_from_slice(&re[..bins]);
        outi[..bins].copy_from_slice(&im[..bins]);
    }

    /// Inverse real FFT of a Hermitian half spectrum → `out[0..nfft]`.
    #[allow(clippy::too_many_arguments)]
    fn irfft(
        br: &[f32],
        bi: &[f32],
        nfft: usize,
        bins: usize,
        out: &mut [f32],
        re: &mut [f32],
        im: &mut [f32],
        twc: &[f32],
        tws: &[f32],
    ) {
        re[..bins].copy_from_slice(&br[..bins]);
        im[..bins].copy_from_slice(&bi[..bins]);
        for k in 1..nfft / 2 {
            re[nfft - k] = br[k];
            im[nfft - k] = -bi[k];
        }
        fft(&mut re[..nfft], &mut im[..nfft], true, twc, tws);
        out[..nfft].copy_from_slice(&re[..nfft]);
    }

    /// Cancel the high-band echo in one block of `b` samples. `adapt` enables
    /// weight updates (cleared by the caller during near-end / double-talk).
    /// Writes the echo-removed high band into `out[0..b]`.
    pub fn process_block(&mut self, mic: &[f32], refb: &[f32], adapt: bool, out: &mut [f32]) {
        let (b, bins, p, nfft) = (self.b, self.bins, self.p, self.nfft);
        if self.faulted {
            out.fill(0.0);
            return;
        }
        if mic.len() != b
            || refb.len() != b
            || out.len() != b
            || mic.iter().chain(refb).any(|v| !v.is_finite())
        {
            self.faulted = true;
            out.fill(0.0);
            return;
        }
        // Overlap-save reference block: [prev | current].
        self.xin[..b].copy_from_slice(&self.prev_ref);
        self.xin[b..].copy_from_slice(&refb[..b]);
        self.prev_ref.copy_from_slice(&refb[..b]);
        // Shift the partition ring down and insert the new spectrum at p=0.
        self.xr.copy_within(0..(p - 1) * bins, bins);
        self.xi.copy_within(0..(p - 1) * bins, bins);
        Self::rfft(
            &self.xin,
            nfft,
            bins,
            &mut self.nxr,
            &mut self.nxi,
            &mut self.re,
            &mut self.im,
            &self.twc,
            &self.tws,
        );
        self.xr[..bins].copy_from_slice(&self.nxr);
        self.xi[..bins].copy_from_slice(&self.nxi);
        // Predict echo spectrum Y = Σ_p H_p * X_p.
        for v in self.yr.iter_mut() {
            *v = 0.0;
        }
        for v in self.yi.iter_mut() {
            *v = 0.0;
        }
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
        Self::irfft(
            &self.yr,
            &self.yi,
            nfft,
            bins,
            &mut self.ytime,
            &mut self.re,
            &mut self.im,
            &self.twc,
            &self.tws,
        );
        for i in 0..b {
            out[i] = mic[i] - self.ytime[b + i];
        }
        if out.iter().any(|v| !v.is_finite()) {
            self.faulted = true;
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
        let permission = if let Some(guard) = &mut self.guard {
            guard.step(mic, refb, &self.ytime[b..])
        } else if adapt {
            1.0
        } else {
            0.0
        };
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
            self.faulted = true;
            out.fill(0.0);
            return;
        }
        if permission == 0.0 {
            return;
        }
        // E = rfft([0; err]); per-bin reference power over all partitions.
        for v in self.ebuf[..b].iter_mut() {
            *v = 0.0;
        }
        Self::rfft(
            &self.ebuf,
            nfft,
            bins,
            &mut self.er,
            &mut self.ei,
            &mut self.re,
            &mut self.im,
            &self.twc,
            &self.tws,
        );
        // Regularise by a fraction of the mean reference power, not a fixed floor:
        // the >8 kHz band carries little energy, so a fixed floor lets bins with
        // near-zero power take huge NLMS steps and the filter diverges (observed
        // as amplified near-end on real audio). A dynamic floor + weight leakage
        // keeps it bounded.
        let eps = (1e-2 * mean_pw).max(1e-6);
        const LEAK: f32 = 0.9995;
        // H_p = LEAK*H_p + mu * conj(X_p) * E / power, then constrain each
        // partition to its first b taps (overlap-save gradient constraint).
        for pi in 0..p {
            let base = pi * bins;
            for k in 0..bins {
                let norm = (self.mu * permission) / (self.norm_pw[k] + eps);
                let (xr, xi) = (self.xr[base + k], self.xi[base + k]);
                let gr = xr * self.er[k] + xi * self.ei[k];
                let gi = xr * self.ei[k] - xi * self.er[k];
                self.hr[base + k] = LEAK * self.hr[base + k] + norm * gr;
                self.hi[base + k] = LEAK * self.hi[base + k] + norm * gi;
            }
            self.hcr.copy_from_slice(&self.hr[base..base + bins]);
            self.hci.copy_from_slice(&self.hi[base..base + bins]);
            Self::irfft(
                &self.hcr,
                &self.hci,
                nfft,
                bins,
                &mut self.htime,
                &mut self.re,
                &mut self.im,
                &self.twc,
                &self.tws,
            );
            for v in self.htime[b..].iter_mut() {
                *v = 0.0;
            }
            Self::rfft(
                &self.htime,
                nfft,
                bins,
                &mut self.hcr,
                &mut self.hci,
                &mut self.re,
                &mut self.im,
                &self.twc,
                &self.tws,
            );
            self.hr[base..base + bins].copy_from_slice(&self.hcr);
            self.hi[base..base + bins].copy_from_slice(&self.hci);
        }
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
mod guard_tests {
    use super::*;
    #[test]
    fn inactive_render_freezes_and_nearend_onset_holds() {
        let mut guard = AdaptationGuard::default();
        let signal = [0.01; 256];
        assert_eq!(guard.step(&signal, &[0.0; 256], &signal), 0.0);
        for _ in 0..100 {
            guard.step(&signal, &signal, &signal);
        }
        assert!(guard.open);
        let near: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 0.2 } else { -0.2 })
            .collect();
        assert_eq!(guard.step(&near, &signal, &signal), 0.0);
        assert!(!guard.open);
        assert!(guard.hold > 0);
    }
    #[test]
    fn numerical_fault_does_not_emit_nonfinite_high_band() {
        let mut aec = HbAec::new(256, 8);
        let mut out = [1.0; 256];
        aec.process_block(&[f32::NAN; 256], &[0.0; 256], true, &mut out);
        assert!(aec.is_faulted());
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
