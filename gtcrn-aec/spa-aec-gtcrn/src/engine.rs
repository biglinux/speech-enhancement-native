//! The split-band canceller: the band up to 8 kHz through the neural model at
//! 16 kHz, the band above through a linear adaptive filter, recombined at 48 kHz.

use gtcrn_aec::{Model, Streamer, hbaec::HbAec};
use ops::resample::{Down3, Up3};

pub(crate) const MAX_CHUNK: usize = 768; // one 16 kHz neural hop at 48 kHz
// Input-to-output delay of both bands, independent of the host block size:
// the pre-filled output hop, the STFT overlap-add hop, and the down- and
// up-sampling filters. 2 * 768 + 2 * 96 = 1728 samples, 36 ms.
const AEC_LATENCY_48K: usize = 2 * MAX_CHUNK + 2 * ops::resample::LPF_DELAY;
const MIC_RING: usize = 8192; // > 2·latency + LPF span + one bounded chunk

/// High-band filter block size and partition count (48 kHz): 64 partitions of
/// 256 samples ≈ 341 ms of echo-tail coverage.
const HB_BLOCK: usize = 256;
const HB_PARTS: usize = 64;

pub(crate) struct Engine {
    model: Model,
    streamer: Streamer,
    down_mic: Down3,
    down_ref: Down3,
    up_out: Up3,
    fifo_mic: Vec<f32>, // 16 kHz, awaiting a full hop
    fifo_ref: Vec<f32>,
    fifo_out: Vec<f32>,     // 48 kHz, awaiting emit
    lpf: Vec<f32>,          // 7.6 kHz low-pass prototype (complement gives the hi band)
    mic_hist: Box<[f32]>,   // raw 48 kHz mic ring, indexed by absolute sample count
    in_count: usize,        // total mic samples pushed
    out_count: usize,       // total samples emitted
    rs: Vec<f32>,           // reused resampler output
    fifo_gain: Vec<f32>,    // per-output-sample low-band suppression ratio, parallel to fifo_out
    gain_scratch: Vec<f32>, // reused per-emit gain buffer
    hb_gain: f32,           // smoothed high-band gain
    ref_hist: Box<[f32]>,   // raw 48 kHz reference ring
    hb: HbAec,              // high-band linear echo canceller
    hb_mic: Vec<f32>,       // accumulating high-band mic block (HB_BLOCK)
    hb_ref: Vec<f32>,       // accumulating high-band reference block
    hbout: Box<[f32]>,      // echo-cancelled high band, indexed by absolute sample count
    hb_t: usize,            // next absolute sample fed into the high-band filter
    hop_gain: [f32; 4],     // low-band output/input amplitude ratio of recent hops
    hops: usize,            // neural hops processed
    previous_mic_energy: f64,
    hb_blk: Vec<f32>, // reused per-block output
}

// `init` builds the engine on the main thread and `run` uses it on the data
// thread through a raw pointer, which the compiler cannot check.
const _: () = {
    const fn is_send<T: Send>() {}
    is_send::<Engine>()
};

impl Engine {
    pub(crate) fn new(model: Model) -> Result<Self, String> {
        let streamer = Streamer::new(&model)?;
        // Reserve a complete hop before emitting audio. Starting empty inserts
        // zeros at callback boundaries until enough slack accumulates, making
        // latency depend on the callback size and desynchronizing the bands.
        // Most queued before an emit: the pre-filled hop of slack, the hop the
        // block completes, and the two samples by which the down-sampler's
        // phase lets a hop complete before its last input arrives.
        let mut fifo_out = Vec::with_capacity(2 * MAX_CHUNK + 2);
        fifo_out.resize(MAX_CHUNK, 0.0);
        let mut fifo_gain = Vec::with_capacity(2 * MAX_CHUNK + 2);
        fifo_gain.resize(MAX_CHUNK, 1.0);
        Ok(Self {
            model,
            streamer,
            down_mic: Down3::new(MAX_CHUNK),
            down_ref: Down3::new(MAX_CHUNK),
            up_out: Up3::new(Streamer::HOP),
            fifo_mic: Vec::with_capacity(2 * Streamer::HOP),
            fifo_ref: Vec::with_capacity(2 * Streamer::HOP),
            fifo_out,
            lpf: ops::resample::lpf_prototype_48k(),
            mic_hist: vec![0.0; MIC_RING].into_boxed_slice(),
            in_count: 0,
            out_count: 0,
            rs: Vec::with_capacity(MAX_CHUNK),
            fifo_gain,
            gain_scratch: Vec::with_capacity(MAX_CHUNK),
            hb_gain: 1.0,
            ref_hist: vec![0.0; MIC_RING].into_boxed_slice(),
            hb: HbAec::new(HB_BLOCK, HB_PARTS),
            hb_mic: Vec::with_capacity(HB_BLOCK),
            hb_ref: Vec::with_capacity(HB_BLOCK),
            hbout: vec![0.0; MIC_RING].into_boxed_slice(),
            hb_t: 0,
            hop_gain: [1.0; 4],
            hops: 0,
            previous_mic_energy: 0.0,
            hb_blk: vec![0.0; HB_BLOCK],
        })
    }

    /// High band (above ~8 kHz) of `hist[bi]`: the sample minus its zero-phase
    /// low-pass. Reads forward to `bi + LPF_DELAY`, so the caller must have
    /// that much history.
    fn band_hi(&self, hist: &[f32], bi: i64) -> f32 {
        let d = ops::resample::LPF_DELAY as i64;
        if bi < d {
            return 0.0; // startup: not enough history yet
        }
        let taps = self.lpf.len();
        let mut lp = 0.0;
        for k in 0..taps {
            lp += self.lpf[k] * hist[((bi + d - k as i64) as usize) % MIC_RING];
        }
        hist[bi as usize % MIC_RING] - lp
    }

    /// Run the high-band canceller over every sample whose low-pass window is
    /// complete, filling `hbout`. The filter adapts only while the low band is
    /// strongly suppressed, a sign that echo dominates (see `hbaec`).
    ///
    /// Each block reads the ratio of the newest hop whose input it has
    /// entirely seen, a choice fixed by sample counts alone, so the output
    /// does not depend on how the host splits the stream.
    fn feed_high_band(&mut self) {
        let d = ops::resample::LPF_DELAY;
        while (self.hb_t + d) < self.in_count {
            let t = self.hb_t as i64;
            self.hb_mic.push(self.band_hi(&self.mic_hist, t));
            self.hb_ref.push(self.band_hi(&self.ref_hist, t));
            self.hb_t += 1;
            if self.hb_mic.len() == HB_BLOCK {
                // The block needed `hb_t + d` input samples; hop k is complete
                // once `(k + 1) * MAX_CHUNK` have arrived.
                let g = match ((self.hb_t + d) / MAX_CHUNK).checked_sub(1) {
                    Some(k) => {
                        debug_assert!(k < self.hops && k + self.hop_gain.len() >= self.hops);
                        self.hop_gain[k % self.hop_gain.len()]
                    }
                    None => 1.0,
                };
                let adapt = g < 0.5;
                self.hb
                    .process_block(&self.hb_mic, &self.hb_ref, adapt, &mut self.hb_blk);
                let start = self.hb_t - HB_BLOCK;
                for (i, &v) in self.hb_blk.iter().enumerate() {
                    self.hbout[(start + i) % MIC_RING] = v;
                }
                self.hb_mic.clear();
                self.hb_ref.clear();
            }
        }
    }

    /// Process one host block of at most `MAX_CHUNK` samples; the bound keeps
    /// the history rings from wrapping before the high band has consumed them.
    pub(crate) fn run(&mut self, rec: &[f32], play: &[f32], out: &mut [f32]) {
        assert!(rec.len() <= MAX_CHUNK && play.len() == rec.len() && out.len() == rec.len());
        for (&m, &r) in rec.iter().zip(play.iter()) {
            self.mic_hist[self.in_count % MIC_RING] = m;
            self.ref_hist[self.in_count % MIC_RING] = r;
            self.in_count += 1;
        }
        self.down_mic.process(rec, &mut self.rs);
        self.fifo_mic.extend_from_slice(&self.rs);
        self.down_ref.process(play, &mut self.rs);
        self.fifo_ref.extend_from_slice(&self.rs);
        let mut consumed = 0;
        while let (Some(m), Some(r)) = (
            self.fifo_mic[consumed..].first_chunk(),
            self.fifo_ref[consumed..].first_chunk(),
        ) {
            let mut o16 = self.streamer.process_hop(&self.model, m, r);
            // Suppression ratio of this hop, which drives the high-band duck. The
            // output frame need not describe the same instant as the input.
            let energy =
                |samples: &[f32]| samples.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
            let me = energy(m);
            let oe = energy(&o16);
            // The STFT output includes the previous microphone hop. Keep that
            // hop's energy in the ceiling so a speech offset is not cut short.
            const MAX_GROWTH: f64 = 4.0; // energy ratio, i.e. 2x amplitude
            let ceiling = me.max(self.previous_mic_energy) * MAX_GROWTH;
            self.previous_mic_energy = me;
            let oe = if oe > ceiling {
                let s = (ceiling / oe).sqrt() as f32;
                for v in &mut o16 {
                    *v *= s;
                }
                ceiling
            } else {
                oe
            };
            let g = if me > 1e-9 {
                (oe / me).sqrt().clamp(0.0, 50.0) as f32
            } else {
                1.0
            };
            self.hop_gain[self.hops % self.hop_gain.len()] = g;
            self.hops += 1;
            self.up_out.process(&o16, &mut self.rs);
            self.fifo_out.extend_from_slice(&self.rs);
            self.fifo_gain.resize(self.fifo_out.len(), g);
            consumed += Streamer::HOP;
        }
        if consumed > 0 {
            self.fifo_mic.drain(..consumed);
            self.fifo_ref.drain(..consumed);
        }
        self.feed_high_band();
        // emit out.len() samples; zero-fill during the startup fill
        let n = out.len();
        let avail = self.fifo_out.len().min(n);
        out[..avail].copy_from_slice(&self.fifo_out[..avail]);
        out[avail..].fill(0.0);
        // Gains aligned with the emitted samples (passthrough where zero-filled).
        self.gain_scratch.clear();
        self.gain_scratch.resize(n, 1.0);
        self.gain_scratch[..avail].copy_from_slice(&self.fifo_gain[..avail]);
        self.fifo_out.drain(..avail);
        self.fifo_gain.drain(..avail);
        // Add the high band delayed by the same latency as the low band (output
        // sample p aligns with mic[p - latency]). It is ducked while the low
        // band is strongly suppressed, which means far-end echo dominates, and
        // passes while the low band is kept (near-end speech, double talk).
        // The one-pole smoothing avoids zipper noise on gain changes.
        const G_LO: f32 = 0.15; // full duck at/below this low-band ratio
        const G_HI: f32 = 0.45; // full passthrough at/above
        const SMOOTH: f32 = 0.0021; // ~10 ms one-pole at 48 kHz
        let mut hb = self.hb_gain;
        for (j, o) in out.iter_mut().enumerate() {
            let target = ((self.gain_scratch[j] - G_LO) / (G_HI - G_LO)).clamp(0.0, 1.0);
            hb += (target - hb) * SMOOTH;
            // The latency exceeds the high-band filter's lag (LPF_DELAY plus
            // one block), so the sample is always ready; before it, silence.
            let hi = match (self.out_count + j).checked_sub(AEC_LATENCY_48K) {
                Some(idx) => {
                    debug_assert!(idx + HB_BLOCK + ops::resample::LPF_DELAY < self.in_count);
                    self.hbout[idx % MIC_RING]
                }
                None => 0.0,
            };
            *o += hb * hi;
        }
        self.hb_gain = hb;
        self.out_count += n;
    }
}

#[cfg(test)]
pub(crate) const TEST_MODEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../model/localvqe-pi-aec-v1-49k-f32.gguf"
);

#[cfg(test)]
pub(crate) fn test_engine() -> Engine {
    let path = std::env::var("AEC_GTCRN_GGUF").unwrap_or_else(|_| TEST_MODEL.into());
    let model = Model::load(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    Engine::new(model).unwrap()
}

#[cfg(test)]
mod split_band_tests {
    use super::{Engine, test_engine};

    /// Run `mic`/`far` through the engine in 480-sample host blocks.
    fn drive(eng: &mut Engine, mic: &[f32], far: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; mic.len()];
        let blocks = mic
            .as_chunks::<480>()
            .0
            .iter()
            .zip(far.as_chunks::<480>().0);
        for ((m, r), o) in blocks.zip(out.as_chunks_mut::<480>().0) {
            eng.run(m, r, o);
        }
        out
    }

    fn sine(freq: f32, amplitude: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 48_000.0).sin())
            .collect()
    }

    // Split `x` into (low, high) band energy with the plugin's own low-pass.
    fn band_energies(x: &[f32]) -> (f64, f64) {
        let lpf = ops::resample::lpf_prototype_48k();
        let d = ops::resample::LPF_DELAY;
        let taps = lpf.len();
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for bi in taps..x.len().saturating_sub(d) {
            let mut low = 0.0f32;
            for k in 0..taps {
                low += lpf[k] * x[bi + d - k];
            }
            let high = x[bi] - low;
            lo += f64::from(low) * f64::from(low);
            hi += f64::from(high) * f64::from(high);
        }
        (lo, hi)
    }

    #[test]
    fn host_block_size_leaves_output_unchanged() {
        let mic: Vec<f32> = (0..48_000)
            .map(|i| 0.1 * (i as f32 * 0.037).sin() + 0.03 * (i as f32 * 1.7).sin())
            .collect();
        let reference = vec![0.0; mic.len()];
        let render = |mut engine: Engine, quantum: usize| {
            let mut out = vec![0.0; mic.len()];
            for ((rec, play), out) in mic
                .chunks(quantum)
                .zip(reference.chunks(quantum))
                .zip(out.chunks_mut(quantum))
            {
                engine.run(rec, play, out);
            }
            out
        };
        let expected = render(test_engine(), 768);
        for quantum in [1, 128, 256, 480, 767] {
            let actual = render(test_engine(), quantum);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 1e-6, "quantum {quantum}: max sample error {error}");
        }
    }

    #[test]
    fn both_bands_have_the_documented_latency() {
        // One click with a quiet bed: the output's strongest sample is the
        // click, delayed by the plugin latency.
        let mut mic = sine(300.0, 1e-3, 3 * 48_000);
        let click = 2 * 48_000;
        mic[click] = 0.5;
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let peak = (0..out.len())
            .max_by(|&a, &b| out[a].abs().total_cmp(&out[b].abs()))
            .unwrap();
        assert_eq!(peak - click, super::AEC_LATENCY_48K);
    }

    #[test]
    fn silent_microphone_emits_no_echo() {
        let mut seed = 17_u32;
        let reference: Vec<f32> = (0..3 * 48_000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                0.2 * ((seed >> 9) as f32 / 8_388_608.0 - 1.0)
            })
            .collect();
        let mic: Vec<f32> = reference
            .iter()
            .enumerate()
            .map(|(i, &r)| if i < 48_000 { 0.4 * r } else { 0.0 })
            .collect();
        let out = drive(&mut test_engine(), &mic, &reference);
        let peak = out[2 * 48_000..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(peak < 1e-7, "silent microphone emitted echo: peak {peak}");
    }

    // The high band must be the mic's complement above 8 kHz: ~0 for an
    // in-band tone, ~full for a tone above it. Isolates the split filter.
    fn hiband_rms(freq: f32) -> f32 {
        let mut eng = test_engine();
        let mic = sine(freq, 0.2, 48_000);
        let (s, e) = (24_000, 43_200);
        let mut acc = 0.0f32;
        let delay = ops::resample::LPF_DELAY;
        for (i, &sample) in mic.iter().enumerate() {
            eng.mic_hist[eng.in_count % super::MIC_RING] = sample;
            eng.in_count += 1;
            // Consume while the FIR window is still present in the ring.
            if i >= delay {
                let bi = i - delay;
                if (s..e).contains(&bi) {
                    let value = eng.band_hi(&eng.mic_hist, bi as i64);
                    acc += value * value;
                }
            }
        }
        (acc / (e - s) as f32).sqrt()
    }

    #[test]
    fn hi_band_is_the_complement() {
        let inrms = 0.2 / 2f32.sqrt();
        let lo = hiband_rms(1000.0);
        let hi = hiband_rms(12000.0);
        assert!(lo < 0.05 * inrms, "1 kHz leaked into hi band: {lo:.4}");
        assert!(hi > 0.9 * inrms, "12 kHz missing from hi band: {hi:.4}");
    }

    // The network cancels only up to 8 kHz; on far-end-only echo the high band
    // must still be removed, by the linear filter and the duck.
    #[test]
    fn far_end_echo_cancelled_in_both_bands() {
        let n = 4 * 48_000;
        let mut seed = 99u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / 8_388_608.0 - 1.0
        };
        let far: Vec<f32> = (0..n)
            .map(|i| {
                let tau = 2.0 * std::f32::consts::PI * i as f32 / 48_000.0;
                0.12 * ((tau * 500.0).sin()
                    + (tau * 3000.0).sin()
                    + (tau * 11000.0).sin()
                    + (tau * 14000.0).sin())
                    + 0.02 * rng()
            })
            .collect();
        let delay = 240;
        let mic: Vec<f32> = (0..n)
            .map(|i| {
                if i >= delay {
                    0.4 * far[i - delay]
                } else {
                    0.0
                }
            })
            .collect();
        let out = drive(&mut test_engine(), &mic, &far);
        // Measure the converged tail only.
        let tail = 2 * 48_000;
        let (mic_lo, mic_hi) = band_energies(&mic[tail..]);
        let (out_lo, out_hi) = band_energies(&out[tail..]);
        let erle = |i: f64, o: f64| 10.0 * (i / o.max(1e-12)).log10();
        let (erle_lo, erle_hi) = (erle(mic_lo, out_lo), erle(mic_hi, out_hi));
        eprintln!("per-band ERLE: <8kHz = {erle_lo:.1} dB, >8kHz = {erle_hi:.1} dB");
        assert!(erle_lo > 12.0, "low band not cancelled: {erle_lo:.1} dB");
        assert!(erle_hi > 12.0, "high band not cancelled: {erle_hi:.1} dB");
    }

    // A 12 kHz tone, above the network's band, must reach the output.
    #[test]
    fn high_band_survives() {
        let mic = sine(12000.0, 0.2, 48_000);
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let rms = |x: &[f32]| (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let (mi, oi) = (rms(&mic[24_000..]), rms(&out[24_000..]));
        assert!(
            oi > 0.5 * mi,
            "12 kHz tone lost: out RMS {oi:.4} vs in {mi:.4}"
        );
    }

    // Without a far-end reference the low band is kept (ratio ~1), so the duck
    // must leave the near-end high band alone.
    #[test]
    fn near_end_high_band_is_preserved() {
        let low = sine(1000.0, 0.2, 2 * 48_000);
        let mic: Vec<f32> = sine(12000.0, 0.2, low.len())
            .iter()
            .zip(&low)
            .map(|(h, l)| h + l)
            .collect();
        let out = drive(&mut test_engine(), &mic, &vec![0.0; mic.len()]);
        let half = mic.len() / 2;
        let (_, mhi) = band_energies(&mic[half..]);
        let (_, ohi) = band_energies(&out[half..]);
        let keep_hi = (ohi / mhi.max(1e-12)).sqrt();
        assert!(keep_hi > 0.8, "near-end >8 kHz was ducked: {keep_hi:.2}x");
    }
}
