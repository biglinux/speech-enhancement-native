//! Streaming 48 kHz ↔ 16 kHz resamplers (exact 3:1 ratio): PipeWire delivers
//! 48 kHz, GTCRN-AEC and the Silero voice detector run at 16 kHz. A windowed-sinc
//! low-pass (cutoff ~7.6 kHz, below both Nyquists) is applied as a polyphase FIR,
//! decimating by 3 on the way in and interpolating by 3 on the way out. Both keep
//! an input history so arbitrary block sizes stream without clicks.

const RATIO: usize = 3;
/// Taps on each side of the centre per 16 kHz phase: the 48 kHz prototype has
/// `2 * HALF * RATIO + 1` = 193 taps.
const HALF: usize = 32;
/// Cutoff in cycles per 48 kHz sample (7.6 kHz).
const CUTOFF: f32 = 7600.0 / 48000.0;

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
    }
}

/// The 48 kHz anti-alias low-pass prototype Down3/Up3 use (7.6 kHz cutoff, 193-tap
/// linear phase, group delay `LPF_DELAY`). Exposed so the split-band AEC can build
/// the complementary high-pass `mic - lpf(mic)`.
#[must_use]
pub fn lpf_prototype_48k() -> Vec<f32> {
    prototype(CUTOFF)
}

/// Group delay of `lpf_prototype_48k` in samples ((193-1)/2 = HALF*RATIO).
pub const LPF_DELAY: usize = HALF * RATIO;

/// Windowed-sinc low-pass prototype at cutoff `fc` (cycles/sample in the higher-rate
/// domain), length `2*HALF*RATIO+1`, Hann-windowed, normalised to unity DC.
fn prototype(fc: f32) -> Vec<f32> {
    let n = 2 * HALF * RATIO + 1;
    let mut h = vec![0.0f32; n];
    let mid = (n / 2) as f32;
    let mut sum = 0.0f32;
    for (i, hi) in h.iter_mut().enumerate() {
        let t = i as f32 - mid;
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
        *hi = 2.0 * fc * sinc(2.0 * fc * t) * w;
        sum += *hi;
    }
    for hi in &mut h {
        *hi /= sum;
    }
    h
}

/// 48 kHz to 16 kHz (decimate by 3).
pub struct Down3 {
    h: Vec<f32>,
    hist: Vec<f32>, // last taps - 1 samples of 48 kHz input
    phase: usize,   // input-sample counter mod 3
    buf: Vec<f32>,  // [history | input] work buffer, reused across calls
}

impl Down3 {
    /// Reserves work storage for blocks of up to `max_input` samples, so the
    /// real-time path does not allocate. Larger blocks still work and grow it.
    #[must_use]
    pub fn new(max_input: usize) -> Self {
        Self {
            h: prototype(CUTOFF),
            hist: vec![0.0; 2 * HALF * RATIO],
            phase: 0,
            buf: Vec::with_capacity(
                max_input
                    .checked_add(2 * HALF * RATIO)
                    .expect("resampler capacity overflow"),
            ),
        }
    }

    /// Feed 48 kHz samples; writes the 16 kHz samples produced into `out`.
    /// `out` and the internal work buffer are reused, so the RT path never
    /// allocates here after warmup.
    pub fn process(&mut self, x: &[f32], out: &mut Vec<f32>) {
        let taps = self.h.len();
        // [history | new input] in one buffer, so the FIR window ending at input i
        // is the contiguous slice `buf[i..i + taps]` the SIMD dot product reads.
        let keep = taps - 1;
        self.buf.clear();
        self.buf.extend_from_slice(&self.hist);
        self.buf
            .extend(x.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        out.clear();
        let mut ph = self.phase;
        for i in 0..x.len() {
            if ph == 0 {
                out.push(crate::vdot_f32(&self.h, &self.buf[i..i + taps]));
            }
            ph = (ph + 1) % RATIO;
        }
        self.phase = (self.phase + x.len()) % RATIO;
        let n = self.buf.len();
        self.hist.copy_from_slice(&self.buf[n - keep..]);
    }
}

/// 16 kHz to 48 kHz (interpolate by 3).
pub struct Up3 {
    h: Vec<f32>,
    hist: Vec<f32>, // last 2 * HALF samples of 16 kHz input
    buf: Vec<f32>,  // [history | input] work buffer, reused across calls
}

impl Up3 {
    /// Reserves work storage for blocks of up to `max_input` samples, so the
    /// real-time path does not allocate. Larger blocks still work and grow it.
    #[must_use]
    pub fn new(max_input: usize) -> Self {
        Self {
            h: prototype(CUTOFF),
            hist: vec![0.0; 2 * HALF],
            buf: Vec::with_capacity(
                max_input
                    .checked_add(2 * HALF)
                    .expect("resampler capacity overflow"),
            ),
        }
    }

    /// Feed 16 kHz samples; writes `3×` as many 48 kHz samples into `out`.
    /// `out` and the internal work buffer are reused (RT: no per-call alloc).
    pub fn process(&mut self, x: &[f32], out: &mut Vec<f32>) {
        let phases = 2 * HALF + 1;
        let keep = phases - 1;
        // [history | new input]: input i sits at buf[keep + i] and the polyphase
        // taps read back from there.
        self.buf.clear();
        self.buf.extend_from_slice(&self.hist);
        self.buf
            .extend(x.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        out.clear();
        for i in 0..x.len() {
            let base = keep + i;
            for p in 0..RATIO {
                let mut acc = 0.0f32;
                let mut k = p;
                let mut j = 0;
                while k < self.h.len() {
                    acc += self.h[k] * self.buf[base - j];
                    k += RATIO;
                    j += 1;
                }
                out.push(acc * RATIO as f32);
            }
        }
        let n = self.buf.len();
        self.hist.copy_from_slice(&self.buf[n - keep..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn down_up_preserves_inband_tone() {
        // A 1 kHz tone (well in-band for both rates) survives 48 to 16 to 48 kHz with a
        // fixed group delay; compare after aligning and skipping filter warmup.
        let sr = 48000.0;
        let n = 48000;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr).sin())
            .collect();
        let mut mid = Vec::new();
        Down3::new(0).process(&tone, &mut mid);
        let mut back = Vec::new();
        Up3::new(0).process(&mid, &mut back);
        // energy is preserved (not collapsed) and output is finite
        assert!(back.iter().all(|v| v.is_finite()));
        let a = 4000usize;
        let ein: f64 = tone[a..n - a]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let eout: f64 = back[a..(n - a).min(back.len())]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let ratio = eout / ein;
        assert!(ratio > 0.7 && ratio < 1.4, "inband energy ratio {ratio:.3}");
    }

    #[test]
    fn down3_matches_the_direct_convolution_at_any_block_size() {
        let mut state = 1u32;
        let x: Vec<f32> = (0..9000)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / 16_777_216.0 - 0.5
            })
            .collect();
        let mut whole = Vec::new();
        Down3::new(0).process(&x, &mut whole);
        let mut d = Down3::new(0);
        let (mut pieces, mut out) = (Vec::new(), Vec::new());
        for block in x.chunks(7) {
            d.process(block, &mut out);
            pieces.extend_from_slice(&out);
        }
        assert_eq!(whole, pieces);
        // Output k is the FIR over the input ending at sample 3k, history zero.
        let h = prototype(CUTOFF);
        for (k, &y) in whole.iter().enumerate() {
            let want: f64 = (0..h.len())
                .filter_map(|j| (3 * k + j).checked_sub(h.len() - 1).map(|i| (j, i)))
                .map(|(j, i)| f64::from(h[j]) * f64::from(x[i]))
                .sum();
            assert!(
                (f64::from(y) - want).abs() < 1e-5,
                "output {k}: {y} vs {want}"
            );
        }
    }

    #[test]
    fn down3_rate() {
        let mut d = Down3::new(0);
        let mut out = Vec::new();
        d.process(&vec![0.0f32; 3000], &mut out);
        assert_eq!(out.len(), 1000);
    }
}
