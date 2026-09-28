//! Streaming 48 kHz ↔ 16 kHz resamplers (exact 3:1 ratio) for the AEC plugin:
//! PipeWire delivers 48 kHz, the GTCRN model runs at 16 kHz. A windowed-sinc
//! low-pass (cutoff ~7.6 kHz, below both Nyquists) is applied as a polyphase FIR,
//! decimating by 3 on the way in and interpolating by 3 on the way out. Both keep
//! an input history so arbitrary block sizes stream without clicks.

const RATIO: usize = 3;
const HALF: usize = 32; // taps each side of centre per phase → 65-tap prototype

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
    }
}

/// The 48 kHz anti-alias low-pass prototype Down3/Up3 use (7.6 kHz cutoff, 193-tap
/// linear phase, group delay `LPF_DELAY`). Exposed so the split-band AEC can build a
/// complementary high-pass (`mic - lpf(mic)`). Replacing the low branch with
/// a neural/resampled signal does not itself guarantee perfect reconstruction.
#[must_use]
pub fn lpf_prototype_48k() -> Vec<f32> {
    prototype(7600.0 / 48000.0)
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

/// 48 kHz → 16 kHz (decimate by 3).
pub struct Down3 {
    h: Vec<f32>,
    hist: Vec<f32>, // last taps of 48 kHz input
    phase: usize,   // input-sample counter mod 3
    buf: Vec<f32>,  // reused [carry tail | new input] work buffer (RT: no per-call alloc)
}

impl Default for Down3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Down3 {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Reserve work storage at initialization; callers must also reserve out.
    /// Inputs above max_input remain supported, but may grow the buffer.
    #[must_use]
    pub fn with_capacity(max_input: usize) -> Self {
        Self {
            h: prototype(7600.0 / 48000.0),
            hist: vec![0.0; 2 * HALF * RATIO + 1],
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
        // Work on one contiguous [carry-over tail | new input] buffer instead of a
        // per-sample `copy_within` memmove; the FIR window ending at input i is
        // `buf[i..i+taps]`, so each dot is a contiguous slice (autovectorizes) and
        // the values are identical to the reference.
        let keep = taps - 1;
        self.buf.clear();
        self.buf.extend_from_slice(&self.hist[1..]); // previous taps-1 tail
        self.buf
            .extend(x.iter().map(|&v| if v.is_finite() { v } else { 0.0 }));
        out.clear();
        let mut ph = self.phase;
        for i in 0..x.len() {
            if ph == 0 {
                let w = &self.buf[i..i + taps];
                let acc: f32 = self.h[..taps].iter().zip(w).map(|(h, v)| h * v).sum();
                out.push(acc);
            }
            ph = (ph + 1) % RATIO;
        }
        self.phase = (self.phase + x.len()) % RATIO;
        let n = self.buf.len();
        self.hist[1..].copy_from_slice(&self.buf[n - keep..]);
    }
}

/// 16 kHz → 48 kHz (interpolate by 3).
pub struct Up3 {
    h: Vec<f32>,
    hist: Vec<f32>, // last taps/RATIO of 16 kHz input
    buf: Vec<f32>,  // reused work buffer (RT: no per-call alloc)
}

impl Default for Up3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Up3 {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Reserve work storage at initialization; callers must also reserve out.
    /// Inputs above max_input remain supported, but may grow the buffer.
    #[must_use]
    pub fn with_capacity(max_input: usize) -> Self {
        Self {
            h: prototype(7600.0 / 48000.0),
            hist: vec![0.0; 2 * HALF + 1],
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
        // [carry tail | new input] so the per-sample `copy_within` memmove goes
        // away; input i sits at buf[keep + i] and the polyphase taps read back from
        // there — same values as the reference.
        self.buf.clear();
        self.buf.extend_from_slice(&self.hist[1..]);
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
        self.hist[1..].copy_from_slice(&self.buf[n - keep..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn down_up_preserves_inband_tone() {
        // A 1 kHz tone (well in-band for both rates) survives 48→16→48 with a
        // fixed group delay; compare after aligning and skipping filter warmup.
        let sr = 48000.0;
        let n = 48000;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr).sin())
            .collect();
        let mut mid = Vec::new();
        Down3::new().process(&tone, &mut mid);
        let mut back = Vec::new();
        Up3::new().process(&mid, &mut back);
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
    fn down3_rate() {
        let mut d = Down3::new();
        let mut out = Vec::new();
        d.process(&vec![0.0f32; 3000], &mut out);
        assert_eq!(out.len(), 1000);
    }
}
