//! Experimental near-only transparency, NOT a double-talk detector.
//! Digital reference silence is usable only when routing has been verified:
//! a missing/broken reference also looks silent. Consequently this is opt-in.
//! The neural and adaptive paths keep running; no model resets or RT allocation.

pub(super) struct ReferenceBypass {
    allowed: Box<[bool]>,
    quiet_samples: usize,
    dry_gain: f32,
}

impl ReferenceBypass {
    // Deliberately conservative and sample-counted, independent of quantum.
    // Not a guarantee about rooms whose reverberation lasts more than 2 s.
    const TAIL_SAMPLES: usize = 2 * 48_000;

    pub(super) fn new(ring_len: usize) -> Self {
        Self {
            allowed: vec![false; ring_len].into_boxed_slice(),
            quiet_samples: 0,
            dry_gain: 0.0,
        }
    }

    pub(super) fn observe(&mut self, reference: f32, input_index: usize) {
        // Only EXACT digital zero, not an arbitrary dBFS threshold which might
        // classify quiet far-end speech as silence. Non-finite counts as active.
        if reference == 0.0 {
            self.quiet_samples = (self.quiet_samples + 1).min(Self::TAIL_SAMPLES);
        } else {
            self.quiet_samples = 0;
        }
        let slot = input_index % self.allowed.len();
        self.allowed[slot] = self.quiet_samples >= Self::TAIL_SAMPLES;
    }

    pub(super) fn mix(&mut self, wet: f32, delayed_dry: f32, input_index: usize) -> f32 {
        let target = if self.allowed[input_index % self.allowed.len()] {
            1.0
        } else {
            0.0
        };
        // Return to AEC within 2 ms when render resumes; open transparency over
        // 20 ms. Latency alignment is the caller's contract, not solved by fading.
        let step = if target < self.dry_gain {
            1.0 / 96.0
        } else {
            1.0 / 960.0
        };
        self.dry_gain += (target - self.dry_gain).clamp(-step, step);
        wet * (1.0 - self.dry_gain) + delayed_dry * self.dry_gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digital_silence_requires_the_whole_tail_and_quantum_does_not_matter() {
        let mut b = ReferenceBypass::new(8192);
        for i in 0..ReferenceBypass::TAIL_SAMPLES - 1 {
            b.observe(0.0, i);
            assert_eq!(b.mix(0.25, 0.75, i), 0.25);
        }
        let start = ReferenceBypass::TAIL_SAMPLES - 1;
        for i in start..start + 1000 {
            b.observe(0.0, i);
            b.mix(0.25, 0.75, i);
        }
        assert_eq!(b.dry_gain, 1.0);
        assert_eq!(b.mix(0.25, 0.75, start + 999), 0.75);
        for i in start + 1000..start + 1100 {
            b.observe(0.1, i);
            b.mix(0.25, 0.75, i);
        }
        assert_eq!(b.dry_gain, 0.0);
    }

    #[test]
    fn quiet_or_invalid_reference_is_not_digital_silence() {
        for r in [1e-20, -1e-20, f32::NAN, f32::INFINITY] {
            let mut b = ReferenceBypass::new(16);
            b.quiet_samples = ReferenceBypass::TAIL_SAMPLES;
            b.observe(r, 5);
            assert!(!b.allowed[5]);
            assert_eq!(b.quiet_samples, 0);
        }
    }
}
