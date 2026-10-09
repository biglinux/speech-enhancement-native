//! Ducking of the residual noise floor in speech pauses.

/// Ducks the residual floor the network leaves in pauses, after the network.
///
/// The model's LSNR stays high for a while after an utterance, so the network keeps
/// applying its mask instead of its hard mute and the residual passes. Here, once
/// the LSNR has clearly dropped and a short hold after speech has expired, the
/// output is ducked by a finite depth, which keeps quiet wanted background at low
/// depths and deepens to a full mute over a sustained silence.
///
/// Scalar f32 with per-sample multiplies only, so it is identical on every SIMD
/// tier. `depth_db <= 0` is an exact pass-through.
pub(crate) struct SilenceExpander {
    /// Current smoothed gain, ramped per sample to avoid zipper noise.
    gain: f32,
    /// Hops of "protect as speech" remaining after the last clearly-speech hop.
    /// Guards word tails and short gaps from being ducked as noise.
    hold: i32,
    /// Consecutive ducked hops since the protective hold expired. Drives the
    /// progressive deepening that mutes long silences fully.
    silent: i32,
    /// Consecutive speech hops, for `OPEN_CONFIRM`.
    speech_run: i32,
}

impl SilenceExpander {
    /// One-pole ramps `1 - exp(-1/(t·sr))`: attack ~2 ms so speech gets its gain
    /// back at once, release ~120 ms so the duck fades in without a step.
    const ATTACK: f32 = 0.010_362_6;
    const RELEASE: f32 = 0.000_173_6;
    /// A hop counts as speech this many dB above the model's gate threshold, so a
    /// marginal frame does not open the expander.
    const OPEN_MARGIN_DB: f32 = 3.0;
    /// Hops after the last speech hop that still pass at unity (300 ms), so word
    /// tails and short gaps are not ducked.
    const HOLD_HOPS: i32 = 30;

    pub(crate) fn new() -> Self {
        Self {
            gain: 1.0,
            hold: 0,
            silent: 0,
            speech_run: 0,
        }
    }

    /// Consecutive speech hops that reopen a deepened silence. A mouse click
    /// flags a single hop; onsets after short gaps still open on the first.
    const OPEN_CONFIRM: i32 = 2;

    /// Ducked hops (0.5 s, after the hold) before the duck deepens toward a full
    /// mute, so gaps between words keep the finite duck.
    const DEEP_START_HOPS: i32 = 50;
    /// Hops (1 s) over which the duck ramps to a full mute, removing the residual
    /// the finite depth leaves.
    const DEEP_RAMP_HOPS: f32 = 100.0;

    /// Ducks one processed hop by up to `depth_db` unless the model's `lsnr` is
    /// clearly above its `gate_db` threshold. Non-finite `depth_db` is a pass-through.
    ///
    /// `floor_open` is a second condition to leave silence, from the plugin's
    /// per-frequency floor curve (`true` without one): it can only tighten muting.
    pub(crate) fn process_hop(
        &mut self,
        buf: &mut [f32],
        depth_db: f32,
        lsnr: f32,
        gate_db: f32,
        floor_open: bool,
    ) {
        if !depth_db.is_finite() || depth_db <= 0.0 || buf.is_empty() {
            return;
        }
        let flagged = lsnr.is_finite() && lsnr >= gate_db + Self::OPEN_MARGIN_DB;
        self.speech_run = if flagged { self.speech_run + 1 } else { 0 };
        let deep = self.silent > Self::DEEP_START_HOPS;
        let open = flagged && floor_open && (!deep || self.speech_run >= Self::OPEN_CONFIRM);
        if open {
            self.hold = Self::HOLD_HOPS;
        } else if self.hold > 0 {
            self.hold -= 1;
        }
        let target = if open || self.hold > 0 {
            self.silent = 0;
            1.0
        } else {
            self.silent += 1;
            let duck = 10f32.powf(-depth_db / 20.0);
            let over = (self.silent - Self::DEEP_START_HOPS) as f32;
            if over > 0.0 {
                duck * (1.0 - (over / Self::DEEP_RAMP_HOPS).min(1.0))
            } else {
                duck
            }
        };
        for s in buf {
            let coeff = if target > self.gain {
                Self::ATTACK
            } else {
                Self::RELEASE
            };
            self.gain += (target - self.gain) * coeff;
            *s *= self.gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SilenceExpander;

    const HOP: usize = 480;
    // A gate threshold and two LSNR values that straddle it: clearly speech and
    // clearly noise (the model reports its local SNR relative to `GATE`).
    const GATE: f32 = -18.0;
    const SPEECH_LSNR: f32 = 0.0; // well above GATE + OPEN_MARGIN
    const NOISE_LSNR: f32 = -30.0; // below GATE

    fn energy(buf: &[f32]) -> f64 {
        buf.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
    }

    fn tone(amp: f32, phase: &mut f32) -> [f32; HOP] {
        let mut b = [0.0f32; HOP];
        for s in &mut b {
            *s = amp * phase.sin();
            *phase += 0.4;
        }
        b
    }

    #[test]
    fn a_long_silence_deepens_to_a_full_mute() {
        // A brief gap keeps the finite duck; a sustained silence is muted fully,
        // removing the low-level residual that -depth dB alone leaves audible.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        let duck = 10f32.powf(-40.0 / 20.0);
        // A gap shorter than DEEP_START_HOPS keeps the finite duck.
        let mut short = 0.0;
        for _ in 0..40 {
            let mut buf = tone(1.0, &mut ph);
            ex.process_hop(&mut buf, 40.0, NOISE_LSNR, GATE, true);
            short = buf[HOP - 1].abs();
        }
        assert!(short > duck * 0.5, "a brief gap must not mute: {short}");
        // Continue into a long silence: the target ramps to a full mute.
        for _ in 0..250 {
            let mut buf = tone(1.0, &mut ph);
            ex.process_hop(&mut buf, 40.0, NOISE_LSNR, GATE, true);
        }
        let mut buf = tone(1.0, &mut ph);
        ex.process_hop(&mut buf, 40.0, NOISE_LSNR, GATE, true);
        let deep = energy(&buf);
        assert!(
            deep < 1e-6,
            "a long silence must mute fully, got energy {deep}"
        );
    }

    #[test]
    fn a_lone_click_in_deep_silence_stays_muted() {
        // Settle into a deep, fully-muted silence, then fire a one-hop impulse
        // that trips the LSNR (a mouse click): it must not reopen the gate.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..250 {
            let mut buf = tone(1.0, &mut ph);
            ex.process_hop(&mut buf, 40.0, NOISE_LSNR, GATE, true);
        }
        // A single speech-flagged hop: the click.
        let mut click = tone(1.0, &mut ph);
        ex.process_hop(&mut click, 40.0, SPEECH_LSNR, GATE, true);
        assert!(
            energy(&click) < 1e-6,
            "a lone click must stay muted, got {}",
            energy(&click)
        );
        // Sustained speech (two+ consecutive flagged hops) must reopen.
        let mut open = 0.0;
        for _ in 0..10 {
            let mut buf = tone(1.0, &mut ph);
            ex.process_hop(&mut buf, 40.0, SPEECH_LSNR, GATE, true);
            open = buf[HOP - 1].abs();
        }
        assert!(open > 0.5, "sustained speech must reopen, got {open}");
    }

    #[test]
    fn depth_zero_is_exact_passthrough() {
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..50 {
            let orig = tone(5e-4, &mut ph);
            let mut buf = orig;
            ex.process_hop(&mut buf, 0.0, NOISE_LSNR, GATE, true);
            assert_eq!(buf, orig, "depth 0 must not touch the samples");
        }
    }

    #[test]
    fn speech_passes_at_unity() {
        // Frames the model is confident are speech (LSNR above the gate) are never
        // ducked, regardless of their level.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        let mut last = 0.0;
        for _ in 0..40 {
            let refb = tone(0.3, &mut 0.0);
            let mut buf = tone(0.3, &mut ph);
            ex.process_hop(&mut buf, 30.0, SPEECH_LSNR, GATE, true);
            last = (energy(&buf) / energy(&refb)).sqrt();
        }
        assert!(last > 0.98, "speech must pass ~unity, got {last:.3}");
    }

    #[test]
    fn post_speech_noise_is_ducked_by_the_depth() {
        // Once LSNR drops below the gate and the speech hold expires, the
        // residual is attenuated toward the finite depth.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..40 {
            let mut b = tone(0.3, &mut ph);
            ex.process_hop(&mut b, 24.0, SPEECH_LSNR, GATE, true);
        }
        // Now feed sustained noise-classified hops long enough to clear the hold
        // and let the ~120 ms release settle.
        let mut last = 1.0;
        for _ in 0..120 {
            let refb = tone(0.05, &mut 0.0);
            let mut b = tone(0.05, &mut ph);
            ex.process_hop(&mut b, 24.0, NOISE_LSNR, GATE, true);
            last = (energy(&b) / energy(&refb)).sqrt();
        }
        // 24 dB is a gain of 0.063; the ramp is well on its way.
        assert!(
            last < 0.2,
            "post-speech noise should be ducked, got {last:.3}"
        );
    }

    #[test]
    fn a_brief_gap_between_words_is_held_not_chopped() {
        // A short dip below the gate (a gap between words) stays within the hold,
        // so the next word passes at unity instead of being ducked.
        let mut ex = SilenceExpander::new();
        let mut ph = 0.0;
        for _ in 0..40 {
            let mut b = tone(0.3, &mut ph);
            ex.process_hop(&mut b, 30.0, SPEECH_LSNR, GATE, true);
        }
        // A ~20 ms gap (2 hops) the model briefly reads as sub-gate.
        for _ in 0..2 {
            let mut b = tone(5e-4, &mut ph);
            ex.process_hop(&mut b, 30.0, NOISE_LSNR, GATE, true);
        }
        let refb = tone(0.3, &mut 0.0);
        let mut word = tone(0.3, &mut ph);
        ex.process_hop(&mut word, 30.0, SPEECH_LSNR, GATE, true);
        let ratio = (energy(&word) / energy(&refb)).sqrt();
        assert!(
            ratio > 0.95,
            "word after a brief gap was chopped: {ratio:.3}"
        );
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut ex = SilenceExpander::new();
            let mut ph = 0.0;
            let mut out = Vec::new();
            for hop in 0..30 {
                let (amp, lsnr) = if hop % 2 == 0 {
                    (0.3, SPEECH_LSNR)
                } else {
                    (5e-4, NOISE_LSNR)
                };
                let mut buf = tone(amp, &mut ph);
                ex.process_hop(&mut buf, 24.0, lsnr, GATE, true);
                out.extend_from_slice(&buf);
            }
            out
        };
        assert_eq!(run(), run());
    }
}
