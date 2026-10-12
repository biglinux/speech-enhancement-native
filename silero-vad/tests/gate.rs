//! The voice gate with either detector, on the shared test speech over noise.
use silero_vad::{CHUNK_48K, VoiceGate};

fn speech_48k() -> Vec<f32> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../testdata/speech.pcm");
    std::fs::read(path)
        .expect("testdata/speech.pcm")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0)
        .collect()
}

/// Deterministic noise at about -46 dBFS.
fn noise(n: usize) -> Vec<f32> {
    let mut state: u32 = 5;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01
        })
        .collect()
}

#[test]
fn detectors_open_only_on_speech() {
    let speech = speech_48k();
    let before = 3 * 48_000;
    let mut x = noise(before);
    x.extend(speech.iter().zip(noise(speech.len())).map(|(s, n)| s + n));
    let onset = before + speech.iter().position(|v| v.abs() > 0.02).unwrap();
    for firered in [false, true] {
        let mut gate = VoiceGate::new(40.0, CHUNK_48K);
        gate.use_firered(firered);
        let mut gains = Vec::with_capacity(x.len());
        for (i, piece) in x.chunks(CHUNK_48K).enumerate() {
            gate.feed(piece);
            // The output lags the input by LAG, as in the plugins.
            for k in 0..piece.len() {
                let t = (i * CHUNK_48K + k) as i64 - VoiceGate::LAG as i64;
                gains.push((t, gate.gain(t)));
            }
        }
        let gain_at = |t: usize| gains.iter().find(|&&(s, _)| s == t as i64).unwrap().1;
        assert!(
            gain_at(before / 2) < 0.02,
            "FireRedVAD {firered}: open on noise"
        );
        let half_second_in = onset + 24_000;
        assert!(
            gain_at(half_second_in) > 0.9,
            "FireRedVAD {firered}: closed on speech"
        );
    }
}
