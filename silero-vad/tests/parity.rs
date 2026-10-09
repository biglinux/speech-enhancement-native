//! The native network against the official ONNX model (`tools/reference.py`).

use ops::resample::{Down3, LPF_DELAY};
use silero_vad::{CHUNK, Silero};

const SECONDS: usize = 30;

fn fixture(name: &str) -> Vec<f32> {
    let path = format!("{}/../testdata/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(path)
        .expect("shared speech fixture")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
        .collect()
}

fn lcg_noise(count: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((f64::from(state >> 8) / 16_777_216.0 - 0.5) as f32) * 0.02
        })
        .collect()
}

/// Speech, half a second of silence, more speech, half a second of noise, repeated.
fn signal(rate: usize) -> Vec<f32> {
    let step = 48_000 / rate;
    let mut block: Vec<f32> = fixture("speech.pcm").into_iter().step_by(step).collect();
    block.extend(std::iter::repeat_n(0.0, rate / 2));
    block.extend(fixture("continuous-speech.pcm").into_iter().step_by(step));
    block.extend(lcg_noise(rate / 2));
    block.iter().copied().cycle().take(SECONDS * rate).collect()
}

fn expected(name: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(path)
        .expect("reference probabilities")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn probabilities(audio16: &[f32]) -> Vec<f32> {
    let mut vad = Silero::new();
    audio16
        .as_chunks::<CHUNK>()
        .0
        .iter()
        .map(|c| vad.process(c))
        .collect()
}

#[test]
fn matches_the_official_model_over_thirty_seconds() {
    let got = probabilities(&signal(16_000));
    let want = expected("expected_16k.f32");
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max);
    assert!(worst < 1e-3, "largest difference {worst}");
}

#[test]
fn decides_like_the_official_model_after_our_resampler() {
    // SciPy's resampler is zero-phase and ours is causal: drop our group delay, then
    // compare decisions, not digits, since the two low-pass filters still differ.
    let mut down = Down3::new(0);
    let mut audio16 = Vec::new();
    down.process(&signal(48_000), &mut audio16);
    let got = probabilities(&audio16[LPF_DELAY / 3..]);
    let want = expected("expected_48k.f32");
    let n = got.len().min(want.len());
    let mean = got[..n]
        .iter()
        .zip(&want[..n])
        .map(|(g, w)| (g - w).abs())
        .sum::<f32>()
        / n as f32;
    let agree = got[..n]
        .iter()
        .zip(&want[..n])
        .filter(|(g, w)| (**g >= 0.2) == (**w >= 0.2))
        .count();
    assert!(mean < 0.01, "mean difference {mean}");
    assert!(agree * 100 >= n * 99, "{agree}/{n} decisions agree");
}
