//! Audio outside the [-1, 1] contract, as a corrupt file gives it.
use firered_vad::{Aed, StreamVad};

/// One second at 16 kHz with a single huge but finite sample in the middle.
fn corrupt() -> Vec<f32> {
    let mut audio: Vec<f32> = (0..16_000).map(|i| 0.3 * (i as f32 * 0.07).sin()).collect();
    audio[8_000] = 1e20;
    audio
}

#[test]
fn huge_samples_keep_answers_finite() {
    let mut aed = Aed::new();
    let mut answers = Vec::new();
    aed.push(&corrupt(), |p| answers.push(*p));
    aed.finish(|p| answers.push(*p));
    assert!(!answers.is_empty());
    assert!(answers.iter().flatten().all(|v| v.is_finite()));

    let mut vad = StreamVad::new(16_000);
    let mut speech = Vec::new();
    vad.push(&corrupt(), |p| speech.push(p));
    assert!(!speech.is_empty());
    assert!(speech.iter().all(|v| v.is_finite()));
}
