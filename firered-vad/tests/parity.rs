//! The native models against the official package (`tools/reference.py`).

use firered_vad::{Aed, StreamVad};

const SECONDS: usize = 20;

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

/// Speech, half a second of silence, more speech, half a second of noise, repeated,
/// taken every third sample down to 16 kHz.
fn signal() -> Vec<f32> {
    let mut block: Vec<f32> = fixture("speech.pcm").into_iter().step_by(3).collect();
    block.extend(std::iter::repeat_n(0.0, 8000));
    block.extend(fixture("continuous-speech.pcm").into_iter().step_by(3));
    block.extend(lcg_noise(8000));
    block
        .iter()
        .copied()
        .cycle()
        .take(SECONDS * 16_000)
        .collect()
}

fn expected(name: &str) -> Vec<f32> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(path)
        .expect("reference outputs")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

#[test]
fn aed_matches_the_official_model() {
    let mut aed = Aed::new();
    let mut got = Vec::new();
    // Uneven blocks, as a converter feeds them.
    for piece in signal().chunks(1237) {
        aed.push(piece, |p| got.extend_from_slice(p));
    }
    aed.finish(|p| got.extend_from_slice(p));
    let worst = worst(&got, &expected("expected_aed.f32"));
    assert!(worst < 1e-4, "largest difference {worst}");
}

#[test]
fn stream_vad_matches_the_official_model() {
    let mut vad = StreamVad::new(512);
    let mut got = Vec::new();
    for piece in signal().chunks(512) {
        vad.push(piece, |p| got.push(p));
    }
    let worst = worst(&got, &expected("expected_stream_vad.f32"));
    assert!(worst < 1e-4, "largest difference {worst}");
}
