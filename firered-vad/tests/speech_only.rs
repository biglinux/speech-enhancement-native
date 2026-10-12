//! The speech-only filter on the shared test speech, alone and over noise.
use firered_vad::SpeechOnly;

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
fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01
        })
        .collect()
}

/// Noise, the speech over the same noise, then noise again.
fn recording() -> (Vec<f32>, usize, usize) {
    let speech = speech_48k();
    let before = 4 * 48_000;
    let mut x = noise(before, 7);
    x.extend(
        speech
            .iter()
            .zip(noise(speech.len(), 8))
            .map(|(s, n)| s + n),
    );
    let end = x.len();
    x.extend(noise(4 * 48_000, 9));
    (x, before, end)
}

fn speech_only(x: &[f32], block: usize) -> Vec<f32> {
    let mut filter = SpeechOnly::new();
    let mut out = Vec::new();
    for piece in x.chunks(block) {
        filter.push(piece, piece, &mut out);
    }
    filter.finish(&mut out);
    out
}

#[test]
fn keeps_speech_with_margins() {
    let (x, start, end) = recording();
    let y = speech_only(&x, 4800);
    assert_eq!(y.len(), x.len());
    // The words and nearly a second before them stay as they were; the margin is
    // counted from the frame that heard the first word.
    let onset = start + speech_48k().iter().position(|v| v.abs() > 0.02).unwrap();
    assert_eq!(y[onset - 43_200..end], x[onset - 43_200..end]);
    // Noise well away from the speech is silenced.
    assert!(y[..start - 60_000].iter().all(|&v| v == 0.0));
    assert!(y[end + 48_000..].iter().all(|&v| v == 0.0));
}

#[test]
fn block_size_changes_nothing() {
    let (x, ..) = recording();
    let whole = speech_only(&x, x.len());
    for block in [1, 480, 1000, 8192] {
        assert_eq!(speech_only(&x, block), whole, "block {block}");
    }
}

#[test]
fn noise_alone_is_silenced() {
    let x = noise(10 * 48_000, 3);
    assert!(speech_only(&x, 4800).iter().all(|&v| v == 0.0));
}

#[test]
fn empty_input_gives_nothing() {
    assert!(speech_only(&[], 4800).is_empty());
}

#[test]
fn keeps_buried_speech_once_denoised() {
    let (clean, start, end) = recording();
    // The raw input: the same words far below loud noise, which the AED does not
    // hear as a voice; the denoised audio has them plainly.
    let raw: Vec<f32> = clean
        .iter()
        .zip(noise(clean.len(), 11))
        .map(|(s, n)| 0.02 * s + 20.0 * n)
        .collect();
    let mut filter = SpeechOnly::new();
    let mut y = Vec::new();
    filter.push(&raw, &clean, &mut y);
    filter.finish(&mut y);
    let onset = start + speech_48k().iter().position(|v| v.abs() > 0.02).unwrap();
    assert_eq!(y[onset..end], clean[onset..end]);
}
