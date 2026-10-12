//! The restarting detector against fresh models, on the shared test speech.
use silero_vad::{CHUNK, Detector, Silero};

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

fn chunks(audio16: &[f32]) -> Vec<[f32; CHUNK]> {
    audio16.as_chunks::<CHUNK>().0.to_vec()
}

#[test]
fn longest_running_model_answers() {
    let speech: Vec<f32> = speech_48k().iter().step_by(3).copied().collect();
    let audio: Vec<f32> = (0..4).flat_map(|_| speech.iter().copied()).collect();
    let audio = chunks(&audio);
    let mut detector = Detector::new(128);
    for (i, chunk) in audio.iter().enumerate() {
        let p = detector.process(chunk);
        // The answering model restarted 64 to 127 chunks ago, or with the stream.
        let age = i % 128;
        let start = if age >= 64 {
            i - age
        } else {
            i.saturating_sub(age + 64)
        };
        if i % 23 == 0 {
            let mut fresh = Silero::new();
            let expected = audio[start..=i]
                .iter()
                .map(|c| fresh.process(c))
                .last()
                .unwrap();
            assert_eq!(p.to_bits(), expected.to_bits(), "chunk {i}");
        }
    }
}
