//! The offline pipeline against the plugin, with the shipped model or the
//! bundle in DPDFNET_TEST_MODEL.
use dpdfnet_native::{AudioProcessor, Bundle, audio::LATENCY, offline};
fn load() -> std::sync::Arc<Bundle> {
    let dir = std::env::var_os("DPDFNET_TEST_MODEL").map_or_else(
        || concat!(env!("CARGO_MANIFEST_DIR"), "/model/dpdfnet2_48khz_hr-w8a16").into(),
        std::path::PathBuf::from,
    );
    Bundle::open(dir).expect("valid model bundle")
}
fn signal(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 * (i as f32 * 0.077 + phase).sin() + 0.02 * (i as f32 * 0.413).cos())
        .collect()
}
fn bytes(frames: &[f32]) -> Vec<u8> {
    frames.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    (0..channels[0].len())
        .flat_map(|i| channels.iter().map(move |c| c[i]))
        .collect()
}
/// What ffmpeg does around the plugin in the converters: reversed lead-in,
/// the recording, zeros for the latency, then both trimmed away.
fn ladspa_chain(x: &[f32]) -> Vec<f32> {
    let mut input = x[..offline::LEAD_IN.min(x.len())].to_vec();
    input.resize(offline::LEAD_IN, 0.0);
    input.reverse();
    input.extend_from_slice(x);
    input.resize(input.len() + LATENCY, 0.0);
    let mut p = AudioProcessor::new(load()).unwrap();
    p.set_attenuation_db(48.0);
    p.reset();
    let mut y = vec![0.0; input.len()];
    p.process(&input, &mut y);
    y[offline::LEAD_IN + LATENCY..].to_vec()
}
fn enhance(input: &[u8], channels: usize, threads: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    offline::enhance(&load(), input, &mut out, channels, threads, 48.0, false)?;
    Ok(out)
}
fn speech() -> Vec<f32> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../testdata/speech.pcm"
    ))
    .expect("testdata/speech.pcm")
    .as_chunks::<2>()
    .0
    .iter()
    .map(|&b| f32::from(i16::from_le_bytes(b)) / 32768.0)
    .collect()
}
#[test]
fn matches_ladspa_chain_on_any_threads() {
    // Not a whole number of hops, long enough for every stage to be busy.
    let channels = [
        signal(25 * 48_000 + 123, 0.0),
        signal(25 * 48_000 + 123, 1.3),
    ];
    let expected: Vec<Vec<f32>> = channels.iter().map(|x| ladspa_chain(x)).collect();
    let expected = bytes(&interleave(&expected));
    let input = bytes(&interleave(&channels));
    for threads in [1, 2, 6, 20] {
        assert_eq!(
            enhance(&input, 2, threads).unwrap(),
            expected,
            "{threads} threads"
        );
    }
}
#[test]
fn recording_shorter_than_the_lead_in() {
    let x = signal(5_000, 0.0);
    assert_eq!(enhance(&bytes(&x), 1, 4).unwrap(), bytes(&ladspa_chain(&x)));
}
#[test]
fn empty_input_and_partial_frames() {
    assert_eq!(enhance(&[], 2, 2).unwrap(), Vec::<u8>::new());
    assert!(enhance(&[0; 12], 2, 2).is_err());
}
#[test]
fn speech_only_filters_the_enhanced_output() {
    let mut voice = vec![0.0; 3 * 48_000];
    voice.extend(speech());
    voice.resize(voice.len() + 3 * 48_000, 0.0);
    let channels = [voice.clone(), signal(voice.len(), 0.4)];
    let expected: Vec<Vec<f32>> = channels
        .iter()
        .map(|raw| {
            let clean = ladspa_chain(raw);
            let mut filter = firered_vad::SpeechOnly::new();
            let mut out = Vec::new();
            filter.push(raw, &clean, &mut out);
            filter.finish(&mut out);
            out
        })
        .collect();
    // The speech is kept, the tone silenced.
    assert!(expected[0].iter().any(|&v| v != 0.0));
    assert!(expected[1].iter().all(|&v| v == 0.0));
    let expected = bytes(&interleave(&expected));
    let input = bytes(&interleave(&channels));
    for threads in [1, 6] {
        let mut out = Vec::new();
        offline::enhance(&load(), &input[..], &mut out, 2, threads, 48.0, true).unwrap();
        assert_eq!(out, expected, "{threads} threads");
    }
}
