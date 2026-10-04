//! DPDFNET_TEST_MODEL=/path cargo test --release --test offline -- --ignored
use dpdfnet_native::{audio::LATENCY, offline, AudioProcessor, Bundle};
fn load() -> std::sync::Arc<Bundle> {
    Bundle::open(std::env::var_os("DPDFNET_TEST_MODEL").expect("set DPDFNET_TEST_MODEL"))
        .expect("valid model bundle")
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
    offline::enhance(&load(), input, &mut out, channels, threads, 48.0)?;
    Ok(out)
}
#[test]
#[ignore = "requires exported trained weights"]
fn matches_the_ladspa_chain_with_any_thread_count() {
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
#[ignore = "requires exported trained weights"]
fn recording_shorter_than_the_lead_in() {
    let x = signal(5_000, 0.0);
    assert_eq!(enhance(&bytes(&x), 1, 4).unwrap(), bytes(&ladspa_chain(&x)));
}
#[test]
#[ignore = "requires exported trained weights"]
fn empty_input_and_partial_frames() {
    assert_eq!(enhance(&[], 2, 2).unwrap(), Vec::<u8>::new());
    assert!(enhance(&[0; 12], 2, 2).is_err());
}
