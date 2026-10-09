//! Streaming behavior with the shipped model, or the bundle in DPDFNET_TEST_MODEL.
use dpdfnet_native::{AudioProcessor, Bundle};
fn load() -> std::sync::Arc<Bundle> {
    let dir = std::env::var_os("DPDFNET_TEST_MODEL").map_or_else(
        || concat!(env!("CARGO_MANIFEST_DIR"), "/model/dpdfnet2_48khz_hr-w8a16").into(),
        std::path::PathBuf::from,
    );
    Bundle::open(dir).expect("valid model bundle")
}
fn signal(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 * (i as f32 * 0.077).sin() + 0.02 * (i as f32 * 0.413).cos())
        .collect()
}
#[test]
fn block_partition_invariance() {
    let b = load();
    let input = signal(12000);
    let mut base = AudioProcessor::new(b.clone()).unwrap();
    let mut expected = vec![0.0; input.len()];
    base.process(&input, &mut expected);
    for size in [1, 7, 64, 128, 256, 480, 512, 960, 1024, 8192] {
        let mut p = AudioProcessor::new(b.clone()).unwrap();
        let mut y = vec![0.0; input.len()];
        for (x, y) in input.chunks(size).zip(y.chunks_mut(size)) {
            p.process(x, y);
        }
        assert_eq!(y, expected, "block size {size}");
        assert!(!p.faulted());
    }
}
#[test]
fn dry_path_has_exactly_declared_latency() {
    let b = load();
    let mut p = AudioProcessor::new(b).unwrap();
    p.set_attenuation_db(0.0);
    p.reset();
    let x = signal(10000);
    let delay = dpdfnet_native::audio::LATENCY;
    let mut padded = x.clone();
    padded.resize(x.len() + delay, 0.0);
    let mut y = vec![0.0; padded.len()];
    p.process(&padded, &mut y);
    assert!(y[..delay].iter().all(|v| v.abs() < 2e-5));
    for (i, (&got, &expected)) in y[delay..].iter().zip(&x).enumerate() {
        assert!((got - expected).abs() < 2e-5, "sample {i}");
    }
}
#[test]
fn reset_is_equivalent_to_new_instance() {
    let b = load();
    let mut p = AudioProcessor::new(b.clone()).unwrap();
    let mut fresh = AudioProcessor::new(b).unwrap();
    let a = signal(4000);
    let mut scratch = vec![0.0; a.len()];
    p.process(&a, &mut scratch);
    p.reset();
    let mut x = signal(5000);
    x.reverse();
    let mut y = vec![0.0; x.len()];
    let mut reference = y.clone();
    p.process(&x, &mut y);
    fresh.process(&x, &mut reference);
    assert_eq!(y, reference);
}
#[test]
fn invalid_samples_do_not_poison_the_state() {
    let b = load();
    let mut p = AudioProcessor::new(b).unwrap();
    let mut x = signal(5000);
    let mut y = vec![0.0; x.len()];
    x[0] = f32::NAN;
    x[41] = f32::INFINITY;
    x[719] = f32::NEG_INFINITY;
    p.process(&x, &mut y);
    assert!(y.iter().all(|v| v.is_finite()));
    assert!(!p.faulted());
}
