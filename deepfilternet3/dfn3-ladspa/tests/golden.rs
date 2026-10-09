//! Golden regression test: the engine output on a fixed input must stay within
//! tolerance of the committed reference. The reference is this int8 pipeline's own
//! output, whose correctness was established stage-by-stage against the upstream
//! DeepFilterNet3 ONNX (encoder/mask 100–140 dB SDR) in the project's validation.
//! Thresholds are set to run the full pipeline so every stage is exercised. The
//! 60 dB tolerance absorbs the AVX2-vs-AVX FMA tier difference while catching any
//! real numeric regression (which degrades output far below that).
//!
//! Regenerate the fixture after an intentional numeric change: `BLESS=1 cargo test
//! --test golden` (then review the diff before committing).
use dfn3_ladspa::{Dfn3, HOP};

const NFRAMES: usize = 80;

fn fixed_input() -> Vec<f32> {
    let mut seed = 1234u32;
    (0..NFRAMES * HOP)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 9) as f32 / 8_388_608.0 - 1.0;
            0.2 * (i as f32 * 0.02).sin() + 0.3 * (i as f32 * 0.005).sin() + 0.05 * noise
        })
        .collect()
}

fn engine_output() -> Vec<f32> {
    let mut e = Dfn3::new();
    // Full pipeline (no gating / stage skipping) so all stages run every frame.
    e.min_db = -100.0;
    e.max_db_erb = 100.0;
    e.max_db_df = 100.0;
    let x = fixed_input();
    let mut out = vec![0.0f32; NFRAMES * HOP];
    let mut ob = [0.0f32; HOP];
    for f in 0..NFRAMES {
        e.process(&x[f * HOP..f * HOP + HOP], &mut ob);
        out[f * HOP..f * HOP + HOP].copy_from_slice(&ob);
    }
    out
}

#[test]
fn golden_output_matches_reference() {
    let out = engine_output();
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golden.bin");
    if std::env::var("BLESS").is_ok() {
        std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures")).unwrap();
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
        return;
    }

    let ref_bytes = std::fs::read(path).expect("golden fixture missing; run once with BLESS=1");
    let (words, _) = ref_bytes.as_chunks::<4>();
    let reference: Vec<f32> = words.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(out.len(), reference.len(), "length changed");

    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (o, r) in out.iter().zip(&reference) {
        sig += (*r as f64).powi(2);
        err += ((*o - *r) as f64).powi(2);
    }
    let sdr = 10.0 * (sig / err.max(1e-30)).log10();
    assert!(
        sdr > 60.0,
        "golden SDR {sdr:.1} dB < 60 dB — numeric regression vs the committed reference"
    );
}
