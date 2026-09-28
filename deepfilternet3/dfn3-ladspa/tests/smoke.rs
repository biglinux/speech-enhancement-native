//! Self-contained behavioural tests: the engine stays finite, suppresses noise,
//! and is deterministic. Correctness against the reference model is verified
//! separately (stage-by-stage vs the official ONNX).
use dfn3_ladspa::{Dfn3, HOP};

static WEIGHTS: &[u8] = include_bytes!("../dfn3_weights.bin");

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 9) as f32 / 8_388_608.0 - 1.0
}

#[test]
fn output_is_finite() {
    let mut e = Dfn3::new(WEIGHTS);
    let mut out = [0.0f32; HOP];
    for f in 0..80 {
        let inp: Vec<f32> = (0..HOP)
            .map(|i| 0.2 * ((f * HOP + i) as f32 * 0.05).sin())
            .collect();
        e.process(&inp, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite at frame {f}");
    }
}

#[test]
fn suppresses_stationary_noise() {
    let mut e = Dfn3::new(WEIGHTS);
    let mut out = [0.0f32; HOP];
    let mut seed = 12345u32;
    let (mut in_e, mut out_e) = (0.0f64, 0.0f64);
    for f in 0..300 {
        let inp: Vec<f32> = (0..HOP).map(|_| 0.05 * lcg(&mut seed)).collect();
        e.process(&inp, &mut out);
        if f > 30 {
            in_e += inp.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            out_e += out.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        }
    }
    assert!(
        out_e < in_e * 0.5,
        "noise not suppressed: in={in_e:.3} out={out_e:.3}"
    );
}

#[test]
fn deterministic() {
    let run = || {
        let mut e = Dfn3::new(WEIGHTS);
        let mut out = [0.0f32; HOP];
        let mut acc = Vec::new();
        for f in 0..40 {
            let inp: Vec<f32> = (0..HOP)
                .map(|i| 0.15 * ((f * HOP + i) as f32 * 0.03).sin())
                .collect();
            e.process(&inp, &mut out);
            acc.extend_from_slice(&out);
        }
        acc
    };
    assert_eq!(run(), run(), "engine is not deterministic");
}
