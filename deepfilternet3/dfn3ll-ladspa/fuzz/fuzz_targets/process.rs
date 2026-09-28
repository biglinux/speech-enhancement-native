#![no_main]
//! Robustness fuzz of the shipped real-time path: adversarial-but-finite audio
//! (clipping, DC, denormals, full-scale, arbitrary frame sequences) must never
//! panic, read out of bounds, or produce a non-finite sample. A real host feeds
//! bounded samples, so non-finite fuzz bytes are sanitised to 0 and magnitudes
//! are clamped to a generous ±8.0 (+18 dBFS) — beyond that is not a real input.
use dfn3ll_ladspa::{Dfn3Ll, Weights, HOP};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;

static WEIGHTS_BLOB: &[u8] = include_bytes!("../../dfn3ll_weights.bin");

// Decode the 10.7 MB weight blob once; each iteration only builds cheap per-instance
// state, which keeps fuzz throughput high (was one full decode per case).
fn weights() -> &'static Weights {
    static W: OnceLock<Weights> = OnceLock::new();
    W.get_or_init(|| Weights::load(WEIGHTS_BLOB))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let mut e = Dfn3Ll::with_weights(weights());
    let mut out = [0.0f32; HOP];
    let mut frame = [0.0f32; HOP];
    let mut i = 0usize;
    // Cap frames so a huge input cannot dominate the run.
    for c in data.chunks_exact(4).take(HOP * 512) {
        let raw = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        frame[i] = if raw.is_finite() { raw.clamp(-8.0, 8.0) } else { 0.0 };
        i += 1;
        if i == HOP {
            e.process(&frame, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
            i = 0;
        }
    }
    if i > 0 {
        frame[i..].fill(0.0);
        e.process(&frame, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }
});
