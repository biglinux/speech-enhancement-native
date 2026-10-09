//! The gate runs inside the plugins' real-time `run()`: after warm-up, feeding input
//! and asking for gains must not touch the heap, at any host block size.
#[path = "../../testdata/heap_calls.rs"]
mod heap_calls;

use heap_calls::heap_calls;
use silero_vad::{CHUNK_48K, VoiceGate};

fn tone(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 * (i as f32 * 0.05).sin() * (i as f32 * 0.0007).sin())
        .collect()
}

#[test]
fn feeding_and_gating_never_allocate_after_warm_up() {
    for block in [1usize, 128, 480, 1024, 4096] {
        let mut gate = VoiceGate::new(40.0, 4096);
        let input = tone(CHUNK_48K * 12);
        let mut t = 0i64;
        // Warm-up: first decision, weights shared, buffers at their working size.
        for piece in input[..CHUNK_48K * 2].chunks(block) {
            gate.feed(piece);
        }
        let calls = heap_calls(|| {
            for piece in input[CHUNK_48K * 2..].chunks(block) {
                gate.feed(piece);
                for _ in piece {
                    gate.gain(t);
                    t += 1;
                }
            }
        });
        assert_eq!(calls, 0, "block {block} allocated");
    }
}

#[test]
fn invalid_input_is_heard_as_silence() {
    let mut gate = VoiceGate::new(40.0, 4096);
    gate.feed(&vec![f32::NAN; CHUNK_48K * 4]);
    gate.feed(&vec![f32::INFINITY; CHUNK_48K * 4]);
    let g = gate.gain(CHUNK_48K as i64 * 7);
    assert!(g.is_finite() && g <= 0.011);
}
