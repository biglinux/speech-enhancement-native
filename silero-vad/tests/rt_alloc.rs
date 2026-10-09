//! The gate runs inside the plugins' real-time `run()`: after warm-up, feeding input
//! and asking for gains must not touch the heap, at any host block size.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use silero_vad::{VoiceGate, CHUNK_48K};

thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if ARMED.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if ARMED.try_with(Cell::get).unwrap_or(false) {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GA: Counting = Counting;

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
        ARMED.with(|a| a.set(true));
        for piece in input[CHUNK_48K * 2..].chunks(block) {
            gate.feed(piece);
            for _ in piece {
                gate.gain(t);
                t += 1;
            }
        }
        ARMED.with(|a| a.set(false));
        assert_eq!(ALLOCS.with(Cell::get), 0, "block {block} allocated");
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
