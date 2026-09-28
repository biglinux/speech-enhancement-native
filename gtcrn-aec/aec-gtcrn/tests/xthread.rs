//! Cross-thread RT gate: the SPA plugin builds+warms the Streamer on one thread
//! and the PipeWire RT callback drives process_hop on another (the instance
//! travels as a C void*). This test replicates that: warm on thread A, then run
//! the counted first hops on a fresh thread B. A thread-local pool is empty on B
//! and allocates; an instance-owned arena does not.
use aec_gtcrn::{Model, Streamer};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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
struct ForceSend<T>(T);
unsafe impl<T> Send for ForceSend<T> {}

fn fixed_io(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut seed = 1234u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 9) as f32 / 8_388_608.0 - 1.0
    };
    let reference: Vec<f32> = (0..n)
        .map(|i| 0.3 * (i as f32 * 0.02).sin() + 0.1 * rng())
        .collect();
    let delay = 137;
    let mic: Vec<f32> = (0..n)
        .map(|i| {
            let e = if i >= delay {
                0.5 * reference[i - delay]
            } else {
                0.0
            };
            e + 0.2 * (i as f32 * 0.011).sin() + 0.02 * rng()
        })
        .collect();
    (mic, reference)
}

#[test]
fn first_hop_on_foreign_thread_is_alloc_free() {
    let Some(path) = std::env::var("AEC_GTCRN_GGUF")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    else {
        eprintln!("AEC_GTCRN_GGUF unset — skipping");
        return;
    };
    let m = Model::load(&path).expect("model");
    let mut s = Streamer::new(&m);
    let hop = 256usize;
    let (mic, reference) = fixed_io(hop * 400);
    // Warm on THIS thread (thread A), like the plugins init/negotiation path.
    let mut o = 0;
    for _ in 0..120 {
        s.process_hop(&m, &mic[o..o + hop], &reference[o..o + hop]);
        o += hop;
    }
    // Move to thread B (the RT callback) and count the FIRST hops there.
    let payload = ForceSend((s, m, mic, reference, o));
    let allocs = std::thread::spawn(move || {
        let ForceSend((mut s, m, mic, reference, mut o)) = payload;
        let _ = ALLOCS.try_with(|c| c.set(0));
        let _ = ARMED.try_with(|a| a.set(true));
        for _ in 0..50 {
            s.process_hop(&m, &mic[o..o + hop], &reference[o..o + hop]);
            o += hop;
        }
        let _ = ARMED.try_with(|a| a.set(false));
        ALLOCS.try_with(Cell::get).unwrap_or(0)
    })
    .join()
    .unwrap();
    eprintln!("foreign-thread first-50-hops allocations: {allocs}");
    assert_eq!(
        allocs, 0,
        "process_hop allocated on the foreign RT thread first call"
    );
}
