//! Real-time gate: `Daf::process` and `Streamer::process_hop` run on the
//! PipeWire data thread and must not touch the heap after warm-up, also on the
//! first call from a thread other than the one that built them. A counting
//! global allocator is armed around runs long enough for a GCC-PHAT update.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use aec_gtcrn::daf::Daf;
use aec_gtcrn::{Model, Streamer};

const HOP: usize = Streamer::HOP;

// Per-thread counters: tests run in parallel and the allocator is global, so
// only the arming thread's own heap calls count. Const-initialised Cells never
// allocate, so reading them inside the allocator is safe.
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static HEAP_CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;
impl Counting {
    fn note() {
        if ARMED.try_with(Cell::get).unwrap_or(false) {
            let _ = HEAP_CALLS.try_with(|c| c.set(c.get() + 1));
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        Self::note();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        Self::note();
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        Self::note();
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GA: Counting = Counting;

/// Heap calls (allocations, frees, reallocations) made by `f` on this thread.
fn heap_calls(f: impl FnOnce()) -> usize {
    HEAP_CALLS.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    HEAP_CALLS.with(Cell::get)
}

fn model() -> Model {
    let path = std::env::var("AEC_GTCRN_GGUF").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../model/localvqe-pi-aec-v1-49k-f32.gguf"
        )
        .into()
    });
    Model::load(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Deterministic echo plus near-end speech stand-in, `n` samples at 16 kHz.
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
            let echo = if i >= delay {
                0.5 * reference[i - delay]
            } else {
                0.0
            };
            echo + 0.2 * (i as f32 * 0.011).sin() + 0.02 * rng()
        })
        .collect();
    (mic, reference)
}

#[test]
fn daf_process_is_alloc_free_after_warmup() {
    let m = model();
    let mut daf = Daf::new(&m.w).expect("daf");
    let block = 128;
    let (mic, reference) = fixed_io(64_000);
    let mut e = vec![0.0f32; block];
    let mut y = vec![0.0f32; block];
    let mut blocks = mic.chunks_exact(block).zip(reference.chunks_exact(block));
    // 200 blocks pass the first GCC-PHAT update (16384 samples); the next
    // 200 contain another one (every 8000 samples).
    for (mic, reference) in blocks.by_ref().take(200) {
        daf.process(mic, reference, block, &mut e, &mut y);
    }
    let calls = heap_calls(|| {
        for (mic, reference) in blocks.take(200) {
            daf.process(mic, reference, block, &mut e, &mut y);
        }
    });
    assert_eq!(calls, 0, "Daf::process used the heap");
}

#[test]
fn process_hop_is_alloc_free_after_warmup() {
    let m = model();
    let mut s = Streamer::new(&m).unwrap();
    let (mic, reference) = fixed_io(HOP * 270);
    let mut hops = mic.as_chunks().0.iter().zip(reference.as_chunks().0);
    for (mic, reference) in hops.by_ref().take(120) {
        s.process_hop(&m, mic, reference);
    }
    let calls = heap_calls(|| {
        for (mic, reference) in hops {
            s.process_hop(&m, mic, reference);
        }
    });
    assert_eq!(calls, 0, "Streamer::process_hop used the heap");
}

// The plugin builds the streamer on the main thread and PipeWire runs it on the
// data thread, so its scratch must belong to the instance, not to a thread.
#[test]
fn first_hops_on_another_thread_are_alloc_free() {
    let m = model();
    let mut s = Streamer::new(&m).unwrap();
    let (mic, reference) = fixed_io(HOP * 170);
    for (mic, reference) in mic
        .as_chunks()
        .0
        .iter()
        .zip(reference.as_chunks().0)
        .take(120)
    {
        s.process_hop(&m, mic, reference);
    }
    let calls = std::thread::spawn(move || {
        let hops = mic.as_chunks().0.iter().zip(reference.as_chunks().0);
        heap_calls(|| {
            for (mic, reference) in hops.skip(120) {
                s.process_hop(&m, mic, reference);
            }
        })
    })
    .join()
    .unwrap();
    assert_eq!(
        calls, 0,
        "process_hop used the heap on its first calls from a new thread"
    );
}
