//! Real-time gate: `Daf::process` and `Streamer::process_hop` must not touch the
//! heap after warm-up; both run on the PipeWire RT thread. A counting
//! `#[global_allocator]` is armed around a run long enough that a GCC-PHAT update
//! also fires under the counter.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use aec_gtcrn::daf::Daf;
use aec_gtcrn::gguf::Gguf;
use aec_gtcrn::{Model, Streamer};

// Per-thread counters: the harness runs tests in parallel and the allocator is
// process-global, so a shared flag would count a sibling test's allocations.
// Counting only the arming thread's own allocations makes each RT test isolated.
// `const`-initialised Cells never allocate, so reading them inside `alloc` is safe.
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn arm(on: bool) {
    let _ = ARMED.try_with(|a| a.set(on));
}
fn allocs() -> usize {
    ALLOCS.try_with(Cell::get).unwrap_or(0)
}
fn reset_allocs() {
    let _ = ALLOCS.try_with(|c| c.set(0));
}

struct Counting;
impl Counting {
    #[inline]
    fn note() {
        let armed = ARMED.try_with(Cell::get).unwrap_or(false);
        if armed {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        Self::note();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        Self::note();
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GA: Counting = Counting;

/// Same deterministic mic + far-end reference as `tests/golden.rs::fixed_io`.
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
    let Some(path) = std::env::var("AEC_GTCRN_GGUF")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    else {
        eprintln!("AEC_GTCRN_GGUF unset — skipping RT alloc gate");
        return;
    };
    let gg = Gguf::load(&path).expect("load gguf");
    let mut daf = Daf::new(&gg).expect("daf");
    let hop = 128usize; // == M
    let (mic, reference) = fixed_io(64_000);
    let mut e = vec![0.0f32; hop];
    let mut y = vec![0.0f32; hop];

    // Warm past the first GCC-PHAT update (n_seen >= G_WIN = 16384), then arm and
    // run long enough that gcc_update fires again (G_HOP = 8000) under the counter.
    let warmup = 200; // 200*128 = 25 600 samples > G_WIN
    let armed = 200; //  200*128 = 25 600 samples > G_HOP
    let mut o = 0;
    for _ in 0..warmup {
        daf.process(
            &mic[o..o + hop],
            &reference[o..o + hop],
            hop,
            &mut e,
            &mut y,
        );
        o += hop;
    }
    reset_allocs();
    arm(true);
    for _ in 0..armed {
        daf.process(
            &mic[o..o + hop],
            &reference[o..o + hop],
            hop,
            &mut e,
            &mut y,
        );
        o += hop;
    }
    arm(false);
    assert_eq!(allocs(), 0, "Daf::process allocated on the RT path");
}

/// Full streaming path (`Streamer::process_hop` = DAF + split-band resampler +
/// GTCRN core) must be allocation-free on the RT thread after warmup: RT-01 pools
/// every GTensor and f32 scratch buffer, keeps dims on the stack, and reuses the
/// output frame, so no hop touches the global allocator.
#[test]
fn streaming_process_hop_is_alloc_free_after_warmup() {
    let Some(path) = std::env::var("AEC_GTCRN_GGUF")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
    else {
        eprintln!("AEC_GTCRN_GGUF unset — skipping streaming RT alloc gate");
        return;
    };
    let m = Model::load(&path).expect("load model");
    let mut s = Streamer::new(&m);
    let hop = 256usize;
    let (mic, reference) = fixed_io(hop * 400);

    let warm = 120; // past DAF GCC-PHAT warmup + pool fill
    let mut o = 0;
    for _ in 0..warm {
        s.process_hop(&m, &mic[o..o + hop], &reference[o..o + hop]);
        o += hop;
    }
    let count = |s: &mut Streamer, o: &mut usize, hops: usize| {
        reset_allocs();
        arm(true);
        for _ in 0..hops {
            s.process_hop(&m, &mic[*o..*o + hop], &reference[*o..*o + hop]);
            *o += hop;
        }
        arm(false);
        allocs()
    };
    let a = count(&mut s, &mut o, 50);
    let b = count(&mut s, &mut o, 100);
    eprintln!("streaming alloc: {a} over 50 hops, {b} over 100 hops");
    assert_eq!(a, 0, "Streamer::process_hop allocated on the RT path");
    assert_eq!(b, 0, "Streamer::process_hop allocated on the RT path");
}
