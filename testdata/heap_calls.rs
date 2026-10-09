//! Global allocator for the tests that prove a real-time path never touches the
//! heap. Included with `#[path]`; it installs itself as the test binary's
//! allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Per thread: tests run in parallel and the allocator is global, so only the
// arming thread's own calls count. Const-initialized Cells never allocate, so
// reading them inside the allocator cannot recurse.
thread_local! {
    static ARMED: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

fn note() {
    if ARMED.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|c| c.set(c.get() + 1));
    }
}

// SAFETY: forwards every call, with the caller's arguments, to the system
// allocator, which upholds the `GlobalAlloc` contract.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        note();
        // SAFETY: the caller's `alloc` contract, passed through.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        note();
        // SAFETY: `p` came from this allocator, which is `System`'s.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        note();
        // SAFETY: `p` came from this allocator, which is `System`'s.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Heap calls (allocations, frees, reallocations) `f` makes on this thread.
pub fn heap_calls(f: impl FnOnce()) -> usize {
    CALLS.with(|c| c.set(0));
    ARMED.with(|a| a.set(true));
    f();
    ARMED.with(|a| a.set(false));
    CALLS.with(Cell::get)
}
