//! Streaming DPDFNet-2 48 kHz HR engine, its LADSPA plugin and a C API.
use std::panic::{AssertUnwindSafe, catch_unwind};

pub mod audio;
mod ffi;
mod kernels;
mod ladspa;
mod layers;
mod model;
pub mod offline;
mod packed;
mod weights;
pub use audio::AudioProcessor;
pub use model::HOP;
pub use weights::{Bundle, Error, Result};

#[cfg(test)]
mod test_support;

/// Runs the body of an `extern "C"` entry: a panic must not unwind into the host.
fn no_unwind<T>(on_panic: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(on_panic)
}
