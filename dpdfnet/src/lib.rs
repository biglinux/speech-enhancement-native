//! Specialized streaming DPDFNet 48 kHz HR executor. No neural-network runtime.
//! Python is used exclusively by the offline exporter and reference tests.
use std::panic::{catch_unwind, AssertUnwindSafe};

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
