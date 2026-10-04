//! Specialized streaming DPDFNet 48 kHz HR executor. No neural-network runtime.
//! Python is used exclusively by the offline exporter and reference tests.
pub mod audio;
mod ffi;
pub mod kernels;
mod ladspa;
pub mod layers;
pub mod model;
pub mod offline;
mod packed;
pub mod weights;
pub use audio::AudioProcessor;
pub use model::{Model, BINS, DF_BINS, FFT, HOP, MODEL_DELAY};
pub use weights::{Bundle, Error, Result};

#[cfg(test)]
mod test_support;
