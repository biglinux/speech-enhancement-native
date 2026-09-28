//! C API for reference comparison and embedding. No Rust references are formed
//! over simultaneous input/output memory: exact in-place operation is supported.
use crate::{AudioProcessor, Bundle, Model, BINS};
use std::{
    ffi::{c_char, CStr},
    panic::{catch_unwind, AssertUnwindSafe},
};
struct Handle {
    audio: AudioProcessor,
}

/// # Safety
/// path must point to a NUL-terminated UTF-8 directory. Returns NULL on any load error.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_create(path: *const c_char) -> *mut std::ffi::c_void {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let path = unsafe { CStr::from_ptr(path) }.to_str().ok()?;
        let bundle = Bundle::open(path).ok()?;
        let audio = AudioProcessor::new(bundle).ok()?;
        Some(Box::into_raw(Box::new(Handle { audio })).cast())
    }));
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}
/// # Safety
/// h is a live handle from create(), not concurrently used or already freed.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_destroy(h: *mut std::ffi::c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h.cast::<Handle>()) });
    }
}
/// # Safety
/// Same handle contract. Only between independent streams; not concurrently with processing.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_reset(h: *mut std::ffi::c_void) {
    if let Some(h) = unsafe { h.cast::<Handle>().as_mut() } {
        h.audio.reset();
    }
}
/// # Safety
/// Input/output each cover n f32 values; either disjoint or exactly identical.
/// Partial overlap is not supported. No concurrent calls on the same handle.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_process(
    h: *mut std::ffi::c_void,
    input: *const f32,
    output: *mut f32,
    n: usize,
    db: f32,
) -> i32 {
    if h.is_null() || input.is_null() || output.is_null() {
        return -1;
    }
    crate::audio::flush_denormals();
    let h = unsafe { &mut *h.cast::<Handle>() };
    h.audio.set_attenuation_db(db);
    for i in 0..n {
        let x = unsafe { input.add(i).read() };
        let y = h.audio.sample(x);
        unsafe { output.add(i).write(y) };
    }
    if h.audio.faulted() {
        -2
    } else {
        0
    }
}
/// # Safety
/// Buffers cover 962 floats, disjoint or exactly identical. Spectrum test API;
/// do not mix spectrum calls and audio calls without reset().
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_spectrum(
    h: *mut std::ffi::c_void,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    if h.is_null() || input.is_null() || output.is_null() {
        return -1;
    }
    let mut spec = [0.0f32; BINS * 2];
    unsafe { std::ptr::copy_nonoverlapping(input, spec.as_mut_ptr(), BINS * 2) };
    let h = unsafe { &mut *h.cast::<Handle>() };
    let y = h.audio.model.process_spectrum(&spec, 0.0);
    unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), output, BINS * 2) };
    if y.iter().all(|v| v.is_finite()) {
        0
    } else {
        -2
    }
}
/// Returns the required length, copying only when capacity suffices.
/// # Safety
/// Same handle contract; output may be null for length query. Otherwise valid for capacity floats.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_trace(
    h: *const std::ffi::c_void,
    id: usize,
    output: *mut f32,
    capacity: usize,
) -> usize {
    if h.is_null() {
        return 0;
    }
    let m: &Model = &unsafe { &*h.cast::<Handle>() }.audio.model;
    if let Some(v) = m.trace(id) {
        if !output.is_null() && capacity >= v.len() {
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), output, v.len()) };
        }
        v.len()
    } else {
        0
    }
}
#[no_mangle]
pub extern "C" fn dpdfnet_native_latency_samples() -> usize {
    crate::audio::LATENCY
}
