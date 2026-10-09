//! C API for reference comparison and embedding. No Rust references are formed
//! over simultaneous input/output memory: exact in-place operation is supported.
use crate::{model::BINS, no_unwind, AudioProcessor, Bundle};
use dfn_ops::DenormalGuard;
use std::ffi::{c_char, c_void, CStr};

/// # Safety
/// path must point to a NUL-terminated UTF-8 directory. Returns NULL on any load error.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_create(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    no_unwind(std::ptr::null_mut(), || {
        let Some(audio) = unsafe { CStr::from_ptr(path) }
            .to_str()
            .ok()
            .and_then(|path| Bundle::open(path).ok())
            .and_then(|bundle| AudioProcessor::new(bundle).ok())
        else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(audio)).cast()
    })
}
/// # Safety
/// h is a live handle from create(), not concurrently used or already freed.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_destroy(h: *mut c_void) {
    if !h.is_null() {
        no_unwind((), || {
            drop(unsafe { Box::from_raw(h.cast::<AudioProcessor>()) })
        });
    }
}
/// # Safety
/// Same handle contract. Only between independent streams; not concurrently with processing.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_reset(h: *mut c_void) {
    if let Some(audio) = unsafe { h.cast::<AudioProcessor>().as_mut() } {
        if !no_unwind(false, || {
            audio.reset();
            true
        }) {
            audio.latch_fault();
        }
    }
}
/// # Safety
/// Input/output each cover n f32 values; either disjoint or exactly identical.
/// Partial overlap is not supported. No concurrent calls on the same handle.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_process(
    h: *mut c_void,
    input: *const f32,
    output: *mut f32,
    n: usize,
    db: f32,
) -> i32 {
    let Some(audio) = (unsafe { h.cast::<AudioProcessor>().as_mut() }) else {
        return -1;
    };
    if input.is_null() || output.is_null() {
        return -1;
    }
    let _denormals = DenormalGuard::new();
    let ran = no_unwind(false, || {
        audio.set_attenuation_db(db);
        for i in 0..n {
            let x = unsafe { input.add(i).read() };
            let y = audio.sample(x);
            unsafe { output.add(i).write(y) };
        }
        true
    });
    if !ran {
        audio.latch_fault();
        unsafe { std::ptr::write_bytes(output, 0, n) };
    }
    if audio.faulted() {
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
    h: *mut c_void,
    input: *const f32,
    output: *mut f32,
) -> i32 {
    let Some(audio) = (unsafe { h.cast::<AudioProcessor>().as_mut() }) else {
        return -1;
    };
    if input.is_null() || output.is_null() {
        return -1;
    }
    let mut spec = [0.0f32; BINS * 2];
    unsafe { std::ptr::copy_nonoverlapping(input, spec.as_mut_ptr(), BINS * 2) };
    let finite = no_unwind(None, || {
        let y = audio.model.process_spectrum(&spec, 0.0);
        unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), output, BINS * 2) };
        Some(y.iter().all(|v| v.is_finite()))
    });
    match finite {
        Some(true) => 0,
        Some(false) => -2,
        None => {
            audio.latch_fault();
            unsafe { std::ptr::write_bytes(output, 0, BINS * 2) };
            -2
        }
    }
}
/// Returns the required length, copying only when capacity suffices.
/// # Safety
/// Same handle contract; output may be null for length query. Otherwise valid for capacity floats.
#[no_mangle]
pub unsafe extern "C" fn dpdfnet_native_trace(
    h: *const c_void,
    id: usize,
    output: *mut f32,
    capacity: usize,
) -> usize {
    let Some(audio) = (unsafe { h.cast::<AudioProcessor>().as_ref() }) else {
        return 0;
    };
    no_unwind(0, || {
        let Some(v) = audio.model.trace(id) else {
            return 0;
        };
        if !output.is_null() && capacity >= v.len() {
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), output, v.len()) };
        }
        v.len()
    })
}
#[no_mangle]
pub extern "C" fn dpdfnet_native_latency_samples() -> usize {
    crate::audio::LATENCY
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    #[allow(deprecated)] // _mm_getcsr is the only stable reader of MXCSR.
    fn process_restores_the_callers_floating_point_environment() {
        use std::arch::x86_64::_mm_getcsr;
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/model/dpdfnet2_48khz_hr-w8a16\0"
        );
        unsafe {
            let h = super::dpdfnet_native_create(dir.as_ptr().cast());
            assert!(!h.is_null());
            let before = _mm_getcsr();
            assert_eq!(before & 0x8040, 0, "FTZ/DAZ already set");
            let mut x = vec![1e-3f32; 4800];
            let n = x.len();
            let p = x.as_mut_ptr();
            assert_eq!(super::dpdfnet_native_process(h, p, p, n, 100.0), 0);
            assert_eq!(_mm_getcsr(), before);
            super::dpdfnet_native_destroy(h);
        }
    }
}
