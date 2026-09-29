//! LADSPA 1.1 ABI, explicitly using int for bitfields and unsigned long for indices.
//! No initialization, locks, files, logging, worker waits or allocations in run().
use crate::{AudioProcessor, Bundle};
use std::{
    ffi::{c_char, c_int, c_ulong, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Mutex, Weak},
};
#[repr(C)]
struct Hint {
    descriptor: c_int,
    lower: f32,
    upper: f32,
}
#[repr(C)]
pub struct Descriptor {
    unique_id: c_ulong,
    label: *const c_char,
    properties: c_int,
    name: *const c_char,
    maker: *const c_char,
    copyright: *const c_char,
    port_count: c_ulong,
    port_descriptors: *const c_int,
    port_names: *const *const c_char,
    port_range_hints: *const Hint,
    implementation_data: *mut c_void,
    instantiate: Option<unsafe extern "C" fn(*const Descriptor, c_ulong) -> *mut c_void>,
    connect_port: Option<unsafe extern "C" fn(*mut c_void, c_ulong, *mut f32)>,
    activate: Option<unsafe extern "C" fn(*mut c_void)>,
    run: Option<unsafe extern "C" fn(*mut c_void, c_ulong)>,
    run_adding: Option<unsafe extern "C" fn(*mut c_void, c_ulong)>,
    set_run_adding_gain: Option<unsafe extern "C" fn(*mut c_void, f32)>,
    deactivate: Option<unsafe extern "C" fn(*mut c_void)>,
    cleanup: Option<unsafe extern "C" fn(*mut c_void)>,
}
// SAFETY: descriptor and all pointed-to metadata are immutable statics. Host may not mutate them.
unsafe impl Sync for Descriptor {}
struct Names([*const c_char; 5]);
unsafe impl Sync for Names {}
static PORTS: [c_int; 5] = [1 | 8, 2 | 8, 1 | 4, 2 | 4, 2 | 4];
static NAMES: Names = Names([
    c"Input".as_ptr(),
    c"Output".as_ptr(),
    c"Attenuation Limit (dB)".as_ptr(),
    c"latency".as_ptr(),
    c"Processing fault".as_ptr(),
]);
static HINTS: [Hint; 5] = [
    Hint {
        descriptor: 0,
        lower: 0.0,
        upper: 0.0,
    },
    Hint {
        descriptor: 0,
        lower: 0.0,
        upper: 0.0,
    },
    Hint {
        descriptor: 1 | 2 | 0x140,
        lower: 0.0,
        upper: 100.0,
    }, // DEFAULT_MAXIMUM, full enhancement
    Hint {
        descriptor: 1 | 0x20,
        lower: 0.0,
        upper: 0.0,
    },
    Hint {
        descriptor: 4,
        lower: 0.0,
        upper: 1.0,
    },
];
static DESC: Descriptor = Descriptor {
    unique_id: 57941,
    label: c"dpdfnet_native_48hr".as_ptr(),
    // Do NOT claim HARD_RT_CAPABLE until measured on the deployment target and allocator-tested.
    properties: 0,
    name: c"DPDFNet native 48k HR".as_ptr(),
    maker: c"BigLinux".as_ptr(),
    copyright: c"MIT OR Apache-2.0; DPDFNet weights Apache-2.0".as_ptr(),
    port_count: 5,
    port_descriptors: PORTS.as_ptr(),
    port_names: NAMES.0.as_ptr(),
    port_range_hints: HINTS.as_ptr(),
    implementation_data: std::ptr::null_mut(),
    instantiate: Some(instantiate),
    connect_port: Some(connect),
    activate: Some(activate),
    run: Some(run),
    run_adding: None,
    set_run_adding_gain: None,
    deactivate: None,
    cleanup: Some(cleanup),
};
// Weak cache: share live weights, release them when the last instance is destroyed.
// This mutex is touched ONLY by instantiate/cleanup, never by run/activate.
static CACHE: Mutex<Option<(std::path::PathBuf, Weak<Bundle>)>> = Mutex::new(None);
struct Instance {
    audio: AudioProcessor,
    ports: [*mut f32; 5],
}
unsafe extern "C" fn instantiate(_: *const Descriptor, rate: c_ulong) -> *mut c_void {
    if rate != 48000 {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> Option<*mut c_void> {
        let dir = std::env::var_os("DPDFNET_NATIVE_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/usr/share/dpdfnet-native/dpdfnet2_48khz_hr-w8a16".into());
        let mut cache = CACHE.lock().ok()?;
        let b = cache
            .as_ref()
            .filter(|(path, _)| path == &dir)
            .and_then(|(_, weak)| weak.upgrade());
        let b = match b {
            Some(b) => b,
            None => {
                let b = Bundle::open(&dir).ok()?;
                *cache = Some((dir, Arc::downgrade(&b)));
                b
            }
        };
        drop(cache);
        let audio = AudioProcessor::new(b).ok()?;
        Some(
            Box::into_raw(Box::new(Instance {
                audio,
                ports: [std::ptr::null_mut(); 5],
            }))
            .cast(),
        )
    }));
    result.ok().flatten().unwrap_or(std::ptr::null_mut())
}
unsafe extern "C" fn connect(h: *mut c_void, p: c_ulong, data: *mut f32) {
    if let Some(i) = unsafe { h.cast::<Instance>().as_mut() } {
        if p < 5 {
            i.ports[p as usize] = data;
        }
    }
}
unsafe extern "C" fn activate(h: *mut c_void) {
    if let Some(i) = unsafe { h.cast::<Instance>().as_mut() } {
        if !i.ports[2].is_null() {
            i.audio.set_attenuation_db(unsafe { *i.ports[2] });
        }
        i.audio.reset();
    }
}
unsafe extern "C" fn run(h: *mut c_void, n: c_ulong) {
    let Some(i) = (unsafe { h.cast::<Instance>().as_mut() }) else {
        return;
    };
    if i.ports[0].is_null() || i.ports[1].is_null() {
        return;
    }
    crate::audio::flush_denormals();
    let db = if i.ports[2].is_null() {
        100.0
    } else {
        unsafe { *i.ports[2] }
    };
    i.audio.set_attenuation_db(db);
    for p in 0..n as usize {
        // Raw read before raw write permits the input and output ports to alias exactly.
        let x = unsafe { i.ports[0].add(p).read() };
        let y = i.audio.sample(x);
        unsafe { i.ports[1].add(p).write(y) };
    }
    if !i.ports[3].is_null() {
        unsafe { *i.ports[3] = crate::audio::LATENCY as f32 };
    }
    if !i.ports[4].is_null() {
        unsafe { *i.ports[4] = if i.audio.faulted() { 1.0 } else { 0.0 } };
    }
}
unsafe extern "C" fn cleanup(h: *mut c_void) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h.cast::<Instance>()) });
    }
    if let Ok(mut cache) = CACHE.lock() {
        if cache.as_ref().is_some_and(|(_, w)| w.strong_count() == 0) {
            *cache = None;
        }
    }
}
#[no_mangle]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESC
    } else {
        std::ptr::null()
    }
}
