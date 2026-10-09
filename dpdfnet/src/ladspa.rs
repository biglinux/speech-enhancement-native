//! LADSPA 1.1 ABI, explicitly using int for bitfields and unsigned long for indices.
//! No initialization, locks, files, logging, worker waits or allocations in run().
use crate::{no_unwind, AudioProcessor, Bundle};
use dfn_ops::DenormalGuard;
use std::{
    ffi::{c_char, c_int, c_ulong, c_void},
    sync::{Arc, Mutex, PoisonError, Weak},
};

const PORT_INPUT: c_int = 0x1;
const PORT_OUTPUT: c_int = 0x2;
const PORT_CONTROL: c_int = 0x4;
const PORT_AUDIO: c_int = 0x8;
const HINT_BOUNDED_BELOW: c_int = 0x1;
const HINT_BOUNDED_ABOVE: c_int = 0x2;
const HINT_TOGGLED: c_int = 0x4;
const HINT_INTEGER: c_int = 0x20;
const HINT_DEFAULT_MAXIMUM: c_int = 0x140;
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
static PORTS: [c_int; 5] = [
    PORT_INPUT | PORT_AUDIO,
    PORT_OUTPUT | PORT_AUDIO,
    PORT_INPUT | PORT_CONTROL,
    PORT_OUTPUT | PORT_CONTROL,
    PORT_OUTPUT | PORT_CONTROL,
];
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
    // Full enhancement by default.
    Hint {
        descriptor: HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE | HINT_DEFAULT_MAXIMUM,
        lower: 0.0,
        upper: 100.0,
    },
    Hint {
        descriptor: HINT_BOUNDED_BELOW | HINT_INTEGER,
        lower: 0.0,
        upper: 0.0,
    },
    Hint {
        descriptor: HINT_TOGGLED,
        lower: 0.0,
        upper: 1.0,
    },
];
static DESC: Descriptor = Descriptor {
    unique_id: 0xE255,
    label: c"dpdfnet_native_48hr".as_ptr(),
    // Not HARD_RT_CAPABLE: the run that completes a hop does the whole inference,
    // so its cost is not proportional to its sample count.
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
// Only instantiate and cleanup take this mutex.
static CACHE: Mutex<Option<(std::path::PathBuf, Weak<Bundle>)>> = Mutex::new(None);
struct Instance {
    audio: AudioProcessor,
    ports: [*mut f32; 5],
}
unsafe extern "C" fn instantiate(_: *const Descriptor, rate: c_ulong) -> *mut c_void {
    if rate != 48000 {
        return std::ptr::null_mut();
    }
    no_unwind(std::ptr::null_mut(), || {
        let dir = Bundle::default_dir();
        let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        let b = cache
            .as_ref()
            .filter(|(path, _)| path == &dir)
            .and_then(|(_, weak)| weak.upgrade());
        let b = match b {
            Some(b) => b,
            None => {
                let Ok(b) = Bundle::open(&dir) else {
                    return std::ptr::null_mut();
                };
                *cache = Some((dir, Arc::downgrade(&b)));
                b
            }
        };
        drop(cache);
        let Ok(audio) = AudioProcessor::new(b) else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(Instance {
            audio,
            ports: [std::ptr::null_mut(); 5],
        }))
        .cast()
    })
}
unsafe extern "C" fn connect(h: *mut c_void, p: c_ulong, data: *mut f32) {
    if let Some(i) = unsafe { h.cast::<Instance>().as_mut() } {
        if p < 5 {
            i.ports[p as usize] = data;
        }
    }
}
unsafe extern "C" fn activate(h: *mut c_void) {
    let Some(i) = (unsafe { h.cast::<Instance>().as_mut() }) else {
        return;
    };
    let control = i.ports[2];
    let audio = &mut i.audio;
    if !no_unwind(false, || {
        if !control.is_null() {
            audio.set_attenuation_db(unsafe { *control });
        }
        audio.reset();
        true
    }) {
        audio.latch_fault();
    }
}
unsafe extern "C" fn run(h: *mut c_void, n: c_ulong) {
    let Some(i) = (unsafe { h.cast::<Instance>().as_mut() }) else {
        return;
    };
    let [input, output, control, latency, fault] = i.ports;
    if input.is_null() || output.is_null() {
        return;
    }
    let n = n as usize;
    let _denormals = DenormalGuard::new();
    let db = if control.is_null() {
        100.0
    } else {
        unsafe { *control }
    };
    let audio = &mut i.audio;
    let ran = no_unwind(false, || {
        audio.set_attenuation_db(db);
        for p in 0..n {
            // Raw read before raw write permits the input and output ports to alias exactly.
            let x = unsafe { input.add(p).read() };
            let y = audio.sample(x);
            unsafe { output.add(p).write(y) };
        }
        true
    });
    if !ran {
        audio.latch_fault();
        unsafe { std::ptr::write_bytes(output, 0, n) };
    }
    if !latency.is_null() {
        unsafe { *latency = crate::audio::LATENCY as f32 };
    }
    if !fault.is_null() {
        unsafe { *fault = if audio.faulted() { 1.0 } else { 0.0 } };
    }
}
unsafe extern "C" fn cleanup(h: *mut c_void) {
    no_unwind((), || {
        if !h.is_null() {
            drop(unsafe { Box::from_raw(h.cast::<Instance>()) });
        }
        let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.as_ref().is_some_and(|(_, w)| w.strong_count() == 0) {
            *cache = None;
        }
    });
}
#[no_mangle]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESC
    } else {
        std::ptr::null()
    }
}
