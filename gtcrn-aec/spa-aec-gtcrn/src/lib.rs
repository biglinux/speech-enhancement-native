//! PipeWire SPA AEC plugin exposing `spa_audio_aec`, backed by the native Rust
//! GTCRN-AEC (`gtcrn-aec`). `module-echo-cancel` loads it via
//! `library.name = aec/libspa-aec-gtcrn`.
//!
//! The graph runs at 48 kHz, mono, planar f32. The band up to 8 kHz is
//! resampled to 16 kHz for the neural canceller, which works in 256-sample
//! hops; FIFOs decouple those hops from the host's block size. The band above
//! 8 kHz goes through a linear adaptive filter instead and is added back,
//! ducked while the neural band is strongly suppressed.
//!
//! Only what `module-echo-cancel` calls is implemented: the factory, the
//! handle's `get_interface`, and `init`/`init2`/`run`/`activate`/`deactivate`.

#![allow(non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::io::Write;
use std::ptr;

use gtcrn_aec::Model;

mod engine;

use engine::{Engine, MAX_CHUNK};
#[cfg(test)]
use engine::{TEST_MODEL, test_engine};

// SPA ABI, mirroring spa/support/plugin.h and spa/interfaces/audio/aec.h.

#[repr(C)]
struct spa_dict_item {
    key: *const c_char,
    value: *const c_char,
}
#[repr(C)]
struct spa_dict {
    flags: u32,
    n_items: u32,
    items: *const spa_dict_item,
}
#[repr(C)]
struct spa_audio_info_raw {
    format: u32,
    flags: u32,
    rate: u32,
    channels: u32,
    position: [u32; 64],
}
#[repr(C)]
struct spa_callbacks {
    funcs: *const c_void,
    data: *mut c_void,
}
#[repr(C)]
struct spa_interface {
    type_: *const c_char,
    version: u32,
    cb: spa_callbacks,
}
#[repr(C)]
struct spa_audio_aec {
    iface: spa_interface,
    name: *const c_char,
    info: *const spa_dict,
    latency: *const c_char,
}
#[repr(C)]
struct spa_handle {
    version: u32,
    get_interface:
        Option<unsafe extern "C" fn(*mut spa_handle, *const c_char, *mut *mut c_void) -> c_int>,
    clear: Option<unsafe extern "C" fn(*mut spa_handle) -> c_int>,
}
#[repr(C)]
struct spa_interface_info {
    type_: *const c_char,
}
#[repr(C)]
struct spa_handle_factory {
    version: u32,
    name: *const c_char,
    info: *const spa_dict,
    get_size: Option<unsafe extern "C" fn(*const spa_handle_factory, *const spa_dict) -> usize>,
    init: Option<
        unsafe extern "C" fn(
            *const spa_handle_factory,
            *mut spa_handle,
            *const spa_dict,
            *const c_void,
            u32,
        ) -> c_int,
    >,
    enum_interface_info: Option<
        unsafe extern "C" fn(
            *const spa_handle_factory,
            *mut *const spa_interface_info,
            *mut u32,
        ) -> c_int,
    >,
}

// Opaque C pointees: only their pointers cross this ABI.
#[repr(C)]
struct spa_hook {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_audio_aec_events {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_pod_builder {
    _private: [u8; 0],
}
#[repr(C)]
struct spa_pod {
    _private: [u8; 0],
}

/// `spa_audio_aec_methods`, in the order of aec.h (version 1:3).
#[repr(C)]
struct spa_audio_aec_methods {
    version: u32,
    add_listener: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut spa_hook,
            *const spa_audio_aec_events,
            *mut c_void,
        ) -> c_int,
    >,
    init: Option<
        unsafe extern "C" fn(*mut c_void, *const spa_dict, *const spa_audio_info_raw) -> c_int,
    >,
    run: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const *const f32,
            *const *const f32,
            *mut *mut f32,
            u32,
        ) -> c_int,
    >,
    set_props: Option<unsafe extern "C" fn(*mut c_void, *const spa_dict) -> c_int>,
    activate: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    deactivate: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    enum_props: Option<unsafe extern "C" fn(*mut c_void, c_int, *mut spa_pod_builder) -> c_int>,
    get_params: Option<unsafe extern "C" fn(*mut c_void, *mut spa_pod_builder) -> c_int>,
    set_params: Option<unsafe extern "C" fn(*mut c_void, *const spa_pod) -> c_int>,
    init2: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const spa_dict,
            *mut spa_audio_info_raw,
            *mut spa_audio_info_raw,
            *mut spa_audio_info_raw,
        ) -> c_int,
    >,
}

// The installed C headers' sizes and offsets on 64-bit targets;
// gtcrn-aec/tools/spa-aec-abi-check.c checks the same numbers against them.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<spa_handle>() == 24);
    assert!(size_of::<spa_handle_factory>() == 48);
    assert!(size_of::<spa_audio_aec>() == 56);
    assert!(size_of::<spa_audio_aec_methods>() == 88);
    assert!(size_of::<spa_audio_info_raw>() == 272);
    assert!(offset_of!(spa_audio_aec_methods, init) == 16);
    assert!(offset_of!(spa_audio_aec_methods, run) == 24);
    assert!(offset_of!(spa_audio_aec_methods, init2) == 80);
};

// SAFETY: the statics of these types hold only pointers to immutable static
// data and function entry points, and nothing writes through them.
unsafe impl Sync for spa_audio_aec_methods {}
// SAFETY: as above.
unsafe impl Sync for spa_interface_info {}
// SAFETY: as above.
unsafe impl Sync for spa_handle_factory {}

const AEC_TYPE: &[u8] = b"Spa:Pointer:Interface:Audio:AEC\0";
const FACTORY_NAME: &[u8] = b"audio.aec\0";
// The block size the plugin asks the host for (one neural hop at 48 kHz),
// not its input-to-output delay. gtcrn-aec/tools/spa-aec-abi-check.c checks
// these strings through the installed C headers.
const AEC_BLOCK: &[u8] = b"768/48000\0";
const AEC_NAME: &[u8] = b"gtcrn\0";
const SPA_AUDIO_FORMAT_F32P: u32 = 0x206;
const SPA_AUDIO_CHANNEL_MONO: u32 = 2;
const DEFAULT_MODEL: &str = "/usr/share/gtcrn-aec-native/localvqe-pi-aec-v1-49k-f32.gguf";
const EINVAL: c_int = 22;
const ENOTSUP: c_int = 95;

#[repr(C)]
struct Handle {
    handle: spa_handle,
    aec: spa_audio_aec,
    engine: *mut Engine, // null until init/init2
}

/// The value of `key` in `args`.
///
/// # Safety
/// `args` is null or points to a valid `spa_dict` whose `n_items` items, and
/// their non-null key and value strings, stay valid for the call.
unsafe fn dict_get(args: *const spa_dict, key: &str) -> Option<String> {
    // SAFETY: the caller guarantees `args` is null or valid.
    let d = unsafe { args.as_ref() }?;
    if d.items.is_null() {
        return None;
    }
    // SAFETY: a non-null `items` holds `n_items` entries.
    let items = unsafe { std::slice::from_raw_parts(d.items, d.n_items as usize) };
    items
        .iter()
        .filter(|it| !it.key.is_null() && !it.value.is_null())
        // SAFETY: non-null keys and values are NUL-terminated strings.
        .find(|it| unsafe { CStr::from_ptr(it.key) }.to_bytes() == key.as_bytes())
        .map(|it| {
            // SAFETY: as above.
            unsafe { CStr::from_ptr(it.value) }
                .to_string_lossy()
                .into_owned()
        })
}

/// The engine of the handle `object`, once `init` has built it.
///
/// # Safety
/// `object` is null or the `Handle` that `f_init` initialized.
unsafe fn engine_from(object: *mut c_void) -> Option<*mut Engine> {
    // SAFETY: the caller guarantees `object` is null or an initialized Handle.
    let h = unsafe { object.cast::<Handle>().as_ref() }?;
    (!h.engine.is_null()).then_some(h.engine)
}

/// # Safety
/// `object` is null or an initialized `Handle` not used by another thread
/// during the call; `args` as for [`dict_get`].
unsafe fn do_init(object: *mut c_void, args: *const spa_dict) -> c_int {
    // SAFETY: the caller guarantees `object` is null or an exclusive Handle.
    let Some(h) = (unsafe { object.cast::<Handle>().as_mut() }) else {
        return -EINVAL;
    };
    if !h.engine.is_null() {
        return 0;
    }
    // SAFETY: forwarded from the caller.
    let path = unsafe { dict_get(args, "gtcrn.model") }.unwrap_or_else(|| DEFAULT_MODEL.into());
    match Model::load(&path).and_then(Engine::new) {
        Ok(engine) => {
            h.engine = Box::into_raw(Box::new(engine));
            0
        }
        Err(error) => {
            // init runs on the main thread, so stderr (the PipeWire log) is
            // usable; a failed write must not abort the host.
            let _ = writeln!(
                std::io::stderr(),
                "spa-aec-gtcrn: cannot use model {path}: {error}"
            );
            // A file that cannot be opened reports why; one that opens but is
            // not a usable model is invalid.
            -std::fs::File::open(&path)
                .err()
                .and_then(|e| e.raw_os_error())
                .unwrap_or(EINVAL)
        }
    }
}

fn valid_format(info: Option<&spa_audio_info_raw>) -> bool {
    info.is_some_and(|i| i.format == SPA_AUDIO_FORMAT_F32P && i.rate == 48_000 && i.channels == 1)
}

/// Asks for a mono stream: `module-echo-cancel` adopts the channel layout
/// `init2` writes back, and its default is stereo.
fn negotiate_mono(info: Option<&mut spa_audio_info_raw>) -> bool {
    let Some(i) = info else { return false };
    if i.channels != 1 {
        i.channels = 1;
        i.position[0] = SPA_AUDIO_CHANNEL_MONO;
    }
    valid_format(Some(i))
}

unsafe extern "C" fn m_init(
    object: *mut c_void,
    args: *const spa_dict,
    info: *const spa_audio_info_raw,
) -> c_int {
    // SAFETY: the host passes null or a valid info, the handle it got from
    // `get_interface` and a null or valid dict, on its main thread.
    unsafe {
        if !valid_format(info.as_ref()) {
            return -EINVAL;
        }
        do_init(object, args)
    }
}
unsafe extern "C" fn m_init2(
    object: *mut c_void,
    args: *const spa_dict,
    first: *mut spa_audio_info_raw,
    second: *mut spa_audio_info_raw,
    third: *mut spa_audio_info_raw,
) -> c_int {
    // aec.h and module-echo-cancel name the three streams differently. All
    // three share one contract, so treat each alike without relying on order.
    // SAFETY: as in `m_init`; the infos are distinct writable structs.
    unsafe {
        let negotiated = [first, second, third].map(|i| negotiate_mono(i.as_mut()));
        if negotiated.contains(&false) {
            return -EINVAL;
        }
        do_init(object, args)
    }
}
// module-echo-cancel calls these from stream state changes on the main thread
// while the data thread may be inside `run`, with no lock between them, so
// they cannot touch the engine. After a pause the engine continues from its
// buffered and adapted state.
unsafe extern "C" fn m_activate(_object: *mut c_void) -> c_int {
    0
}
unsafe extern "C" fn m_deactivate(_object: *mut c_void) -> c_int {
    0
}

unsafe extern "C" fn m_run(
    object: *mut c_void,
    rec: *const *const f32,
    play: *const *const f32,
    out: *mut *mut f32,
    n: u32,
) -> c_int {
    // SAFETY: `object` is the handle the host got from `get_interface`.
    let Some(engine) = (unsafe { engine_from(object) }) else {
        return -EINVAL;
    };
    // No buffers are accessed for a zero-size callback.
    if n == 0 {
        return 0;
    }
    if rec.is_null() || play.is_null() || out.is_null() {
        return -EINVAL;
    }
    // SAFETY: each non-null array holds the one channel pointer of a mono stream.
    let (rec0, play0, out0) = unsafe { (*rec, *play, *out) };
    let n = n as usize;
    if !valid_audio_span(rec0, n)
        || !valid_audio_span(play0, n)
        || !valid_audio_span(out0, n)
        || partial_overlap(rec0, out0, n)
        || partial_overlap(play0, out0, n)
    {
        return -EINVAL;
    }
    // Staging through local chunks allows exact in-place buffers; partial overlap
    // is rejected above because a write could clobber input not yet read.
    // SAFETY: the host serializes `run` and calls nothing else on the engine
    // meanwhile (`activate`/`deactivate` leave it alone).
    let eng = unsafe { &mut *engine };
    let _denormals = ops::DenormalGuard::new();
    let mut mic = [0.0f32; MAX_CHUNK];
    let mut render = [0.0f32; MAX_CHUNK];
    let mut result = [0.0f32; MAX_CHUNK];
    let mut offset = 0;
    while offset < n {
        let count = (n - offset).min(MAX_CHUNK);
        for i in 0..count {
            // SAFETY: the spans hold `n` samples (checked above as far as a
            // pointer can be); reads finish before the writes below.
            let (m, r) = unsafe { (rec0.add(offset + i).read(), play0.add(offset + i).read()) };
            mic[i] = if m.is_finite() { m } else { 0.0 };
            render[i] = if r.is_finite() { r } else { 0.0 };
        }
        eng.run(&mic[..count], &render[..count], &mut result[..count]);
        for (i, &sample) in result[..count].iter().enumerate() {
            // SAFETY: as above; an exact in-place output only overwrites input
            // already read.
            unsafe { out0.add(offset + i).write(sample) };
        }
        offset += count;
    }
    0
}

fn valid_audio_span(p: *const f32, n: usize) -> bool {
    !p.is_null()
        && (p as usize).is_multiple_of(std::mem::align_of::<f32>())
        && n.checked_mul(std::mem::size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .and_then(|bytes| (p as usize).checked_add(bytes))
            .is_some()
}

fn partial_overlap(input: *const f32, output: *const f32, n: usize) -> bool {
    if input == output {
        return false;
    }
    // valid_audio_span has already established non-overflowing ranges.
    let bytes = n * std::mem::size_of::<f32>();
    let (a, b) = (input as usize, output as usize);
    a < b + bytes && b < a + bytes
}

static METHODS: spa_audio_aec_methods = spa_audio_aec_methods {
    version: 3,
    add_listener: None,
    init: Some(m_init),
    run: Some(m_run),
    set_props: None,
    activate: Some(m_activate),
    deactivate: Some(m_deactivate),
    enum_props: None,
    get_params: None,
    set_params: None,
    init2: Some(m_init2),
};

unsafe extern "C" fn h_get_interface(
    handle: *mut spa_handle,
    type_: *const c_char,
    iface: *mut *mut c_void,
) -> c_int {
    if handle.is_null() || type_.is_null() || iface.is_null() {
        return -EINVAL;
    }
    // SAFETY: the loader passes the handle `f_init` initialized (a `Handle`,
    // whose first field is the `spa_handle`), a NUL-terminated type and a
    // writable result pointer.
    unsafe {
        *iface = ptr::null_mut();
        if CStr::from_ptr(type_).to_bytes_with_nul() != AEC_TYPE {
            return -ENOTSUP;
        }
        *iface = (&raw mut (*handle.cast::<Handle>()).aec).cast();
    }
    0
}

unsafe extern "C" fn h_clear(handle: *mut spa_handle) -> c_int {
    // SAFETY: the loader passes null or the handle `f_init` initialized, once,
    // after the last call into it.
    if let Some(h) = unsafe { handle.cast::<Handle>().as_mut() }
        && !h.engine.is_null()
    {
        // SAFETY: `engine` came from `Box::into_raw` in `do_init`.
        drop(unsafe { Box::from_raw(h.engine) });
        h.engine = ptr::null_mut();
    }
    0
}

unsafe extern "C" fn f_get_size(_f: *const spa_handle_factory, _p: *const spa_dict) -> usize {
    std::mem::size_of::<Handle>()
}

unsafe extern "C" fn f_init(
    _f: *const spa_handle_factory,
    handle: *mut spa_handle,
    _info: *const spa_dict,
    _support: *const c_void,
    _n: u32,
) -> c_int {
    if handle.is_null() {
        return -EINVAL;
    }
    let h = handle.cast::<Handle>();
    // SAFETY: the loader allocates `f_get_size()` bytes, suitably aligned, for
    // the handle.
    unsafe {
        h.write(Handle {
            handle: spa_handle {
                version: 0,
                get_interface: Some(h_get_interface),
                clear: Some(h_clear),
            },
            aec: spa_audio_aec {
                iface: spa_interface {
                    type_: AEC_TYPE.as_ptr().cast(),
                    version: 1,
                    cb: spa_callbacks {
                        funcs: (&raw const METHODS).cast(),
                        data: h.cast(), // methods receive this as `object`
                    },
                },
                name: AEC_NAME.as_ptr().cast(),
                info: ptr::null(),
                latency: AEC_BLOCK.as_ptr().cast(),
            },
            engine: ptr::null_mut(),
        });
    }
    0
}

static AEC_IFACE_INFO: spa_interface_info = spa_interface_info {
    type_: AEC_TYPE.as_ptr().cast(),
};

unsafe extern "C" fn f_enum_interface_info(
    _f: *const spa_handle_factory,
    info: *mut *const spa_interface_info,
    index: *mut u32,
) -> c_int {
    if info.is_null() || index.is_null() {
        return -EINVAL;
    }
    // SAFETY: both are non-null pointers the loader owns for the call.
    unsafe {
        if *index == 0 {
            *info = &AEC_IFACE_INFO;
            *index += 1;
            1
        } else {
            0
        }
    }
}

static FACTORY: spa_handle_factory = spa_handle_factory {
    version: 1,
    name: FACTORY_NAME.as_ptr().cast(),
    info: ptr::null(),
    get_size: Some(f_get_size),
    init: Some(f_init),
    enum_interface_info: Some(f_enum_interface_info),
};

/// SPA plugin entry point: enumerate this plugin's handle factories.
///
/// # Safety
/// `factory` and `index` must be valid non-null pointers per the SPA
/// `spa_handle_factory_enum` contract. Called by the PipeWire loader.
#[unsafe(no_mangle)]
#[allow(private_interfaces)]
pub unsafe extern "C" fn spa_handle_factory_enum(
    factory: *mut *const spa_handle_factory,
    index: *mut u32,
) -> c_int {
    if factory.is_null() || index.is_null() {
        return -EINVAL;
    }
    // SAFETY: both are non-null pointers the loader owns for the call.
    unsafe {
        if *index == 0 {
            *factory = &FACTORY;
            *index += 1;
            1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod abi_contract_tests {
    use super::*;

    #[test]
    fn unsupported_streams_fail_before_loading() {
        let mut stream = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        assert!(valid_format(Some(&stream)));
        assert!(!valid_format(None));
        stream.rate = 44_100;
        assert_eq!(
            // SAFETY: m_init rejects the rate before it touches the null object.
            unsafe { m_init(ptr::null_mut(), ptr::null(), &stream) },
            -EINVAL
        );
        stream.rate = 48_000;
        stream.channels = 2;
        assert!(!valid_format(Some(&stream)));
        stream.channels = 1;
        stream.format = 0;
        assert!(!valid_format(Some(&stream)));
    }

    #[test]
    fn init2_makes_default_stereo_mono() {
        let model = std::ffi::CString::new(TEST_MODEL).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let stereo = || {
            let mut position = [0; 64];
            position[..2].copy_from_slice(&[3, 4]); // FL, FR
            spa_audio_info_raw {
                format: SPA_AUDIO_FORMAT_F32P,
                flags: 0,
                rate: 48_000,
                channels: 2,
                position,
            }
        };
        let mut infos = [stereo(), stereo(), stereo()];
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        let [a, b, c] = &mut infos;
        // SAFETY: `handle` has room for a Handle, the infos outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            assert_eq!(m_init2(h.cast(), &dict, a, b, c), 0);
            h_clear(h.cast());
        }
        for info in infos {
            assert_eq!(
                (info.channels, info.position[0]),
                (1, SPA_AUDIO_CHANNEL_MONO)
            );
        }
        let mut wrong_rate = stereo();
        wrong_rate.rate = 44_100;
        assert!(!negotiate_mono(Some(&mut wrong_rate)));
    }

    #[test]
    fn dict_get_skips_null_entries() {
        let items = [
            spa_dict_item {
                key: ptr::null(),
                value: c"x".as_ptr(),
            },
            spa_dict_item {
                key: c"gtcrn.model".as_ptr(),
                value: ptr::null(),
            },
            spa_dict_item {
                key: c"gtcrn.model".as_ptr(),
                value: c"/m.gguf".as_ptr(),
            },
        ];
        let dict = spa_dict {
            flags: 0,
            n_items: 3,
            items: items.as_ptr(),
        };
        let no_items = spa_dict {
            flags: 0,
            n_items: 2,
            items: ptr::null(),
        };
        // SAFETY: `dict` points to live NUL-terminated strings; `no_items` and the
        // null dict exercise the null checks.
        unsafe {
            assert_eq!(dict_get(&dict, "gtcrn.model").as_deref(), Some("/m.gguf"));
            assert_eq!(dict_get(&no_items, "gtcrn.model"), None);
            assert_eq!(dict_get(ptr::null(), "gtcrn.model"), None);
        }
    }

    /// `init` with `gtcrn.model` set to `model`, on a fresh handle.
    fn init_with_model(model: &str) -> c_int {
        let model = std::ffi::CString::new(model).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let stream = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        // SAFETY: `handle` has room for a Handle, the buffers outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            let result = m_init(h.cast(), &dict, &stream);
            h_clear(h.cast());
            result
        }
    }

    /// Echo of a noisy far end (50 ms delay) plus a near-end burst, 48 kHz.
    fn echo_and_near_end(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut seed = 5u32;
        let far: Vec<f32> = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let env = 0.6 + 0.4 * (i as f32 * 4e-4).sin();
                env * 0.2 * ((seed >> 9) as f32 / 8_388_608.0 - 1.0)
            })
            .collect();
        let mic = (0..n)
            .map(|i| {
                let echo = if i >= 2400 { 0.4 * far[i - 2400] } else { 0.0 };
                let tau = 2.0 * std::f32::consts::PI * i as f32 / 48_000.0;
                let near = if (120_000..168_000).contains(&i) {
                    0.2 * (tau * 300.0).sin() + 0.05 * (tau * 12_000.0).sin()
                } else {
                    0.0
                };
                echo + near
            })
            .collect();
        (mic, far)
    }

    /// Run the signals through the C ABI in host blocks of `block` samples,
    /// writing the output over the microphone buffer when `in_place`.
    fn run_through_abi(mic: &[f32], far: &[f32], block: usize, in_place: bool) -> Vec<f32> {
        let model = std::ffi::CString::new(TEST_MODEL).unwrap();
        let items = [spa_dict_item {
            key: c"gtcrn.model".as_ptr(),
            value: model.as_ptr(),
        }];
        let dict = spa_dict {
            flags: 0,
            n_items: 1,
            items: items.as_ptr(),
        };
        let stream = spa_audio_info_raw {
            format: SPA_AUDIO_FORMAT_F32P,
            flags: 0,
            rate: 48_000,
            channels: 1,
            position: [0; 64],
        };
        let mut rec = mic.to_vec();
        let mut out = vec![0.0; mic.len()];
        let mut handle = std::mem::MaybeUninit::<Handle>::uninit();
        // SAFETY: `handle` has room for a Handle, the buffers outlive the calls,
        // and the handle is cleared once.
        unsafe {
            let h = handle.as_mut_ptr();
            assert_eq!(
                f_init(ptr::null(), h.cast(), ptr::null(), ptr::null(), 0),
                0
            );
            assert_eq!(m_init(h.cast(), &dict, &stream), 0);
            let mut offset = 0;
            while offset < mic.len() {
                let n = block.min(mic.len() - offset);
                let r = [rec[offset..].as_ptr()];
                let p = [far[offset..].as_ptr()];
                let mut o = [if in_place {
                    rec[offset..].as_mut_ptr()
                } else {
                    out[offset..].as_mut_ptr()
                }];
                assert_eq!(
                    m_run(h.cast(), r.as_ptr(), p.as_ptr(), o.as_mut_ptr(), n as u32),
                    0
                );
                offset += n;
            }
            h_clear(h.cast());
        }
        if in_place { rec } else { out }
    }

    #[test]
    fn active_reference_output_ignores_block_size() {
        let (mic, far) = echo_and_near_end(4 * 48_000);
        let expected = run_through_abi(&mic, &far, 480, false);
        assert_eq!(run_through_abi(&mic, &far, 480, true), expected, "in place");
        for block in [1, 128, 1024, 8192] {
            let actual = run_through_abi(&mic, &far, block, false);
            let first = actual.iter().zip(&expected).position(|(a, b)| a != b);
            assert_eq!(first, None, "block {block} differs from block 480");
        }
    }

    #[test]
    fn hostile_samples_keep_output_finite() {
        let (mut mic, mut far) = echo_and_near_end(2 * 48_000);
        for (i, v) in [f32::NAN, f32::INFINITY, f32::MAX, -f32::MAX]
            .into_iter()
            .enumerate()
        {
            mic[10_000 * (i + 1)..][..700].fill(v);
            far[10_000 * (i + 1) + 5_000..][..700].fill(v);
        }
        let out = run_through_abi(&mic, &far, 480, false);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn init_reports_unusable_model_reason() {
        assert_eq!(init_with_model("/nonexistent/model.gguf"), -2); // ENOENT
        assert_eq!(init_with_model(env!("CARGO_MANIFEST_PATH")), -EINVAL);
        assert_eq!(init_with_model(TEST_MODEL), 0);
    }

    #[test]
    fn ffi_spans_allow_only_exact_inplace() {
        let mut samples = [0.0f32; 16];
        let p = samples.as_mut_ptr();
        assert!(valid_audio_span(p, 16));
        assert!(!partial_overlap(p, p, 8));
        assert!(partial_overlap(p, p.wrapping_add(1), 8));
        assert!(!partial_overlap(p, p.wrapping_add(8), 8));
        assert!(!valid_audio_span(std::ptr::null(), 8));
    }
}

#[cfg(test)]
#[path = "../../../testdata/heap_calls.rs"]
mod heap_calls;

// The engine runs on PipeWire's data thread: after warm-up, and on the first
// callback from a thread other than the one that built it, it must not touch
// the heap.
#[cfg(test)]
mod rt_alloc_tests {
    use super::{MAX_CHUNK, test_engine};
    use crate::heap_calls::heap_calls;

    #[test]
    fn warm_engine_run_never_allocates() {
        let mut eng = test_engine();
        let q = 480;
        let mic: Vec<f32> = (0..q * 280)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48_000.0).sin())
            .collect();
        let zero = vec![0.0f32; q];
        let mut out = vec![0.0f32; q];
        let mut blocks = mic.chunks_exact(q);
        // Warm past the FIFO fill and the first GCC-PHAT update.
        for block in blocks.by_ref().take(200) {
            eng.run(block, &zero, &mut out);
        }
        let calls = heap_calls(|| {
            for block in blocks {
                eng.run(block, &zero, &mut out);
            }
        });
        assert_eq!(calls, 0, "Engine::run used the heap");
    }

    // A host block that ends one or two samples before a hop boundary,
    // followed by a full block, leaves the most output queued.
    #[test]
    fn near_hop_blocks_never_allocate() {
        let mut eng = test_engine();
        let (mic, render) = ([0.05; MAX_CHUNK], [0.1; MAX_CHUNK]);
        let mut out = [0.0; MAX_CHUNK];
        eng.run(
            &mic[..MAX_CHUNK - 2],
            &render[..MAX_CHUNK - 2],
            &mut out[..MAX_CHUNK - 2],
        );
        let calls = heap_calls(|| eng.run(&mic, &render, &mut out));
        assert_eq!(calls, 0, "a full block after a short one used the heap");
    }

    #[test]
    fn other_thread_first_callback_never_allocates() {
        let mut eng = test_engine();
        let calls = std::thread::spawn(move || {
            let (mic, render) = ([0.05; MAX_CHUNK], [0.1; MAX_CHUNK]);
            let mut out = [0.0; MAX_CHUNK];
            heap_calls(|| eng.run(&mic, &render, &mut out))
        })
        .join()
        .unwrap();
        assert_eq!(calls, 0, "the first callback used the heap");
    }
}
