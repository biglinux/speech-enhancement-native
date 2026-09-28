//! LADSPA mono wrapper around the Rust DFN3 engine.
//!
//! Hand-written LADSPA 1.1 ABI (no binding crate). One mono denoiser instance
//! that buffers the host's arbitrary block size into the model's fixed 480-sample
//! hops. Weights are embedded, so the `.so` is self-contained.

use crate::weights::W;
use crate::{Dfn3Ll, HOP, SR};
use std::os::raw::{c_char, c_ulong, c_void};
use std::ptr;
use std::sync::{Arc, OnceLock};

/// The embedded weight blob, 64-byte aligned under r11-align64 (otherwise 4) so the engine can view
/// its f32 tensors in place (zero-copy) instead of parsing a second heap copy.
#[repr(C)]
#[cfg_attr(feature = "r11-align64", repr(align(64)))]
#[cfg_attr(not(feature = "r11-align64"), repr(align(4)))]
struct AlignedBlob<const N: usize>([u8; N]);
static WEIGHTS: AlignedBlob<
    { include_bytes!(concat!(env!("OUT_DIR"), "/embedded_weights.bin")).len() },
> = AlignedBlob(*include_bytes!(concat!(
    env!("OUT_DIR"),
    "/embedded_weights.bin"
)));

/// The weight table is immutable and identical for every instance, so view it
/// once and share it; instances and `activate()` resets just clone the `Arc`.
/// Viewed in place from the aligned blob (no second copy) and locked into RAM so
/// mlock is best-effort: successful locking must be verified on the host.
pub(crate) fn shared_weights() -> Arc<W> {
    static SHARED_W: OnceLock<Arc<W>> = OnceLock::new();
    SHARED_W
        .get_or_init(|| {
            let w = W::from_static_aligned(&WEIGHTS.0);
            w.mlock();
            Arc::new(w)
        })
        .clone()
}

type Data = f32;
type Handle = *mut c_void;

#[repr(C)]
struct PortRangeHint {
    hint_descriptor: i32,
    lower: Data,
    upper: Data,
}

#[repr(C)]
pub struct Descriptor {
    unique_id: c_ulong,
    label: *const c_char,
    properties: i32,
    name: *const c_char,
    maker: *const c_char,
    copyright: *const c_char,
    port_count: c_ulong,
    port_descriptors: *const i32,
    port_names: *const *const c_char,
    port_range_hints: *const PortRangeHint,
    implementation_data: *mut c_void,
    instantiate: Option<extern "C" fn(*const Descriptor, c_ulong) -> Handle>,
    connect_port: Option<extern "C" fn(Handle, c_ulong, *mut Data)>,
    activate: Option<extern "C" fn(Handle)>,
    run: Option<extern "C" fn(Handle, c_ulong)>,
    run_adding: Option<extern "C" fn(Handle, c_ulong)>,
    set_run_adding_gain: Option<extern "C" fn(Handle, Data)>,
    deactivate: Option<extern "C" fn(Handle)>,
    cleanup: Option<extern "C" fn(Handle)>,
}
// The descriptor is immutable after init and only read by the host.
unsafe impl Sync for Descriptor {}

const PORT_INPUT: i32 = 0x1;
const PORT_OUTPUT: i32 = 0x2;
const PORT_CONTROL: i32 = 0x4;
const PORT_AUDIO: i32 = 0x8;
const HINT_BOUNDED_BELOW: i32 = 0x1;
const HINT_BOUNDED_ABOVE: i32 = 0x2;
// LADSPA default hints. Non-log ranges interpolate linearly, so with the ranges
// HIGH puts the Min SNR gate at -10 dB (its CPU saving); MAXIMUM puts the
// stage-skips at 40 dB — off, since the DFN3 benchmark showed they degraded
// speech for no CPU gain and the LL shares the mechanism.
const HINT_DEFAULT_MINIMUM: i32 = 0x40; // lower bound
const HINT_DEFAULT_MIDDLE: i32 = 0xC0; // 0.5*lower + 0.5*upper
const HINT_DEFAULT_HIGH: i32 = 0x100; // 0.25*lower + 0.75*upper
const HINT_DEFAULT_MAXIMUM: i32 = 0x140; // upper bound

const FIFO_CAP: usize = 2048;

struct Instance {
    eng: Dfn3Ll,
    p_in: *const Data,
    p_out: *mut Data,
    p_atten: *const Data,
    p_min_db: *const Data,
    p_max_erb: *const Data,
    p_max_df: *const Data,
    p_depth: *const Data,
    p_post_filter: *const Data,
    p_startup_ms: *const Data,
    p_anchors: [*const Data; NUM_ANCHORS],
    startup_control_pending: bool,
    // Post-model level gate that ducks the residual floor in pauses.
    expander: dfn_ops::SilenceExpander,
    acc: [f32; HOP],
    acc_n: usize,
    fifo: [f32; FIFO_CAP],
    fifo_head: usize,
    fifo_len: usize,
    // Last dB seen on the attenuation port, so the powf conversion runs only when
    // the control moves — not on every run(), which at block size 1 would be tens
    // of thousands of powf per second.
    last_atten_db: f32,
    startup_samples: usize,
}

extern "C" fn instantiate(_d: *const Descriptor, sr: c_ulong) -> Handle {
    // The model is fixed at 48 kHz; reject any other rate rather than silently
    // treating 44.1/96 kHz audio as 48 kHz. Hosts should resample around us.
    if sr != SR as c_ulong {
        return ptr::null_mut();
    }
    let inst = Box::new(Instance {
        eng: Dfn3Ll::from_shared(shared_weights()),
        p_in: ptr::null(),
        p_out: ptr::null_mut(),
        p_atten: ptr::null(),
        p_min_db: ptr::null(),
        p_max_erb: ptr::null(),
        p_max_df: ptr::null(),
        p_depth: ptr::null(),
        p_post_filter: ptr::null(),
        p_startup_ms: ptr::null(),
        p_anchors: [ptr::null(); NUM_ANCHORS],
        startup_control_pending: true,
        expander: dfn_ops::SilenceExpander::new(),
        acc: [0.0; HOP],
        acc_n: 0,
        fifo: [0.0; FIFO_CAP],
        fifo_head: 0,
        fifo_len: 0,
        last_atten_db: f32::NAN,
        startup_samples: SR,
    });
    Box::into_raw(inst) as Handle
}

extern "C" fn connect_port(h: Handle, port: c_ulong, data: *mut Data) {
    // SAFETY: `h` is the Instance Box leaked in instantiate(); valid for the instance lifetime.
    let inst = unsafe { &mut *(h as *mut Instance) };
    match port {
        0 => inst.p_in = data as *const Data,
        1 => inst.p_out = data,
        2 => inst.p_atten = data as *const Data,
        3 => inst.p_min_db = data as *const Data,
        4 => inst.p_max_erb = data as *const Data,
        5 => inst.p_max_df = data as *const Data,
        6 => inst.p_depth = data as *const Data,
        7 => inst.p_post_filter = data as *const Data,
        8 => inst.p_startup_ms = data as *const Data,
        p if (9..9 + NUM_ANCHORS as c_ulong).contains(&p) => {
            inst.p_anchors[(p - 9) as usize] = data as *const Data;
        }
        _ => {}
    }
}

/// Per-frequency silence floor. Returns true = the processed output spectrum rises
/// above the user's drawn curve this hop (leaving silence is allowed), false = it
/// sits under the curve (force mute). All-sentinel anchors → true, i.e. no curve,
/// so the plugin stays bit-identical to the scalar path (the golden vectors run
/// with unconnected anchor ports → sentinel → this returns true every hop).
fn floor_open(band_db: &[f32; crate::NB_ERB], anchors: &[f32; NUM_ANCHORS]) -> bool {
    if !anchors
        .iter()
        .any(|&a| a.is_finite() && a > ANCHOR_SENTINEL)
    {
        return true;
    }
    const EXCESS_MARGIN: f32 = 3.0;
    // Calibration: engine peak-bin band level reads ~8 dB below the GUI bars.
    const FLOOR_CALIB_DB: f32 = 8.0;
    let bin_hz = SR as f32 / crate::FFT as f32;
    // Log-frequency linear interpolation of the anchor curve at `f` Hz (clamped).
    let interp = |f: f32| -> f32 {
        if f <= ANCHOR_HZ[0] {
            return anchors[0];
        }
        let lf = f.ln();
        for k in 1..NUM_ANCHORS {
            if f <= ANCHOR_HZ[k] {
                let t = (lf - ANCHOR_HZ[k - 1].ln()) / (ANCHOR_HZ[k].ln() - ANCHOR_HZ[k - 1].ln());
                return anchors[k - 1] + t * (anchors[k] - anchors[k - 1]);
            }
        }
        anchors[NUM_ANCHORS - 1]
    };
    let mut off = 0usize;
    let mut excess = 0.0f32;
    let mut run_over = 0i32;
    let mut adjacency = false;
    for b in 0..crate::NB_ERB {
        let w = crate::ERB_WIDTHS[b];
        let center = (off as f32 + w as f32 * 0.5) * bin_hz;
        off += w;
        // Voice band (200 Hz–4 kHz) counts full; rumble/hiss edges count less so a
        // low-frequency hum or HF hiss alone is far less likely to hold the gate open.
        let weight = if (200.0..=4000.0).contains(&center) {
            1.0
        } else {
            0.3
        };
        let over = band_db[b] + FLOOR_CALIB_DB - interp(center);
        if over > 0.0 {
            excess += weight * over;
            run_over += 1;
            if run_over >= 3 {
                adjacency = true;
            }
        } else {
            run_over = 0;
        }
    }
    excess > EXCESS_MARGIN || adjacency
}

extern "C" fn run(h: Handle, n: c_ulong) {
    // SAFETY: `h` is the Instance Box leaked in instantiate(); valid for the instance lifetime.
    let inst = unsafe { &mut *(h as *mut Instance) };
    if inst.p_in.is_null() || inst.p_out.is_null() {
        return;
    }
    dfn_ops::flush_denormals();
    // Attenuation limit (dB), DeepFilterNet semantics: 0 dB = no noise reduction
    // (output is the noisy input), >=100 dB = full reduction, in between limits how
    // much the noisy signal is mixed back. Internally atten_lim is the noisy-mix
    // fraction: 1.0 keeps the input, 0.0 is full enhancement.
    if !inst.p_atten.is_null() {
        // SAFETY: connect_port set p_atten to a valid control port (LADSPA host contract).
        let db = unsafe { *inst.p_atten };
        // Recompute only when the control moved (compares unequal to NAN on the
        // first call, so the initial value is always applied).
        if db != inst.last_atten_db {
            inst.eng.atten_lim = dfn_ops::atten_lim_from_db(db);
            inst.last_atten_db = db;
        }
    }
    // Existing upstream spectral post-filter; no extra model or lookahead.
    // Beta 0 preserves the original engine path. Apply to BOTH startup and
    // steady engines below, otherwise the same control changes at handoff.
    if !inst.p_post_filter.is_null() {
        // SAFETY: the host keeps connected LADSPA control storage alive.
        let beta = unsafe { *inst.p_post_filter };
        if beta.is_finite() {
            inst.eng.post_filter_beta = beta.clamp(0.0, 1.0);
        }
    }
    // Activation-only. Never restart a mute timer on a live control change.
    // Keep the legacy 1000 ms when unconnected; a validated call preset may
    // request 0..1000 ms without discarding an arbitrary first second of voice.
    if inst.startup_control_pending && n > 0 {
        if !inst.p_startup_ms.is_null() {
            // SAFETY: connected scalar input port owned by the host.
            let ms = unsafe { *inst.p_startup_ms };
            if ms.is_finite() {
                inst.startup_samples =
                    (ms.clamp(0.0, 1000.0) * SR as f32 / 1000.0).round() as usize;
            }
        }
        inst.startup_control_pending = false;
    }
    // Stage-skip thresholds (LSNR dB): default to off via the port hints (they
    // degraded speech for no CPU gain); the noise gate still saves CPU, and the
    // filter-chain config can override any of them.
    // Non-finite control values are ignored (the engine keeps its current value).
    // SAFETY: each pointer, when non-null, was set to a valid control port by the host.
    if !inst.p_min_db.is_null() {
        let v = unsafe { *inst.p_min_db };
        if v.is_finite() {
            inst.eng.min_db = v;
        }
    }
    if !inst.p_max_erb.is_null() {
        let v = unsafe { *inst.p_max_erb };
        if v.is_finite() {
            inst.eng.max_db_erb = v;
        }
    }
    if !inst.p_max_df.is_null() {
        let v = unsafe { *inst.p_max_df };
        if v.is_finite() {
            inst.eng.max_db_df = v;
        }
    }
    // Silence-expander depth (dB): 0 = off. Capped below by the attenuation limit
    // (in dB) so a low intensity — which keeps wanted background — also limits the
    // ducking. `min` with a NaN atten (port unconnected) returns the depth.
    // SAFETY: p_depth, when non-null, was set to a valid control port by the host.
    let depth = if inst.p_depth.is_null() {
        0.0
    } else {
        let v = unsafe { *inst.p_depth };
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };
    let depth_eff = depth.min(inst.last_atten_db);
    // Per-frequency silence-floor curve (control ports 9..). Read once per run()
    // (host holds control values constant across the call). Sentinel = off.
    let mut anchors = [ANCHOR_SENTINEL; NUM_ANCHORS];
    for (k, a) in anchors.iter_mut().enumerate() {
        let p = inst.p_anchors[k];
        if !p.is_null() {
            // SAFETY: non-null anchor ports were set to valid control ports by the host.
            let v = unsafe { *p };
            if v.is_finite() {
                *a = v;
            }
        }
    }
    // Raw pointer indexing (no overlapping slices) so the host may connect the same
    // buffer to input and output (LADSPA allows in-place) without aliasing UB.
    for i in 0..n as usize {
        // SAFETY: p_in valid for `n` samples (host connected it before run(); LADSPA ABI).
        let x = unsafe { *inst.p_in.add(i) };
        // Sanitise at the trust boundary: a single NaN/Inf would otherwise pass the
        // mean-square silence gate and permanently poison the recurrent GRU state.
        inst.acc[inst.acc_n] = if x.is_finite() { x } else { 0.0 };
        inst.acc_n += 1;
        if inst.acc_n == HOP {
            let acc = inst.acc;
            let mut frame = [0.0f32; HOP];
            // One continuously-advancing engine produces every hop. The old
            // synthetic "warm" startup engine overwrote the output for the first
            // ~1 s of speech and then handed off to this engine — a discontinuous
            // switch that surfaced as noise starting a few seconds into use
            // (external audit, Rank 4).
            inst.eng.process(&acc, &mut frame);
            // Duck the residual floor during pauses (no-op at depth 0). `fopen` is
            // the per-frequency floor test on this hop's processed spectrum: AND-ed
            // with the model's LSNR gate, so a drawn curve can only tighten muting.
            let fopen = floor_open(&inst.eng.band_db, &anchors);
            inst.expander
                .process_hop(&mut frame, depth_eff, inst.eng.lsnr, inst.eng.min_db, fopen);
            // enqueue 480 processed samples
            for &s in frame.iter() {
                if inst.fifo_len < FIFO_CAP {
                    let tail = (inst.fifo_head + inst.fifo_len) % FIFO_CAP;
                    inst.fifo[tail] = s;
                    inst.fifo_len += 1;
                }
            }
            inst.acc_n = 0;
        }
        // drain one sample (zeros during the initial fill latency)
        let y = if inst.fifo_len > 0 {
            let v = inst.fifo[inst.fifo_head];
            inst.fifo_head = (inst.fifo_head + 1) % FIFO_CAP;
            inst.fifo_len -= 1;
            v
        } else {
            0.0
        };
        // Process real input during the configured startup mute and keep draining
        // the FIFO, so the muted audio is discarded rather than delayed.
        let y = if inst.startup_samples > 0 {
            inst.startup_samples -= 1;
            0.0
        } else {
            y
        };
        // SAFETY: p_out valid for `n` samples (host connected it before run(); LADSPA ABI).
        unsafe { *inst.p_out.add(i) = y };
    }
}

extern "C" fn activate(h: Handle) {
    // SAFETY: `h` is the Instance Box leaked in instantiate(); valid for the instance lifetime.
    let inst = unsafe { &mut *(h as *mut Instance) };
    // Discard the previous stream and start a fresh engine.
    inst.eng = Dfn3Ll::from_shared(shared_weights());
    inst.startup_samples = SR;
    inst.startup_control_pending = true;
    inst.expander = dfn_ops::SilenceExpander::new();
    // A fresh engine has default attenuation, regardless of the cached control.
    // Force every activation to reapply the connected value on its first run.
    inst.last_atten_db = f32::NAN;
    inst.acc_n = 0;
    inst.fifo_head = 0;
    inst.fifo_len = 0;
}

extern "C" fn cleanup(h: Handle) {
    if !h.is_null() {
        // SAFETY: `h` is the Box leaked in instantiate(); reclaimed exactly once.
        unsafe { drop(Box::from_raw(h as *mut Instance)) };
    }
}

// ---- static descriptor tables ----
const LABEL: &[u8] = b"deep_filter_net3_ll_rs_mono\0";
const NAME: &[u8] = b"DeepFilterNet3-LL (Rust) mono noise reducer\0";
const MAKER: &[u8] = b"BigLinux\0";
const COPYRIGHT: &[u8] = b"Code: MIT OR Apache-2.0\0";
const PN_IN: &[u8] = b"Audio In\0";
const PN_OUT: &[u8] = b"Audio Out\0";
const PN_ATTEN: &[u8] = b"Attenuation Limit (dB)\0";
const PN_MIN_DB: &[u8] = b"Min SNR gate (dB)\0";
const PN_MAX_ERB: &[u8] = b"Skip-all SNR (dB)\0";
const PN_MAX_DF: &[u8] = b"Skip-DF SNR (dB)\0";
const PN_DEPTH: &[u8] = b"Silence expander depth (dB)\0";
const PN_POST_FILTER: &[u8] = b"Post filter beta\0";
const PN_STARTUP_MS: &[u8] = b"Startup mute (ms)\0";
// Per-frequency silence-floor anchors: the user-drawn curve the processed spectrum
// must rise above (per band) to leave silence. AND-ed with the model's LSNR gate,
// so it can only tighten muting. Default sentinel (<= ANCHOR_SENTINEL) = no curve
// on that anchor → the plugin is bit-identical to the scalar path (golden intact).
pub const NUM_ANCHORS: usize = 10;
pub const ANCHOR_HZ: [f32; NUM_ANCHORS] = [
    60.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0,
];
pub const ANCHOR_SENTINEL: f32 = -190.0;
const PN_A0: &[u8] = b"Silence floor 60 Hz (dB)\0";
const PN_A1: &[u8] = b"Silence floor 125 Hz (dB)\0";
const PN_A2: &[u8] = b"Silence floor 250 Hz (dB)\0";
const PN_A3: &[u8] = b"Silence floor 500 Hz (dB)\0";
const PN_A4: &[u8] = b"Silence floor 1 kHz (dB)\0";
const PN_A5: &[u8] = b"Silence floor 2 kHz (dB)\0";
const PN_A6: &[u8] = b"Silence floor 4 kHz (dB)\0";
const PN_A7: &[u8] = b"Silence floor 8 kHz (dB)\0";
const PN_A8: &[u8] = b"Silence floor 12 kHz (dB)\0";
const PN_A9: &[u8] = b"Silence floor 16 kHz (dB)\0";

const N_PORTS: usize = 9 + NUM_ANCHORS;
static PORT_DESCRIPTORS: [i32; N_PORTS] = [
    PORT_INPUT | PORT_AUDIO,
    PORT_OUTPUT | PORT_AUDIO,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
];
struct Names([*const c_char; N_PORTS]);
unsafe impl Sync for Names {}
static PORT_NAMES: Names = Names([
    PN_IN.as_ptr() as *const c_char,
    PN_OUT.as_ptr() as *const c_char,
    PN_ATTEN.as_ptr() as *const c_char,
    PN_MIN_DB.as_ptr() as *const c_char,
    PN_MAX_ERB.as_ptr() as *const c_char,
    PN_MAX_DF.as_ptr() as *const c_char,
    PN_DEPTH.as_ptr() as *const c_char,
    PN_POST_FILTER.as_ptr() as *const c_char,
    PN_STARTUP_MS.as_ptr() as *const c_char,
    PN_A0.as_ptr() as *const c_char,
    PN_A1.as_ptr() as *const c_char,
    PN_A2.as_ptr() as *const c_char,
    PN_A3.as_ptr() as *const c_char,
    PN_A4.as_ptr() as *const c_char,
    PN_A5.as_ptr() as *const c_char,
    PN_A6.as_ptr() as *const c_char,
    PN_A7.as_ptr() as *const c_char,
    PN_A8.as_ptr() as *const c_char,
    PN_A9.as_ptr() as *const c_char,
]);
const CTRL: i32 = HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE;
// Anchor floor threshold: dBFS range, default MINIMUM (= -200 = sentinel/off).
const ANCHOR_HINT: PortRangeHint = PortRangeHint {
    hint_descriptor: CTRL | HINT_DEFAULT_MINIMUM,
    lower: -200.0,
    upper: 0.0,
};
static PORT_HINTS: [PortRangeHint; N_PORTS] = [
    PortRangeHint {
        hint_descriptor: 0,
        lower: 0.0,
        upper: 0.0,
    },
    PortRangeHint {
        hint_descriptor: 0,
        lower: 0.0,
        upper: 0.0,
    },
    // Attenuation limit: default 100 dB (full noise reduction), 0 dB = bypass.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MAXIMUM,
        lower: 0.0,
        upper: 100.0,
    },
    // Min SNR gate: range -40..0, HIGH -> -10 dB (default).
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_HIGH,
        lower: -40.0,
        upper: 0.0,
    },
    // Skip-all SNR: range 0..40, MAXIMUM -> 40 dB (default = skip off). The DFN3
    // benchmark showed the stage-skip degraded speech for no CPU gain; the LL
    // shares the mechanism, so it defaults off too. The Min SNR noise gate above
    // still provides the CPU saving.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MAXIMUM,
        lower: 0.0,
        upper: 40.0,
    },
    // Skip-DF SNR: range 0..40, MAXIMUM -> 40 dB (default = skip off).
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MAXIMUM,
        lower: 0.0,
        upper: 40.0,
    },
    // Silence expander depth: range 0..40 dB, MIDDLE -> 20 dB (default on). Ducks
    // the residual floor in pauses; capped at run time by the Attenuation Limit,
    // so low intensity keeps wanted background. 0 dB = off (exact pass-through).
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MIDDLE,
        lower: 0.0,
        upper: 40.0,
    },
    // Post-filter beta defaults to the minimum (0 = off): a host that follows the
    // hint must not silently apply a strong post-filter. 0x100 is DEFAULT_HIGH
    // (0.75), not the minimum — that was the bug (external audit).
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MINIMUM,
        lower: 0.0,
        upper: 1.0,
    },
    // Activation-only; legacy default retained until onset/noise A/B approval.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_MAXIMUM,
        lower: 0.0,
        upper: 1000.0,
    },
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
    ANCHOR_HINT,
];
static DESCRIPTOR: Descriptor = Descriptor {
    unique_id: 0x00DF_3159,
    label: LABEL.as_ptr() as *const c_char,
    // Not HARD_RT_CAPABLE: run() time is signal/state-dependent (silence and
    // LSNR-based stage skipping), which the strict hard-RT contract forbids. The
    // CPU savings are worth more than the advisory flag, and PipeWire loads it anyway.
    properties: 0,
    name: NAME.as_ptr() as *const c_char,
    maker: MAKER.as_ptr() as *const c_char,
    copyright: COPYRIGHT.as_ptr() as *const c_char,
    port_count: N_PORTS as c_ulong,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.0.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    implementation_data: ptr::null_mut(),
    instantiate: Some(instantiate),
    connect_port: Some(connect_port),
    activate: Some(activate),
    run: Some(run),
    run_adding: None,
    set_run_adding_gain: None,
    deactivate: None,
    cleanup: Some(cleanup),
};

/// LADSPA host entry point.
#[cfg_attr(not(feature = "r11-probe"), no_mangle)]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESCRIPTOR
    } else {
        ptr::null()
    }
}

#[cfg(feature = "r11-probe")]
pub(crate) fn probe_weight_bytes() -> &'static [u8] {
    &WEIGHTS.0
}
