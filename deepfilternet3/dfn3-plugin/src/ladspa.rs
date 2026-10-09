//! The LADSPA 1.1 mono plugin around a [`Denoiser`], hand-written against the C ABI.
//!
//! Buffers the host's block size into 480-sample hops, ducks the residual floor in
//! pauses (`SilenceExpander`) and mutes the output while Silero hears no speech
//! in the raw input ([`VoiceGate`]). Each plugin crate builds its descriptor with
//! [`descriptor`] and exports it from `ladspa_descriptor`.

use crate::expander::SilenceExpander;
use crate::{Denoiser, Network, ERB_WIDTHS, FFT, HOP, NB_ERB, SR};
use dfn_ops::{atten_lim_from_db, DenormalGuard};
use silero_vad::VoiceGate;
use std::ffi::CStr;
use std::os::raw::{c_char, c_ulong, c_void};
use std::ptr;

type Handle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct PortRangeHint {
    hint_descriptor: i32,
    lower: f32,
    upper: f32,
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
    connect_port: Option<extern "C" fn(Handle, c_ulong, *mut f32)>,
    activate: Option<extern "C" fn(Handle)>,
    run: Option<extern "C" fn(Handle, c_ulong)>,
    run_adding: Option<extern "C" fn(Handle, c_ulong)>,
    set_run_adding_gain: Option<extern "C" fn(Handle, f32)>,
    deactivate: Option<extern "C" fn(Handle)>,
    cleanup: Option<extern "C" fn(Handle)>,
}
// SAFETY: immutable after construction; the pointers are to 'static data.
unsafe impl Sync for Descriptor {}

const PORT_INPUT: i32 = 0x1;
const PORT_OUTPUT: i32 = 0x2;
const PORT_CONTROL: i32 = 0x4;
const PORT_AUDIO: i32 = 0x8;
const HINT_BOUNDED: i32 = 0x1 | 0x2;
// Defaults interpolate linearly between the bounds.
const DEFAULT_MINIMUM: i32 = 0x40;
const DEFAULT_MIDDLE: i32 = 0xC0;
const DEFAULT_HIGH: i32 = 0x100; // 0.25 * lower + 0.75 * upper
const DEFAULT_MAXIMUM: i32 = 0x140;

// Port indices are public contract: hosts address controls by index, so new ports
// are only appended.
const IN: usize = 0;
const OUT: usize = 1;
const ATTEN: usize = 2;
const MIN_DB: usize = 3;
const MAX_DB_ERB: usize = 4;
const MAX_DB_DF: usize = 5;
const DEPTH: usize = 6;
const POST_FILTER: usize = 7;
const STARTUP_MS: usize = 8;
const ANCHORS: usize = 9;
const VOICE_GATE: usize = ANCHORS + NUM_ANCHORS;
const PORT_COUNT: usize = VOICE_GATE + 1;

/// Frequencies of the silence-floor curve's anchors.
const NUM_ANCHORS: usize = 10;
const ANCHOR_HZ: [f32; NUM_ANCHORS] = [
    60.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0,
];
/// An anchor at or below this level is off; with every anchor off there is no curve.
const ANCHOR_SENTINEL: f32 = -190.0;

const NAMES: [&CStr; PORT_COUNT] = [
    c"Audio In",
    c"Audio Out",
    c"Attenuation Limit (dB)",
    c"Min SNR gate (dB)",
    c"Skip-all SNR (dB)",
    c"Skip-DF SNR (dB)",
    c"Silence expander depth (dB)",
    c"Post filter beta",
    c"Startup mute (ms)",
    c"Silence floor 60 Hz (dB)",
    c"Silence floor 125 Hz (dB)",
    c"Silence floor 250 Hz (dB)",
    c"Silence floor 500 Hz (dB)",
    c"Silence floor 1 kHz (dB)",
    c"Silence floor 2 kHz (dB)",
    c"Silence floor 4 kHz (dB)",
    c"Silence floor 8 kHz (dB)",
    c"Silence floor 12 kHz (dB)",
    c"Silence floor 16 kHz (dB)",
    c"Voice gate depth (dB)",
];

struct Names([*const c_char; PORT_COUNT]);
// SAFETY: pointers to 'static C string literals.
unsafe impl Sync for Names {}
static PORT_NAMES: Names = Names({
    let mut p = [ptr::null(); PORT_COUNT];
    let mut i = 0;
    while i < PORT_COUNT {
        p[i] = NAMES[i].as_ptr();
        i += 1;
    }
    p
});

static PORT_DESCRIPTORS: [i32; PORT_COUNT] = {
    let mut d = [PORT_INPUT | PORT_CONTROL; PORT_COUNT];
    d[IN] = PORT_INPUT | PORT_AUDIO;
    d[OUT] = PORT_OUTPUT | PORT_AUDIO;
    d
};

const fn bounded(default: i32, lower: f32, upper: f32) -> PortRangeHint {
    PortRangeHint {
        hint_descriptor: HINT_BOUNDED | default,
        lower,
        upper,
    }
}

static PORT_HINTS: [PortRangeHint; PORT_COUNT] = {
    // Silence-floor anchors: dBFS, default -200, below the sentinel.
    let mut h = [bounded(DEFAULT_MINIMUM, -200.0, 0.0); PORT_COUNT];
    let audio = PortRangeHint {
        hint_descriptor: 0,
        lower: 0.0,
        upper: 0.0,
    };
    h[IN] = audio;
    h[OUT] = audio;
    // 100 dB is full noise reduction, 0 dB passes the input through.
    h[ATTEN] = bounded(DEFAULT_MAXIMUM, 0.0, 100.0);
    // -10 dB.
    h[MIN_DB] = bounded(DEFAULT_HIGH, -40.0, 0.0);
    // 40 dB, so no stage is skipped: skipping at 30 and 20 dB dropped the SI-SDR
    // improvement on noisy speech from 11 to 1 dB and saved no CPU.
    h[MAX_DB_ERB] = bounded(DEFAULT_MAXIMUM, 0.0, 40.0);
    h[MAX_DB_DF] = bounded(DEFAULT_MAXIMUM, 0.0, 40.0);
    // 20 dB; capped at run time by the attenuation limit, so a low limit, which
    // keeps wanted background, also limits the ducking. 0 is an exact pass-through.
    h[DEPTH] = bounded(DEFAULT_MIDDLE, 0.0, 40.0);
    // Off: a host that applies the hints must not get a strong post filter.
    h[POST_FILTER] = bounded(DEFAULT_MINIMUM, 0.0, 1.0);
    // 1000 ms, read once per activation.
    h[STARTUP_MS] = bounded(DEFAULT_MAXIMUM, 0.0, 1000.0);
    // 40 dB, on; 60 or more mutes completely, 0 is off. On, it adds the delay that
    // brings either engine to `VoiceGate::LAG` (66 ms), read once per activation.
    h[VOICE_GATE] = bounded(DEFAULT_MIDDLE, 0.0, 80.0);
    h
};

/// The descriptor of a plugin over network `N`. Label, name and unique ID are what
/// hosts store in their configurations; they must never change.
pub const fn descriptor<N: Network>(
    unique_id: c_ulong,
    label: &'static CStr,
    name: &'static CStr,
) -> Descriptor {
    Descriptor {
        unique_id,
        label: label.as_ptr(),
        // Not HARD_RT_CAPABLE: the cost of a hop depends on the signal (silence
        // and stage skipping), which the hard-RT contract forbids.
        properties: 0,
        name: name.as_ptr(),
        maker: c"BigLinux".as_ptr(),
        copyright: c"Code: MIT OR Apache-2.0".as_ptr(),
        port_count: PORT_COUNT as c_ulong,
        port_descriptors: PORT_DESCRIPTORS.as_ptr(),
        port_names: PORT_NAMES.0.as_ptr(),
        port_range_hints: PORT_HINTS.as_ptr(),
        implementation_data: ptr::null_mut(),
        instantiate: Some(instantiate::<N>),
        connect_port: Some(connect_port::<N>),
        activate: Some(activate::<N>),
        run: Some(run::<N>),
        run_adding: None,
        set_run_adding_gain: None,
        deactivate: None,
        cleanup: Some(cleanup::<N>),
    }
}

struct Instance<N> {
    eng: Denoiser<N>,
    ports: [*mut f32; PORT_COUNT],
    /// The startup mute and whether the voice gate runs are read at the first run
    /// after activation.
    first_run: bool,
    startup_samples: usize,
    gate: VoiceGate,
    gate_on: bool,
    /// Delays the output to the input sample the gate has decided on.
    gate_line: Vec<f32>,
    gate_pos: usize,
    out_index: i64,
    expander: SilenceExpander,
    acc: [f32; HOP],
    acc_n: usize,
    /// The last processed hop, read out while the next one fills.
    frame: [f32; HOP],
    /// Last control values, so their powf conversions run only when they move.
    last_atten_db: f32,
    last_gate_db: f32,
}

impl<N: Network> Instance<N> {
    /// Input-to-output delay of the denoiser: HOP - 1 samples of buffering, FFT - HOP
    /// of overlap-add, and the network's lookahead.
    const ENGINE_DELAY: usize = FFT - 1 + N::LOOKAHEAD * HOP;

    fn new() -> Self {
        Self {
            eng: Denoiser::new(),
            ports: [ptr::null_mut(); PORT_COUNT],
            first_run: true,
            startup_samples: SR,
            gate: VoiceGate::new(0.0, HOP),
            gate_on: false,
            gate_line: vec![0.0; VoiceGate::LAG - Self::ENGINE_DELAY],
            gate_pos: 0,
            out_index: 0,
            expander: SilenceExpander::new(),
            acc: [0.0; HOP],
            acc_n: 0,
            frame: [0.0; HOP],
            last_atten_db: f32::NAN,
            last_gate_db: f32::NAN,
        }
    }

    /// The value of a control port; unconnected and non-finite both read as `None`.
    fn control(&self, port: usize) -> Option<f32> {
        let p = self.ports[port];
        // SAFETY: a connected port points at host control storage valid during run().
        (!p.is_null())
            .then(|| unsafe { *p })
            .filter(|v| v.is_finite())
    }
}

/// Whether the processed spectrum rises above the user's silence-floor curve this
/// hop, which leaving silence requires. Without a curve it always does.
fn floor_open(band_db: &[f32; NB_ERB], anchors: &[f32; NUM_ANCHORS]) -> bool {
    if !anchors.iter().any(|&a| a > ANCHOR_SENTINEL) {
        return true;
    }
    const EXCESS_MARGIN: f32 = 3.0;
    // The engine's 960-point peak-bin level reads ~8 dB below the GUI's 4096-point
    // windowed spectrum bars at residual levels, measured against the bars; the
    // curve is drawn against the bars.
    const FLOOR_CALIB_DB: f32 = 8.0;
    let bin_hz = SR as f32 / FFT as f32;
    // The curve at `f` Hz: linear in dB over log frequency, clamped at the ends.
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
    for b in 0..NB_ERB {
        let w = ERB_WIDTHS[b];
        let center = (off as f32 + w as f32 * 0.5) * bin_hz;
        off += w;
        // The voice band counts fully, so hum or hiss alone rarely opens.
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

extern "C" fn instantiate<N: Network>(_: *const Descriptor, sr: c_ulong) -> Handle {
    // The model is fixed at 48 kHz; hosts resample around it.
    if sr != SR as c_ulong {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new(Instance::<N>::new())).cast()
}

extern "C" fn connect_port<N: Network>(h: Handle, port: c_ulong, data: *mut f32) {
    // SAFETY: `h` is the Instance leaked by instantiate(), alive until cleanup().
    let inst = unsafe { &mut *h.cast::<Instance<N>>() };
    if let Some(p) = inst.ports.get_mut(port as usize) {
        *p = data;
    }
}

extern "C" fn activate<N: Network>(h: Handle) {
    // SAFETY: `h` is the Instance leaked by instantiate(), alive until cleanup().
    let inst = unsafe { &mut *h.cast::<Instance<N>>() };
    // A new stream: everything but the port connections starts over.
    let ports = inst.ports;
    *inst = Instance::new();
    inst.ports = ports;
}

extern "C" fn run<N: Network>(h: Handle, n: c_ulong) {
    // SAFETY: `h` is the Instance leaked by instantiate(), alive until cleanup().
    let inst = unsafe { &mut *h.cast::<Instance<N>>() };
    let (input, output) = (inst.ports[IN], inst.ports[OUT]);
    if input.is_null() || output.is_null() {
        return;
    }
    let _fp_env = DenormalGuard::new();
    let n = n as usize;

    if let Some(db) = inst.control(ATTEN) {
        if db != inst.last_atten_db {
            inst.eng.atten_lim = atten_lim_from_db(db);
            inst.last_atten_db = db;
        }
    }
    if let Some(beta) = inst.control(POST_FILTER) {
        inst.eng.post_filter_beta = beta.clamp(0.0, 1.0);
    }
    if let Some(v) = inst.control(MIN_DB) {
        inst.eng.min_db = v;
    }
    if let Some(v) = inst.control(MAX_DB_ERB) {
        inst.eng.max_db_erb = v;
    }
    if let Some(v) = inst.control(MAX_DB_DF) {
        inst.eng.max_db_df = v;
    }
    let gate_db = inst.control(VOICE_GATE).map_or(0.0, |v| v.max(0.0));
    // Read once, so a live control change neither restarts the mute nor changes
    // the delay the gate adds.
    if inst.first_run && n > 0 {
        if let Some(ms) = inst.control(STARTUP_MS) {
            inst.startup_samples = (ms.clamp(0.0, 1000.0) * SR as f32 / 1000.0).round() as usize;
        }
        inst.gate_on = gate_db > 0.0;
        inst.first_run = false;
    }
    if inst.gate_on && gate_db != inst.last_gate_db {
        inst.gate.set_depth(gate_db);
        inst.last_gate_db = gate_db;
    }
    // The attenuation limit caps the expander's depth; NaN while it is unset.
    let depth = inst.control(DEPTH).unwrap_or(0.0).min(inst.last_atten_db);
    let anchors: [f32; NUM_ANCHORS] =
        std::array::from_fn(|k| inst.control(ANCHORS + k).unwrap_or(ANCHOR_SENTINEL));

    // In slices of at most a hop: the detector hears each slice before any of it
    // leaves, so every decision an output waits for exists, and it never runs so
    // far ahead that the gate has forgotten the decisions the output needs.
    for start in (0..n).step_by(HOP) {
        let len = HOP.min(n - start);
        // Copied out first, since the host may pass one buffer as input and output.
        let mut slice = [0.0f32; HOP];
        for (k, s) in slice[..len].iter_mut().enumerate() {
            // SAFETY: the host connected the input to at least `n` samples.
            let x = unsafe { *input.add(start + k) };
            // A NaN would pass the silence test and poison the GRU state for good.
            *s = if x.is_finite() { x } else { 0.0 };
        }
        if inst.gate_on {
            inst.gate.feed(&slice[..len]);
        }
        for (k, &x) in slice[..len].iter().enumerate() {
            inst.acc[inst.acc_n] = x;
            inst.acc_n += 1;
            if inst.acc_n == HOP {
                inst.eng.process(&inst.acc, &mut inst.frame);
                let open = floor_open(&inst.eng.band_db, &anchors);
                inst.expander.process_hop(
                    &mut inst.frame,
                    depth,
                    inst.eng.lsnr,
                    inst.eng.min_db,
                    open,
                );
                inst.acc_n = 0;
            }
            // A hop is complete exactly when the previous one has been read out.
            let mut y = inst.frame[inst.acc_n];
            if inst.gate_on {
                let delayed = std::mem::replace(&mut inst.gate_line[inst.gate_pos], y);
                inst.gate_pos += 1;
                if inst.gate_pos == inst.gate_line.len() {
                    inst.gate_pos = 0;
                }
                y = delayed * inst.gate.gain(inst.out_index - VoiceGate::LAG as i64);
            }
            inst.out_index += 1;
            // The muted audio is still processed, so it is discarded, not delayed.
            if inst.startup_samples > 0 {
                inst.startup_samples -= 1;
                y = 0.0;
            }
            // SAFETY: the host connected the output to at least `n` samples.
            unsafe { *output.add(start + k) = y };
        }
    }
}

extern "C" fn cleanup<N: Network>(h: Handle) {
    if !h.is_null() {
        // SAFETY: `h` is the Box leaked by instantiate(), reclaimed exactly once.
        unsafe { drop(Box::from_raw(h.cast::<Instance<N>>())) };
    }
}
