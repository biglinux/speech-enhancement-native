//! Mono lookahead brickwall limiter as a hand-written LADSPA 1.1 plugin.
//!
//! The microphone pipeline's final "clamp" node prevents clipping only by
//! hard-clipping, which distorts. A transparent limiter delays the audio by a
//! short lookahead, tracks the upcoming peak, and rides the gain down *before*
//! the peak reaches the output so the ceiling is met without a hard corner. A
//! final clamp stays as an absolute guarantee. Latency is the lookahead in
//! samples. LADSPA 1.1 has no generic latency field; the graph owner must account
//! for it explicitly (or use a documented latency control-port extension).
//!
//! No mono lookahead limiter LADSPA ships on the system (fastLookaheadLimiter is
//! stereo; hardLimiter is a bare clamp), hence this crate. Pure `f32`, no deps.

use std::collections::VecDeque;
use std::os::raw::{c_char, c_ulong, c_void};
use std::ptr;

/// Lookahead in milliseconds. 3 ms is enough to smooth transients at 48 kHz
/// while adding only 3 ms of algorithmic latency.
const LOOKAHEAD_MS: f32 = 3.0;

/// Pure DSP core, independent of the C ABI so it can be unit-tested directly.
pub struct Limiter {
    delay: Vec<f32>,            // audio delay ring, length = lookahead
    mags: VecDeque<(u64, f32)>, // monotonic-decreasing window of (index, |x|)
    idx: u64,
    gain: f32,
    ceiling: f32,      // linear peak ceiling
    release_step: f32, // per-sample linear gain recovery
    look: usize,
}

impl Limiter {
    #[must_use]
    pub fn new(sample_rate: f32, ceiling_db: f32, release_s: f32) -> Self {
        let look = ((sample_rate * LOOKAHEAD_MS / 1000.0).round() as usize).max(1);
        Self {
            delay: vec![0.0; look],
            mags: VecDeque::with_capacity(look + 1),
            idx: 0,
            gain: 1.0,
            ceiling: db_to_lin(ceiling_db),
            release_step: 1.0 / (release_s.max(1e-3) * sample_rate),
            look,
        }
    }

    /// Reset signal history without allocating or invalidating connected ports.
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
        self.mags.clear();
        self.idx = 0;
        self.gain = 1.0;
    }

    /// Latency in samples the plugin adds (the lookahead).
    #[must_use]
    pub fn latency(&self) -> usize {
        self.look
    }

    pub fn set_ceiling_db(&mut self, ceiling_db: f32) {
        self.ceiling = db_to_lin(ceiling_db);
    }

    pub fn set_release_s(&mut self, release_s: f32, sample_rate: f32) {
        self.release_step = 1.0 / (release_s.max(1e-3) * sample_rate);
    }

    /// Process disjoint Rust slices. The LADSPA wrapper also permits exact in-place.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (&x, out) in input.iter().zip(output) {
            *out = self.process_sample(x);
        }
    }

    fn process_sample(&mut self, x: f32) -> f32 {
        // Do not retain NaN/Inf in the lookahead history or peak envelope.
        let x = if x.is_finite() { x } else { 0.0 };
        // Sliding max of |x| over the lookahead window (this output sample
        // plus its `look` successors), via a monotonic-decreasing deque.
        let oldest = self.idx.saturating_sub(self.look as u64);
        while self.mags.front().is_some_and(|&(j, _)| j < oldest) {
            self.mags.pop_front();
        }
        let mag = x.abs();
        while self.mags.back().is_some_and(|&(_, m)| m <= mag) {
            self.mags.pop_back();
        }
        // At most `look` old entries remain before this insertion.
        self.mags.push_back((self.idx, mag));
        let peak = self.mags.front().map_or(0.0, |&(_, m)| m);

        // Target gain that would hold the windowed peak at the ceiling.
        let target = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };
        // Ride down immediately (the peak is still `look` samples from the
        // output), recover linearly by the release step.
        if target < self.gain {
            self.gain = target;
        } else {
            self.gain = (self.gain + self.release_step).min(target);
        }

        // Output the delayed sample scaled by the gain, then clamp as an
        // absolute brickwall guarantee.
        let slot = (self.idx as usize) % self.look;
        let delayed = self.delay[slot];
        self.delay[slot] = x;
        let output = (delayed * self.gain).clamp(-self.ceiling, self.ceiling);
        self.idx += 1;
        output
    }
}

fn db_to_lin(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ── LADSPA 1.1 ABI ──────────────────────────────────────────────────────────

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
unsafe impl Sync for Descriptor {}

const PORT_INPUT: i32 = 0x1;
const PORT_OUTPUT: i32 = 0x2;
const PORT_CONTROL: i32 = 0x4;
const PORT_AUDIO: i32 = 0x8;
const HINT_BOUNDED_BELOW: i32 = 0x1;
const HINT_BOUNDED_ABOVE: i32 = 0x2;
const HINT_DEFAULT_HIGH: i32 = 0x100; // 0.25*lower + 0.75*upper

struct Instance {
    lim: Limiter,
    sr: f32,
    p_in: *const Data,
    p_out: *mut Data,
    p_ceiling: *const Data,
    p_release: *const Data,
    last_ceiling: f32,
    last_release: f32,
}

extern "C" fn instantiate(_d: *const Descriptor, sr: c_ulong) -> Handle {
    // Reject nonsensical host formats before allocation or reciprocal arithmetic.
    if !(8_000..=384_000).contains(&sr) {
        return ptr::null_mut();
    }
    let sr = sr as f32;
    let inst = Box::new(Instance {
        lim: Limiter::new(sr, -1.0, 0.2),
        sr,
        p_in: ptr::null(),
        p_out: ptr::null_mut(),
        p_ceiling: ptr::null(),
        p_release: ptr::null(),
        last_ceiling: -1.0,
        last_release: 0.2,
    });
    Box::into_raw(inst) as Handle
}

extern "C" fn connect_port(h: Handle, port: c_ulong, data: *mut Data) {
    if h.is_null() {
        return;
    }
    // SAFETY: `h` is the Instance leaked in instantiate(); valid for its lifetime.
    let inst = unsafe { &mut *(h as *mut Instance) };
    match port {
        0 => inst.p_in = data,
        1 => inst.p_out = data,
        2 => inst.p_ceiling = data,
        3 => inst.p_release = data,
        _ => {}
    }
}

extern "C" fn run(h: Handle, n: c_ulong) {
    if h.is_null() || n == 0 {
        return;
    }
    // SAFETY: `h` is a live Instance; the host connected every port to buffers
    // at least `n` long before calling run().
    let inst = unsafe { &mut *(h as *mut Instance) };
    let n = n as usize;
    if inst.p_in.is_null() || inst.p_out.is_null() {
        return;
    }
    // Control ports move rarely; convert only on change.
    if !inst.p_ceiling.is_null() {
        let db = unsafe { *inst.p_ceiling };
        if db.is_finite() && db != inst.last_ceiling {
            inst.lim.set_ceiling_db(db.clamp(-20.0, 0.0));
            inst.last_ceiling = db;
        }
    }
    if !inst.p_release.is_null() {
        let rel = unsafe { *inst.p_release };
        if rel.is_finite() && rel != inst.last_release {
            inst.lim.set_release_s(rel.clamp(0.01, 2.0), inst.sr);
            inst.last_release = rel;
        }
    }
    for i in 0..n {
        // LADSPA permits the same audio buffer for input and output. Read the
        // input value before writing, and never create overlapping references.
        // The host still owns pointer validity and the n-sample extent.
        let x = unsafe { inst.p_in.add(i).read() };
        let y = inst.lim.process_sample(x);
        unsafe { inst.p_out.add(i).write(y) };
    }
}

extern "C" fn activate(h: Handle) {
    if h.is_null() {
        return;
    }
    // SAFETY: live handle from instantiate; LADSPA serializes lifecycle calls.
    unsafe {
        (&mut *(h as *mut Instance)).lim.reset();
    }
}

extern "C" fn deactivate(_h: Handle) {}

extern "C" fn cleanup(h: Handle) {
    if !h.is_null() {
        // SAFETY: `h` is the Box leaked in instantiate(); reclaimed exactly once.
        unsafe { drop(Box::from_raw(h as *mut Instance)) };
    }
}

const LABEL: &[u8] = b"biglinux_lookahead_limiter_mono\0";
const NAME: &[u8] = b"BigLinux mono lookahead limiter\0";
const MAKER: &[u8] = b"BigLinux\0";
const COPYRIGHT: &[u8] = b"Code: MIT OR Apache-2.0\0";
const PN_IN: &[u8] = b"Audio In\0";
const PN_OUT: &[u8] = b"Audio Out\0";
const PN_CEILING: &[u8] = b"Ceiling (dB)\0";
const PN_RELEASE: &[u8] = b"Release (s)\0";

static PORT_DESCRIPTORS: [i32; 4] = [
    PORT_INPUT | PORT_AUDIO,
    PORT_OUTPUT | PORT_AUDIO,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
];
struct Names([*const c_char; 4]);
unsafe impl Sync for Names {}
static PORT_NAMES: Names = Names([
    PN_IN.as_ptr() as *const c_char,
    PN_OUT.as_ptr() as *const c_char,
    PN_CEILING.as_ptr() as *const c_char,
    PN_RELEASE.as_ptr() as *const c_char,
]);
const CTRL: i32 = HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE;
static PORT_HINTS: [PortRangeHint; 4] = [
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
    // Ceiling: -20..0 dB, HIGH -> -1 dB (0.25*-20 + 0.75*0 = -5? no) — see note.
    // 0.25*lower + 0.75*upper = 0.25*-20 + 0.75*0 = -5 dB default. A -1 dBFS
    // ceiling is the usual target; callers set it explicitly, the hint is only a
    // host fallback.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_HIGH,
        lower: -20.0,
        upper: 0.0,
    },
    // Release: 0.01..2 s, HIGH -> 1.5 s. Callers set it explicitly.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_HIGH,
        lower: 0.01,
        upper: 2.0,
    },
];
static DESCRIPTOR: Descriptor = Descriptor {
    unique_id: 0x00B6_4C31,
    label: LABEL.as_ptr() as *const c_char,
    properties: 0,
    name: NAME.as_ptr() as *const c_char,
    maker: MAKER.as_ptr() as *const c_char,
    copyright: COPYRIGHT.as_ptr() as *const c_char,
    port_count: 4,
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
    deactivate: Some(deactivate),
    cleanup: Some(cleanup),
};

/// LADSPA host entry point.
#[no_mangle]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 {
        &DESCRIPTOR
    } else {
        ptr::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_is_the_lookahead() {
        let lim = Limiter::new(48_000.0, -1.0, 0.2);
        assert_eq!(lim.latency(), 144); // 3 ms @ 48 kHz
    }

    #[test]
    fn output_never_exceeds_the_ceiling() {
        let mut lim = Limiter::new(48_000.0, -1.0, 0.2);
        let ceil = db_to_lin(-1.0);
        let mut ph = 0.0f32;
        let mut out = vec![0.0f32; 480];
        for block in 0..200 {
            // Loud sine (amp 2.0, ~+6 dBFS) plus occasional bigger transients.
            let input: Vec<f32> = (0..480)
                .map(|i| {
                    ph += 0.05;
                    let spike = if (block * 480 + i) % 997 == 0 {
                        8.0
                    } else {
                        0.0
                    };
                    2.0 * ph.sin() + spike
                })
                .collect();
            lim.process(&input, &mut out);
            for &s in &out {
                assert!(s.abs() <= ceil + 1e-6, "output {s} exceeds ceiling {ceil}");
            }
        }
    }

    #[test]
    fn sub_ceiling_signal_passes_through_delayed() {
        // A steady tone well under the ceiling must come out unchanged (unity
        // gain), just delayed by the lookahead.
        let mut lim = Limiter::new(48_000.0, -1.0, 0.2);
        let look = lim.latency();
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.03).sin()).collect();
        let mut out = vec![0.0f32; n];
        lim.process(&input, &mut out);
        // After the delay, output equals the delayed input (gain stayed 1.0).
        let mut worst = 0.0f32;
        for i in (look + 100)..n {
            worst = worst.max((out[i] - input[i - look]).abs());
        }
        assert!(worst < 1e-6, "sub-ceiling signal altered by {worst}");
    }

    #[test]
    fn a_transient_is_attenuated_before_it_reaches_the_output() {
        // The gain must be riding down when the peak arrives (lookahead), so the
        // limited peak is not just hard-clipped: check the gain envelope engaged
        // by confirming the pre-peak samples are already scaled below unity.
        let mut lim = Limiter::new(48_000.0, -1.0, 0.5);
        let n = 2048;
        let look = lim.latency();
        let mut input = vec![0.1f32; n]; // quiet steady
        input[1000] = 6.0; // a big transient
        let mut out = vec![0.0f32; n];
        lim.process(&input, &mut out);
        // The steady 0.1 output around the peak's arrival (index 1000+... wait
        // output is delayed: the peak lands at output index 1000 + look). Just
        // before it, the quiet 0.1 input should already be ducked (< 0.1).
        let arrival = 1000 + look;
        let pre = out[arrival - 2];
        assert!(
            pre.abs() < 0.1,
            "gain did not pre-duck before the peak: {pre}"
        );
    }
}

#[cfg(test)]
mod revision_contract_tests {
    use super::*;

    #[test]
    fn ladspa_in_place_matches_disjoint_buffers() {
        let input: Vec<f32> = (0..4096)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 20.0)
            .collect();
        let mut expected = vec![0.0; input.len()];
        let mut reference = Limiter::new(48_000.0, -1.0, 0.2);
        reference.process(&input, &mut expected);
        let mut actual = input;
        let handle = instantiate(&DESCRIPTOR, 48_000);
        assert!(!handle.is_null());
        connect_port(handle, 0, actual.as_mut_ptr());
        connect_port(handle, 1, actual.as_mut_ptr());
        run(handle, actual.len() as c_ulong);
        cleanup(handle);
        assert_eq!(actual, expected);
    }

    #[test]
    fn strictly_decreasing_window_never_grows_the_deque() {
        let mut limiter = Limiter::new(48_000.0, -1.0, 0.2);
        let capacity = limiter.mags.capacity();
        // A decreasing signal retains every candidate until it expires.
        for i in 0..4096 {
            limiter.process_sample(1.0 - i as f32 / 8192.0);
            assert_eq!(limiter.mags.capacity(), capacity);
            assert!(limiter.mags.len() <= limiter.look + 1);
        }
    }

    #[test]
    fn nonfinite_audio_does_not_poison_the_history() {
        let mut limiter = Limiter::new(48_000.0, -1.0, 0.2);
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(limiter.process_sample(x).is_finite());
        }
        for _ in 0..4096 {
            assert!(limiter.process_sample(0.25).is_finite());
        }
        assert!(instantiate(&DESCRIPTOR, 0).is_null());
    }
}
