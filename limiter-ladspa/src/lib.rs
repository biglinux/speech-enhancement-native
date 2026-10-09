//! Mono lookahead peak limiter as a LADSPA 1.1 plugin.
//!
//! The audio is delayed by a 3 ms lookahead, so the gain can ramp down before a
//! peak reaches the output instead of clipping it. A final clamp at the ceiling
//! catches rounding. The plugin adds `LOOKAHEAD_MS` of latency, which LADSPA 1.1
//! cannot report; the graph owner has to account for it.

use std::collections::VecDeque;
use std::os::raw::{c_char, c_ulong, c_void};
use std::ptr;

const LOOKAHEAD_MS: f32 = 3.0;

const CEILING_DB: (f32, f32) = (-20.0, 0.0);
const RELEASE_S: (f32, f32) = (0.01, 2.0);

/// The value LADSPA `DEFAULT_HIGH` gives a port, so hosts that apply the hints and
/// the plugin itself start from the same settings.
const fn default_high((lower, upper): (f32, f32)) -> f32 {
    0.25 * lower + 0.75 * upper
}

struct Limiter {
    delay: Vec<f32>,            // audio delay ring, `look` samples
    mags: VecDeque<(u64, f32)>, // decreasing window of (index, |x|) over look + 1 inputs
    idx: u64,
    envelope: f32,     // gain the window needs, recovering by `release_step`
    history: Vec<f32>, // the last look + 1 envelope values
    sum: f64,          // sum of `history`
    ceiling: f32,      // linear peak ceiling
    release_step: f32, // per-sample linear gain recovery
    look: usize,
}

impl Limiter {
    fn new(sample_rate: f32, ceiling_db: f32, release_s: f32) -> Self {
        let look = ((sample_rate * LOOKAHEAD_MS / 1000.0).round() as usize).max(1);
        let mut limiter = Self {
            delay: vec![0.0; look],
            mags: VecDeque::with_capacity(look + 1),
            idx: 0,
            envelope: 1.0,
            history: vec![1.0; look + 1],
            sum: 0.0,
            ceiling: db_to_lin(ceiling_db),
            release_step: 0.0,
            look,
        };
        limiter.set_release_s(release_s, sample_rate);
        limiter.reset();
        limiter
    }

    /// Clears the signal history without allocating.
    fn reset(&mut self) {
        self.delay.fill(0.0);
        self.mags.clear();
        self.idx = 0;
        self.envelope = 1.0;
        self.history.fill(1.0);
        self.sum = self.history.len() as f64;
    }

    fn set_ceiling_db(&mut self, ceiling_db: f32) {
        self.ceiling = db_to_lin(ceiling_db);
    }

    fn set_release_s(&mut self, release_s: f32, sample_rate: f32) {
        self.release_step = 1.0 / (release_s.max(1e-3) * sample_rate);
    }

    #[cfg(test)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), output.len());
        for (&x, out) in input.iter().zip(output) {
            *out = self.process_sample(x);
        }
    }

    fn process_sample(&mut self, x: f32) -> f32 {
        let x = if x.is_finite() { x } else { 0.0 };
        // Sliding max of |x| over the sample leaving the delay line and the `look`
        // samples behind it.
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

        let need = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };
        self.envelope = need.min(self.envelope + self.release_step);

        // The gain is the mean envelope over the window. Every envelope value in it
        // already allows for the sample leaving the delay line, so the mean does
        // too, and a new peak is approached in look + 1 equal steps.
        let slot = (self.idx % self.history.len() as u64) as usize;
        self.sum += f64::from(self.envelope) - f64::from(self.history[slot]);
        self.history[slot] = self.envelope;
        if slot == 0 {
            // Bound the rounding drift of the running sum.
            self.sum = self.history.iter().copied().map(f64::from).sum();
        }
        let gain = (self.sum / self.history.len() as f64) as f32;

        let slot = (self.idx % self.look as u64) as usize;
        let delayed = self.delay[slot];
        self.delay[slot] = x;
        self.idx += 1;
        (delayed * gain).clamp(-self.ceiling, self.ceiling)
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
const HINT_DEFAULT_HIGH: i32 = 0x100;

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
    let ceiling = default_high(CEILING_DB);
    let release = default_high(RELEASE_S);
    let inst = Box::new(Instance {
        lim: Limiter::new(sr, ceiling, release),
        sr,
        p_in: ptr::null(),
        p_out: ptr::null_mut(),
        p_ceiling: ptr::null(),
        p_release: ptr::null(),
        last_ceiling: ceiling,
        last_release: release,
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
            inst.lim
                .set_ceiling_db(db.clamp(CEILING_DB.0, CEILING_DB.1));
            inst.last_ceiling = db;
        }
    }
    if !inst.p_release.is_null() {
        let rel = unsafe { *inst.p_release };
        if rel.is_finite() && rel != inst.last_release {
            inst.lim
                .set_release_s(rel.clamp(RELEASE_S.0, RELEASE_S.1), inst.sr);
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
    // Defaults: -5 dB ceiling, 1.5 s release.
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_HIGH,
        lower: CEILING_DB.0,
        upper: CEILING_DB.1,
    },
    PortRangeHint {
        hint_descriptor: CTRL | HINT_DEFAULT_HIGH,
        lower: RELEASE_S.0,
        upper: RELEASE_S.1,
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
    fn the_lookahead_is_3_ms() {
        assert_eq!(Limiter::new(48_000.0, -1.0, 0.2).look, 144);
    }

    #[test]
    fn hint_defaults_are_the_instantiate_defaults() {
        assert_eq!(default_high(CEILING_DB), -5.0);
        assert_eq!(default_high(RELEASE_S), 1.5025);
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
        let mut lim = Limiter::new(48_000.0, -1.0, 0.2);
        let look = lim.look;
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| 0.2 * (i as f32 * 0.03).sin()).collect();
        let mut out = vec![0.0f32; n];
        lim.process(&input, &mut out);
        let mut worst = 0.0f32;
        for i in (look + 100)..n {
            worst = worst.max((out[i] - input[i - look]).abs());
        }
        assert!(worst < 1e-6, "sub-ceiling signal altered by {worst}");
    }

    #[test]
    fn a_transient_is_attenuated_before_it_reaches_the_output() {
        let mut lim = Limiter::new(48_000.0, -1.0, 0.5);
        let n = 2048;
        let look = lim.look;
        let mut input = vec![0.1f32; n]; // quiet steady
        input[1000] = 6.0; // a big transient
        let mut out = vec![0.0f32; n];
        lim.process(&input, &mut out);
        // The peak leaves the delay line at 1000 + look; the carrier before it is
        // already attenuated.
        let arrival = 1000 + look;
        let pre = out[arrival - 2];
        assert!(
            pre.abs() < 0.1,
            "gain did not pre-duck before the peak: {pre}"
        );
    }

    #[test]
    fn the_gain_ramps_down_over_the_lookahead_instead_of_stepping() {
        let mut lim = Limiter::new(48_000.0, -1.0, 0.5);
        let look = lim.look;
        let mut input = vec![0.1f32; 2048];
        input[1000] = 6.0;
        let mut out = vec![0.0f32; input.len()];
        lim.process(&input, &mut out);
        // The steady 0.1 carrier shows the gain. Reaching the peak's gain in `look`
        // equal steps bounds every step by the total drop over `look`.
        let target = db_to_lin(-1.0) / 6.0;
        let max_step = 0.1 * (1.0 - target) / look as f32;
        for i in 999..1000 + look - 1 {
            let step = (out[i] - out[i + 1]).abs();
            assert!(
                step <= max_step * 1.01,
                "step {step} at {i} exceeds {max_step}"
            );
        }
        assert!(out[1000 + look].abs() <= db_to_lin(-1.0) + 1e-6);
    }

    #[test]
    fn ladspa_in_place_matches_disjoint_buffers() {
        let input: Vec<f32> = (0..4096)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 20.0)
            .collect();
        let mut expected = vec![0.0; input.len()];
        let mut reference =
            Limiter::new(48_000.0, default_high(CEILING_DB), default_high(RELEASE_S));
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
    }

    #[test]
    fn implausible_sample_rates_are_rejected() {
        assert!(instantiate(&DESCRIPTOR, 0).is_null());
        assert!(instantiate(&DESCRIPTOR, 1_000_000).is_null());
    }
}
