//! LADSPA 1.1 plugin. The ABI uses int for bitfields and unsigned long for indices.
//! No initialization, locks, files, logging, worker waits or allocations in run().
use crate::{
    AudioProcessor, Bundle, HOP,
    audio::{LATENCY, bounded_input},
    no_unwind,
};
use ops::DenormalGuard;
use silero_vad::VoiceGate;
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
const HINT_DEFAULT_MINIMUM: c_int = 0x40;
const HINT_DEFAULT_MIDDLE: c_int = 0xC0;
const HINT_DEFAULT_MAXIMUM: c_int = 0x140;
// Port indices are public contract: hosts address controls by index, so new ports
// are only appended.
const PORT_COUNT: usize = 8;
const VOICE_GATE: usize = 5;
const VOICE_DETECTOR: usize = 6;
const VOICE_SENSITIVITY: usize = 7;
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
struct Names([*const c_char; PORT_COUNT]);
// SAFETY: the names point to immutable C string literals.
unsafe impl Sync for Names {}
static PORTS: [c_int; PORT_COUNT] = [
    PORT_INPUT | PORT_AUDIO,
    PORT_OUTPUT | PORT_AUDIO,
    PORT_INPUT | PORT_CONTROL,
    PORT_OUTPUT | PORT_CONTROL,
    PORT_OUTPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
    PORT_INPUT | PORT_CONTROL,
];
static NAMES: Names = Names([
    c"Input".as_ptr(),
    c"Output".as_ptr(),
    c"Attenuation Limit (dB)".as_ptr(),
    c"latency".as_ptr(),
    c"Processing fault".as_ptr(),
    c"Voice gate depth (dB)".as_ptr(),
    c"Voice detector (0 Silero, 1 FireRedVAD)".as_ptr(),
    c"Voice sensitivity (1 sensitive, 3 normal, 5 strict)".as_ptr(),
]);
static HINTS: [Hint; PORT_COUNT] = [
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
    // Off. Above 0 it mutes the output while the detector hears no speech, by that
    // many dB (60 or more is silence), and the latency grows to `VoiceGate::LAG`
    // (66 ms); read once per activation.
    Hint {
        descriptor: HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE | HINT_DEFAULT_MINIMUM,
        lower: 0.0,
        upper: 80.0,
    },
    // Silero, read once per activation like the gate.
    Hint {
        descriptor: HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE | HINT_INTEGER | HINT_DEFAULT_MINIMUM,
        lower: 0.0,
        upper: 1.0,
    },
    // 3, the normal level; read every run, so it follows a live change.
    Hint {
        descriptor: HINT_BOUNDED_BELOW | HINT_BOUNDED_ABOVE | HINT_INTEGER | HINT_DEFAULT_MIDDLE,
        lower: 1.0,
        upper: 5.0,
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
    port_count: PORT_COUNT as c_ulong,
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
    ports: [*mut f32; PORT_COUNT],
    /// Whether the voice gate runs and which detector it hears with are read at the
    /// first run after activation.
    first_run: bool,
    gate: VoiceGate,
    gate_on: bool,
    /// Delays the output to the input sample the gate has decided on.
    gate_line: Vec<f32>,
    gate_pos: usize,
    out_index: i64,
}
impl Instance {
    fn new(audio: AudioProcessor) -> Self {
        const { assert!(VoiceGate::LAG > LATENCY) };
        Self {
            audio,
            ports: [std::ptr::null_mut(); PORT_COUNT],
            first_run: true,
            gate: VoiceGate::new(0.0, HOP),
            gate_on: false,
            gate_line: vec![0.0; VoiceGate::LAG - LATENCY],
            gate_pos: 0,
            out_index: 0,
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
        Box::into_raw(Box::new(Instance::new(audio))).cast()
    })
}
unsafe extern "C" fn connect(h: *mut c_void, p: c_ulong, data: *mut f32) {
    // SAFETY: the host passes null or a handle from instantiate() that is not
    // used concurrently (LADSPA 1.1 instance contract).
    if let Some(i) = unsafe { h.cast::<Instance>().as_mut() }
        && (p as usize) < PORT_COUNT
    {
        i.ports[p as usize] = data;
    }
}
// Ports may be connected after activate, so the control is first read by the
// next run, where it applies without a ramp.
unsafe extern "C" fn activate(h: *mut c_void) {
    // SAFETY: the host passes null or a handle from instantiate() that is not
    // used concurrently (LADSPA 1.1 instance contract).
    let Some(i) = (unsafe { h.cast::<Instance>().as_mut() }) else {
        return;
    };
    // A new stream: the gate starts over and is chosen again at the first run.
    let Instance {
        audio,
        gate,
        gate_line,
        ..
    } = &mut *i;
    if !no_unwind(false, || {
        audio.reset();
        *gate = VoiceGate::new(0.0, HOP);
        gate_line.fill(0.0);
        true
    }) {
        audio.latch_fault();
    }
    i.first_run = true;
    i.gate_on = false;
    i.gate_pos = 0;
    i.out_index = 0;
}
unsafe extern "C" fn run(h: *mut c_void, n: c_ulong) {
    // SAFETY: the host passes null or a handle from instantiate() that is not
    // used concurrently (LADSPA 1.1 instance contract).
    let Some(i) = (unsafe { h.cast::<Instance>().as_mut() }) else {
        return;
    };
    let [input, output, control, latency, fault, ..] = i.ports;
    if input.is_null() || output.is_null() {
        return;
    }
    let n = n as usize;
    let _denormals = DenormalGuard::new();
    let db = if control.is_null() {
        100.0
    } else {
        // SAFETY: a connected control port points to one float.
        unsafe { *control }
    };
    let gate_db = i.control(VOICE_GATE).map_or(0.0, |v| v.max(0.0));
    let sensitivity = i.control(VOICE_SENSITIVITY);
    // Read once, so a live control change never changes the delay the gate adds.
    if i.first_run && n > 0 {
        i.gate_on = gate_db > 0.0;
        i.gate
            .use_firered(i.control(VOICE_DETECTOR).is_some_and(|v| v >= 0.5));
        i.first_run = false;
    }
    let ran = no_unwind(false, || {
        i.audio.set_attenuation_db(db);
        if i.gate_on {
            i.gate.set_depth(gate_db);
            if let Some(level) = sensitivity {
                i.gate.set_sensitivity(level.round().clamp(1.0, 5.0) as u8);
            }
        }
        // In slices of at most a hop: the detector hears each slice before any of
        // it leaves, so every decision an output waits for exists.
        for start in (0..n).step_by(HOP) {
            let len = HOP.min(n - start);
            // Copied out first, since the host may pass one buffer as input and
            // output.
            let mut slice = [0.0f32; HOP];
            for (k, s) in slice[..len].iter_mut().enumerate() {
                // SAFETY: the host connects audio buffers of at least `n` samples.
                *s = bounded_input(unsafe { input.add(start + k).read() });
            }
            if i.gate_on {
                i.gate.feed(&slice[..len]);
            }
            for (k, &x) in slice[..len].iter().enumerate() {
                let mut y = i.audio.sample(x);
                if i.gate_on {
                    let delayed = std::mem::replace(&mut i.gate_line[i.gate_pos], y);
                    i.gate_pos = (i.gate_pos + 1) % i.gate_line.len();
                    y = delayed * i.gate.gain(i.out_index - VoiceGate::LAG as i64);
                }
                i.out_index += 1;
                // SAFETY: as above.
                unsafe { output.add(start + k).write(y) };
            }
        }
        true
    });
    let audio = &mut i.audio;
    if !ran {
        audio.latch_fault();
        // SAFETY: the output buffer holds at least `n` samples.
        unsafe { std::ptr::write_bytes(output, 0, n) };
    }
    if !latency.is_null() {
        let delay = if i.gate_on || (i.first_run && gate_db > 0.0) {
            VoiceGate::LAG
        } else {
            LATENCY
        };
        // SAFETY: connected control ports point to one float each.
        unsafe { *latency = delay as f32 };
    }
    if !fault.is_null() {
        // SAFETY: as above.
        unsafe { *fault = if audio.faulted() { 1.0 } else { 0.0 } };
    }
}
unsafe extern "C" fn cleanup(h: *mut c_void) {
    no_unwind((), || {
        if !h.is_null() {
            // SAFETY: `h` came from `Box::into_raw` in instantiate(), and the host
            // calls cleanup once and never uses the handle afterwards.
            drop(unsafe { Box::from_raw(h.cast::<Instance>()) });
        }
        let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.as_ref().is_some_and(|(_, w)| w.strong_count() == 0) {
            *cache = None;
        }
    });
}
#[unsafe(no_mangle)]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    if index == 0 { &DESC } else { std::ptr::null() }
}
#[cfg(test)]
mod tests {
    use super::*;

    const ATTEN: usize = 2;
    const LATENCY_PORT: usize = 3;

    fn bundle() -> Arc<Bundle> {
        Bundle::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/model/dpdfnet2_48khz_hr-w8a16"
        ))
        .expect("valid model bundle")
    }

    fn pcm(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f32::from(i16::from_le_bytes(v)) / 32768.0)
            .collect()
    }

    fn noise(count: usize, level: f32) -> Vec<f32> {
        let mut state: u32 = 7;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((f64::from(state >> 8) / 16_777_216.0 - 0.5) as f32) * level
            })
            .collect()
    }

    /// Two seconds of noise, then speech over it, then a second of noise.
    fn noisy_speech() -> Vec<f32> {
        let speech = pcm(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../testdata/continuous-speech.pcm"
        )));
        let mut input = noise(48_000 * 3 + speech.len(), 0.02);
        for (i, s) in speech.iter().enumerate() {
            input[48_000 * 2 + i] += s;
        }
        input
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// Runs `input` through a fresh plugin in blocks of `block`; returns the output
    /// and the latency it reported.
    fn run_plugin(controls: &[(usize, f32)], input: &[f32], block: usize) -> (Vec<f32>, f32) {
        let audio = AudioProcessor::new(bundle()).unwrap();
        let h: *mut c_void = Box::into_raw(Box::new(Instance::new(audio))).cast();
        let mut values: Vec<f32> = controls.iter().map(|&(_, v)| v).collect();
        let mut latency = -1.0f32;
        let mut inbuf = vec![0.0f32; block];
        let mut outbuf = vec![0.0f32; block];
        let mut out = Vec::with_capacity(input.len());
        // SAFETY: `h` is live until cleanup below, and every connected buffer
        // outlives the runs.
        unsafe {
            for (k, &(port, _)) in controls.iter().enumerate() {
                connect(h, port as c_ulong, &raw mut values[k]);
            }
            connect(h, LATENCY_PORT as c_ulong, &raw mut latency);
            connect(h, 0, inbuf.as_mut_ptr());
            connect(h, 1, outbuf.as_mut_ptr());
            activate(h);
            for piece in input.chunks(block) {
                inbuf[..piece.len()].copy_from_slice(piece);
                run(h, piece.len() as c_ulong);
                out.extend_from_slice(&outbuf[..piece.len()]);
            }
            cleanup(h);
        }
        (out, latency)
    }

    #[test]
    fn gate_off_is_the_engine() {
        let input = noisy_speech()[..48_000].to_vec();
        let mut engine = AudioProcessor::new(bundle()).unwrap();
        engine.reset();
        let mut expected = vec![0.0; input.len()];
        engine.process(&input, &mut expected);
        let (out, latency) = run_plugin(&[], &input, 480);
        assert_eq!(out, expected);
        assert_eq!(latency, LATENCY as f32);
    }

    #[test]
    fn gate_keeps_only_speech() {
        // Attenuation 0 is the aligned dry signal, so the gate alone mutes.
        let input = noisy_speech();
        let (off, _) = run_plugin(&[(ATTEN, 0.0)], &input, 480);
        let (on, latency) = run_plugin(&[(ATTEN, 0.0), (VOICE_GATE, 40.0)], &input, 480);
        assert_eq!(latency, VoiceGate::LAG as f32);
        // The dry signal moved by the gate's extra delay.
        let shift = VoiceGate::LAG - LATENCY;
        assert!(rms(&off[48_000..96_000 - 4_000]) > 0.005);
        assert!(rms(&on[24_000..96_000 - 4_000]) <= rms(&off[24_000..96_000 - 4_000]) * 0.05);
        let talk = 96_000 + 9_600..input.len() - 48_000;
        let kept = rms(&on[talk.start + shift..talk.end + shift]) / rms(&off[talk]);
        assert!(kept > 0.9, "speech energy kept {kept}");
    }

    #[test]
    fn gated_output_ignores_block_size() {
        // Into the speech, so the gate both closes and opens.
        let input = noisy_speech()[96_000 - 24_000..96_000 + 24_000].to_vec();
        let controls = [(ATTEN, 0.0), (VOICE_GATE, 40.0)];
        let (expected, _) = run_plugin(&controls, &input, 480);
        for block in [1, 7, 8192] {
            assert_eq!(
                run_plugin(&controls, &input, block).0,
                expected,
                "block {block}"
            );
        }
    }

    #[test]
    fn sensitivity_sets_gate_opening() {
        let input = noisy_speech()[..96_000].to_vec();
        let noise_at = |level: f32| {
            let controls = [(ATTEN, 0.0), (VOICE_GATE, 40.0), (VOICE_SENSITIVITY, level)];
            rms(&run_plugin(&controls, &input, 480).0[24_000..96_000 - 4_000])
        };
        let (sensitive, strict) = (noise_at(1.0), noise_at(5.0));
        assert!(sensitive > strict, "level 1 {sensitive}, level 5 {strict}");
    }
}
