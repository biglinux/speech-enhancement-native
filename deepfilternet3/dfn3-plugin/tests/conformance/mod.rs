//! The behaviour every DeepFilterNet3 plugin shows, through its LADSPA descriptor as
//! a host drives it, and of its bare engine. Each plugin crate's
//! `tests/conformance.rs` includes this module next to its `Net`, `DESCRIPTOR` and
//! `ENGINE_DELAY`. (Both plugins cannot share one test binary: each exports
//! `ladspa_descriptor`.)
use std::os::raw::{c_char, c_ulong, c_void};

use super::{Net, DESCRIPTOR, ENGINE_DELAY};
use dfn3_plugin::{Denoiser, HOP};
use silero_vad::VoiceGate;

type Handle = *mut c_void;

/// The LADSPA 1.1 descriptor layout, as a host sees it.
#[repr(C)]
struct Desc {
    unique_id: c_ulong,
    label: *const c_char,
    properties: i32,
    name: *const c_char,
    maker: *const c_char,
    copyright: *const c_char,
    port_count: c_ulong,
    port_descriptors: *const i32,
    port_names: *const *const c_char,
    port_range_hints: *const c_void,
    implementation_data: *mut c_void,
    instantiate: Option<extern "C" fn(*const Desc, c_ulong) -> Handle>,
    connect_port: Option<extern "C" fn(Handle, c_ulong, *mut f32)>,
    activate: Option<extern "C" fn(Handle)>,
    run: Option<extern "C" fn(Handle, c_ulong)>,
    run_adding: Option<extern "C" fn(Handle, c_ulong)>,
    set_run_adding_gain: Option<extern "C" fn(Handle, f32)>,
    deactivate: Option<extern "C" fn(Handle)>,
    cleanup: Option<extern "C" fn(Handle)>,
}

const ATTEN: c_ulong = 2;
const DEPTH: c_ulong = 6;
const POST_FILTER: c_ulong = 7;
const STARTUP_MS: c_ulong = 8;
const VOICE_GATE: c_ulong = 19;
const GATE_ON: (c_ulong, f32) = (VOICE_GATE, 40.0);
const GATE_OFF: (c_ulong, f32) = (VOICE_GATE, 0.0);
const NO_STARTUP_MUTE: (c_ulong, f32) = (STARTUP_MS, 0.0);

fn desc() -> &'static Desc {
    // SAFETY: `Desc` mirrors the repr(C) descriptor layout.
    unsafe { &*(&DESCRIPTOR as *const _ as *const Desc) }
}

/// The bare engine over `input` in hops, with default settings.
fn engine(input: &[f32]) -> Vec<f32> {
    let mut e = Denoiser::<Net>::new();
    let mut out = vec![0.0; input.len() / HOP * HOP];
    for (x, y) in input
        .as_chunks::<HOP>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<HOP>().0)
    {
        e.process(x, y);
    }
    out
}

fn pcm(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&v| f32::from(i16::from_le_bytes(v)) / 32768.0)
        .collect()
}

/// Three seconds of synthetic speech (`testdata/README.md`).
fn speech() -> Vec<f32> {
    pcm(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/continuous-speech.pcm"
    )))
}

fn lcg_noise(count: usize, level: f32) -> Vec<f32> {
    let mut state: u32 = 7;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((f64::from(state >> 8) / 16_777_216.0 - 0.5) as f32) * level
        })
        .collect()
}

/// Two seconds of steady noise, then speech over the same noise, then noise: the
/// voice gate opens and closes on it.
fn noisy_speech() -> Vec<f32> {
    let speech = speech();
    let mut input = lcg_noise(48_000 * 3 + speech.len(), 0.02);
    for (i, s) in speech.iter().enumerate() {
        input[48_000 * 2 + i] += s;
    }
    input
}

/// A running plugin instance with the given controls connected.
struct Run {
    d: &'static Desc,
    h: Handle,
    // Boxed so the host-side storage stays put while connected.
    controls: Box<[f32]>,
}

impl Run {
    fn new(controls: &[(c_ulong, f32)]) -> Self {
        let d = desc();
        let h = (d.instantiate.unwrap())(d, 48000);
        assert!(!h.is_null());
        let mut values: Box<[f32]> = controls.iter().map(|&(_, v)| v).collect();
        for (k, &(port, _)) in controls.iter().enumerate() {
            (d.connect_port.unwrap())(h, port, values[k..].as_mut_ptr());
        }
        Self {
            d,
            h,
            controls: values,
        }
    }

    fn activate(&self) {
        (self.d.activate.unwrap())(self.h);
    }

    /// Runs `input` in blocks of `block` through separate input and output buffers,
    /// or through one buffer when `in_place`.
    fn process(&self, input: &[f32], block: usize, in_place: bool) -> Vec<f32> {
        let (connect, run) = (self.d.connect_port.unwrap(), self.d.run.unwrap());
        let mut inbuf = vec![0.0f32; block];
        let mut outbuf = vec![0.0f32; block];
        connect(self.h, 0, inbuf.as_mut_ptr());
        let out_ptr = if in_place {
            inbuf.as_mut_ptr()
        } else {
            outbuf.as_mut_ptr()
        };
        connect(self.h, 1, out_ptr);
        let mut out = Vec::with_capacity(input.len());
        for piece in input.chunks(block) {
            inbuf[..piece.len()].copy_from_slice(piece);
            run(self.h, piece.len() as c_ulong);
            let result = if in_place { &inbuf } else { &outbuf };
            out.extend_from_slice(&result[..piece.len()]);
        }
        out
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        (self.d.cleanup.unwrap())(self.h);
    }
}

/// Instantiates, activates and runs `input` in 480-sample blocks.
fn run_with(controls: &[(c_ulong, f32)], input: &[f32]) -> Vec<f32> {
    let run = Run::new(controls);
    run.activate();
    run.process(input, 480, false)
}

#[test]
fn rates_other_than_48k_are_rejected() {
    let d = desc();
    assert!((d.instantiate.unwrap())(d, 44100).is_null());
}

#[test]
fn startup_mutes_the_first_second_then_speech_passes() {
    let out = run_with(&[], &speech());
    assert!(out[..48000].iter().all(|&v| v == 0.0));
    assert!(out[48000..72000].iter().any(|v| v.abs() > 0.01));
}

#[test]
fn a_zero_startup_mute_passes_the_first_second() {
    let out = run_with(&[(POST_FILTER, 0.02), NO_STARTUP_MUTE], &speech()[..48000]);
    assert!(out.iter().all(|v| v.is_finite()));
    let peak = out.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(peak > 0.001, "speech was discarded");
}

#[test]
fn after_the_startup_mute_the_output_is_the_engine_output_one_hop_late() {
    let input = speech();
    let out = run_with(&[], &input);
    let mut bare = vec![0.0; HOP - 1];
    bare.extend(engine(&input));
    assert!(out[48000..].iter().any(|v| v.abs() > 0.01));
    assert!(out[48000..] == bare[48000..out.len()]);
}

#[test]
fn the_output_does_not_depend_on_the_block_size() {
    let input = noisy_speech();
    for gate in [GATE_OFF, GATE_ON] {
        let run = Run::new(&[(DEPTH, 20.0), gate]);
        run.activate();
        let reference = run.process(&input, 480, false);
        for (block, in_place) in [
            (1, false),
            (127, false),
            (1024, false),
            (16384, false),
            (65536, false),
            (480, true),
            (16384, true),
        ] {
            run.activate();
            let got = run.process(&input, block, in_place);
            assert!(
                got == reference,
                "gate {}: block {block}, in place {in_place}",
                gate.1
            );
        }
    }
}

#[test]
fn every_control_port_tolerates_non_finite_values() {
    let input = noisy_speech();
    for v in [
        0.0f32,
        6.0,
        100.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let controls: Vec<_> = (2..desc().port_count).map(|port| (port, v)).collect();
        let out = run_with(&controls, &input[..48000 * 2]);
        assert!(out.iter().all(|y| y.is_finite()), "controls {v}");
    }
}

#[test]
fn a_non_finite_voice_gate_depth_mid_stream_keeps_the_output_finite() {
    let input = noisy_speech();
    let mut run = Run::new(&[GATE_ON, NO_STARTUP_MUTE]);
    run.activate();
    let mut out = run.process(&input[..48000], 480, false);
    for v in [f32::NAN, f32::INFINITY, 40.0] {
        run.controls[0] = v;
        out.extend(run.process(&input[..24000], 480, false));
    }
    assert!(out.iter().all(|y| y.is_finite()));
}

#[test]
fn non_finite_input_is_heard_as_silence() {
    let mut input = noisy_speech();
    input[48_000 * 2 + 7] = f32::NAN;
    input[48_000 * 3] = f32::INFINITY;
    for gate in [GATE_OFF, GATE_ON] {
        let run = Run::new(&[gate]);
        run.activate();
        let out = run.process(&input, 480, true);
        assert!(out.iter().all(|v| v.is_finite()), "gate {}", gate.1);
        // The speech after it still passes: the state was not poisoned.
        assert!(out[48_000 * 3 + 9_600..48_000 * 4]
            .iter()
            .any(|v| v.abs() > 0.01));
    }
}

#[test]
fn reactivation_restarts_the_stream_and_reapplies_unchanged_controls() {
    let input = noisy_speech();
    for controls in [
        [(ATTEN, 0.0), GATE_OFF],
        [(ATTEN, 12.0), GATE_ON],
        [(ATTEN, 60.0), GATE_ON],
    ] {
        let run = Run::new(&controls);
        run.activate();
        let first = run.process(&input, 480, false);
        if let Some(deactivate) = run.d.deactivate {
            deactivate(run.h);
        }
        run.activate();
        let second = run.process(&input, 480, false);
        assert!(first == second, "controls {controls:?}");
    }
}

#[test]
fn a_run_before_activate_is_a_fresh_stream() {
    let input = noisy_speech();
    let activated = run_with(&[GATE_ON], &input);
    let unactivated = Run::new(&[GATE_ON]).process(&input, 480, false);
    assert!(activated == unactivated);
}

#[test]
fn zero_db_attenuation_passes_the_noise_and_100_db_removes_it() {
    let input = lcg_noise(480 * 300, 0.2);
    let energy =
        |out: Vec<f32>| -> f64 { out[480 * 100..].iter().map(|v| f64::from(*v).powi(2)).sum() };
    let bypass = energy(run_with(&[(ATTEN, 0.0)], &input));
    let full = energy(run_with(&[(ATTEN, 100.0)], &input));
    assert!(bypass > full * 2.0, "{bypass:.4} vs {full:.4}");
}

/// Lag of the strongest correlation between `input` and `output`, past the startup.
fn lag(input: &[f32], output: &[f32]) -> usize {
    let span = 48_000..input.len() - 4_000;
    (0..4_000)
        .max_by(|&a, &b| {
            let score = |l: usize| -> f64 {
                span.clone()
                    .map(|i| f64::from(input[i]) * f64::from(output[i + l]))
                    .sum()
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap()
}

#[test]
fn the_delay_is_the_engine_delay_and_the_voice_gate_raises_it_to_its_lag() {
    // Attenuation 0 passes the input through the analysis and synthesis unchanged,
    // so the correlation peak is the delay.
    let input = lcg_noise(48_000 * 3, 0.2);
    let off = run_with(&[(ATTEN, 0.0), NO_STARTUP_MUTE, GATE_OFF], &input);
    assert_eq!(lag(&input, &off), ENGINE_DELAY);
    let on = run_with(&[(ATTEN, 0.0), NO_STARTUP_MUTE, GATE_ON], &input);
    assert_eq!(lag(&input, &on), VoiceGate::LAG);
}

#[test]
fn the_voice_gate_mutes_noise_and_keeps_speech() {
    let input = noisy_speech();
    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let off = run_with(&[NO_STARTUP_MUTE, GATE_OFF], &input);
    let on = run_with(&[NO_STARTUP_MUTE, GATE_ON], &input);
    let quiet = 24_000..96_000 - 4_000;
    assert!(
        rms(&on[quiet.clone()]) <= rms(&off[quiet]) * 0.05,
        "noise not muted"
    );
    let talk = 96_000 + 9_600..96_000 + speech().len();
    let kept = rms(&on[talk.clone()]) / rms(&off[talk]);
    assert!(kept > 0.9, "speech energy kept {kept}");
}

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 9) as f32 / 8_388_608.0 - 1.0
}

#[test]
fn the_engine_output_is_finite() {
    let mut e = Denoiser::<Net>::new();
    let mut out = [0.0f32; HOP];
    for f in 0..80 {
        let inp: Vec<f32> = (0..HOP)
            .map(|i| 0.2 * ((f * HOP + i) as f32 * 0.05).sin())
            .collect();
        e.process(&inp, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite at frame {f}");
    }
}

#[test]
fn the_engine_suppresses_stationary_noise() {
    let mut e = Denoiser::<Net>::new();
    let mut out = [0.0f32; HOP];
    let mut seed = 12345u32;
    let (mut in_e, mut out_e) = (0.0f64, 0.0f64);
    for f in 0..300 {
        let inp: Vec<f32> = (0..HOP).map(|_| 0.05 * lcg(&mut seed)).collect();
        e.process(&inp, &mut out);
        if f > 30 {
            in_e += inp.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            out_e += out.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        }
    }
    assert!(out_e < in_e * 0.5, "in {in_e:.3}, out {out_e:.3}");
}

#[test]
fn the_engine_is_deterministic() {
    let run = || {
        let mut e = Denoiser::<Net>::new();
        let mut out = [0.0f32; HOP];
        let mut acc = Vec::new();
        for f in 0..40 {
            let inp: Vec<f32> = (0..HOP)
                .map(|i| 0.15 * ((f * HOP + i) as f32 * 0.03).sin())
                .collect();
            e.process(&inp, &mut out);
            acc.extend_from_slice(&out);
        }
        acc
    };
    assert_eq!(run(), run());
}

#[test]
fn speech_passes_again_after_long_noise_and_silence() {
    let speech = speech();
    let hops = speech.as_chunks::<HOP>().0;
    let mut e = Denoiser::<Net>::new();
    let mut out = [0.0; HOP];
    let energy = |e: &mut Denoiser<Net>, frames: &[[f32; HOP]]| -> f64 {
        let mut out = [0.0; HOP];
        let mut sum = 0.0;
        for frame in frames {
            e.process(frame, &mut out);
            sum += out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        }
        sum
    };
    energy(&mut e, hops);
    let mut seed = 17_u32;
    let noise: Vec<[f32; HOP]> = (0..6000)
        .map(|_| {
            std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                0.001 * (2.0 * (seed >> 9) as f32 / 8_388_608.0 - 1.0)
            })
        })
        .collect();
    let mut gated = 0;
    for frame in &noise {
        e.process(frame, &mut out);
        if e.lsnr < e.min_db {
            gated += 1;
        }
    }
    assert!(gated > 100, "noise did not exercise the gate: {gated} hops");
    energy(&mut e, &[[0.0; HOP]; 60]);
    let burst = energy(&mut e, &hops[..50]);
    assert!(burst > 0.1, "a short utterance was muted: {burst}");
    let resumed = energy(&mut e, hops);
    assert!(resumed > 1.0, "speech did not resume: {resumed}");
}
