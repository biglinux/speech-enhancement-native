//! The behavior every DeepFilterNet3 plugin shows, through its LADSPA descriptor as
//! a host drives it, and of its bare engine. Each plugin crate's
//! `tests/conformance.rs` includes this module next to its `Net`, `DESCRIPTOR` and
//! `ENGINE_DELAY`. (Both plugins cannot share one test binary: each exports
//! `ladspa_descriptor`.)
use std::os::raw::c_ulong;

use super::{DESCRIPTOR, ENGINE_DELAY, Net};
#[path = "../../../../testdata/heap_calls.rs"]
mod heap_calls;
use dfn3_plugin::ladspa::{Descriptor, Handle};
use dfn3_plugin::{Denoiser, HOP, SILENCE_SKIP_MAX};
use heap_calls::heap_calls;
use silero_vad::VoiceGate;

const ATTEN: c_ulong = 2;
const MIN_DB: c_ulong = 3;
const MAX_DB_ERB: c_ulong = 4;
const MAX_DB_DF: c_ulong = 5;
const DEPTH: c_ulong = 6;
const POST_FILTER: c_ulong = 7;
const STARTUP_MS: c_ulong = 8;
const VOICE_GATE: c_ulong = 19;
const LATENCY: c_ulong = 20;
const GATE_ON: (c_ulong, f32) = (VOICE_GATE, 40.0);
const GATE_OFF: (c_ulong, f32) = (VOICE_GATE, 0.0);
const NO_STARTUP_MUTE: (c_ulong, f32) = (STARTUP_MS, 0.0);

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
    d: &'static Descriptor,
    h: Handle,
    // Boxed so the host-side storage stays put while connected.
    controls: Box<[f32]>,
}

impl Run {
    fn new(controls: &[(c_ulong, f32)]) -> Self {
        let d = &DESCRIPTOR;
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
    let d = &DESCRIPTOR;
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
        let controls: Vec<_> = (2..DESCRIPTOR.port_count).map(|port| (port, v)).collect();
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
fn non_finite_and_huge_input_do_not_poison_the_state() {
    let mut input = noisy_speech();
    input[48_000 * 2 + 7] = f32::NAN;
    input[48_000 * 3] = f32::INFINITY;
    input[48_000 * 3 + 1] = 1e30;
    for gate in [GATE_OFF, GATE_ON] {
        let run = Run::new(&[gate]);
        run.activate();
        let out = run.process(&input, 480, true);
        assert!(out.iter().all(|v| v.is_finite()), "gate {}", gate.1);
        // The speech after it still passes: the state was not poisoned.
        assert!(
            out[48_000 * 3 + 9_600..48_000 * 4]
                .iter()
                .any(|v| v.abs() > 0.01)
        );
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
fn the_latency_port_reports_the_delay_the_voice_gate_raises_to_its_lag() {
    // Attenuation 0 passes the input through the analysis and synthesis unchanged,
    // so the correlation peak is the delay.
    let input = lcg_noise(48_000 * 3, 0.2);
    for (gate, delay) in [(GATE_OFF, ENGINE_DELAY), (GATE_ON, VoiceGate::LAG)] {
        let run = Run::new(&[(ATTEN, 0.0), NO_STARTUP_MUTE, gate, (LATENCY, -1.0)]);
        run.activate();
        let out = run.process(&input, 480, false);
        assert_eq!(lag(&input, &out), delay, "gate {}", gate.1);
        assert_eq!(run.controls[3], delay as f32, "gate {}", gate.1);
    }
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
        let inp: [f32; HOP] = std::array::from_fn(|i| 0.2 * ((f * HOP + i) as f32 * 0.05).sin());
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
        let inp: [f32; HOP] = std::array::from_fn(|_| 0.05 * lcg(&mut seed));
        e.process(&inp, &mut out);
        if f > 30 {
            in_e += inp.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            out_e += out.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        }
    }
    assert!(out_e < in_e * 0.5, "in {in_e:.3}, out {out_e:.3}");
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

#[test]
fn digital_silence_skips_the_network_and_speech_resumes_at_once() {
    let speech = speech();
    let hops = speech.as_chunks::<HOP>().0;
    let mut e = Denoiser::<Net>::new();
    let mut out = [0.0; HOP];
    for frame in &hops[..100] {
        e.process(frame, &mut out);
    }
    for k in 0..SILENCE_SKIP_MAX + 100 {
        e.process(&[0.0; HOP], &mut out);
        // Past the skip threshold the network does not run: its LSNR is the floor.
        if k >= SILENCE_SKIP_MAX {
            assert_eq!(e.lsnr, -15.0, "hop {k} ran the network");
            assert!(out.iter().all(|&v| v == 0.0));
        }
    }
    let mut energy = 0.0;
    for frame in &hops[100..150] {
        e.process(frame, &mut out);
        energy += out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    }
    assert!(energy > 0.1, "speech after silence was muted: {energy}");
}

/// Golden regression: the engine output on a fixed input must stay within 60 dB SDR
/// of `tests/fixtures/golden.bin`, this pipeline's own output. The thresholds run
/// every stage on every hop; 60 dB absorbs the FMA difference between SIMD tiers
/// and catches any real numeric regression, which lands far below.
///
/// After an intentional numeric change, regenerate the fixture with
/// `BLESS=1 cargo test --test conformance golden` and review the result.
#[test]
fn golden_output_matches_reference() {
    const NFRAMES: usize = 80;
    let mut seed = 1234u32;
    let input: Vec<f32> = (0..NFRAMES * HOP)
        .map(|i| {
            0.2 * (i as f32 * 0.02).sin() + 0.3 * (i as f32 * 0.005).sin() + 0.05 * lcg(&mut seed)
        })
        .collect();
    let mut e = Denoiser::<Net>::new();
    e.min_db = -100.0;
    e.max_db_erb = 100.0;
    e.max_db_df = 100.0;
    let mut out = vec![0.0f32; NFRAMES * HOP];
    for (x, y) in input
        .as_chunks::<HOP>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<HOP>().0)
    {
        e.process(x, y);
    }
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/golden.bin");
    if std::env::var_os("BLESS").is_some() {
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).expect("write the golden fixture");
        return;
    }
    let bytes = std::fs::read(path).expect("golden fixture missing; run once with BLESS=1");
    let reference: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(out.len(), reference.len(), "length changed");
    let (mut sig, mut err) = (0.0f64, 0.0f64);
    for (o, r) in out.iter().zip(&reference) {
        sig += f64::from(*r).powi(2);
        err += f64::from(o - r).powi(2);
    }
    let sdr = 10.0 * (sig / err.max(1e-30)).log10();
    assert!(sdr > 60.0, "golden SDR {sdr:.1} dB is below 60 dB");
}

#[test]
fn run_never_allocates() {
    let input = noisy_speech();
    for block in [1usize, 480, 1024, 16384] {
        for gate in [GATE_OFF, GATE_ON] {
            let run = Run::new(&[gate, (DEPTH, 20.0), (LATENCY, 0.0)]);
            run.activate();
            let (connect, plugin_run) = (run.d.connect_port.unwrap(), run.d.run.unwrap());
            let mut buf = vec![0.0f32; block];
            connect(run.h, 0, buf.as_mut_ptr());
            connect(run.h, 1, buf.as_mut_ptr());
            let mut calls = 0;
            for piece in input.chunks(block) {
                buf[..piece.len()].copy_from_slice(piece);
                calls += heap_calls(|| plugin_run(run.h, piece.len() as c_ulong));
            }
            assert_eq!(calls, 0, "block {block}, gate {}", gate.1);
        }
    }
}

/// The value a host that applies the hints starts a control at.
fn hint_default(port: c_ulong) -> f32 {
    // SAFETY: `port_range_hints` has `port_count` entries.
    let h = unsafe { *DESCRIPTOR.port_range_hints.add(port as usize) };
    let (l, u) = (h.lower, h.upper);
    match h.hint_descriptor & 0x3C0 {
        0x40 => l,
        0x80 => 0.75 * l + 0.25 * u,
        0xC0 => 0.5 * l + 0.5 * u,
        0x100 => 0.25 * l + 0.75 * u,
        0x140 => u,
        other => panic!("port {port}: default hint {other:#x}"),
    }
}

#[test]
fn the_engine_defaults_are_the_port_defaults() {
    // A host that leaves a control unconnected gets the engine's value, one that
    // applies the hints gets the hint's: both must be the same setting.
    let e = Denoiser::<Net>::new();
    assert_eq!(e.min_db, hint_default(MIN_DB));
    assert_eq!(e.max_db_erb, hint_default(MAX_DB_ERB));
    assert_eq!(e.max_db_df, hint_default(MAX_DB_DF));
    assert_eq!(e.post_filter_beta, hint_default(POST_FILTER));
}
