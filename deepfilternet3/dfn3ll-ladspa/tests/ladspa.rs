//! LADSPA ABI coverage: the shipped C entry points (instantiate SR gate, in-place
//! run, NaN sanitisation) are only reachable through the descriptor, not the engine.
use std::os::raw::{c_char, c_ulong, c_void};

type Handle = *mut c_void;

// Mirror of crate::ladspa::Descriptor (repr(C), same field order) so the function
// pointers land at the right offsets.
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

fn desc() -> &'static Desc {
    unsafe { &*(dfn3ll_ladspa::ladspa_descriptor_ptr(0) as *const Desc) }
}

// Feed `input` through the plugin in fixed-size blocks and collect the output.
fn process_blocks(d: &Desc, block: usize, input: &[f32]) -> Vec<f32> {
    let h = (d.instantiate.unwrap())(d as *const Desc, 48000);
    let mut inbuf = vec![0.0f32; block];
    let mut outbuf = vec![0.0f32; block];
    (d.connect_port.unwrap())(h, 0, inbuf.as_mut_ptr());
    (d.connect_port.unwrap())(h, 1, outbuf.as_mut_ptr());
    (d.activate.unwrap())(h);
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let n = block.min(input.len() - i);
        inbuf[..n].copy_from_slice(&input[i..i + n]);
        (d.run.unwrap())(h, n as c_ulong);
        out.extend_from_slice(&outbuf[..n]);
        i += n;
    }
    (d.cleanup.unwrap())(h);
    out
}

#[test]
fn abi_startup_mute_is_block_size_independent() {
    let d = desc();
    let input: Vec<f32> = include_bytes!("../../dfn3-ladspa/tests/fixtures/speech.pcm")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&v| f32::from(i16::from_le_bytes(v)) / 32768.0)
        .collect();
    let reference = process_blocks(d, 480, &input);
    assert!(
        reference[..48000].iter().all(|&v| v == 0.0),
        "startup must be muted for one second"
    );
    assert!(
        reference[48000..72000].iter().any(|v| v.abs() > 0.01),
        "voice must pass while speech context is still being validated"
    );
    // After speech establishes real context, the artificial seed must have no
    // effect. Account for the wrapper's 479-sample input accumulation delay.
    let mut engine = dfn3ll_ladspa::Dfn3Ll::new(include_bytes!("../dfn3ll_weights.bin"));
    let mut normal = vec![0.0; 479];
    for frame in input.as_chunks::<480>().0 {
        let mut out = [0.0; 480];
        engine.process(frame, &mut out);
        normal.extend_from_slice(&out);
    }
    // This fixture establishes its continuous second by 2 s. The earlier
    // assertion checks that validation did not mute the preceding speech.
    assert!(
        reference[96000..].iter().any(|v| v.abs() > 0.01),
        "the context comparison must include audible speech"
    );
    assert!(
        reference[96000..] == normal[96000..reference.len()],
        "synthetic context affected established speech"
    );
    for block in [1usize, 32, 127, 512, 1024] {
        let got = process_blocks(d, block, &input);
        assert!(
            got == reference,
            "block size {block} changed the output stream"
        );
    }
}

#[test]
fn abi_control_values_stay_finite() {
    let d = desc();
    for db in [
        0.0f32,
        6.0,
        100.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let h = (d.instantiate.unwrap())(d as *const Desc, 48000);
        const N: usize = 480;
        let mut inbuf = [0.0f32; N];
        let mut outbuf = [0.0f32; N];
        // Drive every control port (atten + the three thresholds) with the value,
        // so the threshold sanitisation on ports 3/4/5 is exercised too.
        let (mut c2, mut c3, mut c4, mut c5) = (db, db, db, db);
        (d.connect_port.unwrap())(h, 0, inbuf.as_mut_ptr());
        (d.connect_port.unwrap())(h, 1, outbuf.as_mut_ptr());
        (d.connect_port.unwrap())(h, 2, &mut c2 as *mut f32);
        (d.connect_port.unwrap())(h, 3, &mut c3 as *mut f32);
        (d.connect_port.unwrap())(h, 4, &mut c4 as *mut f32);
        (d.connect_port.unwrap())(h, 5, &mut c5 as *mut f32);
        (d.activate.unwrap())(h);
        for b in 0..120 {
            for (i, s) in inbuf.iter_mut().enumerate() {
                *s = 0.05 * ((b * N + i) as f32 * 0.05).sin();
            }
            (d.run.unwrap())(h, N as c_ulong);
            assert!(
                outbuf.iter().all(|v| v.is_finite()),
                "non-finite output with attenuation control = {db}"
            );
        }
        (d.cleanup.unwrap())(h);
    }
}

#[test]
fn abi_attenuation_semantics() {
    let d = desc();
    // Output energy over stationary noise for a given Attenuation Limit (dB).
    let energy = |atten: f32| -> f64 {
        let h = (d.instantiate.unwrap())(d as *const Desc, 48000);
        const N: usize = 480;
        let mut inbuf = [0.0f32; N];
        let mut outbuf = [0.0f32; N];
        let mut a = atten;
        (d.connect_port.unwrap())(h, 0, inbuf.as_mut_ptr());
        (d.connect_port.unwrap())(h, 1, outbuf.as_mut_ptr());
        (d.connect_port.unwrap())(h, 2, &mut a as *mut f32);
        (d.activate.unwrap())(h);
        let mut seed = 99u32;
        let mut e = 0.0f64;
        for b in 0..300 {
            for s in inbuf.iter_mut() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *s = 0.1 * ((seed >> 9) as f32 / 8_388_608.0 - 1.0);
            }
            (d.run.unwrap())(h, N as c_ulong);
            if b > 100 {
                e += outbuf.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            }
        }
        (d.cleanup.unwrap())(h);
        e
    };
    // 0 dB = bypass (keeps the noisy input), 100 dB = full reduction (suppresses
    // it). The old inverted mapping produced the opposite and would fail here.
    let bypass = energy(0.0);
    let full = energy(100.0);
    assert!(
        bypass > full * 2.0,
        "0 dB (bypass) energy {bypass:.4} should clearly exceed 100 dB (full reduction) {full:.4}"
    );
}

#[test]
fn abi_sr_gate_inplace_and_nan_sanitised() {
    let d = unsafe { &*(dfn3ll_ladspa::ladspa_descriptor_ptr(0) as *const Desc) };
    assert_eq!(d.port_count, 19);
    let inst = d.instantiate.unwrap();
    let connect = d.connect_port.unwrap();
    let activate = d.activate.unwrap();
    let run = d.run.unwrap();
    let cleanup = d.cleanup.unwrap();

    // sample rate other than 48 kHz is rejected
    let dp = d as *const Desc;
    assert!(inst(dp, 44100).is_null(), "non-48k must be rejected");
    let h = inst(dp, 48000);
    assert!(!h.is_null());

    const N: usize = 480;
    let mut buf = [0.0f32; N];
    let mut ctl = 0.0f32;
    connect(h, 0, buf.as_mut_ptr());
    connect(h, 1, buf.as_mut_ptr()); // same buffer: in-place
    connect(h, 2, &mut ctl as *mut f32);
    activate(h);

    // In-place processing, with one NaN injected, must never emit a non-finite
    // sample and must recover afterwards (the input NaN is sanitised at the boundary).
    for b in 0..300 {
        for (i, s) in buf.iter_mut().enumerate() {
            *s = 0.05 * ((b * N + i) as f32 * 0.05).sin();
        }
        if b == 20 {
            buf[7] = f32::NAN;
        }
        run(h, N as c_ulong);
        assert!(
            buf.iter().all(|v| v.is_finite()),
            "non-finite output at block {b}"
        );
    }
    cleanup(h);
}

#[test]
fn abi_reactivation_reapplies_unchanged_attenuation() {
    let d = desc();
    // Compare the same signal before/after activation with a control whose
    // value is deliberately unchanged. All state must reset, controls must not.
    for level in [0.0f32, 12.0, 60.0] {
        let h = (d.instantiate.unwrap())(d as *const Desc, 48000);
        assert!(!h.is_null());
        let mut input = [0.0f32; 480];
        let mut output = [0.0f32; 480];
        let mut attenuation = level;
        let mut depth = 0.0f32;
        (d.connect_port.unwrap())(h, 0, input.as_mut_ptr());
        (d.connect_port.unwrap())(h, 1, output.as_mut_ptr());
        (d.connect_port.unwrap())(h, 2, &mut attenuation);
        (d.connect_port.unwrap())(h, 6, &mut depth);
        let mut reference = Vec::new();
        for activation in 0..2 {
            (d.activate.unwrap())(h);
            for frame in 0..132 {
                for (i, x) in input.iter_mut().enumerate() {
                    let t = (frame * 480 + i) as f32;
                    *x = 0.07 * (t * 0.043).sin() + 0.015 * (t * 0.193).sin();
                }
                (d.run.unwrap())(h, 480);
                if activation == 0 {
                    reference.extend_from_slice(&output);
                } else {
                    assert_eq!(
                        &output[..],
                        &reference[frame * 480..(frame + 1) * 480],
                        "control {level} changed on activation, frame {frame}"
                    );
                }
            }
            if let Some(deactivate) = d.deactivate {
                deactivate(h);
            }
        }
        (d.cleanup.unwrap())(h);
    }
}

#[test]
fn abi_new_ports_are_appended_without_renumbering_existing_controls() {
    let d = desc();
    for (port, expected) in [
        (6, "Silence expander depth (dB)"),
        (7, "Post filter beta"),
        (8, "Startup mute (ms)"),
    ] {
        // SAFETY: descriptor tables contain port_count entries.
        let name = unsafe { std::ffi::CStr::from_ptr(*d.port_names.add(port)) };
        assert_eq!(name.to_str().unwrap(), expected);
    }
}

#[test]
fn abi_opt_in_short_startup_does_not_discard_the_first_second() {
    let d = desc();
    let h = (d.instantiate.unwrap())(d, 48000);
    assert!(!h.is_null());
    let mut beta = 0.02f32;
    let mut mute_ms = 0.0f32;
    let mut outbuf = [0.0f32; 480];
    let input: Vec<f32> = include_bytes!("../../dfn3-ladspa/tests/fixtures/continuous-speech.pcm")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f32::from(i16::from_le_bytes(*v)) / 32768.0)
        .collect();
    (d.connect_port.unwrap())(h, 1, outbuf.as_mut_ptr());
    (d.connect_port.unwrap())(h, 7, &mut beta);
    (d.connect_port.unwrap())(h, 8, &mut mute_ms);
    (d.activate.unwrap())(h);
    let mut peak = 0.0f32;
    for frame in input.as_chunks::<480>().0.iter().take(100) {
        // The input port is read-only despite LADSPA's mutable pointer ABI.
        (d.connect_port.unwrap())(h, 0, frame.as_ptr().cast_mut());
        (d.run.unwrap())(h, 480);
        assert!(outbuf.iter().all(|v| v.is_finite()));
        peak = peak.max(outbuf.iter().fold(0.0f32, |a, v| a.max(v.abs())));
    }
    (d.cleanup.unwrap())(h);
    assert!(peak > 0.001, "speech was still discarded for a second");
}
