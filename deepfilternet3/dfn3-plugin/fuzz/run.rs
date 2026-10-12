//! Fuzzes a DeepFilterNet3 plugin's `run()` the way a host drives it: any block
//! size, separate or in-place buffers, run before activate, reactivation, hostile
//! control values (the voice detector choice too) and NaN/inf audio. The output must stay finite. Each plugin's
//! fuzz crate builds this file against its own descriptor, as `plugin`.
//!
//! Input: the block size (u16 LE), flags, one selector per input control, then the
//! audio as f32 LE.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::os::raw::c_ulong;

/// Every input control: the voice detector choice follows the latency output.
const CONTROL_PORTS: [usize; 19] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21,
];
const LATENCY: c_ulong = 20;
const HEADER: usize = 3 + CONTROL_PORTS.len();
/// Control selectors 1..=8; 0 leaves the port unconnected and larger values pick a
/// point in the port's range.
const HOSTILE: [f32; 8] = [
    0.0,
    -1.0,
    1e9,
    -1e9,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::MAX,
];
/// 0.64 s, so a case stays fast.
const MAX_SAMPLES: usize = 480 * 64;

const IN_PLACE: u8 = 1;
const ACTIVATE: u8 = 2;
const REACTIVATE: u8 = 4;

fuzz_target!(|data: &[u8]| {
    let Some((header, audio)) = data.split_at_checked(HEADER) else {
        return;
    };
    let d = &plugin::DESCRIPTOR;
    let block = usize::from(u16::from_le_bytes([header[0], header[1]])) % 4096 + 1;
    let flags = header[2];
    let input: Vec<f32> = audio
        .as_chunks::<4>()
        .0
        .iter()
        .take(MAX_SAMPLES)
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    // SAFETY: instantiate() takes no pointer it reads.
    let h = unsafe { (d.instantiate.unwrap())(d, 48_000) };
    assert!(!h.is_null());
    let (connect, run, activate) = (
        d.connect_port.unwrap(),
        d.run.unwrap(),
        d.activate.unwrap(),
    );
    let mut controls = [0.0f32; CONTROL_PORTS.len()];
    for (k, &sel) in header[3..].iter().enumerate() {
        controls[k] = match sel {
            0 => continue,
            1..=8 => HOSTILE[usize::from(sel - 1)],
            _ => {
                // SAFETY: `port_range_hints` has `port_count` entries.
                let hint = unsafe { *d.port_range_hints.add(CONTROL_PORTS[k]) };
                hint.lower + f32::from(sel - 9) / 246.0 * (hint.upper - hint.lower)
            }
        };
    }
    let mut latency = 0.0f32;
    let mut buf = vec![0.0f32; block];
    let mut separate = vec![0.0f32; block];
    let in_place = flags & IN_PLACE != 0;
    let out_ptr = if in_place {
        buf.as_mut_ptr()
    } else {
        separate.as_mut_ptr()
    };
    // SAFETY: `h` lives until cleanup(); the controls, latency and both buffers
    // (`block` samples each) outlive every run.
    unsafe {
        for (k, &sel) in header[3..].iter().enumerate() {
            if sel != 0 {
                connect(h, CONTROL_PORTS[k] as c_ulong, &raw mut controls[k]);
            }
        }
        connect(h, LATENCY, &raw mut latency);
        connect(h, 0, buf.as_mut_ptr());
        connect(h, 1, out_ptr);
        if flags & ACTIVATE != 0 {
            activate(h);
        }
        run(h, 0);
    }
    let pieces = input.len().div_ceil(block);
    for (i, piece) in input.chunks(block).enumerate() {
        buf[..piece.len()].copy_from_slice(piece);
        // SAFETY: as above, and `piece` is at most `block` long.
        unsafe {
            if flags & REACTIVATE != 0 && i == pieces / 2 {
                activate(h);
            }
            run(h, piece.len() as c_ulong);
        }
        let out = if in_place { &buf } else { &separate };
        assert!(out[..piece.len()].iter().all(|v| v.is_finite()));
    }
    assert!(latency.is_finite() && latency > 0.0, "latency {latency}");
    // SAFETY: `h` came from instantiate() and is reclaimed once, here.
    unsafe { (d.cleanup.unwrap())(h) };
});
