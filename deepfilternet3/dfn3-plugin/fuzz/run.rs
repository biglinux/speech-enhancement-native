//! Fuzzes a DeepFilterNet3 plugin's `run()` the way a host drives it: any block
//! size, separate or in-place buffers, run before activate, reactivation, hostile
//! control values and NaN/inf audio. The output must stay finite. Each plugin's
//! fuzz crate builds this file against its own descriptor, as `plugin`.
//!
//! Input: the block size (u16 LE), flags, one selector per input control, then the
//! audio as f32 LE.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::os::raw::c_ulong;

const FIRST_CONTROL: usize = 2;
const CONTROLS: usize = 18;
const LATENCY: c_ulong = 20;
const HEADER: usize = 3 + CONTROLS;
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

    let h = (d.instantiate.unwrap())(d, 48_000);
    assert!(!h.is_null());
    let (connect, run) = (d.connect_port.unwrap(), d.run.unwrap());
    let mut controls = [0.0f32; CONTROLS];
    for (k, &sel) in header[3..].iter().enumerate() {
        controls[k] = match sel {
            0 => continue,
            1..=8 => HOSTILE[usize::from(sel - 1)],
            _ => {
                // SAFETY: `port_range_hints` has `port_count` entries.
                let hint = unsafe { *d.port_range_hints.add(FIRST_CONTROL + k) };
                hint.lower + f32::from(sel - 9) / 246.0 * (hint.upper - hint.lower)
            }
        };
    }
    for (k, &sel) in header[3..].iter().enumerate() {
        if sel != 0 {
            connect(h, (FIRST_CONTROL + k) as c_ulong, &raw mut controls[k]);
        }
    }
    let mut latency = 0.0f32;
    connect(h, LATENCY, &raw mut latency);
    let mut buf = vec![0.0f32; block];
    let mut separate = vec![0.0f32; block];
    let in_place = flags & IN_PLACE != 0;
    connect(h, 0, buf.as_mut_ptr());
    connect(
        h,
        1,
        if in_place {
            buf.as_mut_ptr()
        } else {
            separate.as_mut_ptr()
        },
    );
    if flags & ACTIVATE != 0 {
        (d.activate.unwrap())(h);
    }
    run(h, 0);
    let pieces = input.len().div_ceil(block);
    for (i, piece) in input.chunks(block).enumerate() {
        if flags & REACTIVATE != 0 && i == pieces / 2 {
            (d.activate.unwrap())(h);
        }
        buf[..piece.len()].copy_from_slice(piece);
        run(h, piece.len() as c_ulong);
        let out = if in_place { &buf } else { &separate };
        assert!(out[..piece.len()].iter().all(|v| v.is_finite()));
    }
    assert!(latency.is_finite() && latency > 0.0, "latency {latency}");
    (d.cleanup.unwrap())(h);
});
