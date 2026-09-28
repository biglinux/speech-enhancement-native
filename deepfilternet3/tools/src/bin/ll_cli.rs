//! Validation/bench CLI: process a mono WAV through the Rust DeepFilterNet3-LL engine.
//! Usage: ll_cli <weights.bin> <in.wav> <out.wav>
//! Input must be 48 kHz mono (16-bit or float). Output: 48 kHz f32 WAV.

use dfn3ll_ladspa::{Dfn3Ll, HOP};
use std::io::{Read, Write};

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut b)
        .unwrap();
    assert_eq!(&b[0..4], b"RIFF");
    let mut pos = 12;
    let (mut ch, mut sr, mut bps, mut fmt) = (1u16, 48000u32, 16u16, 1u16);
    let mut data: Vec<f32> = Vec::new();
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let sz = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            fmt = u16::from_le_bytes([b[body], b[body + 1]]);
            ch = u16::from_le_bytes([b[body + 2], b[body + 3]]);
            sr = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
            bps = u16::from_le_bytes([b[body + 14], b[body + 15]]);
        } else if id == b"data" {
            let d = &b[body..body + sz];
            let bytes_per = (bps / 8) as usize * ch as usize;
            let frames = sz / bytes_per;
            for f in 0..frames {
                let s = body + f * bytes_per; // take channel 0
                let v = match (fmt, bps) {
                    (1, 16) => i16::from_le_bytes([b[s], b[s + 1]]) as f32 / 32768.0,
                    (3, 32) => f32::from_le_bytes([b[s], b[s + 1], b[s + 2], b[s + 3]]),
                    _ => panic!("unsupported fmt={fmt} bps={bps}"),
                };
                data.push(v);
            }
            let _ = d;
        }
        pos = body + sz + (sz & 1);
    }
    (data, sr)
}

fn write_wav_f32(path: &str, samples: &[f32], sr: u32) {
    let mut f = std::fs::File::create(path).unwrap();
    let data_size = (samples.len() * 4) as u32;
    let mut h = Vec::new();
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_size).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&3u16.to_le_bytes()); // float
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&sr.to_le_bytes());
    h.extend_from_slice(&(sr * 4).to_le_bytes());
    h.extend_from_slice(&4u16.to_le_bytes());
    h.extend_from_slice(&32u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_size.to_le_bytes());
    f.write_all(&h).unwrap();
    let mut buf = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    f.write_all(&buf).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let wbytes = std::fs::read(&a[1]).unwrap();
    let (x, sr) = read_wav(&a[2]);
    assert_eq!(sr, 48000, "expect 48k input");
    let mut eng = Dfn3Ll::new(&wbytes);
    if let Some(pf) = a.get(4).and_then(|s| s.parse::<f32>().ok()) {
        eng.post_filter_beta = pf;
    }
    if let Some(al) = a.get(5).and_then(|s| s.parse::<f32>().ok()) {
        // Arg is dB, converted the same way as the LADSPA plugin (not the raw
        // internal noisy-mix fraction the engine field holds).
        eng.atten_lim = dfn3ll_ladspa::atten_lim_from_db(al);
    }
    if let Some(v) = a.get(6).and_then(|s| s.parse::<f32>().ok()) {
        eng.min_db = v;
    }
    if let Some(v) = a.get(7).and_then(|s| s.parse::<f32>().ok()) {
        eng.max_db_erb = v;
    }
    if let Some(v) = a.get(8).and_then(|s| s.parse::<f32>().ok()) {
        eng.max_db_df = v;
    }
    let nframes = x.len() / HOP;
    let mut out = vec![0.0f32; nframes * HOP];
    let mut ob = [0.0f32; HOP];
    // No warm-up: the engine is stateful, so a warm-up loop would prepend its
    // frames to the recurrent state and contaminate the written output. The p50/p99
    // over thousands of frames are unaffected by the handful of cold-start frames.
    let mut ts = vec![0.0f64; nframes];
    // stage tally by lsnr ladder (mirrors process()): full / erb-only / skip-all / zeros
    let (mut full, mut erb_only, mut skip_all, mut zeros) = (0u64, 0u64, 0u64, 0u64);
    for f in 0..nframes {
        let t = std::time::Instant::now();
        eng.process(&x[f * HOP..f * HOP + HOP], &mut ob);
        ts[f] = t.elapsed().as_secs_f64() * 1e3;
        out[f * HOP..f * HOP + HOP].copy_from_slice(&ob);
        let l = eng.lsnr;
        if l < eng.min_db {
            zeros += 1;
        } else if l > eng.max_db_erb {
            skip_all += 1;
        } else if l > eng.max_db_df {
            erb_only += 1;
        } else {
            full += 1;
        }
    }
    write_wav_f32(&a[3], &out, 48000);
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = ts.iter().sum::<f64>() / nframes as f64;
    let n = nframes as f64;
    eprintln!(
        "thr={}/{}/{} frames={nframes} mean={:.3} p50={:.3} p99={:.3} max={:.3} ms  RTF={:.3}",
        eng.min_db,
        eng.max_db_erb,
        eng.max_db_df,
        mean,
        ts[nframes / 2],
        ts[(n * 0.99) as usize],
        ts[nframes - 1],
        mean / 10.0
    );
    eprintln!(
        "stages: full={full} ({:.1}%) erb_only={erb_only} ({:.1}%) skip_all={skip_all} ({:.1}%) zeros={zeros} ({:.1}%)",
        100.0 * full as f64 / n,
        100.0 * erb_only as f64 / n,
        100.0 * skip_all as f64 / n,
        100.0 * zeros as f64 / n
    );
}
