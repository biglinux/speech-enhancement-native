//! Denoise a WAV file with an engine, or time the engine per hop.

use dfn3_plugin::{atten_lim_from_db, Denoiser, Network, HOP, SR};
use std::process::ExitCode;
use std::time::Instant;

/// Duration of one hop, the real-time budget of a `process` call.
const HOP_MS: f64 = HOP as f64 * 1000.0 / SR as f64;

/// Runs the command line of the CLI called `name` over engine `N`.
pub fn main<N: Network>(name: &str) -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("denoise") if (3..=7).contains(&args.len()) => denoise::<N>(&args[1..]),
        Some("bench") if args.len() == 1 => {
            bench::<N>();
            Ok(())
        }
        _ => {
            eprintln!(
                "usage: {name} denoise IN.wav OUT.wav [POST_FILTER_BETA [ATTEN_DB [MIN_DB [MAX_DB_ERB [MAX_DB_DF]]]]]\n       {name} bench\n\
                 IN.wav is 48 kHz mono, 16-bit PCM or 32-bit float; OUT.wav is 32-bit float."
            );
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{name}: {e}");
            ExitCode::FAILURE
        }
    }
}

fn denoise<N: Network>(args: &[String]) -> Result<(), String> {
    let bytes = std::fs::read(&args[0]).map_err(|e| format!("{}: {e}", args[0]))?;
    let x = read_wav(&bytes).map_err(|e| format!("{}: {e}", args[0]))?;
    let mut eng = Denoiser::<N>::new();
    let mut controls = Vec::new();
    for a in &args[2..] {
        controls.push(a.parse::<f32>().map_err(|e| format!("{a}: {e}"))?);
    }
    let mut controls = controls.into_iter();
    if let Some(v) = controls.next() {
        eng.post_filter_beta = v;
    }
    if let Some(db) = controls.next() {
        eng.atten_lim = atten_lim_from_db(db);
    }
    if let Some(v) = controls.next() {
        eng.min_db = v;
    }
    if let Some(v) = controls.next() {
        eng.max_db_erb = v;
    }
    if let Some(v) = controls.next() {
        eng.max_db_df = v;
    }

    // No warm-up: the engine is stateful and every hop is part of the output.
    let hops = x.len() / HOP;
    let mut out = vec![0.0f32; hops * HOP];
    let mut ms = Vec::with_capacity(hops);
    // Which stages ran, by the same LSNR ladder as `Denoiser::process`.
    let (mut full, mut erb_only, mut skip_all, mut zeros) = (0, 0, 0, 0);
    for (input, output) in x
        .as_chunks::<HOP>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<HOP>().0)
    {
        let t = Instant::now();
        eng.process(input, output);
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        if eng.lsnr < eng.min_db {
            zeros += 1;
        } else if eng.lsnr > eng.max_db_erb {
            skip_all += 1;
        } else if eng.lsnr > eng.max_db_df {
            erb_only += 1;
        } else {
            full += 1;
        }
    }
    std::fs::write(&args[1], wav_f32(&out)).map_err(|e| format!("{}: {e}", args[1]))?;
    if hops == 0 {
        return Ok(());
    }
    eprintln!(
        "thresholds {}/{}/{} hops {hops} {}",
        eng.min_db,
        eng.max_db_erb,
        eng.max_db_df,
        timing(&mut ms)
    );
    let pct = |n: i32| 100.0 * f64::from(n) / hops as f64;
    eprintln!(
        "stages: full {full} ({:.1}%) erb-only {erb_only} ({:.1}%) skip-all {skip_all} ({:.1}%) zeros {zeros} ({:.1}%)",
        pct(full),
        pct(erb_only),
        pct(skip_all),
        pct(zeros)
    );
    Ok(())
}

fn bench<N: Network>() {
    const HOPS: usize = 3000;
    let input: Vec<[f32; HOP]> = (0..HOPS)
        .map(|f| {
            std::array::from_fn(|j| {
                0.05 * ((f * HOP + j) as f32 * 0.02).sin()
                    + 0.01 * ((((f * 7 + j) % 997) as f32 / 498.0) - 1.0)
            })
        })
        .collect();
    let mut eng = Denoiser::<N>::new();
    let mut out = [0.0f32; HOP];
    for frame in &input[..100] {
        eng.process(frame, &mut out);
    }
    let mut ms: Vec<f64> = input
        .iter()
        .map(|frame| {
            let t = Instant::now();
            eng.process(frame, &mut out);
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    println!("{}", timing(&mut ms));
}

/// Mean, median, 99th percentile and maximum per hop, and the real-time factor.
fn timing(ms: &mut [f64]) -> String {
    ms.sort_by(f64::total_cmp);
    let n = ms.len();
    let mean = ms.iter().sum::<f64>() / n as f64;
    format!(
        "mean {mean:.3} p50 {:.3} p99 {:.3} max {:.3} ms/hop, RTF {:.3}",
        ms[n / 2],
        ms[n * 99 / 100],
        ms[n - 1],
        mean / HOP_MS
    )
}

/// The samples of a 48 kHz mono WAV file in 16-bit PCM or 32-bit float.
fn read_wav(b: &[u8]) -> Result<Vec<f32>, String> {
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut fmt, mut data) = (None, None);
    let mut pos = 12;
    while pos + 8 <= b.len() {
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = b.get(pos + 8..pos + 8 + size).ok_or("truncated chunk")?;
        match &b[pos..pos + 4] {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + size % 2;
    }
    let fmt = fmt.filter(|f| f.len() >= 16).ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    let u16_at = |o: usize| u16::from_le_bytes([fmt[o], fmt[o + 1]]);
    let (tag, channels, bits) = (u16_at(0), u16_at(2), u16_at(14));
    let rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
    if channels != 1 {
        return Err(format!("{channels} channels, the engine takes mono"));
    }
    if rate != SR as u32 {
        return Err(format!("{rate} Hz, the engine takes {SR} Hz"));
    }
    match (tag, bits) {
        (1, 16) => Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| f32::from(i16::from_le_bytes(*s)) / 32768.0)
            .collect()),
        (3, 32) => Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|s| f32::from_le_bytes(*s))
            .collect()),
        _ => Err(format!(
            "format tag {tag} with {bits} bits; use 16-bit PCM or 32-bit float"
        )),
    }
}

/// A 48 kHz mono 32-bit float WAV file.
fn wav_f32(samples: &[f32]) -> Vec<u8> {
    let data_size = (samples.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + samples.len() * 4);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_size).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(SR as u32).to_le_bytes());
    b.extend_from_slice(&(SR as u32 * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_size.to_le_bytes());
    for s in samples {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}
