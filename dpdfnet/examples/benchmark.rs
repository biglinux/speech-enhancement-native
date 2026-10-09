//! Times the complete mono audio path under the plugin's FTZ/DAZ setting and
//! fails on any heap use while processing.
//! cargo run --profile release-unwind -p dpdfnet-native --example benchmark -- MODEL_DIRECTORY
use dpdfnet_native::{AudioProcessor, Bundle, HOP};
use std::{process::ExitCode, time::Instant};
#[path = "../../testdata/heap_calls.rs"]
mod heap_calls;
use heap_calls::heap_calls;
fn go() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 || a.len() > 3 {
        return Err("usage: benchmark MODEL_DIRECTORY [AUDIO_SECONDS=10]".into());
    }
    let secs: usize = a
        .get(2)
        .map_or(Ok(10), |s| s.parse::<usize>())
        .map_err(|e| e.to_string())?;
    if secs == 0 || secs > 600 {
        return Err("seconds must be 1..600".into());
    }
    let bundle = Bundle::open(&a[1])?;
    let mut p = AudioProcessor::new(bundle.clone())?;
    let hops = secs * 100;
    let mut samples = vec![0.0f32; hops * HOP];
    let mut state = 0x71f337a5u32;
    for (i, x) in samples.iter_mut().enumerate() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = state as f32 / u32::MAX as f32 - 0.5;
        *x = 0.02 * noise + 0.03 * (i as f32 * 0.02879793).sin();
    }
    let mut out = vec![0.0; HOP];
    let mut times = vec![0.0f64; hops];
    for _ in 0..16 {
        p.process(&samples[..HOP], &mut out);
    }
    p.reset();
    let mut elapsed = 0.0;
    let mut checksum = 0.0f64;
    let heap = heap_calls(|| {
        let all = Instant::now();
        for (n, chunk) in samples.as_chunks::<HOP>().0.iter().enumerate() {
            let start = Instant::now();
            p.process(chunk, &mut out);
            times[n] = start.elapsed().as_secs_f64() * 1e6;
            checksum += out[HOP / 2] as f64;
        }
        elapsed = all.elapsed().as_secs_f64();
    });
    times.sort_by(f64::total_cmp);
    let report = serde_json::json!({"audio_seconds":secs,"wall_seconds":elapsed,"rtf":elapsed/secs as f64,
        "hop_us_median":times[hops/2],"hop_us_p95":times[(hops*95/100).min(hops-1)],
        "hop_us_p99":times[(hops*99/100).min(hops-1)],"hop_us_max":times[hops-1],
        "hot_heap_calls":heap,
        "fault":p.faulted(),"checksum":checksum,"weight_bytes":bundle.weight_bytes(),
        "diagnostics":{"force_avx1":cfg!(feature = "force-avx1"),"force_sse41":cfg!(feature = "force-sse41")},
        "note":"Wall-clock measurement in this process; obtain CPU time and RSS separately with /usr/bin/time -v."});
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    if heap != 0 || p.faulted() {
        return Err("RT acceptance checks failed".into());
    }
    Ok(())
}
fn main() -> ExitCode {
    match go() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("benchmark: {e}");
            ExitCode::FAILURE
        }
    }
}
