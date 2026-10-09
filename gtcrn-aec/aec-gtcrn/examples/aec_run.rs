//! Offline AEC over raw mono f32 files, for parity against the LocalVQE CLI.
//!   aec_run <model.gguf> <mic.f32> <ref.f32> <out.f32>
use std::io::{Read, Write};

fn read_f32(path: &str) -> Vec<f32> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut b)
        .unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: aec_run <model.gguf> <mic.f32> <ref.f32> <out.f32>");
        std::process::exit(2);
    }
    let m = aec_gtcrn::Model::load(&a[1]).expect("load model");
    let mic = read_f32(&a[2]);
    let reference = read_f32(&a[3]);
    let stream = std::env::var("AEC_STREAM").is_ok();
    let n = mic.len().min(reference.len());
    let t0 = std::time::Instant::now();
    let out = if stream {
        aec_gtcrn::run_aec_stream(&m, &mic, &reference)
    } else {
        aec_gtcrn::run_aec(&m, &mic, &reference)
    }
    .expect("run");
    let elapsed = t0.elapsed().as_secs_f64();
    let audio = n as f64 / 16_000.0;
    eprintln!(
        "aec_run: {} {:.0} samples  wall={:.3}s  audio={:.3}s  RTF={:.4}",
        if stream { "stream" } else { "batch" },
        n as f64,
        elapsed,
        audio,
        elapsed / audio,
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[4]).unwrap());
    for v in out {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}
