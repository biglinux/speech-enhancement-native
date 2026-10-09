//! Offline AEC over raw mono f32 files, for parity against the LocalVQE CLI.
//!   aec_run <model.gguf> <mic.f32> <ref.f32> <out.f32>
//! `AEC_STREAM=1` selects the streaming path instead of the whole-file one.
use std::io::Write;
use std::process::ExitCode;

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn run(model: &str, mic: &str, reference: &str, out: &str) -> Result<(), String> {
    let m = gtcrn_aec::Model::load(model)?;
    let mic = read_f32(mic)?;
    let reference = read_f32(reference)?;
    let stream = std::env::var_os("AEC_STREAM").is_some();
    let n = mic.len().min(reference.len());
    let t0 = std::time::Instant::now();
    let result = if stream {
        gtcrn_aec::run_aec_stream(&m, &mic, &reference)?
    } else {
        gtcrn_aec::run_aec(&m, &mic, &reference)?
    };
    let elapsed = t0.elapsed().as_secs_f64();
    let audio = n as f64 / 16_000.0;
    eprintln!(
        "aec_run: {} {n} samples  wall={elapsed:.3}s  audio={audio:.3}s  RTF={:.4}",
        if stream { "stream" } else { "batch" },
        elapsed / audio,
    );
    let file = std::fs::File::create(out).map_err(|e| format!("create {out}: {e}"))?;
    let mut w = std::io::BufWriter::new(file);
    for v in result {
        w.write_all(&v.to_le_bytes())
            .map_err(|e| format!("write {out}: {e}"))?;
    }
    w.flush().map_err(|e| format!("write {out}: {e}"))
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    let [_, model, mic, reference, out] = a.as_slice() else {
        eprintln!("usage: aec_run <model.gguf> <mic.f32> <ref.f32> <out.f32>");
        return ExitCode::from(2);
    };
    match run(model, mic, reference, out) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("aec_run: {e}");
            ExitCode::FAILURE
        }
    }
}
