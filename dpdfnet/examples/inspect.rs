//! Loads a model bundle and prints its summary, or the reason it fails to load.
use dpdfnet_native::{AudioProcessor, Bundle, audio::LATENCY};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: inspect MODEL_DIRECTORY");
        return ExitCode::from(2);
    }
    let result = Bundle::open(&args[1]).and_then(|b| {
        let weight_bytes = b.weight_bytes();
        let meta = b.manifest()?;
        AudioProcessor::new(b)?;
        let summary = serde_json::json!({"loaded": true, "weight_bytes": weight_bytes,
            "sample_rate": meta["sample_rate"], "depth": meta["depth"],
            "quantization": meta["quantization"], "latency_samples": LATENCY});
        println!("{summary}");
        Ok(())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inspect: cannot load the model: {e}");
            ExitCode::FAILURE
        }
    }
}
