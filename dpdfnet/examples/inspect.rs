//! Loads a model bundle and prints its summary, or the reason it fails to load.
use dpdfnet_native::{audio::LATENCY, AudioProcessor, Bundle};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: inspect MODEL_DIRECTORY");
        std::process::exit(2);
    }
    let result = Bundle::open(&args[1]).and_then(|b| {
        let weight_bytes = b.weight_bytes();
        let meta = b.manifest()?;
        AudioProcessor::new(b)?;
        let summary = serde_json::json!({"loaded": true, "weight_bytes": weight_bytes,
            "sample_rate": 48000, "depth": meta["depth"], "quantization": meta["quantization"],
            "latency_samples": LATENCY});
        println!("{summary}");
        Ok(())
    });
    if let Err(e) = result {
        eprintln!("MODEL LOAD FAILED: {e}");
        std::process::exit(1);
    }
}
