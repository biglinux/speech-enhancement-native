use dpdfnet_native::{Bundle, Model};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: inspect MODEL_DIRECTORY");
        std::process::exit(2);
    }
    let result=Bundle::open(&args[1]).and_then(|b|{
        let bytes=b.weight_bytes();let meta=b.manifest()?;let metadata_bytes=b.metadata_bytes();let _model=Model::new(b)?;
        println!("{}",serde_json::json!({"loaded":true,"weight_bytes":bytes,"sample_rate":48000,
            "depth":meta["depth"],"quantization":meta["quantization"],"metadata_bytes":metadata_bytes,
            "latency_samples":dpdfnet_native::audio::LATENCY}));
        Ok(())
    });
    if let Err(e) = result {
        eprintln!("MODEL LOAD FAILED: {e}");
        std::process::exit(1);
    }
}
