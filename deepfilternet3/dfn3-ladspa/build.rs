#[allow(dead_code)]
#[path = "../../ops/src/pack_format.rs"]
mod pack_format;
fn main() {
    println!("cargo:rerun-if-env-changed=DFN3_WEIGHTS");
    println!("cargo:rerun-if-changed=../../ops/src/pack_format.rs");
    let path = std::env::var_os("DFN3_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("dfn3_weights.bin")
        });
    println!("cargo:rerun-if-changed={}", path.display());
    let raw=std::fs::read(&path).unwrap_or_else(|e|panic!("Weights missing/unreadable: {}: {e}. Restore the ORIGINAL dfn3_weights.bin or set DFN3_WEIGHTS; no dummy weights are generated.",path.display()));
    // The engine reads the pair-packed layout; row-major files are packed here.
    let bytes = pack_format::convert(&raw, pack_format::Geometry::DFN3)
        .expect("invalid model weight format");
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_weights.bin"),
        bytes,
    )
    .expect("write weights");
}
