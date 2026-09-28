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
    let g = pack_format::Geometry::DFN3;
    let sections = pack_format::sections(&raw, g).expect("invalid model weight format");
    let packed = std::env::var_os("CARGO_FEATURE_R11_PACKED").is_some();
    if sections.packed && !packed {
        panic!("DFNPAIR1 requires r11-packed; provide legacy row-major weights for the control");
    }
    let bytes = if packed {
        pack_format::convert(&raw, g).expect("packing failed")
    } else {
        raw
    };
    std::fs::write(
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_weights.bin"),
        bytes,
    )
    .expect("write weights");
}
