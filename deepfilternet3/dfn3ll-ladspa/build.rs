//! Writes the weights `include_bytes!` embeds into `OUT_DIR`, in the pair-packed
//! layout the engine reads; a row-major file is packed here.

#[expect(dead_code, reason = "the build script only converts")]
#[path = "../../ops/src/pack_format.rs"]
mod pack_format;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=DFN3LL_WEIGHTS");
    println!("cargo:rerun-if-changed=../../ops/src/pack_format.rs");
    let path = std::env::var_os("DFN3LL_WEIGHTS").map_or_else(
        || {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("dfn3ll_weights.bin")
        },
        PathBuf::from,
    );
    println!("cargo:rerun-if-changed={}", path.display());
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; set DFN3LL_WEIGHTS to the model weights",
            path.display()
        )
    });
    let packed = pack_format::convert(&raw, pack_format::Geometry::DFN3LL)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("embedded_weights.bin");
    std::fs::write(&out, packed).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}
