# Weight tooling

## `quantize_int8.py`

Converts a DeepFilterNet **f32** weight blob into the **int8-hybrid** format the
engines load, and regenerates that crate's `src/weights.rs` accessor table.

```
ONNX (upstream)  ──►  f32 <name>_weights.bin + f32 src/weights.rs  ──►  quantize_int8.py  ──►  int8 blob + hybrid weights.rs
   (Rikorose/DeepFilterNet)        (extraction, upstream-specific)         (this script)          (committed in the crate)
```

The extraction step (ONNX → f32 `*_weights.bin` laid out by an f32 `weights.rs`)
is upstream-model-specific and lives outside this repo; the crates ship the int8
output of this script.

### What it does

Only the GRU projection matrices (`*_gru<n>_W` / `*_gru<n>_R`, shape `[3*HID, HID]`)
are quantized — they are ~94% of the compute and the memory-bound part. Each is
quantized **per output row, symmetric int8**:

```
scale[i] = max(|row_i|) / 127
q[i,j]   = round(w[i,j] / scale[i])   clamped to [-127, 127]
```

Everything else (convs, grouped linears, biases) stays f32. At runtime the GEMV is
exact-integer (`dfn-ops::matvec_i8_i16`, `pmaddwd`):

```
y[i] = (Σ q[i,j] · xq[j]) · scale[i] · xscale
```

with the activation quantized to int16 per vector (`quantize_i16`). This W8A16
scheme is bit-identical across the SIMD tiers and validated stage-by-stage against
the upstream ONNX (encoder/mask 100–140 dB SDR).

### Usage

```bash
python3 scripts/quantize_int8.py <crate_dir> <HID>
#   dfn3-ladspa   (DeepFilterNet3):     HID 256  ->  ~2.67 MB (fits a 3 MB L3)
#   dfn3ll-ladspa (DeepFilterNet3-LL):  HID 512  ->  ~10.7 MB
```

Run once on the f32 crate; it overwrites `<name>_weights.bin` and `src/weights.rs`
in place, then run `cargo fmt` (the generated `weights.rs` is valid Rust but not
rustfmt-formatted). Requires `numpy`. The int8 blob is byte-reproducible.
