# Weight tooling

## `quantize_int8.py`

Records how the committed `dfn3-ladspa/dfn3_weights.bin` and
`dfn3ll-ladspa/dfn3ll_weights.bin` were made from the upstream DeepFilterNet3
ONNX models (Rikorose/DeepFilterNet):

1. An extraction step, not in this repository, wrote each model's tensors as an
   f32 blob and a Rust table of their offsets.
2. This script quantizes the GRU matrices of that blob and writes the int8-hybrid
   blob that is committed.
3. Each crate's `build.rs` repacks the committed blob into the tiled layout the
   engine reads (`ops/src/pack_format.rs`) and embeds it in the plugin.

Without the f32 input the script cannot be rerun from this checkout.

### What it does

Only the GRU projection matrices (`*_gru<n>_W` / `*_gru<n>_R`, shape
`[3*HID, HID]`) are quantized. They hold about 94% of the multiply-adds and are
the memory-bound part. Each is quantized per output row, symmetric int8:

```
scale[i] = max(|w[i,:]|) / 127
q[i,j]   = round(w[i,j] / scale[i])   clamped to [-127, 127]
```

Everything else (convolutions, grouped linears, biases) stays f32. At run time the
product is exact in integers (`ops::gru_cell_packed`, `pmaddwd`):

```
y[i] = (Σ q[i,j] · xq[j]) · scale[i] · xscale
```

with the activation quantized to int16 per vector. This W8A16 scheme gives the
same bits on every SIMD tier.

### Usage

```bash
python3 quantize_int8.py F32_BLOB F32_TABLE HID OUT_BLOB
#   DeepFilterNet3:     HID 256, about 2.67 MB
#   DeepFilterNet3-LL:  HID 512, about 10.7 MB
```

It prints the offset of every tensor in its section; `src/weights.rs` of the
crate must hold the same offsets. Requires `numpy`.
