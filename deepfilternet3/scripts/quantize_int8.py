#!/usr/bin/env python3
"""Quantize a DeepFilterNet f32 weight blob to the int8-hybrid blob the engines embed.

This is how the committed dfn3_weights.bin and dfn3ll_weights.bin were made. Its
input comes from an ONNX extraction step that is not in this repository:

  F32_BLOB   f32 little-endian tensors, back to back
  F32_TABLE  the extraction's Rust accessor table, one line per tensor:
             `pub fn NAME(&self) -> &[f32] { self.s(OFFSET, LEN) }`

Output: OUT_BLOB = [ f32 non-GRU tensors | int8 GRU W/R | f32 per-row scales ], and
on stdout the offset of every tensor in its section, which the crate's
src/weights.rs table must match (GRU matrices as NAME_q and NAME_s).

Each GRU matrix (a name ending in `_gru<n>_W` or `_gru<n>_R`, shape [3*HID, HID]) is
quantized per output row, symmetric int8:
  scale[i] = max(|w[i,:]|) / 127,   q[i,j] = round(w[i,j] / scale[i]) in [-127, 127]
Everything else stays f32.

Usage: quantize_int8.py F32_BLOB F32_TABLE HID OUT_BLOB
  HID is 256 for DeepFilterNet3 and 512 for DeepFilterNet3-LL.
"""

import re
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    blob_path, table_path, hid, out_path = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
    )

    buf = np.fromfile(blob_path, dtype="<f4")
    with open(table_path) as f:
        layout = [
            (m[0], int(m[1]), int(m[2]))
            for m in re.findall(
                r"pub fn (\w+)\(&self\) -> &\[f32\] \{ self\.s\((\d+), (\d+)\) \}",
                f.read(),
            )
        ]
    if not layout:
        sys.exit(f"{table_path} has no f32 accessor lines")

    f32buf, i8buf, scbuf = [], [], []
    for name, off, n in layout:
        data = buf[off : off + n]
        if re.search(r"_gru\d*_[WR]$", name):
            if n != 3 * hid * hid:
                sys.exit(
                    f"{name}: expected {3 * hid * hid} values for HID {hid}, got {n}"
                )
            w = data.reshape(3 * hid, hid)
            sc = np.abs(w).max(axis=1) / 127.0
            sc[sc == 0] = 1.0
            q = np.round(w / sc[:, None]).clip(-127, 127).astype(np.int8)
            print(f"{name}_q i8 {len(i8buf)} {n}")
            print(f"{name}_s scale {len(scbuf)} {3 * hid}")
            i8buf.extend(q.reshape(-1).tolist())
            scbuf.extend(sc.tolist())
        else:
            print(f"{name} f32 {len(f32buf)} {n}")
            f32buf.extend(data.tolist())

    with open(out_path, "wb") as fo:
        np.array(f32buf, dtype="<f4").tofile(fo)
        np.array(i8buf, dtype=np.int8).tofile(fo)
        np.array(scbuf, dtype="<f4").tofile(fo)


if __name__ == "__main__":
    main()
