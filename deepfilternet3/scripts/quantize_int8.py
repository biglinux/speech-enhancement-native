#!/usr/bin/env python3
"""Quantize a DeepFilterNet f32 weight blob to the int8-hybrid format the engines
load, and regenerate that crate's src/weights.rs accessors.

Input  (produced by the upstream-ONNX extraction step, not this script):
  <crate>/<name>_weights.bin   f32 little-endian, laid out per <crate>/src/weights.rs
  <crate>/src/weights.rs       f32 accessor table `pub fn NAME(&self) -> &[f32] { self.s(OFF, N) }`

Output (overwrites both in place — run once on the f32 crate):
  <crate>/<name>_weights.bin   [ f32 non-GRU tensors | int8 GRU W/R | f32 per-row scales ]
  <crate>/src/weights.rs       hybrid accessors: NAME() for f32, NAME_q()/NAME_s() for GRU

Scheme: each GRU weight matrix (name matching `_gru<n>_[WR]`, shape [3*HID, HID]) is
quantized per output row, symmetric int8:  scale[i] = max(|row_i|)/127,
q[i,j] = round(w[i,j]/scale[i]) clamped to [-127,127]. Everything else stays f32.
Runtime dequant:  y[i] = (Σ q[i,j]·xq[j]) · scale[i] · xscale  (see dfn-ops::matvec_i8_i16).

Usage:  quantize_int8.py <crate_dir> <HID>
  e.g.  quantize_int8.py ../dfn3-ladspa 256     (DeepFilterNet3)
        quantize_int8.py ../dfn3ll-ladspa 512   (DeepFilterNet3-LL)
"""
import glob
import re
import sys

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    crate, hid = sys.argv[1], int(sys.argv[2])
    bin_paths = [p for p in glob.glob(f"{crate}/*_weights.bin")]
    if len(bin_paths) != 1:
        sys.exit(f"expected exactly one *_weights.bin in {crate}, found {bin_paths}")
    bin_path = bin_paths[0]
    rs_path = f"{crate}/src/weights.rs"

    buf = np.fromfile(bin_path, dtype="<f4")
    src = open(rs_path).read()
    layout = [
        (m[0], int(m[1]), int(m[2]))
        for m in re.findall(
            r"pub fn (\w+)\(&self\) -> &\[f32\] \{ self\.s\((\d+), (\d+)\) \}", src
        )
    ]
    if not layout:
        sys.exit(f"{rs_path} is not the f32 accessor table (already quantized?)")

    is_gru = lambda n: re.search(r"_gru\d*_[WR]$", n) is not None
    f32buf, i8buf, scbuf = [], [], []
    f32_acc, gru_acc = [], []
    for name, off, n in layout:
        data = buf[off : off + n]
        if is_gru(name):
            if n != 3 * hid * hid:
                sys.exit(f"{name}: expected {3*hid*hid} elems for HID={hid}, got {n}")
            w = data.reshape(3 * hid, hid)
            sc = np.abs(w).max(axis=1) / 127.0
            sc[sc == 0] = 1.0
            q = np.round(w / sc[:, None]).clip(-127, 127).astype(np.int8)
            gru_acc.append((name, len(i8buf), n, len(scbuf), 3 * hid))
            i8buf.extend(q.reshape(-1).tolist())
            scbuf.extend(sc.tolist())
        else:
            f32_acc.append((name, len(f32buf), n))
            f32buf.extend(data.tolist())

    with open(bin_path, "wb") as fo:
        np.array(f32buf, dtype="<f4").tofile(fo)
        np.array(i8buf, dtype=np.int8).tofile(fo)
        np.array(scbuf, dtype="<f4").tofile(fo)

    lines = [
        "// Auto-generated hybrid weight table: non-GRU tensors f32, GRU W/R int8 + per-row scale.",
        "#![allow(dead_code, non_snake_case)]",
        "use std::borrow::Cow;",
        "pub struct W {\n    f32buf: Cow<'static, [f32]>,\n    i8buf: Cow<'static, [i8]>,\n    scbuf: Cow<'static, [f32]>,\n}",
        "impl W {",
        "    /// Owning decode (copies into Vecs); used by the CLIs and file-loading tests.",
        "    pub fn load(bytes: &[u8]) -> Self {",
        f"        let (f, i, s) = ({len(f32buf)}usize, {len(i8buf)}usize, {len(scbuf)}usize);",
        "        let f32buf: Vec<f32> = bytes[..f * 4]",
        "            .as_chunks::<4>()\n            .0\n            .iter()\n            .map(|c| f32::from_le_bytes(*c))\n            .collect();",
        "        let i8buf: Vec<i8> = bytes[f * 4..f * 4 + i].iter().map(|&b| b as i8).collect();",
        "        let scbuf: Vec<f32> = bytes[f * 4 + i..f * 4 + i + s * 4]",
        "            .as_chunks::<4>()\n            .0\n            .iter()\n            .map(|c| f32::from_le_bytes(*c))\n            .collect();",
        "        W { f32buf: Cow::Owned(f32buf), i8buf: Cow::Owned(i8buf), scbuf: Cow::Owned(scbuf) }",
        "    }",
        "    /// Zero-copy view over a 'static, 4-byte-aligned blob (native-endian, x86_64).",
        "    pub fn from_static_aligned(bytes: &'static [u8]) -> Self {",
        f"        let (f, i, s) = ({len(f32buf)}usize, {len(i8buf)}usize, {len(scbuf)}usize);",
        '        assert!(bytes.len() >= f * 4 + i + s * 4, "weight blob too small");',
        '        assert_eq!(bytes.as_ptr() as usize % 4, 0, "weight blob must be 4-byte aligned");',
        "        // SAFETY: 4-aligned blob, f32 sections at 4-divisible offsets, little-endian target.",
        "        let f32buf = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), f) };",
        "        let i8buf = unsafe { std::slice::from_raw_parts(bytes[f * 4..].as_ptr().cast::<i8>(), i) };",
        "        let scbuf = unsafe { std::slice::from_raw_parts(bytes[f * 4 + i..].as_ptr().cast::<f32>(), s) };",
        "        W { f32buf: Cow::Borrowed(f32buf), i8buf: Cow::Borrowed(i8buf), scbuf: Cow::Borrowed(scbuf) }",
        "    }",
        "    /// Pin the weight buffers in RAM (best effort) so realtime processing never",
        "    /// faults on them under memory pressure.",
        "    pub fn mlock(&self) {",
        "        dfn_ops::mlock_slice(&self.f32buf);",
        "        dfn_ops::mlock_slice(&self.i8buf);",
        "        dfn_ops::mlock_slice(&self.scbuf);",
        "    }",
        "    #[inline]\n    fn f(&self, o: usize, n: usize) -> &[f32] { &self.f32buf[o..o + n] }",
        "    #[inline]\n    fn i(&self, o: usize, n: usize) -> &[i8] { &self.i8buf[o..o + n] }",
        "    #[inline]\n    fn sc(&self, o: usize, n: usize) -> &[f32] { &self.scbuf[o..o + n] }",
        "}",
        "impl W {",
    ]
    for name, off, n in f32_acc:
        lines.append(f"    #[inline]\n    pub fn {name}(&self) -> &[f32] {{ self.f({off}, {n}) }}")
    for name, io, n, so, nr in gru_acc:
        lines.append(f"    #[inline]\n    pub fn {name}_q(&self) -> &[i8] {{ self.i({io}, {n}) }}")
        lines.append(f"    #[inline]\n    pub fn {name}_s(&self) -> &[f32] {{ self.sc({so}, {nr}) }}")
    lines.append("}")
    open(rs_path, "w").write("\n".join(lines) + "\n")

    total = len(f32buf) * 4 + len(i8buf) + len(scbuf) * 4
    print(f"{bin_path}: f32={len(f32buf)} i8={len(i8buf)} sc={len(scbuf)} "
          f"({total} bytes, {total/1e6:.2f} MB), {len(gru_acc)} GRU matrices int8")


if __name__ == "__main__":
    main()
