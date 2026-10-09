#!/usr/bin/env python3
"""Losslessly rearrange W8A16 matrix bytes; no calibration or requantization.

Schema 2 makes old executors REJECT the new layout rather than silently reading
it as row-major. Float/scales/bias bytes and file length remain unchanged. Use a
new destination; the source is never modified. --unpack restores schema-1 rows.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

ROW = "row-major-v1"
PAIR = "pair-output8-v1"


def walk(value, path="root"):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, f"{path}[{i}]")


def positive_int(value, key, maximum):
    n = value.get(key)
    if type(n) is not int or not 0 < n <= maximum:
        raise ValueError(f"Invalid {key}: {n!r}")
    return n


def repack_array(weight: np.ndarray, unpack: bool = False) -> np.ndarray:
    """Input/output shaped [rows,cols], but packed result's dimensions are opaque."""
    a = np.asarray(weight, dtype=np.int8)
    if a.ndim != 2:
        raise ValueError("Expected a matrix")
    rows, cols = a.shape
    if rows == 0 or rows % 8 or cols == 0 or cols % 2:
        raise ValueError("Rows must be a positive multiple of eight; columns even")
    if unpack:
        return a.reshape(rows // 8, cols // 2, 8, 2).transpose(0, 2, 1, 3).reshape(rows, cols).copy()
    return a.reshape(rows // 8, 8, cols // 2, 2).transpose(0, 2, 1, 3).reshape(rows, cols).copy()


def transform(manifest: dict, blob: bytes, unpack: bool = False) -> tuple[dict, bytes, dict]:
    if manifest.get("schema") not in (1, 2) or manifest.get("architecture") != "dpdfnet-48hr-v1":
        raise ValueError("Unsupported bundle schema/architecture")
    if len(blob) > 64 * 1024 * 1024:
        raise ValueError("Weight blob exceeds native limit")
    digest = hashlib.sha256(blob).hexdigest()
    if manifest.get("weights_sha256") != digest or manifest.get("weight_bytes") != len(blob):
        raise ValueError("Source weight checksum/length mismatch")
    m = copy.deepcopy(manifest)
    data = bytearray(blob)
    targets = []
    all_tensors = []
    for path, item in walk(m):
        if "dtype" in item and "offset" in item and "len" in item:
            if item["dtype"] not in ("f32", "i8"):
                raise ValueError(f"Unsupported tensor dtype at {path}")
            size = 4 if item["dtype"] == "f32" else 1
            off, count = item["offset"], item["len"]
            if type(off) is not int or type(count) is not int or off < 0 or count < 0 or off % size or off + count * size > len(blob):
                raise ValueError(f"Tensor range/alignment failure at {path}")
            all_tensors.append((off, off + count * size, item["dtype"], path))
        if item.get("kind") != "w8a16":
            continue
        rows = positive_int(item, "rows", 3072)
        cols = positive_int(item, "cols", 1024)
        layout = item.get("layout", ROW)
        if layout not in (ROW, PAIR) or (layout == PAIR and m["schema"] != 2):
            raise ValueError(f"Invalid layout/schema at {path}")
        ref = item.get("weight", {})
        if ref.get("dtype") != "i8" or ref.get("len") != rows * cols:
            raise ValueError(f"Wrong quantized weight dimensions at {path}")
        off = ref.get("offset")
        if type(off) is not int or off < 0 or off + rows * cols > len(blob):
            raise ValueError(f"Invalid matrix offset at {path}")
        q = np.frombuffer(blob, dtype=np.int8, count=rows * cols, offset=off)
        if np.any(q == -128):
            raise ValueError(f"Asymmetric int8 minimum is not supported: {path}")
        sr = item.get("scales", {})
        so = sr.get("offset")
        if (sr.get("dtype") != "f32" or sr.get("len") != rows or type(so) is not int
                or so < 0 or so % 4 or so + rows * 4 > len(blob)):
            raise ValueError(f"Invalid scale tensor at {path}")
        scales = np.frombuffer(blob, dtype="<f4", count=rows, offset=so)
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError(f"Invalid quantization scales at {path}")
        if rows % 8 or cols % 2:
            raise ValueError(f"Rows must be a multiple of eight and columns even at {path}")
        want = ROW if unpack else PAIR
        targets.append((path, item, off, rows, cols, layout, want))
    changed = []
    seen = {}
    for path, item, off, rows, cols, layout, want in targets:
        end = off + rows * cols
        for a, z, dtype, other in all_tensors:
            if max(a, off) < min(z, end) and (a != off or z != end or dtype != "i8"):
                raise ValueError(f"Aliased/overlapping tensor: {path} and {other}")
        key = (off, end)
        spec = (rows, cols, layout)
        if key in seen and seen[key] != spec:
            raise ValueError(f"Ambiguous aliased matrix shape/layout: {path}")
        if key not in seen and layout != want:
            q = np.frombuffer(blob, dtype=np.int8, count=rows * cols, offset=off).reshape(rows, cols)
            data[off:end] = repack_array(q, unpack=unpack).tobytes()
            changed.append(dict(path=path, rows=rows, cols=cols, bytes=rows * cols))
        seen[key] = spec
        if want == ROW:
            item.pop("layout", None)
        else:
            item["layout"] = PAIR
    m["schema"] = 1 if unpack else 2
    m["weights_sha256"] = hashlib.sha256(data).hexdigest()
    result = dict(operation="unpack" if unpack else "pack", lossless=True,
                  source_weights_sha256=digest, target_weights_sha256=m["weights_sha256"],
                  weight_bytes=len(data), changed_matrices=changed,
                  changed_weight_bytes=sum(x["bytes"] for x in changed),
                  note="No weights/scales requantized; no audio quality or CPU result implied.")
    return m, bytes(data), result


def convert(source: Path, destination: Path, unpack: bool = False) -> dict:
    source, destination = source.resolve(), destination.absolute()
    if destination.exists():
        raise FileExistsError(f"Destination exists; use a fresh directory: {destination}")
    mp, bp = source / "manifest.json", source / "weights.bin"
    if mp.stat().st_size > 4 * 1024 * 1024 or bp.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Bundle exceeds native size bounds")
    original_manifest = mp.read_bytes()
    original_blob = bp.read_bytes()
    m, blob, report = transform(json.loads(original_manifest), original_blob, unpack)
    report["source_manifest_sha256"] = hashlib.sha256(original_manifest).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".dpdfnet-pack-", dir=destination.parent))
    try:
        (temp / "weights.bin").write_bytes(blob)
        (temp / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        (temp / "packing.json").write_text(json.dumps(report, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(temp, destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unpack", action="store_true")
    a = p.parse_args()
    try:
        report = convert(a.source, a.output, a.unpack)
    except (OSError, ValueError) as e:
        p.error(str(e))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
