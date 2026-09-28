#!/usr/bin/env python3
"""Offline WAV renderer for controlled listening tests, NOT the real-time path.

Requires mono 48 kHz WAV. Never silently downmix, resample, normalize or clip.
Writes floating-point WAV and trims exactly the declared structural delay.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from native_api import Native

def read_mono(path: Path) -> np.ndarray:
    rate, x = wavfile.read(path)
    if rate != 48000 or x.ndim != 1:
        raise ValueError("Input must be MONO, 48000 Hz. Convert explicitly before this comparison.")
    if x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128) / 128
    elif np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(np.float32) / float(2 ** (8 * x.dtype.itemsize - 1))
    elif np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    else:
        raise ValueError(f"Unsupported WAV representation: {x.dtype}")
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Input is empty or contains non-finite samples")
    return np.ascontiguousarray(x)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--lib", type=Path, default=Path("target/release-unwind/libdpdfnet_native.so"))
    p.add_argument("--block", type=int, default=480)
    p.add_argument("--attenuation-db", type=float, default=100)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.block <= 0 or a.block > 65536 or not 0 <= a.attenuation_db <= 100:
        raise ValueError("Block must be 1..65536; attenuation must be 0..100 dB")
    if a.input.resolve() == a.output.resolve():
        raise ValueError("Refusing to overwrite the input WAV")
    if a.output.exists():
        raise FileExistsError("Output already exists; select a new output path")
    target = a.report or a.output.with_suffix(a.output.suffix + ".json")
    if target.resolve() in (a.input.resolve(), a.output.resolve()):
        raise ValueError("Report path must not overwrite an audio file")
    x = read_mono(a.input)
    with Native(a.lib, a.bundle) as native:
        # Prime control only, not model audio: a zero-length call is well-defined.
        native.process(np.empty(0, np.float32), db=a.attenuation_db)
        native.reset()
        delay = native.lib.dpdfnet_native_latency_samples()
        padded = np.pad(x, (0, delay))
        y = np.empty_like(padded)
        for start in range(0, len(padded), a.block):
            y[start:start+a.block] = native.process(padded[start:start+a.block], db=a.attenuation_db)
        y = y[delay:delay+len(x)].copy()
    if not np.isfinite(y).all():
        raise ValueError("Native output contains non-finite values")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(a.output, 48000, y)
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    report = dict(input=str(a.input), output=str(a.output), samples=len(x), sample_rate=48000,
                  latency_trimmed_samples=delay, attenuation_db=a.attenuation_db, block=a.block,
                  input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                  weights_sha256=manifest["weights_sha256"], quantization=manifest["quantization"],
                  output_peak=float(np.max(np.abs(y))), samples_above_full_scale=int(np.sum(np.abs(y)>1)),
                  automatic_normalization=False, automatic_limiter=False,
                  note="Floating-point WAV: peaks above 1 are preserved and reported, not clipped.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
