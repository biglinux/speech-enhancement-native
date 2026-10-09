#!/usr/bin/env python3
"""Expected Silero probabilities for `tests/parity.rs`, from the official model.

The input is rebuilt by the test from the shared `testdata` speech, so only the
probabilities are committed. Needs the silero-vad 6.2.3 package (its ONNX model
and `OnnxWrapper`), torch, onnxruntime and scipy:

    python3 reference.py /path/to/silero_vad-6.2.3-package-dir ../tests/fixtures
"""

import sys
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "testdata"
SECONDS = 30


def lcg_noise(count):
    """Deterministic noise the Rust test reproduces bit for bit."""
    state, out = 0x12345678, np.empty(count, np.float32)
    for i in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((state >> 8) / 16777216.0 - 0.5) * np.float32(0.02)
    return out


def pcm(name):
    return np.fromfile(FIXTURES / name, "<i2").astype(np.float32) / np.float32(32768.0)


def signal(rate):
    """Speech, half a second of silence, more speech, half a second of noise, repeated."""
    step = 48000 // rate
    gap = rate // 2
    block = np.concatenate(
        [
            pcm("speech.pcm")[::step],
            np.zeros(gap, np.float32),
            pcm("continuous-speech.pcm")[::step],
            lcg_noise(gap),
        ]
    )
    whole = np.tile(block, -(-SECONDS * rate // len(block)))
    return whole[: SECONDS * rate].astype(np.float32)


def probabilities(model, audio16):
    import torch

    model.reset_states()
    return np.array(
        [
            model(torch.from_numpy(audio16[i : i + 512].copy()), 16000).item()
            for i in range(0, len(audio16) - 511, 512)
        ],
        np.float32,
    )


def main(package, out):
    from scipy.signal import resample_poly

    sys.path.insert(0, str(Path(package).resolve()))
    from silero_vad import load_silero_vad

    model = load_silero_vad(onnx=True, opset_version=15)
    out = Path(out)
    core = probabilities(model, signal(16000))
    core.tofile(out / "expected_16k.f32")
    wide = probabilities(model, resample_poly(signal(48000), 1, 3).astype(np.float32))
    wide.tofile(out / "expected_48k.f32")
    print(
        f"{len(core)} + {len(wide)} probabilities, speech share {np.mean(core > 0.5):.2f}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
