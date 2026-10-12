#!/usr/bin/env python3
"""Expected FireRedVAD outputs for `tests/parity.rs`, from the official package.

The input is rebuilt by the test from the shared `testdata` speech, so only the
outputs are committed: the AED's speech, singing and music probabilities and the
Stream-VAD's speech probability. Needs the fireredvad 0.0.2 package (kaldi-native-fbank,
kaldiio, torch) and the release's model directories:

    python3 reference.py FireRedVAD ../tests/fixtures
"""

import argparse
import sys
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parents[2] / "testdata"
SECONDS = 20


def lcg_noise(count):
    """Deterministic noise the Rust test reproduces bit for bit."""
    state, out = 0x12345678, np.empty(count, np.float32)
    for i in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((state >> 8) / 16777216.0 - 0.5) * np.float32(0.02)
    return out


def pcm(name):
    return np.fromfile(FIXTURES / name, "<i2").astype(np.float32) / np.float32(32768.0)


def signal():
    """Speech, half a second of silence, more speech, half a second of noise, repeated,
    taken every third sample down to 16 kHz."""
    block = np.concatenate(
        [
            pcm("speech.pcm")[::3],
            np.zeros(8000, np.float32),
            pcm("continuous-speech.pcm")[::3],
            lcg_noise(8000),
        ]
    )
    return np.resize(block, SECONDS * 16000)


def load(directory):
    import torch
    from fireredvad.core.audio_feat import AudioFeat
    from fireredvad.core.detect_model import DetectModel

    torch.serialization.add_safe_globals([argparse.Namespace])
    package = torch.load(
        f"{directory}/model.pth.tar", map_location="cpu", weights_only=True
    )
    model = DetectModel(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()
    return AudioFeat(f"{directory}/cmvn.ark"), model


def main(release, out):
    out = Path(out)
    wav = signal() * np.float32(32768.0)
    for name, directory in (("aed", "AED"), ("stream_vad", "Stream-VAD")):
        features, model = load(f"{release}/{directory}")
        feat, _ = features.extract(wav)
        probs = model.forward(feat.unsqueeze(0))[0].squeeze(0).numpy()
        probs.astype("<f4").tofile(out / f"expected_{name}.f32")
        print(f"{name}: {probs.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
