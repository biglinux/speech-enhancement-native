#!/usr/bin/env python3
"""Generate a deterministic AEC test set from real speech + captured room noise.

Per scenario and sample rate it writes three aligned mono wavs:
  ref.wav          far-end (loudspeaker) signal, the AEC reference
  mic.wav          what the microphone captures (echo + near-end + noise)
  nearend_clean.wav the near-end voice alone at the mic, the quality target
Plus echo.wav (echo component alone) so the scorer can mask far-end-active frames.

Scenarios: farend_only (ERLE), nearend_only (must not damage voice),
doubletalk (suppress echo AND keep voice), delaychange (re-adaptation).

Deterministic for a given set of source clips (fixed seed). Run:
  python3 gen_testset.py --far far.wav --near near.wav --noise noise.wav \
      --out ~/.cache/aec-eval/testset
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

HOME = Path(os.path.expanduser("~"))

DURATION_S = 10.0
RATES = (16000, 48000)
SEED = 20260912


def load_mono(path: Path, rate: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data.mean(axis=1)  # to mono
    if sr != rate:
        # resample_poly needs integer up/down; reduce by gcd
        g = np.gcd(sr, rate)
        x = resample_poly(x, rate // g, sr // g).astype(np.float32)
    return x


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Tile/trim x to exactly n samples."""
    if len(x) < n:
        reps = int(np.ceil(n / len(x)))
        x = np.tile(x, reps)
    return x[:n].copy()


def set_rms_db(x: np.ndarray, db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    return (x * (10 ** (db / 20) / rms)).astype(np.float32)


def synth_rir(rate: int, t60: float, rng: np.random.Generator) -> np.ndarray:
    """Simple exponentially-decaying sparse-reflection room impulse response."""
    n = int(t60 * rate)
    t = np.arange(n)
    decay = np.exp(-6.9 * t / n)  # ~ -60 dB at t60
    rir = rng.standard_normal(n).astype(np.float32) * decay
    rir[0] = 1.0  # direct path
    rir /= np.max(np.abs(rir))
    return rir.astype(np.float32)


def make_echo(ref: np.ndarray, rate: int, delay_ms: float, atten_db: float,
              rng: np.random.Generator) -> np.ndarray:
    """Loudspeaker echo path: RIR convolution + bulk delay + soft-clip nonlinearity."""
    rir = synth_rir(rate, t60=0.35, rng=rng)
    echo = fftconvolve(ref, rir)[: len(ref)].astype(np.float32)
    # loudspeaker nonlinearity: gentle soft-clip so a linear-only AEC (Speex) leaves residue
    drive = echo * 3.0
    echo = np.tanh(drive).astype(np.float32) / 3.0
    d = int(delay_ms * rate / 1000)
    echo = np.concatenate([np.zeros(d, np.float32), echo])[: len(ref)]
    return set_rms_db(echo, atten_db)


def write(out: Path, rate: int, scen: str, ref, mic, near, echo) -> None:
    d = out / f"{rate}" / scen
    d.mkdir(parents=True, exist_ok=True)
    for name, sig in (("ref", ref), ("mic", mic), ("nearend_clean", near), ("echo", echo)):
        sf.write(str(d / f"{name}.wav"), np.clip(sig, -1.0, 1.0), rate, subtype="FLOAT")


def build(out: Path, rate: int, sources: argparse.Namespace) -> None:
    rng = np.random.default_rng(SEED)
    n = int(DURATION_S * rate)
    far = fit(load_mono(sources.far, rate), n)
    near = fit(load_mono(sources.near, rate), n)
    noise = fit(load_mono(sources.noise, rate), n)
    far = set_rms_db(far, -16.0)
    near_lvl = set_rms_db(near, -20.0)
    noise_lvl = set_rms_db(noise, -48.0)
    silence = np.zeros(n, np.float32)

    # farend_only: only echo (+noise) reaches the mic -> ERLE
    echo = make_echo(far, rate, delay_ms=45, atten_db=-15.0, rng=rng)
    write(out, rate, "farend_only", far, echo + noise_lvl, silence, echo)

    # nearend_only: far-end silent, mic is just the near-end voice (+noise); AEC must not damage
    write(out, rate, "nearend_only", silence, near_lvl + noise_lvl, near_lvl, silence)

    # doubletalk: both active
    echo2 = make_echo(far, rate, delay_ms=45, atten_db=-15.0, rng=rng)
    write(out, rate, "doubletalk", far, echo2 + near_lvl + noise_lvl, near_lvl, echo2)

    # delaychange: far-end only, delay jumps 45->95 ms at the midpoint (re-adaptation)
    h = n // 2
    e_a = make_echo(far[:h], rate, delay_ms=45, atten_db=-15.0, rng=rng)
    e_b = make_echo(far[h:], rate, delay_ms=95, atten_db=-15.0, rng=rng)
    echo3 = np.concatenate([e_a, e_b])[:n]
    write(out, rate, "delaychange", far, echo3 + noise_lvl, silence, echo3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", type=Path, required=True, help="far-end (loudspeaker) speech")
    ap.add_argument("--near", type=Path, required=True, help="near-end speech, a second speaker")
    ap.add_argument("--noise", type=Path, required=True, help="recorded room noise")
    ap.add_argument("--out", default=str(HOME / ".cache/aec-eval/testset"))
    args = ap.parse_args()
    out = Path(args.out)
    for rate in RATES:
        build(out, rate, args)
        print(f"wrote {rate} Hz scenarios under {out}/{rate}")


if __name__ == "__main__":
    main()
