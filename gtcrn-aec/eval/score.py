#!/usr/bin/env python3
"""Score one AEC candidate's outputs against the test set.

The candidate dir mirrors the test set: <rate>/<scenario>/cleaned.wav
Metrics per scenario:
  ERLE_dB        echo removed on far-end-active frames (farend_only, delaychange)
  nearSNR_dB     near-end voice preservation vs nearend_clean (nearend_only, doubletalk)
  LSD_dB         log-spectral distance to nearend_clean (lower = less coloration)
Outputs JSON on stdout. Cleaned is delay/gain aligned to the target before scoring.

  python3 score.py --testset ~/.cache/aec-eval/testset --cand <dir> --name webrtc
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly, stft

try:
    from pesq import pesq as _pesq
    from pystoi import stoi as _stoi
    HAVE_PERCEPTUAL = True
except ImportError:
    HAVE_PERCEPTUAL = False


def _to16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return x
    g = np.gcd(rate, 16000)
    return resample_poly(x, 16000 // g, rate // g).astype(np.float32)


def perceptual(est: np.ndarray, clean: np.ndarray, rate: int) -> dict:
    """PESQ (wideband MOS-LQO) + STOI intelligibility vs the clean near-end."""
    if not HAVE_PERCEPTUAL:
        return {}
    n = min(len(est), len(clean))
    e, c = _to16k(est[:n], rate), _to16k(clean[:n], rate)
    m = min(len(e), len(c))
    e, c = e[:m], c[:m]
    out = {}
    try:
        out["PESQ"] = round(float(_pesq(16000, c, e, "wb")), 2)
    except Exception:
        pass
    try:
        out["STOI"] = round(float(_stoi(c, e, 16000, extended=False)), 3)
    except Exception:
        pass
    return out

FRAME = 0.02  # 20 ms analysis frames for activity/segmental metrics


def load(path: Path, n: int | None = None) -> np.ndarray:
    x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x[:n] if n else x


def align(x: np.ndarray, ref: np.ndarray, rate: int) -> np.ndarray:
    """Delay-compensate x to ref. `d>0` means x lags ref by d samples (an AEC's
    algorithmic delay), so we drop x's first d samples; d<0 pads the front."""
    m = min(len(x), len(ref), 4 * rate)
    if rate <= 0 or m == 0 or not np.isfinite(x).all() or not np.isfinite(ref).all():
        raise ValueError("expected positive sample rate and nonempty finite audio")
    a = x[:m].astype(np.float64) - x[:m].mean(dtype=np.float64)
    b = ref[:m].astype(np.float64) - ref[:m].mean(dtype=np.float64)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return x[:min(len(x), len(ref))]
    corr = correlate(a, b, mode="full", method="fft")  # index m-1+d peaks when x lags ref by d
    lags = np.arange(-m + 1, m)
    max_lag = rate // 4  # up to 250 ms
    w = np.abs(lags) <= max_lag
    d = int(lags[w][np.argmax(corr[w])])
    if d > 0:
        x = x[d:]
    elif d < 0:
        x = np.concatenate([np.zeros(-d, np.float32), x])
    n = min(len(x), len(ref))
    return x[:n]


def gain_match(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = min(len(x), len(target))
    x, target = x[:n], target[:n]
    denom = float(np.dot(x, x)) + 1e-12
    g = float(np.dot(x, target)) / denom
    return x * g


def frame_energy(x: np.ndarray, rate: int) -> np.ndarray:
    w = int(FRAME * rate)
    nf = len(x) // w
    return (x[: nf * w].reshape(nf, w) ** 2).mean(axis=1)


def erle_db(mic_echo: np.ndarray, cleaned: np.ndarray, rate: int) -> float:
    n = min(len(mic_echo), len(cleaned))
    me, cl = frame_energy(mic_echo[:n], rate), frame_energy(cleaned[:n], rate)
    m = min(len(me), len(cl))
    me, cl = me[:m], cl[:m]
    active = me > (me.max() * 1e-3)  # far-end-active frames
    if active.sum() == 0:
        return 0.0
    return float(10 * np.log10((me[active].sum() + 1e-12) / (cl[active].sum() + 1e-12)))


def seg_snr_db(est: np.ndarray, clean: np.ndarray, rate: int) -> float:
    n = min(len(est), len(clean))
    est, clean = est[:n], clean[:n]
    w = int(FRAME * rate)
    nf = n // w
    vals = []
    for i in range(nf):
        c = clean[i * w:(i + 1) * w]
        e = est[i * w:(i + 1) * w]
        cp = float(np.dot(c, c))
        if cp < 1e-7:  # skip near-end-silent frames
            continue
        noise = float(np.dot(c - e, c - e)) + 1e-12
        vals.append(10 * np.log10(cp / noise))
    return float(np.clip(np.mean(vals), -10, 60)) if vals else 0.0


def lsd_db(est: np.ndarray, clean: np.ndarray, rate: int) -> float:
    n = min(len(est), len(clean))
    _, _, E = stft(est[:n], rate, nperseg=int(FRAME * rate))
    _, _, C = stft(clean[:n], rate, nperseg=int(FRAME * rate))
    m = min(E.shape[1], C.shape[1])
    le = np.log10(np.abs(E[:, :m]) ** 2 + 1e-8)
    lc = np.log10(np.abs(C[:, :m]) ** 2 + 1e-8)
    # only frames where the target has energy
    act = (np.abs(C[:, :m]) ** 2).sum(axis=0) > 1e-6
    if act.sum() == 0:
        return 0.0
    return float(np.mean(np.sqrt(np.mean((10 * (le - lc))[:, act] ** 2, axis=0))))


def score(testset: Path, cand: Path, rate: int) -> dict:
    out = {}
    for scen in ("farend_only", "nearend_only", "doubletalk", "delaychange"):
        base = testset / f"{rate}" / scen
        cleaned_p = cand / f"{rate}" / scen / "cleaned.wav"
        if not cleaned_p.exists():
            continue
        cleaned = load(cleaned_p)
        m = {}
        if scen in ("farend_only", "delaychange"):
            echo = load(base / "echo.wav")
            m["ERLE_dB"] = round(erle_db(echo, cleaned, rate), 2)
        if scen in ("nearend_only", "doubletalk"):
            near = load(base / "nearend_clean.wav")
            al = gain_match(align(cleaned, near, rate), near)
            m["nearSNR_dB"] = round(seg_snr_db(al, near, rate), 2)
            m["LSD_dB"] = round(lsd_db(al, near, rate), 2)
            m.update(perceptual(al, near, rate))
        out[scen] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="~/.cache/aec-eval/testset")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rates", default="16000,48000")
    args = ap.parse_args()
    testset = Path(args.testset).expanduser()
    cand = Path(args.cand).expanduser()
    res = {"name": args.name, "rates": {}}
    for rate in (int(r) for r in args.rates.split(",")):
        s = score(testset, cand, rate)
        if s:
            res["rates"][str(rate)] = s
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
