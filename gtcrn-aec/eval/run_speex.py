#!/usr/bin/env python3
"""SpeexDSP AEC runner (ctypes -> libspeexdsp), classic linear DSP baseline.

speex_echo_cancellation() adapts a filter of `tail` taps to model delay+RIR, then
speex_preprocess_run() with the echo state applied removes residual echo. int16 API.

  python3 run_speex.py --testset ~/.cache/aec-eval/testset --out <dir>
Writes <out>/<rate>/<scenario>/cleaned.wav and prints a resource JSON line.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf

lib = C.CDLL("libspeexdsp.so.1")
lib.speex_echo_state_init.restype = C.c_void_p
lib.speex_echo_state_init.argtypes = [C.c_int, C.c_int]
lib.speex_echo_cancellation.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
lib.speex_echo_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
lib.speex_echo_state_destroy.argtypes = [C.c_void_p]
lib.speex_preprocess_state_init.restype = C.c_void_p
lib.speex_preprocess_state_init.argtypes = [C.c_int, C.c_int]
lib.speex_preprocess_ctl.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
lib.speex_preprocess_run.argtypes = [C.c_void_p, C.c_void_p]
lib.speex_preprocess_state_destroy.argtypes = [C.c_void_p]

SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_PREPROCESS_SET_ECHO_STATE = 24


def to_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def process(mic: np.ndarray, ref: np.ndarray, rate: int, use_pre: bool) -> np.ndarray:
    frame = 256 if rate <= 16000 else 512
    tail = int(0.4 * rate)  # cover 95 ms bulk delay + RIR
    st = lib.speex_echo_state_init(frame, tail)
    r = C.c_int(rate)
    lib.speex_echo_ctl(st, SPEEX_ECHO_SET_SAMPLING_RATE, C.byref(r))
    pre = None
    if use_pre:
        # Residual echo suppression only — denoise/AGC left at library defaults; the
        # baseline runs without this (plan §7.1: SpeexDSP sem preprocessor residual).
        pre = lib.speex_preprocess_state_init(frame, rate)
        lib.speex_preprocess_ctl(pre, SPEEX_PREPROCESS_SET_ECHO_STATE, C.cast(st, C.c_void_p))

    m = to_i16(mic)
    p = to_i16(ref)
    n = (min(len(m), len(p)) // frame) * frame
    out = np.zeros(n, np.int16)
    buf = (C.c_int16 * frame)()
    for i in range(0, n, frame):
        rec = m[i:i + frame].ctypes.data_as(C.c_void_p)
        play = p[i:i + frame].ctypes.data_as(C.c_void_p)
        lib.speex_echo_cancellation(st, rec, play, buf)
        if pre is not None:
            lib.speex_preprocess_run(pre, buf)
        out[i:i + frame] = np.frombuffer(buf, np.int16)
    if pre is not None:
        lib.speex_preprocess_state_destroy(pre)
    lib.speex_echo_state_destroy(st)
    return out.astype(np.float32) / 32768.0, frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="~/.cache/aec-eval/testset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rates", default="16000,48000")
    ap.add_argument("--preprocess", action="store_true", help="add residual echo suppressor")
    args = ap.parse_args()
    ts = Path(args.testset).expanduser()
    out = Path(args.out).expanduser()
    proc_s = 0.0
    audio_s = 0.0
    latency_ms = {}
    for rate in (int(r) for r in args.rates.split(",")):
        for scen in ("farend_only", "nearend_only", "doubletalk", "delaychange"):
            base = ts / f"{rate}" / scen
            mic, _ = sf.read(str(base / "mic.wav"), dtype="float32", always_2d=True)
            ref, _ = sf.read(str(base / "ref.wav"), dtype="float32", always_2d=True)
            mic, ref = mic.mean(1), ref.mean(1)
            t0 = time.perf_counter()
            cleaned, frame = process(mic, ref, rate, args.preprocess)
            proc_s += time.perf_counter() - t0
            audio_s += len(cleaned) / rate
            latency_ms[str(rate)] = round(1000 * frame / rate, 1)
            d = out / f"{rate}" / scen
            d.mkdir(parents=True, exist_ok=True)
            sf.write(str(d / "cleaned.wav"), cleaned, rate, subtype="FLOAT")
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({
        "name": "speexdsp",
        "rt_factor": round(proc_s / audio_s, 4),
        "latency_ms": latency_ms,
        "peak_rss_mb": round(rss_mb, 1),
        "weight_kb": 0,
    }))


if __name__ == "__main__":
    main()
