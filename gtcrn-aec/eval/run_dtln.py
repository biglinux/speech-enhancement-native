#!/usr/bin/env python3
"""DTLN-aec runner (LiteRT/tflite) — two-stage stateful LSTM AEC, 16 kHz only.

Replicates the reference frame loop from breizhn/DTLN-aec run_aec.py: 512-sample
block / 128 hop, stage-1 predicts an STFT-magnitude mask, stage-2 refines in a
learned time-domain, overlap-add. Runs over the 16 kHz test set.

  python3 run_dtln.py --size 128 --models <dir> --testset <ts> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter

BLOCK = 512
SHIFT = 128


def process(mic: np.ndarray, lpb: np.ndarray, it1: Interpreter, it2: Interpreter):
    i1, o1 = it1.get_input_details(), it1.get_output_details()
    i2, o2 = it2.get_input_details(), it2.get_output_details()
    st1 = np.zeros(i1[1]["shape"], np.float32)
    st2 = np.zeros(i2[1]["shape"], np.float32)
    pad = np.zeros(BLOCK - SHIFT, np.float32)
    audio = np.concatenate([pad, mic, pad]).astype(np.float32)
    lpb = np.concatenate([pad, lpb, pad]).astype(np.float32)
    out = np.zeros(len(audio), np.float32)
    inbuf = np.zeros(BLOCK, np.float32)
    lpbuf = np.zeros(BLOCK, np.float32)
    outbuf = np.zeros(BLOCK, np.float32)
    nblk = (audio.shape[0] - (BLOCK - SHIFT)) // SHIFT
    t0 = time.perf_counter()
    for k in range(nblk):
        inbuf[:-SHIFT] = inbuf[SHIFT:]
        inbuf[-SHIFT:] = audio[k * SHIFT:k * SHIFT + SHIFT]
        lpbuf[:-SHIFT] = lpbuf[SHIFT:]
        lpbuf[-SHIFT:] = lpb[k * SHIFT:k * SHIFT + SHIFT]
        fft = np.fft.rfft(inbuf).astype("complex64")
        mag = np.abs(fft).reshape(1, 1, -1).astype("float32")
        lmag = np.abs(np.fft.rfft(lpbuf)).reshape(1, 1, -1).astype("float32")
        it1.set_tensor(i1[0]["index"], mag)
        it1.set_tensor(i1[2]["index"], lmag)
        it1.set_tensor(i1[1]["index"], st1)
        it1.invoke()
        mask = it1.get_tensor(o1[0]["index"])
        st1 = it1.get_tensor(o1[1]["index"])
        est = np.fft.irfft(fft * mask).reshape(1, 1, -1).astype("float32")
        it2.set_tensor(i2[0]["index"], est)
        it2.set_tensor(i2[2]["index"], lpbuf.reshape(1, 1, -1).astype("float32"))
        it2.set_tensor(i2[1]["index"], st2)
        it2.invoke()
        oblk = it2.get_tensor(o2[0]["index"])
        st2 = it2.get_tensor(o2[1]["index"])
        outbuf[:-SHIFT] = outbuf[SHIFT:]
        outbuf[-SHIFT:] = 0.0
        outbuf += np.squeeze(oblk)
        out[k * SHIFT:k * SHIFT + SHIFT] = outbuf[:SHIFT]
    proc = time.perf_counter() - t0
    return out[BLOCK - SHIFT:BLOCK - SHIFT + len(mic)], proc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="128", choices=["128", "256", "512"])
    ap.add_argument("--models", default="~/.cache/aec-eval/DTLN-aec/pretrained_models")
    ap.add_argument("--testset", default="~/.cache/aec-eval/testset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    md = Path(args.models).expanduser()
    it1 = Interpreter(model_path=str(md / f"dtln_aec_{args.size}_1.tflite")); it1.allocate_tensors()
    it2 = Interpreter(model_path=str(md / f"dtln_aec_{args.size}_2.tflite")); it2.allocate_tensors()
    ts = Path(args.testset).expanduser()
    out = Path(args.out).expanduser()
    proc_s = audio_s = 0.0
    for scen in ("farend_only", "nearend_only", "doubletalk", "delaychange"):
        base = ts / "16000" / scen
        mic = sf.read(str(base / "mic.wav"), dtype="float32", always_2d=True)[0].mean(1)
        lpb = sf.read(str(base / "ref.wav"), dtype="float32", always_2d=True)[0].mean(1)
        cleaned, proc = process(mic, lpb, it1, it2)
        proc_s += proc
        audio_s += len(mic) / 16000
        d = out / "16000" / scen
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / "cleaned.wav"), np.clip(cleaned, -1, 1), 16000, subtype="FLOAT")
    weight_kb = sum(os.path.getsize(md / f"dtln_aec_{args.size}_{s}.tflite") for s in (1, 2)) / 1024
    print(json.dumps({
        "name": f"dtln_{args.size}",
        "rt_factor": round(proc_s / audio_s, 4),
        "latency_ms": {"16000": round(1000 * BLOCK / 16000, 1)},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "weight_kb": round(weight_kb, 1),
    }))


if __name__ == "__main__":
    main()
