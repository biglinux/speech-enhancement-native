#!/usr/bin/env python3
"""LocalVQE runner (ggml CLI) — DeepVQE-derived neural AEC, 16 kHz.

  python3 run_localvqe.py --bin <localvqe> --model <gguf> --name <label> \
      --testset <ts> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--testset", default="~/.cache/aec-eval/testset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    binp = Path(args.bin).expanduser()
    model = Path(args.model).expanduser()
    ts = Path(args.testset).expanduser()
    out = Path(args.out).expanduser()
    proc_s = audio_s = 0.0
    for scen in ("farend_only", "nearend_only", "doubletalk", "delaychange"):
        base = ts / "16000" / scen
        d = out / "16000" / scen
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        subprocess.run([str(binp), str(model), "--in-wav", str(base / "mic.wav"),
                        str(base / "ref.wav"), "--out-wav", str(d / "cleaned.wav")],
                       check=True, capture_output=True)
        proc_s += time.perf_counter() - t0
        audio_s += sf.info(str(base / "mic.wav")).duration
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(json.dumps({
        "name": args.name,
        "rt_factor": round(proc_s / audio_s, 4),
        "latency_ms": {"16000": 16.0},  # 256-sample streaming hop
        "peak_rss_mb": round(rss, 1),
        "weight_kb": round(os.path.getsize(model) / 1024, 1),
    }))


if __name__ == "__main__":
    main()
