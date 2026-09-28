#!/usr/bin/env python3
"""WebRTC AEC runner: drives the shipped libspa-aec-webrtc via the C harness.

  python3 run_webrtc.py --bin <run_webrtc> --plugin <so> --testset <ts> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PLUGIN = "/usr/lib/spa-0.2/aec/libspa-aec-webrtc.so"


def wav_to_f32(path: Path, rate: int) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ar", str(rate),
                          "-ac", "1", "-f", "f32le", "-"], check=True, capture_output=True)
    return np.frombuffer(out.stdout, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="~/.cache/aec-eval/bin/run_webrtc")
    ap.add_argument("--plugin", default=PLUGIN)
    ap.add_argument("--testset", default="~/.cache/aec-eval/testset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rates", default="16000,48000")
    args = ap.parse_args()
    binp = Path(args.bin).expanduser()
    ts = Path(args.testset).expanduser()
    out = Path(args.out).expanduser()
    proc_s = audio_s = 0.0
    lat = {}
    with tempfile.TemporaryDirectory(dir=Path("~/.cache/aec-eval").expanduser()) as tmp:
        tmp = Path(tmp)
        for rate in (int(r) for r in args.rates.split(",")):
            lat[str(rate)] = round(1000 * (rate // 100) / rate, 1)  # 10 ms frame
            for scen in ("farend_only", "nearend_only", "doubletalk", "delaychange"):
                base = ts / f"{rate}" / scen
                for name in ("mic", "ref"):
                    x = wav_to_f32(base / f"{name}.wav", rate)
                    x.tofile(tmp / f"{name}.f32")
                t0 = time.perf_counter()
                subprocess.run([str(binp), args.plugin, str(rate),
                                str(tmp / "mic.f32"), str(tmp / "ref.f32"), str(tmp / "out.f32")],
                               check=True, capture_output=True)
                proc_s += time.perf_counter() - t0
                cleaned = np.fromfile(tmp / "out.f32", np.float32)
                audio_s += len(cleaned) / rate
                d = out / f"{rate}" / scen
                d.mkdir(parents=True, exist_ok=True)
                sf.write(str(d / "cleaned.wav"), np.clip(cleaned, -1, 1), rate, subtype="FLOAT")
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    print(json.dumps({
        "name": "webrtc",
        "rt_factor": round(proc_s / audio_s, 4),
        "latency_ms": lat,
        "peak_rss_mb": round(rss, 1),
        "weight_kb": 0,
    }))


if __name__ == "__main__":
    main()
