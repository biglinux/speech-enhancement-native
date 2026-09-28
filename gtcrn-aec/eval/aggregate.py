#!/usr/bin/env python3
"""Aggregate per-candidate score.json + res.json into a comparison table + RESULTS.md.

  python3 aggregate.py --cand ~/.cache/aec-eval/cand --out RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["passthrough", "webrtc", "speexdsp",
         "dtln_128", "dtln_256", "dtln_512",
         "localvqe_2.7k", "localvqe_49k", "localvqe_200k"]

HDR = ["cand", "FE-ERLE", "DC-ERLE", "NE-PESQ", "NE-STOI", "DT-PESQ", "DT-STOI",
       "RTf", "lat_ms", "wt_KB"]


def row(name: str, cand: Path):
    sp = cand / name / "score.json"
    rp = cand / name / "res.json"
    if not sp.exists():
        return None
    q = json.load(open(sp))["rates"].get("16000", {})
    r = json.load(open(rp)) if rp.exists() else {}
    fe = q.get("farend_only", {})
    dc = q.get("delaychange", {})
    ne = q.get("nearend_only", {})
    dt = q.get("doubletalk", {})
    return [name,
            fe.get("ERLE_dB", "-"), dc.get("ERLE_dB", "-"),
            ne.get("PESQ", "-"), ne.get("STOI", "-"),
            dt.get("PESQ", "-"), dt.get("STOI", "-"),
            r.get("rt_factor", "-"), r.get("latency_ms", "-"), r.get("weight_kb", "-")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="~/.cache/aec-eval/cand")
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()
    cand = Path(args.cand).expanduser()
    rows = [r for r in (row(n, cand) for n in ORDER) if r]
    w = [max(len(str(x)) for x in [HDR[i]] + [r[i] for r in rows]) for i in range(len(HDR))]
    def fmt(vals):
        return " | ".join(str(v).rjust(w[i]) for i, v in enumerate(vals))
    lines = [fmt(HDR), "-+-".join("-" * wi for wi in w)]
    lines += [fmt(r) for r in rows]
    table = "\n".join(lines)
    print(table)
    Path(args.out).write_text(
        "# AEC candidates — quality + resource (16 kHz test set)\n\n"
        "```\n" + table + "\n```\n\n"
        "FE-ERLE/DC-ERLE: echo removed (dB, higher better) on far-end-only / delay-change.\n"
        "NE-*: near-end preservation with no echo (PESQ 1-4.5, STOI 0-1, higher better).\n"
        "DT-*: near-end quality during double-talk. RTf: process/audio (<1 = real-time).\n"
        "lat_ms: algorithmic frame latency. wt_KB: model weight size.\n")


if __name__ == "__main__":
    main()
