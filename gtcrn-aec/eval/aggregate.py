#!/usr/bin/env python3
"""Rebuild the table in RESULTS.md from per-candidate score.json + res.json.

Only the text between the aggregate markers is replaced; the prose around it
is kept.

  python3 aggregate.py --cand <dir> --out RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = [
    "passthrough",
    "webrtc",
    "speexdsp",
    "dtln_128",
    "dtln_256",
    "dtln_512",
    "localvqe_2.7k",
    "localvqe_49k",
    "localvqe_200k",
]

HDR = [
    "cand",
    "FE-ERLE",
    "DC-ERLE",
    "NE-PESQ",
    "NE-STOI",
    "DT-PESQ",
    "DT-STOI",
    "RTf",
    "lat_ms",
    "wt_KB",
]

START = "<!-- aggregate:start -->"
END = "<!-- aggregate:end -->"


def row(name: str, cand: Path):
    sp = cand / name / "score.json"
    rp = cand / name / "res.json"
    if not sp.exists():
        return None
    q = json.loads(sp.read_text())["rates"].get("16000", {})
    r = json.loads(rp.read_text()) if rp.exists() else {}
    lat = r.get("latency_ms", {})
    if isinstance(lat, dict):
        lat = lat.get("16000", "-")
    fe = q.get("farend_only", {})
    dc = q.get("delaychange", {})
    ne = q.get("nearend_only", {})
    dt = q.get("doubletalk", {})
    return [
        name,
        fe.get("ERLE_dB", "-"),
        dc.get("ERLE_dB", "-"),
        ne.get("PESQ", "-"),
        ne.get("STOI", "-"),
        dt.get("PESQ", "-"),
        dt.get("STOI", "-"),
        r.get("rt_factor", "-"),
        lat,
        r.get("weight_kb", "-"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", default="RESULTS.md")
    args = ap.parse_args()
    cand = Path(args.cand).expanduser()
    rows = [r for r in (row(n, cand) for n in ORDER) if r]
    w = [
        max(len(str(x)) for x in [HDR[i]] + [r[i] for r in rows])
        for i in range(len(HDR))
    ]

    def fmt(vals):
        return " | ".join(str(v).rjust(w[i]) for i, v in enumerate(vals))

    lines = [fmt(HDR), "-+-".join("-" * wi for wi in w)]
    lines += [fmt(r) for r in rows]
    table = "\n".join(lines)
    print(table)
    out = Path(args.out)
    text = out.read_text()
    head, sep, rest = text.partition(START)
    _, sep2, tail = rest.partition(END)
    if not sep or not sep2:
        raise SystemExit(f"{out}: missing {START} / {END} markers")
    out.write_text(f"{head}{START}\n```\n{table}\n```\n{END}{tail}")


if __name__ == "__main__":
    main()
