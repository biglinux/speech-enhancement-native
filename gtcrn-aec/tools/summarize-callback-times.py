#!/usr/bin/env python3
"""Summarize measured callbacks from CSV. Never generates synthetic measurements.
Required columns: duration_ns,frames,rate,phase. Record times outside DSP state.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def percentile(ordered: list[float], p: float) -> float:
    # Nearest rank; finite nonempty samples only. No interpolated maximum.
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--max-utilization', type=float, default=1.0,
                        help='Allowed fraction of frames/rate deadline, default 1.0')
    args = parser.parse_args()
    if not 0 < args.max_utilization <= 1:
        parser.error('max utilization must be finite and in (0, 1]')
    groups = defaultdict(list)
    try:
        with args.csv_file.open(newline='', encoding='utf-8') as source:
            rows = csv.DictReader(source)
            if not {'duration_ns', 'frames', 'rate', 'phase'}.issubset(rows.fieldnames or []):
                raise ValueError('missing required CSV columns')
            for index, row in enumerate(rows, 2):
                duration, frames, rate = float(row['duration_ns']), int(row['frames']), int(row['rate'])
                phase = row['phase']
                if (not math.isfinite(duration) or duration < 0 or frames <= 0 or rate <= 0
                        or not phase or len(phase) > 80):
                    raise ValueError(f'invalid callback record at line {index}')
                groups[(phase, frames, rate)].append(duration)
        if not groups:
            raise ValueError('empty timing input is not a passing result')
        summary = []
        for (phase, frames, rate), samples in sorted(groups.items()):
            samples.sort()
            deadline = frames * 1e9 / rate
            summary.append({'phase': phase, 'frames': frames, 'rate': rate, 'count': len(samples),
                'deadline_ns': deadline, 'p50_ns': percentile(samples, .5), 'p95_ns': percentile(samples, .95),
                'p99_ns': percentile(samples, .99), 'p999_ns': percentile(samples, .999), 'max_ns': samples[-1],
                'max_utilization': samples[-1] / deadline,
                'budget_exceedances': sum(value > deadline * args.max_utilization for value in samples),
                'deadline_exceedances': sum(value > deadline for value in samples)})
        print(json.dumps({'requested_budget_fraction': args.max_utilization, 'groups': summary,
            'scope': 'These measured callbacks only; not an end-to-end latency or universal RT proof.'}, indent=2))
        return int(any(row['budget_exceedances'] for row in summary))
    except (OSError, UnicodeError, csv.Error, ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
