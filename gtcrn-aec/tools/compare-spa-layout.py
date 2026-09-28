#!/usr/bin/env python3
"""Compare logs produced by the C checker and the Rust layout test; no builds."""
import argparse
import re
import sys
from pathlib import Path

FIELDS = {'handle', 'factory', 'aec', 'methods', 'raw',
          'methods.init', 'methods.run', 'methods.init2'}
LIMIT = 2 * 1024 * 1024


def read_layout(path: Path, marker: str) -> dict[str, int]:
    with path.open('rb') as source:
        raw = source.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError(f'{path}: log exceeds {LIMIT} bytes')
    text = raw.decode('utf-8')
    lines = [line.split(marker, 1)[1].strip() for line in text.splitlines() if marker in line]
    if len(lines) != 1:
        raise ValueError(f'{path}: expected exactly one {marker} record')
    pairs = re.findall(r'([a-z.0-9]+)=(\d+)', lines[0])
    values = {key: int(value) for key, value in pairs}
    if len(pairs) != len(values) or set(values) != FIELDS:
        raise ValueError(f'{path}: missing, duplicate or unexpected layout fields')
    if marker == 'header-layout:' and 'PASS: C header / Rust SPA interface contract' not in text:
        raise ValueError(f'{path}: C checker did not report completing its checks')
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('c_log', type=Path)
    parser.add_argument('rust_log', type=Path)
    args = parser.parse_args()
    try:
        c = read_layout(args.c_log, 'header-layout:')
        rust = read_layout(args.rust_log, 'rust-layout:')
        differences = [f'{key}: C={c[key]}, Rust={rust[key]}' for key in sorted(FIELDS) if c[key] != rust[key]]
        if differences:
            raise ValueError('layout mismatch: ' + '; '.join(differences))
    except (OSError, UnicodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print('Compared layout fields match. This does not certify streaming or soundness.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
