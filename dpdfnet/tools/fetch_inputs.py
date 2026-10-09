#!/usr/bin/env python3
"""Downloads the pinned upstream sources and checkpoint."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

# Constants only: importing export_model would unnecessarily require torch during download.
COMMIT = "9bd9844a227bb6aa57e55588d8d0e961fcff1c46"
BLOBS = {
    "onnx_model/dpdfnet_48khz_hr.py": "138b99a0d8d17f23302accc60ae8aa7ccceeb380",
    "onnx_model/layers.py": "509d9393cf44f7ea2034cd2f824dfc34af1a380f",
    "onnx_model/multiframe.py": "a9a2785934269f291a382aab084de75bd0332b9f",
    "onnx_model/init_norms.py": "66995131e000a27f20196e7a2e63d2120ef18da0",
    "onnx_model/utils.py": "b792ead64cec6e5c12fe9fac4327de9f15e87834",
    "model/utils.py": "01d3b8b825f41f52c7851b0538131ad341ce6a5e",
    "LICENSE": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
}


def get(url: str, limit: int = 64 * 1024 * 1024) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "BigLinux-DPDFNet-native-preparation/1"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Download exceeds configured size limit")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("inputs"))
    p.add_argument("--depth", choices=(2, 8), type=int, default=2)
    p.add_argument(
        "--hf-revision", help="40-character commit from Ceva-IP/DPDFNet on Hugging Face"
    )
    p.add_argument("--checkpoint-sha256", help="Expected checkpoint checksum")
    p.add_argument("--source-only", action="store_true")
    a = p.parse_args()
    if a.hf_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", a.hf_revision):
        p.error("--hf-revision must be a full 40-character commit SHA")
    try:
        fetch(a)
    except (OSError, ValueError) as e:
        p.error(str(e))


def fetch(a: argparse.Namespace) -> None:
    upstream = a.directory / "upstream"
    for file, expected in BLOBS.items():
        data = get(f"https://raw.githubusercontent.com/ceva-ip/DPDFNet/{COMMIT}/{file}")
        actual = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream checksum mismatch: {file}")
        target = upstream / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if a.source_only:
        return
    revision = a.hf_revision
    if revision is None:
        meta = json.loads(
            get("https://huggingface.co/api/models/Ceva-IP/DPDFNet", 2 * 1024 * 1024)
        )
        revision = meta["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hugging Face revision must be a full commit SHA")
    filename = f"dpdfnet{a.depth}_48khz_hr.pth"
    data = get(
        f"https://huggingface.co/Ceva-IP/DPDFNet/resolve/{revision}/checkpoints/{filename}"
    )
    digest = hashlib.sha256(data).hexdigest()
    if a.checkpoint_sha256 and a.checkpoint_sha256.lower() != digest:
        raise ValueError("Checkpoint checksum mismatch")
    target = a.directory / "checkpoints" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    provenance = dict(
        source_commit=COMMIT,
        huggingface_revision=revision,
        checkpoint=filename,
        checkpoint_sha256=digest,
        independent_checksum_supplied=bool(a.checkpoint_sha256),
    )
    (target.parent / (filename + ".provenance.json")).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
