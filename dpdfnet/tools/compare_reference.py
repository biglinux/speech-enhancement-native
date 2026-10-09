#!/usr/bin/env python3
"""Compares the Rust engine layer by layer with Ceva's PyTorch model or the W8A16
oracle, and reports the first layer that diverges.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from export_model import load_model, TRACE_NAMES
from native_api import Native


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lib", type=Path, default=Path("target/release-unwind/libdpdfnet_native.so")
    )
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--quantized-reference", action="store_true")
    p.add_argument("--atol", type=float, default=5e-4)
    p.add_argument("--rtol", type=float, default=5e-4)
    p.add_argument("--report", type=Path, default=Path("comparison.json"))
    a = p.parse_args()
    if a.frames < 8:
        p.error("use at least eight frames to exercise delayed paths")
    try:
        manifest = json.loads((a.bundle / "manifest.json").read_text())
        checkpoint = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    except (OSError, ValueError) as e:
        p.error(str(e))
    if checkpoint != manifest["checkpoint_sha256"]:
        p.error("reference and native checkpoint differ")
    if manifest["quantization"] != "f32" and not a.quantized_reference:
        p.error(
            "use --quantized-reference for W8A16; comparing against FP32 is a separate quality assessment"
        )
    torch.set_num_threads(1)
    model = load_model(a.upstream, a.checkpoint, manifest["depth"])
    if a.quantized_reference:
        from quantized_reference import install

        install(model, manifest["quantization"], set(manifest["float_grus"]))
    traces = {}
    hooks = []

    def hook(index, channels_first=False):
        def receive(_module, _args, output):
            y = output[0] if isinstance(output, tuple) else output
            if channels_first:
                y = y.permute(0, 2, 3, 1)
            traces[index] = (
                y.detach().cpu().numpy().astype(np.float32).reshape(-1).copy()
            )

        return receive

    modules = [
        model.erb_norm,
        model.spec_norm,
        model.enc.erb_conv0,
        model.enc.erb_conv1,
        model.enc.erb_conv2,
        model.enc.erb_conv3,
        model.enc.df_conv0,
        model.enc.df_conv1,
        model.enc.dprnn_erb,
        model.enc.dprnn_df,
        model.enc.emb_gru,
        model.erb_dec,
        model.df_dec,
    ]
    for i, module in enumerate(modules):
        hooks.append(module.register_forward_hook(hook(i, 2 <= i <= 9)))
    state = model.initial_state(dtype=torch.float32)
    rng = np.random.default_rng(621751)
    # Includes real-only FFT endpoints, silence, broadband noise, and nonstationary levels.
    signals = rng.normal(size=(a.frames, 960)).astype(np.float32)
    signals *= np.logspace(-5, -1, a.frames).astype(np.float32)[:, None]
    signals[:4] = 0
    rows = [
        dict(name=name, max_abs=0.0, max_relative_rms=0.0, passed=True)
        for name in TRACE_NAMES
    ]
    first_failure = None
    try:
        native = Native(a.lib, a.bundle)
    except (OSError, RuntimeError) as e:
        p.error(str(e))
    with native, torch.inference_mode():
        for frame, signal in enumerate(signals):
            spectrum = np.fft.rfft(signal).astype(np.complex64)
            packed = np.stack([spectrum.real, spectrum.imag], axis=-1).astype(
                np.float32
            )
            x = torch.from_numpy(packed[None, None]) * np.float32(model.wnorm)
            reference, state = model(x, state)
            traces[13] = (
                (reference / np.float32(model.wnorm)).numpy().reshape(-1).copy()
            )
            native.spectrum(packed)
            for i, row in enumerate(rows):
                got, expected = native.trace(i), traces[i]
                if got.shape != expected.shape:
                    raise ValueError(
                        f"Trace shape mismatch {row['name']}: {got.shape} vs {expected.shape}"
                    )
                err = got.astype(np.float64) - expected
                row["max_abs"] = max(row["max_abs"], float(np.max(np.abs(err))))
                row["max_relative_rms"] = max(
                    row["max_relative_rms"],
                    float(np.linalg.norm(err) / (np.linalg.norm(expected) + 1e-12)),
                )
                ok = bool(
                    np.isfinite(got).all()
                    and np.allclose(got, expected, atol=a.atol, rtol=a.rtol)
                )
                row["passed"] &= ok
                if not ok and first_failure is None:
                    first_failure = dict(frame=frame, trace=i, name=row["name"])
    for h in hooks:
        h.remove()
    report = dict(
        passed=first_failure is None,
        first_failure=first_failure,
        frames=a.frames,
        reference="independent W8A16 oracle"
        if a.quantized_reference
        else "original PyTorch",
        atol=a.atol,
        rtol=a.rtol,
        checkpoint_sha256=manifest["checkpoint_sha256"],
        layers=rows,
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
