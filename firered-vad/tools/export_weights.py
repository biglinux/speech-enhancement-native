#!/usr/bin/env python3
"""Write a FireRedVAD model as the raw f32 blob the crate embeds.

Input is a model directory of the FireRedVAD release (Apache-2.0,
huggingface.co/FireRedTeam/FireRedVAD): `model.pth.tar` and `cmvn.ark`. The
checkpoint is read with `weights_only=True`, allowing only the `argparse.Namespace`
its hyperparameters are stored in. Tensors are written little-endian f32 in the
order below, linear weights transposed to `[in][out]` and FSMN filters to
`[tap][channel]`; `src/lib.rs` reads them back the same way. Needs `torch` and
`kaldiio`.

    python3 export_weights.py FireRedVAD/AED ../model/firered_aed.bin
    python3 export_weights.py FireRedVAD/Stream-VAD ../model/firered_stream_vad.bin
"""

import argparse
import sys

import numpy as np

FEATURES, HIDDEN, PROJECTION, ORDER, BLOCKS = 80, 256, 128, 20, 8


def main(src, dst):
    import kaldiio
    import torch

    torch.serialization.add_safe_globals([argparse.Namespace])
    package = torch.load(f"{src}/model.pth.tar", map_location="cpu", weights_only=True)
    args, state = package["args"], package["model_state_dict"]
    shape = (args.idim, args.R, args.M, args.H, args.P, args.N1, args.S1)
    if shape != (FEATURES, BLOCKS, 1, HIDDEN, PROJECTION, ORDER, 1) or args.N2 not in (0, ORDER):
        sys.exit(f"unsupported FireRedVAD configuration: {vars(args)}")
    if args.N2 and args.S2 != 1:
        sys.exit(f"unsupported lookahead stride {args.S2}")

    stats = kaldiio.load_mat(f"{src}/cmvn.ark")
    count = stats[0, FEATURES]
    mean = stats[0, :FEATURES] / count
    variance = np.maximum(stats[1, :FEATURES] / count - mean * mean, 1e-20)
    tensors = [mean, 1.0 / np.sqrt(variance)]

    def take(name, expected):
        tensor = state[name].numpy()
        if tensor.shape != expected:
            sys.exit(f"{name}: expected {expected}, found {tensor.shape}")
        return tensor

    def linear(name, n_in, n_out, bias=True):
        tensors.append(take(f"{name}.weight", (n_out, n_in)).T)
        if bias:
            tensors.append(take(f"{name}.bias", (n_out,)))

    def fsmn(name):
        for side in ("lookback", "lookahead")[: 2 if args.N2 else 1]:
            tensors.append(take(f"{name}.{side}_filter.weight", (PROJECTION, 1, ORDER))[:, 0, :].T)

    linear("dfsmn.fc1.0", FEATURES, HIDDEN)
    linear("dfsmn.fc2.0", HIDDEN, PROJECTION)
    fsmn("dfsmn.fsmn1")
    for block in range(BLOCKS - 1):
        linear(f"dfsmn.fsmns.{block}.fc1.0", PROJECTION, HIDDEN)
        linear(f"dfsmn.fsmns.{block}.fc2", HIDDEN, PROJECTION, bias=False)
        fsmn(f"dfsmn.fsmns.{block}.fsmn")
    linear("dfsmn.dnns.0", PROJECTION, HIDDEN)
    linear("out", HIDDEN, args.odim)

    out = b"".join(np.ascontiguousarray(t, dtype="<f4").tobytes() for t in tensors)
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out)} bytes, {args.odim} outputs, lookahead {args.N2}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
