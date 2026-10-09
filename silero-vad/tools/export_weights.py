#!/usr/bin/env python3
"""Write the Silero VAD 16 kHz weights as the raw f32 blob the crate embeds.

Input is `silero_vad_16k_op15.onnx` from the silero-vad 6.2.3 wheel (MIT,
github.com/snakers4/silero-vad). Its weights equal the opset-16 ONNX and the
TorchScript model's; the wheel's `silero_vad_16k.safetensors` holds a different
checkpoint and must not be used. Tensors are written in `ORDER`, little-endian f32,
with nothing between them; `src/lib.rs` slices them by the same shapes. Needs `onnx`.

    python3 export_weights.py silero_vad_16k_op15.onnx ../model/silero_vad_16k.bin
"""

import sys

import numpy as np

ORDER = [
    ("model.stft.forward_basis_buffer", (258, 1, 256)),
    ("model.encoder.0.reparam_conv.weight", (128, 129, 3)),
    ("model.encoder.0.reparam_conv.bias", (128,)),
    ("model.encoder.1.reparam_conv.weight", (64, 128, 3)),
    ("model.encoder.1.reparam_conv.bias", (64,)),
    ("model.encoder.2.reparam_conv.weight", (64, 64, 3)),
    ("model.encoder.2.reparam_conv.bias", (64,)),
    ("model.encoder.3.reparam_conv.weight", (128, 64, 3)),
    ("model.encoder.3.reparam_conv.bias", (128,)),
    ("model.decoder.rnn.weight_ih", (512, 128)),
    ("model.decoder.rnn.weight_hh", (512, 128)),
    ("model.decoder.rnn.bias_ih", (512,)),
    ("model.decoder.rnn.bias_hh", (512,)),
    ("model.decoder.decoder.2.weight", (1, 128, 1)),
    ("model.decoder.decoder.2.bias", (1,)),
]


def main(src, dst):
    import onnx
    from onnx import numpy_helper

    tensors = {t.name: numpy_helper.to_array(t) for t in onnx.load(src).graph.initializer}
    out = bytearray()
    for name, shape in ORDER:
        tensor = tensors[name]
        if tensor.dtype != np.float32 or tensor.shape != shape:
            sys.exit(f"{name}: expected float32 {shape}, found {tensor.dtype} {tensor.shape}")
        out += tensor.astype("<f4").tobytes()
    open(dst, "wb").write(out)
    print(f"{dst}: {len(out)} bytes, {len(out) // 4} weights")


if __name__ == "__main__":
    main(*sys.argv[1:3])
