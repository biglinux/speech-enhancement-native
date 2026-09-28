"""Independent, deliberately slow W8A16 oracle for differential testing.
Integer sums are int64 here, to check native int32 kernels without sharing code.
The oracle keeps the original PyTorch r,z,n order (native uses z,r,n).
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from export_model import npf, quantize_rows

def activation_quant(x):
    x = np.asarray(x, dtype=np.float32)
    peak = np.max(np.abs(x))
    scale = np.float32(peak / np.float32(16383)) if peak > 0 else np.float32(1)
    if scale == 0:
        return np.zeros_like(x, dtype=np.int16), scale
    with np.errstate(over="ignore"):
        inverse = np.float32(1) / scale
    v = x * inverse if np.isfinite(inverse) else x / scale
    # Rust f32::round rounds ties away from zero. numpy.rint does not.
    # Do not add 0.5 before floor in FP32: nextafter(0.5, 0) + 0.5
    # rounds to 1.0 and incorrectly quantizes a value below the tie upward.
    magnitude = np.abs(v)
    whole = np.floor(magnitude)
    rounded = whole + ((magnitude - whole) >= np.float32(0.5)).astype(np.float32)
    q = np.copysign(rounded, v)
    return np.clip(q, -16383, 16383).astype(np.int16), scale

class Projection:
    def __init__(self, weight, quantized):
        self.w = npf(weight)
        self.quantized = quantized
        if quantized:
            self.q, self.scale = quantize_rows(self.w)
            self.q64 = self.q.astype(np.int64)
    def __call__(self, x):
        if not self.quantized:
            return self.w @ x
        q, scale = activation_quant(x)
        total = self.q64 @ q.astype(np.int64)
        return total.astype(np.float32) * self.scale * scale

class Cell:
    def __init__(self, module, suffix="", quantized=True):
        self.w = Projection(getattr(module, "weight_ih" + suffix), quantized)
        self.r = Projection(getattr(module, "weight_hh" + suffix), quantized)
        self.bi = npf(getattr(module, "bias_ih" + suffix))
        self.br = npf(getattr(module, "bias_hh" + suffix))
        self.hidden = module.hidden_size
    def __call__(self, x, h):
        n = self.hidden
        wx, rh = self.w(x), self.r(h)
        with np.errstate(over="ignore"):
            r = 1 / (1 + np.exp(-(wx[:n] + rh[:n] + self.bi[:n] + self.br[:n])))
            z = 1 / (1 + np.exp(-(wx[n:2*n] + rh[n:2*n] + self.bi[n:2*n] + self.br[n:2*n])))
            new = np.tanh(wx[2*n:] + self.bi[2*n:] + r * (rh[2*n:] + self.br[2*n:]))
        return ((1 - z) * new + z * h).astype(np.float32)

class QCell(nn.Module):
    def __init__(self, module, quantized):
        super().__init__()
        self.cell = Cell(module, quantized=quantized)
    def forward(self, x, h):
        xn, hn = npf(x), npf(h)
        out = np.stack([self.cell(a, b) for a, b in zip(xn, hn)])
        return torch.from_numpy(out)

class QBiGRU(nn.Module):
    def __init__(self, module, quantized_forward, quantized_backward):
        super().__init__()
        self.f = Cell(module, "_l0", quantized_forward)
        self.b = Cell(module, "_l0_reverse", quantized_backward)
    def forward(self, x, h=None):
        if h is not None:
            raise ValueError("Intra-frequency GRU must reset for every audio frame")
        a = npf(x)
        batch, freq, _ = a.shape
        n = self.f.hidden
        y = np.empty((batch, freq, 2*n), np.float32)
        states = np.empty((2, batch, n), np.float32)
        for b in range(batch):
            hf, hb = np.zeros(n, np.float32), np.zeros(n, np.float32)
            for f in range(freq):
                hf = self.f(a[b, f], hf)
                y[b, f, :n] = hf
            for f in reversed(range(freq)):
                hb = self.b(a[b, f], hb)
                y[b, f, n:] = hb
            states[:, b] = np.stack([hf, hb])
        return torch.from_numpy(y), torch.from_numpy(states)

def install(model, mode, float_grus):
    def quant(name):
        return mode == "w8a16" and name not in float_grus
    for branch_name, branch in [("enc.dprnn_erb", model.enc.dprnn_erb), ("enc.dprnn_df", model.enc.dprnn_df)]:
        for i, b in enumerate(branch.blocks):
            prefix = f"{branch_name}.blocks.{i}"
            b.intra_gru = QBiGRU(b.intra_gru, quant(prefix + ".forward"), quant(prefix + ".backward"))
            b.inter_gru.grucell = QCell(b.inter_gru.grucell, quant(prefix + ".temporal"))
    for name, module in [("enc.emb_gru", model.enc.emb_gru), ("erb_dec.emb_gru", model.erb_dec.emb_gru),
                         ("df_dec.df_gru", model.df_dec.df_gru)]:
        for i, g in enumerate(module.gru):
            g.grucell = QCell(g.grucell, quant(f"{name}.gru.{i}"))
