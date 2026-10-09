#!/usr/bin/env python3
"""Offline-only conversion of the exact Ceva streaming model to native weights.

No ONNX, OpenVINO or torch dependency is used by the resulting Rust library.
Unknown layers, checkpoint mismatches, unsafe pickle fallback and silent reshape
fixups are deliberately rejected. GRU gates are explicitly converted rzn -> zrn.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

UPSTREAM_COMMIT = "9bd9844a227bb6aa57e55588d8d0e961fcff1c46"
SOURCE_BLOBS = {
    "onnx_model/dpdfnet_48khz_hr.py": "138b99a0d8d17f23302accc60ae8aa7ccceeb380",
    "onnx_model/layers.py": "509d9393cf44f7ea2034cd2f824dfc34af1a380f",
    "onnx_model/multiframe.py": "a9a2785934269f291a382aab084de75bd0332b9f",
    "onnx_model/init_norms.py": "66995131e000a27f20196e7a2e63d2120ef18da0",
    "onnx_model/utils.py": "b792ead64cec6e5c12fe9fac4327de9f15e87834",
    "model/utils.py": "01d3b8b825f41f52c7851b0538131ad341ce6a5e",
}
TRACE_NAMES = ["magnitude_features", "complex_features", "erb0", "erb1", "erb2", "erb3",
               "df0", "df1", "erb_dual", "df_dual", "embedding", "mask", "coefficients", "spectrum"]

def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def verify_sources(upstream: Path) -> None:
    for filename, expected in SOURCE_BLOBS.items():
        data = (upstream / filename).read_bytes()
        check(git_blob(data) == expected, f"Upstream file differs from audited commit: {filename}")

def load_model(upstream: Path, checkpoint: Path, depth: int):
    verify_sources(upstream)
    sys.path.insert(0, str(upstream.resolve()))
    module = importlib.import_module("onnx_model.dpdfnet_48khz_hr")
    check(Path(module.__file__).resolve() == (upstream / "onnx_model/dpdfnet_48khz_hr.py").resolve(),
          "Python imported a different upstream model")
    model = module.DPDFNet48HR(
        conv_kernel_inp=(3, 3), conv_ch=64, enc_gru_dim=256,
        erb_dec_gru_dim=256, df_dec_gru_dim=256, enc_lin_groups=32, lin_groups=16,
        upsample_conv_type="subpixel", group_linear_type="loop", point_wise_type="cnn",
        separable_first_conv=True, dprnn_num_blocks=depth,
    )
    # Never retry weights_only=False. Pickle can execute arbitrary code.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    check(isinstance(state, dict) and all(isinstance(x, torch.Tensor) for x in state.values()),
          "Checkpoint must contain only a tensor state_dict")
    model.load_state_dict(module.correct_state_dict(state), strict=True)
    model.eval()
    return model

def npf(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    a = np.asarray(t, dtype="<f4")
    check(np.isfinite(a).all(), "Non-finite weights")
    return np.ascontiguousarray(a)

def gate_zrn(a: np.ndarray) -> np.ndarray:
    check(a.shape[0] % 3 == 0, "Invalid GRU gate axis")
    r, z, n = np.split(a, 3, axis=0)
    return np.concatenate([z, r, n], axis=0)

def quantize_rows(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = npf(a)
    scales = np.max(np.abs(a), axis=1) / np.float32(127)
    scales = np.where(scales > 0, scales, np.float32(1)).astype("<f4")
    q = np.clip(np.rint(a / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales

class Writer:
    def __init__(self):
        self.data = bytearray()
        self.tensors: list[dict[str, Any]] = []
    def tensor(self, a, dtype="f32") -> dict:
        a = npf(a) if dtype == "f32" else np.asarray(a, dtype=np.int8)
        while len(self.data) % 64:
            self.data.append(0)
        ref = {"offset": len(self.data), "len": int(a.size), "dtype": dtype}
        self.data.extend(np.ascontiguousarray(a).tobytes())
        self.tensors.append({**ref, "shape": list(a.shape)})
        return ref

class Exporter:
    def __init__(self, mode: str = "w8a16", float_grus: set[str] | None = None):
        self.writer = Writer()
        self.mode = mode
        self.float_grus = float_grus or set()
        self.seen_float_grus: set[str] = set()
        self.covered: set[int] = set()
        self.fused_depthwise_pointwise_pairs = 0
    def touch(self, module):
        self.covered.update(id(p) for p in module.parameters())
    def matrix(self, a: np.ndarray, mode: str) -> dict:
        a = npf(a)
        check(a.ndim == 2 and 0 < a.shape[0] <= 3072 and 0 < a.shape[1] <= 1024, "Invalid recurrent matrix")
        result = {"kind": mode, "rows": a.shape[0], "cols": a.shape[1]}
        if mode == "w8a16":
            q, scales = quantize_rows(a)
            result.update(weight=self.writer.tensor(q, "i8"), scales=self.writer.tensor(scales))
        elif mode == "f32":
            result["weight"] = self.writer.tensor(a)
        else:
            raise ValueError(f"Unknown matrix mode {mode}")
        return result
    def gru(self, module, name: str, reverse: bool = False) -> dict:
        self.touch(module)
        if isinstance(module, nn.GRU):
            check(module.num_layers == 1, f"Expected one-layer GRU at {name}")
            suffix = "_l0_reverse" if reverse else "_l0"
            check(not reverse or module.bidirectional, "Reverse weights requested from unidirectional GRU")
        elif isinstance(module, nn.GRUCell):
            check(not reverse, "GRUCell cannot be bidirectional")
            suffix = ""
        else:
            raise ValueError(f"Unsupported GRU class at {name}: {type(module)}")
        h = module.hidden_size
        check(h in (64, 256), "Only audited widths 64 and 256 supported")
        w = gate_zrn(npf(getattr(module, "weight_ih" + suffix)))
        r = gate_zrn(npf(getattr(module, "weight_hh" + suffix)))
        bi = gate_zrn(npf(getattr(module, "bias_ih" + suffix)))
        br = gate_zrn(npf(getattr(module, "bias_hh" + suffix)))
        mode = self.mode
        if name in self.float_grus:
            self.seen_float_grus.add(name)
            mode = "f32"
        return dict(hidden=h, gate_order="zrn", reset="after", input=self.matrix(w, mode),
                    recurrent=self.matrix(r, mode), bias=self.writer.tensor(np.concatenate([bi, br])))
    @staticmethod
    def activation(module) -> str:
        return {nn.Identity: "none", nn.ReLU: "relu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}.get(type(module), "unsupported")
    def linear(self, module) -> dict:
        act = "none"
        if isinstance(module, nn.Sequential):
            check(len(module) == 2, "Expected projection followed by activation")
            act = self.activation(module[1])
            module = module[0]
        check(act != "unsupported", "Unsupported linear activation")
        name = type(module).__name__
        self.touch(module)
        if isinstance(module, nn.Linear):
            groups = 1
            w = npf(module.weight)[None, :, :].transpose(0, 2, 1)
            bias = npf(module.bias) if module.bias is not None else np.zeros(module.out_features, np.float32)
        elif name == "GroupedLinear":
            check(not module.shuffle, "GroupedLinear shuffle is unsupported")
            groups = module.groups
            w = np.stack([npf(l.weight).T for l in module.layers])
            bias = np.concatenate([npf(l.bias) if l.bias is not None else np.zeros(l.out_features, np.float32)
                                   for l in module.layers])
        elif name == "GroupedLinearEinsum":
            groups = module.groups
            w = npf(module.weight)
            bias = npf(module.bias)
        else:
            raise ValueError(f"Unsupported projection {name}")
        g, ip, op = w.shape
        check(g == groups, "Grouped projection shape error")
        return dict(input=g * ip, output=g * op, groups=g, activation=act,
                    weight=self.writer.tensor(w), bias=self.writer.tensor(bias))
    def squeezed(self, module, name: str) -> dict:
        check(module.gru_skip is None, "Unexpected squeezed residual")
        check(isinstance(module.linear_out, nn.Identity) or isinstance(module.linear_out, nn.Sequential),
              "Unsupported squeezed output")
        grus = [self.gru(g.grucell, f"{name}.gru.{i}") for i, g in enumerate(module.gru)]
        check(1 <= len(grus) <= 2, "Unsupported squeezed depth")
        return dict(linear_in=self.linear(module.linear_in), grus=grus,
                    linear_out=None if isinstance(module.linear_out, nn.Identity) else self.linear(module.linear_out))
    def layer_norm(self, module: nn.LayerNorm) -> dict:
        check(isinstance(module, nn.LayerNorm) and tuple(module.normalized_shape) == (64,), "LayerNorm must normalize channels only")
        self.touch(module)
        return dict(eps=module.eps, gamma=self.writer.tensor(npf(module.weight)), beta=self.writer.tensor(npf(module.bias)))
    def dual(self, module, name: str) -> dict:
        check(isinstance(module.input_proj, nn.Identity) and isinstance(module.output_proj, nn.Identity), "Unexpected DPRNN projection")
        blocks = []
        for i, block in enumerate(module.blocks):
            prefix = f"{name}.blocks.{i}"
            blocks.append(dict(
                forward=self.gru(block.intra_gru, prefix + ".forward"),
                backward=self.gru(block.intra_gru, prefix + ".backward", reverse=True),
                temporal=self.gru(block.inter_gru.grucell, prefix + ".temporal"),
                fc_intra=self.linear(block.fc_intra), fc_inter=self.linear(block.fc_inter),
                ln_intra=self.layer_norm(block.ln_intra), ln_inter=self.layer_norm(block.ln_inter),
            ))
        check(len(blocks) in (2, 8), "Unsupported DPRNN depth")
        return dict(frequencies=module.blocks[0].num_feat, channels=module.blocks[0].hidden_dim, blocks=blocks)
    def raw_conv(self, module) -> dict:
        self.touch(module)
        if type(module).__name__ == "GroupedConv2D":
            convs = list(module.convs)
            first = convs[0]
            groups = len(convs)
            check(all(c.groups == 1 and c.kernel_size == first.kernel_size and c.stride == first.stride and
                      c.padding == first.padding and c.dilation == first.dilation for c in convs), "Grouped conv mismatch")
            w = np.concatenate([npf(c.weight) for c in convs], axis=0)
            bias = np.concatenate([npf(c.bias) if c.bias is not None else np.zeros(c.out_channels, np.float32) for c in convs])
            ci = first.in_channels * groups
        elif isinstance(module, nn.Conv2d):
            first = module
            groups = module.groups
            w = npf(module.weight)
            bias = npf(module.bias) if module.bias is not None else np.zeros(module.out_channels, np.float32)
            ci = module.in_channels
        else:
            raise ValueError(f"Unsupported convolution {type(module)}")
        check(first.dilation == (1, 1) and first.stride[0] == 1 and first.padding[0] == 0,
              "Only audited convolution geometry is supported")
        co, _, kt, kf = w.shape
        return dict(kind="conv", ci=ci, co=co, kt=kt, kf=kf, stride=first.stride[1], pad=first.padding[1],
                    groups=groups, activation="none", _w=w.copy(), _b=bias.copy())
    def pipeline(self, module, shape: tuple[int, int, int]) -> dict:
        # shape: time, frequency, channel. Feature histories are handled by native rings.
        check(isinstance(module, nn.Sequential), "Convolution pipeline must be Sequential")
        ops = []
        for layer in module:
            cls = type(layer).__name__
            if isinstance(layer, nn.Identity):
                continue
            if isinstance(layer, nn.Conv2d) or cls == "GroupedConv2D":
                ops.append(self.raw_conv(layer))
            elif cls == "SubPixelConv2D":
                branches = [self.raw_conv(c) for c in layer.convs]
                check(len(branches) == layer.fstride and len(branches) in (2, 3), "Subpixel phase mismatch")
                ops.append(dict(kind="subpixel", branches=branches))
            elif isinstance(layer, nn.BatchNorm2d):
                self.touch(layer)
                check(not layer.training and layer.running_mean is not None and layer.running_var is not None,
                      "BatchNorm folding requires eval and running statistics")
                check(bool(ops), "BatchNorm with no preceding conv")
                targets = ops[-1].get("branches", [ops[-1]])
                for conv in targets:
                    check(conv["activation"] == "none", "Cannot fold BN through an activation")
                    gamma = npf(layer.weight) if layer.affine else np.ones(conv["co"], np.float32)
                    beta = npf(layer.bias) if layer.affine else np.zeros(conv["co"], np.float32)
                    scale = gamma / np.sqrt(npf(layer.running_var) + np.float32(layer.eps))
                    conv["_w"] *= scale[:, None, None, None]
                    conv["_b"] = (conv["_b"] - npf(layer.running_mean)) * scale + beta
            elif self.activation(layer) != "unsupported":
                check(bool(ops), "Activation without convolution")
                for conv in ops[-1].get("branches", [ops[-1]]):
                    check(conv["activation"] == "none", "Multiple activations not supported")
                    conv["activation"] = self.activation(layer)
            else:
                raise ValueError(f"Unsupported layer in convolution pipeline: {cls}")
        check(bool(ops), "Empty pipeline")
        # A depthwise 1x1 with no activation followed by a dense 1x1 is one affine
        # map. Fold it offline: W'[o,i]=Wpoint[o,i]*Wdepth[i]; b'=Wpoint*bdepth+bpoint.
        # Only fuse an actually present pair. The pinned upstream suppresses the
        # pointwise stage in its 1x1 skip convolutions; never invent that stage.
        # Record actual fusion count in the manifest (zero is a valid outcome).
        # Never fuse spatial/temporal depthwise kernels: that would densify expensive work.
        fused = []
        i = 0
        while i < len(ops):
            first = ops[i]
            second = ops[i + 1] if i + 1 < len(ops) else None
            can_fuse = (first["kind"] == "conv" and second is not None and second["kind"] == "conv"
                        and first["ci"] == first["co"] == first["groups"]
                        and first["kt"] == first["kf"] == first["stride"] == 1 and first["pad"] == 0
                        and first["activation"] == "none" and second["ci"] == first["co"]
                        and second["kt"] == second["kf"] == second["stride"] == second["groups"] == 1
                        and second["pad"] == 0)
            if can_fuse:
                point = second["_w"][:, :, 0, 0].copy()
                second["_b"] = (point @ first["_b"]) + second["_b"]
                second["_w"] = (point * first["_w"][:, 0, 0, 0][None, :])[:, :, None, None]
                fused.append(second)
                self.fused_depthwise_pointwise_pairs += 1
                i += 2
            else:
                fused.append(first)
                i += 1
        ops = fused
        t, f, c = shape
        for op in ops:
            targets = op.get("branches", [op])
            result_f = []
            for conv in targets:
                check(conv["ci"] == c and conv["kt"] == t, "Convolution shape mismatch")
                g, co, kt, kf = conv["groups"], conv["co"], conv["kt"], conv["kf"]
                ip, opg = c // g, co // g
                # PyTorch [group*out,input,kt,kf] -> [kt,kf,group,input,out].
                w = conv.pop("_w").reshape(g, opg, ip, kt, kf).transpose(3, 4, 0, 2, 1)
                conv["weight"] = self.writer.tensor(w)
                conv["bias"] = self.writer.tensor(conv.pop("_b"))
                result_f.append((f + 2 * conv["pad"] - kf) // conv["stride"] + 1)
            check(len(set(result_f)) == 1, "Subpixel branches disagree")
            f, c, t = result_f[0] * len(targets), targets[0]["co"], 1
        return dict(in_t=shape[0], in_f=shape[1], in_c=shape[2], out_f=f, out_c=c, ops=ops)
    def model(self, m, depth: int) -> dict:
        check(m.mask_method == "before_df" and m.df_lookahead == 2 and m.df_order == 5 and m.nb_df == 96 and m.freq_bins == 481,
              "Unexpected model geometry or mask method")
        check(not m.erb_norm.dynamic_var and m.erb_norm.alpha == m.spec_norm.alpha and m.erb_norm.eps == m.spec_norm.eps,
              "Unexpected normalization mode")
        check(m.stft.n_fft == 960 and m.stft.hop == 480 and m.stft.win_len == 960, "Wrong STFT geometry")
        check(isinstance(m.enc.combine, nn.Module) and type(m.enc.combine).__name__ == "Concat", "Encoder must concatenate embeddings")
        check(np.all(npf(m.erb_norm.var0) == 1600), "Normalization variance is not 40^2")
        # Verify the buffering contracts rather than inferring them from the model name.
        check(m.mask.spec_buffer.delay_frames == 2 and m.df_op.coefs_buffer.delay_frames == 2,
              "Unexpected mask/coefficient delay")
        check(m.df_op.spec_buffer.time_steps == 5 and m.df_op.spec_buffer.delay_frames == 0, "Unexpected deep-filter history")
        e, d, df = m.enc, m.erb_dec, m.df_dec
        encoder = dict(
            erb0=self.pipeline(e.erb_conv0, (3, 480, 1)), erb1=self.pipeline(e.erb_conv1, (1, 480, 64)),
            erb2=self.pipeline(e.erb_conv2, (1, 160, 64)), erb3=self.pipeline(e.erb_conv3, (1, 80, 64)),
            df0=self.pipeline(e.df_conv0, (3, 96, 2)), df1=self.pipeline(e.df_conv1, (1, 96, 64)),
            erb_dual=self.dual(e.dprnn_erb, "enc.dprnn_erb"), df_dual=self.dual(e.dprnn_df, "enc.dprnn_df"),
            erb_fc=self.linear(e.erb_fc_emb), df_fc=self.linear(e.df_fc_emb), gru=self.squeezed(e.emb_gru, "enc.emb_gru"),
        )
        mask_decoder = dict(gru=self.squeezed(d.emb_gru, "erb_dec.emb_gru"), fc=self.linear(d.erb_fc_emb))
        for name, layer, shape in [
            ("skip3", d.conv3p, (1, 40, 64)), ("up3", d.convt3, (1, 40, 64)),
            ("skip2", d.conv2p, (1, 80, 64)), ("up2", d.convt2, (1, 80, 64)),
            ("skip1", d.conv1p, (1, 160, 64)), ("up1", d.convt1, (1, 160, 64)),
            ("skip0", d.conv0p, (1, 480, 64)), ("out", d.conv0_out, (1, 480, 64)),
        ]:
            mask_decoder[name] = self.pipeline(layer, shape)
        df_decoder = dict(gru=self.squeezed(df.df_gru, "df_dec.df_gru"), skip=self.linear(df.df_skip),
                          out=self.linear(df.df_out), path=self.pipeline(df.df_convp, (5, 96, 64)))
        # The lsnr head is computed but unused by this upstream forward. No threshold/gate is added.
        intentionally_pruned = {id(p) for p in e.lsnr_fc.parameters()}
        uncovered = [name for name, p in m.named_parameters() if id(p) not in self.covered | intentionally_pruned]
        check(not uncovered, f"Unexported parameters (do not ignore): {uncovered}")
        check(self.float_grus == self.seen_float_grus, f"Unknown --float-gru names: {self.float_grus - self.seen_float_grus}")
        return dict(schema=1, architecture="dpdfnet-48hr-v1", depth=depth, sample_rate=48000, fft=960, hop=480,
                    df_bins=96, mask_method="before_df", lookahead=2, wnorm=float(m.wnorm),
                    window=self.writer.tensor(npf(m.stft.w)),
                    norm=dict(alpha=m.erb_norm.alpha, one_minus_alpha=1.0-m.erb_norm.alpha, eps=m.erb_norm.eps, std=40.0,
                              mu0=self.writer.tensor(npf(m.erb_norm.initial_state())),
                              s0=self.writer.tensor(npf(m.spec_norm.initial_state()))),
                    encoder=encoder, mask_decoder=mask_decoder, df_decoder=df_decoder,
                    upstream_commit=UPSTREAM_COMMIT, quantization=self.mode, float_grus=sorted(self.float_grus),
                    intentionally_pruned=["enc.lsnr_fc (unused by upstream output)"],
                    optimizations=dict(batch_norm_folded=True,
                        depthwise_pointwise_1x1_pairs=self.fused_depthwise_pointwise_pairs,
                        native_layout="time,frequency,channel"),
                    trace_names=TRACE_NAMES)

def write_bundle(exporter: Exporter, manifest: dict, output: Path, checkpoint: Path | None = None):
    output.mkdir(parents=True, exist_ok=True)
    data = bytes(exporter.writer.data)
    manifest["weights_sha256"] = hashlib.sha256(data).hexdigest()
    if checkpoint is not None:
        manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest["checkpoint_filename"] = checkpoint.name
    manifest["weight_bytes"] = len(data)
    (output / "weights.bin").write_bytes(data)
    # Manifest is written last; readers reject mismatched pair by SHA-256 during updates.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "tensor_inventory.json").write_text(json.dumps(exporter.writer.tensors, indent=2) + "\n")

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--depth", type=int, choices=(2, 8), default=2)
    p.add_argument("--mode", choices=("f32", "w8a16"), default="w8a16")
    p.add_argument("--float-gru", action="append", default=[], help="Exact canonical GRU name to retain in f32")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    try:
        model = load_model(args.upstream, args.checkpoint, args.depth)
        exporter = Exporter(args.mode, set(args.float_gru))
        with torch.inference_mode():
            manifest = exporter.model(model, args.depth)
        write_bundle(exporter, manifest, args.output, args.checkpoint)
    except (OSError, ValueError) as e:
        p.error(str(e))
    print(json.dumps({"output": str(args.output), "weight_bytes": manifest["weight_bytes"],
                      "sha256": manifest["weights_sha256"], "mode": args.mode}, indent=2))

if __name__ == "__main__":
    main()
