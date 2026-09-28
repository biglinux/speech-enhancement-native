"""Executed without Rust: tests exporter layouts and DSP equations against PyTorch.
These tests DO NOT certify that the Rust source compiles or matches end to end.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
import pytest
import torch
from torch import nn
from export_model import Exporter, quantize_rows
from quantized_reference import Cell, activation_quant

torch.set_num_threads(1)

def read(ex, ref):
    dtype = "<f4" if ref["dtype"] == "f32" else "i1"
    return np.frombuffer(ex.writer.data, dtype=dtype, count=ref["len"], offset=ref["offset"])

def pipeline_numpy(ex, config, x):
    # Deliberately express the native storage indexing directly, rather than invert a PyTorch conv.
    for operation in config["ops"]:
        results = []
        for op in operation.get("branches", [operation]):
            t, f, ci = x.shape
            co, groups, kt, kf = (op[k] for k in ("co", "groups", "kt", "kf"))
            ip, outpg = ci // groups, co // groups
            w = read(ex, op["weight"]).reshape(kt, kf, groups, ip, outpg)
            of = (f + 2 * op["pad"] - kf) // op["stride"] + 1
            y = np.broadcast_to(read(ex, op["bias"]), (of, co)).copy()
            for fo in range(of):
                for ti in range(kt):
                    for k in range(kf):
                        fi = fo * op["stride"] + k - op["pad"]
                        if not 0 <= fi < f:
                            continue
                        for g in range(groups):
                            y[fo, g*outpg:(g+1)*outpg] += x[ti, fi, g*ip:(g+1)*ip] @ w[ti, k, g]
            if op["activation"] == "relu":
                y = np.maximum(y, 0)
            elif op["activation"] == "sigmoid":
                y = 1 / (1 + np.exp(-y))
            results.append(y)
        # [F, phase, channel], NOT [phase, F, channel].
        x = np.stack(results, axis=1).reshape(1, -1, co)
    return x

class GroupedConv2D(nn.Module):
    def __init__(self, ci, co, kt, kf, groups, stride=1):
        super().__init__()
        self.groups = groups
        self.convs = nn.ModuleList([nn.Conv2d(ci//groups, co//groups, (kt,kf), stride=(1,stride), padding=(0,kf//2), bias=False)
                                   for _ in range(groups)])
    def forward(self,x):
        return torch.cat([c(a) for c,a in zip(self.convs,x.chunk(self.groups,dim=1))],dim=1)

class SubPixelConv2D(nn.Module):
    def __init__(self, ci, factor):
        super().__init__()
        self.fstride = factor
        self.convs = nn.ModuleList([nn.Conv2d(ci,ci,(1,3),padding=(0,1),groups=ci,bias=False) for _ in range(factor)])
    def forward(self,x):
        b,c,t,f=x.shape
        return torch.stack([conv(x) for conv in self.convs],dim=-1).reshape(b,c,t,f*self.fstride)

@pytest.mark.parametrize("ci,co,groups,kt,stride",[(1,64,1,3,1),(2,64,2,3,1),(64,64,64,1,3),(64,10,2,5,1),(64,1,1,1,1)])
def test_convolution_and_bn_layout(ci,co,groups,kt,stride):
    torch.manual_seed(25)
    conv=GroupedConv2D(ci,co,kt,3,groups,stride) if groups>1 and ci!=co else nn.Conv2d(ci,co,(kt,3),padding=(0,1),stride=(1,stride),groups=groups,bias=False)
    bn=nn.BatchNorm2d(co)
    with torch.no_grad():
        bn.running_mean.normal_();bn.running_var.uniform_(0.5,2);bn.weight.normal_();bn.bias.normal_()
    module=nn.Sequential(conv,bn,nn.ReLU()).eval()
    x=torch.randn(1,ci,kt,17)
    ex=Exporter("f32");spec=ex.pipeline(module,(kt,17,ci))
    actual=pipeline_numpy(ex,spec,x.permute(0,2,3,1).numpy()[0])
    expected=module(x).detach().permute(0,2,3,1).numpy()[0]
    np.testing.assert_allclose(actual,expected,atol=5e-6,rtol=2e-5)

@pytest.mark.parametrize("factor",[2,3])
def test_subpixel_phase_and_pointwise_bn(factor):
    torch.manual_seed(83)
    module=nn.Sequential(SubPixelConv2D(8,factor),nn.Conv2d(8,8,1,bias=False),nn.BatchNorm2d(8),nn.ReLU()).eval()
    x=torch.randn(1,8,1,13)
    ex=Exporter();spec=ex.pipeline(module,(1,13,8))
    actual=pipeline_numpy(ex,spec,x.permute(0,2,3,1).numpy()[0])
    expected=module(x).detach().permute(0,2,3,1).numpy()[0]
    np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-5)

@pytest.mark.parametrize("width",[64,256])
def test_exported_gru_gate_order_matches_torch(width):
    torch.manual_seed(123)
    cell=nn.GRUCell(width,width).eval()
    x=torch.randn(width);h=torch.randn(width)
    ex=Exporter("f32");g=ex.gru(cell,"test")
    wi=read(ex,g["input"]["weight"]).reshape(3*width,width)
    wr=read(ex,g["recurrent"]["weight"]).reshape(3*width,width)
    bias=read(ex,g["bias"])
    wx=wi@x.numpy();rh=wr@h.numpy();n=width
    z=1/(1+np.exp(-(wx[:n]+rh[:n]+bias[:n]+bias[3*n:4*n])))
    r=1/(1+np.exp(-(wx[n:2*n]+rh[n:2*n]+bias[n:2*n]+bias[4*n:5*n])))
    new=np.tanh(wx[2*n:]+bias[2*n:3*n]+r*(rh[2*n:]+bias[5*n:]))
    y=(1-z)*new+z*h.numpy()
    np.testing.assert_allclose(y,cell(x,h).detach().numpy(),atol=6e-7,rtol=3e-5)
    np.testing.assert_allclose(Cell(cell,quantized=False)(x.numpy(),h.numpy()),y,atol=6e-7,rtol=3e-5)

@pytest.mark.parametrize("width",[64,256,1024])
def test_integer_accumulation_bound(width):
    w=np.full((3,width),127,dtype=np.int8)
    x=np.full(width,16383,dtype=np.int16)
    total=w.astype(np.int64)@x.astype(np.int64)
    assert np.max(total)<2**31
    np.testing.assert_array_equal(total,np.full(3,127*16383*width))

def test_activation_rounding_ties_and_silence():
    x=np.array([16383.,0.5,-0.5,1.5,-1.5],np.float32)
    q,s=activation_quant(x)
    assert s==1
    np.testing.assert_array_equal(q,[16383,1,-1,2,-2])
    q,s=activation_quant(np.zeros(64,np.float32))
    assert s==1 and not q.any()

def test_quantization_row_error_bound():
    rng=np.random.default_rng(49)
    w=rng.normal(size=(192,64)).astype(np.float32)
    q,s=quantize_rows(w)
    assert q.min()>=-127 and q.max()<=127
    assert np.all(np.abs(q.astype(np.float32)*s[:,None]-w)<=s[:,None]*0.5001)

def test_layer_norm_biased_variance():
    torch.manual_seed(231)
    ln=nn.LayerNorm(64).eval()
    x=torch.randn(19,64)
    a=x.numpy();mu=a.mean(-1,keepdims=True);var=((a-mu)**2).mean(-1,keepdims=True)
    y=(a-mu)/np.sqrt(var+ln.eps)
    np.testing.assert_allclose(y,ln(x).detach().numpy(),atol=5e-7,rtol=2e-6)

def test_before_df_delays_against_vectorized_oracle():
    rng=np.random.default_rng(801)
    raw=np.zeros((5,481,2),np.float32);masked=raw.copy();cs=np.zeros((3,96,5,2),np.float32)
    for frame in range(20):
        spec=rng.normal(size=(481,2)).astype(np.float32)
        mask=rng.random(481).astype(np.float32);co=rng.normal(size=(96,5,2)).astype(np.float32)
        raw=np.concatenate([raw[1:],spec[None]],0)
        masked=np.concatenate([masked[1:],(raw[2]*mask[:,None])[None]],0)
        cs=np.concatenate([cs[1:],co[None]],0)
        native=masked[2].copy()
        for f in range(96):
            re=im=np.float32(0)
            for tap in range(5):
                s=masked[tap,f];c=cs[0,f,tap]
                re+=s[0]*c[0]-s[1]*c[1];im+=s[0]*c[1]+s[1]*c[0]
            native[f]=[re,im]
        sm=torch.from_numpy(masked[:,:96]).permute(1,0,2)
        cc=torch.from_numpy(cs[0])
        oracle=torch.view_as_real((torch.view_as_complex(sm.contiguous())*torch.view_as_complex(cc.contiguous())).sum(1)).numpy()
        np.testing.assert_allclose(native[:96],oracle,atol=2e-6,rtol=2e-5)

def test_reflected_nyquist_is_penultimate_bin():
    x=torch.arange(480,dtype=torch.float32).reshape(1,1,1,480)
    y=torch.nn.functional.pad(x,(0,1,0,0),mode="reflect")
    assert y.reshape(-1)[480]==478

def test_python_alpha_complement_not_recomputed_in_f32():
    # PyTorch receives independently rounded scalar constants alpha and (1-alpha).
    alpha=0.98
    a=np.float32(alpha);b=np.float32(1.0-alpha)
    assert b != np.float32(1.0)-a
    state=torch.tensor([17.4,-73.2],dtype=torch.float32)
    x=torch.tensor([0.1,44.7],dtype=torch.float32)
    expected=(alpha*state+(1-alpha)*x).numpy()
    actual=a*state.numpy()+b*x.numpy()
    np.testing.assert_array_equal(actual,expected)

@pytest.mark.parametrize("freq",[5,40,48])
def test_bidirectional_oracle_resets_each_frame(freq):
    from quantized_reference import QBiGRU
    torch.manual_seed(300)
    original=nn.GRU(64,64,batch_first=True,bidirectional=True).eval()
    oracle=QBiGRU(original,False,False)
    x=torch.randn(1,freq,64)
    for _ in range(2):
        a,h=oracle(x);b,hb=original(x)
        np.testing.assert_allclose(a.detach().numpy(),b.detach().numpy(),atol=5e-7,rtol=1e-4)
        np.testing.assert_allclose(h.detach().numpy(),hb.detach().numpy(),atol=5e-7,rtol=1e-4)

def test_recurrent_bias_inside_reset_is_required():
    # Regression guard against moving recurrent candidate bias outside the reset gate.
    rng=np.random.default_rng(21)
    r=rng.random(64).astype(np.float32);rh=rng.normal(size=64).astype(np.float32);br=np.ones(64,np.float32)
    correct=r*(rh+br)
    wrong=r*rh+br
    assert np.max(np.abs(correct-wrong))>0.5

@pytest.mark.parametrize("bias", [False, True])
def test_affine_depthwise_pointwise_fusion(bias):
    torch.manual_seed(721)
    module = nn.Sequential(nn.Conv2d(64,64,1,groups=64,bias=bias),
                           nn.Conv2d(64,64,1,bias=bias),nn.BatchNorm2d(64),nn.ReLU()).eval()
    with torch.no_grad():
        module[2].running_mean.normal_(); module[2].running_var.uniform_(0.1,2)
        module[2].weight.normal_(); module[2].bias.normal_()
    x=torch.randn(1,64,1,19)
    ex=Exporter(); spec=ex.pipeline(module,(1,19,64))
    assert len(spec["ops"])==1 and ex.fused_depthwise_pointwise_pairs==1
    got=pipeline_numpy(ex,spec,x.permute(0,2,3,1).numpy()[0])
    expected=module(x).detach().permute(0,2,3,1).numpy()[0]
    np.testing.assert_allclose(got,expected,atol=5e-6,rtol=3e-5)

def test_no_fusion_through_activation():
    module=nn.Sequential(nn.Conv2d(8,8,1,groups=8),nn.ReLU(),nn.Conv2d(8,8,1)).eval()
    ex=Exporter(); config=ex.pipeline(module,(1,9,8))
    assert len(config["ops"])==2 and ex.fused_depthwise_pointwise_pairs==0

def test_tiny_activation_reciprocal_does_not_saturate():
    x=np.array([1e-38,0.5e-38,0,-0.5e-38],np.float32)
    q,scale=activation_quant(x)
    assert np.isfinite(scale) and scale>0
    assert 7500<q[1]<8500 and q[2]==0 and q[3]==-q[1]
    q,scale=activation_quant(np.full(4,np.nextafter(np.float32(0),np.float32(1)),np.float32))
    assert scale==0 and not q.any()


def test_activation_rounding_half_neighbors():
    # Pin the dynamic activation scale to exactly 1. Test each half-integer
    # and its adjacent FP32 values, in both signs, throughout the valid range.
    half = np.arange(16383, dtype=np.float32) + np.float32(0.5)
    positive = np.concatenate([
        np.nextafter(half, np.float32(0)),
        half,
        np.nextafter(half, np.float32(np.inf)),
    ])
    x = np.concatenate([
        np.array([16383, -16383, 0], dtype=np.float32), positive, -positive,
    ])
    q, scale = activation_quant(x)
    assert scale == np.float32(1)
    # FP64 is used ONLY for this bounded test's independent rounding oracle.
    # Every FP32 input and its sum with 0.5 here is exactly representable in FP64.
    exact = np.abs(x.astype(np.float64))
    expected = np.copysign(np.floor(exact + 0.5), x).astype(np.int16)
    np.testing.assert_array_equal(q, expected)
