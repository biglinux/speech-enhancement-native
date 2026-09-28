#!/usr/bin/env python3
"""C ABI and LADSPA smoke tests after native compilation; use --bundle for weights."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np
from native_api import Native, P

class Hint(C.Structure):
    _fields_=[("descriptor",C.c_int),("lower",C.c_float),("upper",C.c_float)]
class Descriptor(C.Structure):
    pass
Instantiate=C.CFUNCTYPE(C.c_void_p,C.POINTER(Descriptor),C.c_ulong)
Connect=C.CFUNCTYPE(None,C.c_void_p,C.c_ulong,P)
Activate=C.CFUNCTYPE(None,C.c_void_p)
Run=C.CFUNCTYPE(None,C.c_void_p,C.c_ulong)
Cleanup=C.CFUNCTYPE(None,C.c_void_p)
Descriptor._fields_=[("id",C.c_ulong),("label",C.c_char_p),("properties",C.c_int),("name",C.c_char_p),
    ("maker",C.c_char_p),("copyright",C.c_char_p),("port_count",C.c_ulong),
    ("ports",C.POINTER(C.c_int)),("names",C.POINTER(C.c_char_p)),("hints",C.POINTER(Hint)),
    ("implementation_data",C.c_void_p),("instantiate",Instantiate),("connect",Connect),("activate",Activate),
    ("run",Run),("run_adding",C.c_void_p),("set_gain",C.c_void_p),("deactivate",C.c_void_p),("cleanup",Cleanup)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib",type=Path,default=Path("target/release-unwind/libdpdfnet_native.so"))
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--report",type=Path,default=Path("abi-results.json"))
    args=p.parse_args()
    rng=np.random.default_rng(915)
    x=rng.normal(0,0.02,10000).astype(np.float32)
    with Native(args.lib,args.bundle) as a, Native(args.lib,args.bundle) as b:
        normal=a.process(x);inplace=b.process(x,inplace=True)
        np.testing.assert_array_equal(normal,inplace)
    os.environ["DPDFNET_NATIVE_MODEL"]=str(args.bundle.resolve())
    lib=C.CDLL(str(args.lib.resolve()))
    lib.ladspa_descriptor.argtypes=[C.c_ulong];lib.ladspa_descriptor.restype=C.POINTER(Descriptor)
    assert not lib.ladspa_descriptor(1)
    desc=lib.ladspa_descriptor(0);d=desc.contents
    assert d.port_count==5 and d.label==b"dpdfnet_native_48hr"
    assert not d.instantiate(desc,44100)
    h=d.instantiate(desc,48000)
    assert h, "LADSPA instantiate failed"
    try:
        y=np.zeros_like(x);db=C.c_float(100);latency=C.c_float();fault=C.c_float()
        d.connect(h,0,x.ctypes.data_as(P));d.connect(h,1,y.ctypes.data_as(P))
        for index,control in [(2,db),(3,latency),(4,fault)]:d.connect(h,index,C.pointer(control))
        d.activate(h);d.run(h,x.size)
        np.testing.assert_array_equal(y,normal)
        assert latency.value==2880 and fault.value==0
        # Exact aliasing through the actual LADSPA callbacks, not just the auxiliary C API.
        z=x.copy();d.connect(h,0,z.ctypes.data_as(P));d.connect(h,1,z.ctypes.data_as(P))
        d.activate(h);d.run(h,z.size);np.testing.assert_array_equal(z,normal)
    finally:
        d.cleanup(h)
    report=dict(passed=True,tests=["C ABI in-place","descriptor layout","unsupported rate rejection",
        "LADSPA vs C API output","latency port","LADSPA in-place","activate resets stream"],
        descriptor_size=C.sizeof(Descriptor),unsigned_long_size=C.sizeof(C.c_ulong))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
