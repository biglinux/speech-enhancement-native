"""ctypes interface used by tests; never imported by the Rust plugin."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
from typing import Self
import numpy as np
P = C.POINTER(C.c_float)
class Native:
    def __init__(self, library: Path, bundle: Path) -> None:
        self.lib = C.CDLL(str(library.resolve()))
        self.lib.dpdfnet_native_create.argtypes = [C.c_char_p]
        self.lib.dpdfnet_native_create.restype = C.c_void_p
        self.lib.dpdfnet_native_destroy.argtypes = [C.c_void_p]
        self.lib.dpdfnet_native_destroy.restype = None
        self.lib.dpdfnet_native_reset.argtypes = [C.c_void_p]
        self.lib.dpdfnet_native_reset.restype = None
        self.lib.dpdfnet_native_process.argtypes = [C.c_void_p, P, P, C.c_size_t, C.c_float]
        self.lib.dpdfnet_native_process.restype = C.c_int
        self.lib.dpdfnet_native_spectrum.argtypes = [C.c_void_p, P, P]
        self.lib.dpdfnet_native_spectrum.restype = C.c_int
        self.lib.dpdfnet_native_trace.argtypes = [C.c_void_p, C.c_size_t, P, C.c_size_t]
        self.lib.dpdfnet_native_trace.restype = C.c_size_t
        self.lib.dpdfnet_native_latency_samples.argtypes = []
        self.lib.dpdfnet_native_latency_samples.restype = C.c_size_t
        self.handle = self.lib.dpdfnet_native_create(str(bundle.resolve()).encode())
        if not self.handle:
            raise RuntimeError("Native load failed. Run `cargo run --profile release-unwind -p dpdfnet-native "
                               "--example inspect -- BUNDLE` for the load error.")
    def close(self) -> None:
        if self.handle:
            self.lib.dpdfnet_native_destroy(self.handle)
            self.handle = None
    def reset(self) -> None:
        self.lib.dpdfnet_native_reset(self.handle)
    def spectrum(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if x.size != 962:
            raise ValueError("Spectrum must have 481 complex bins")
        y = np.empty_like(x)
        rc = self.lib.dpdfnet_native_spectrum(self.handle, x.ctypes.data_as(P), y.ctypes.data_as(P))
        if rc:
            raise RuntimeError(f"Spectrum processing failed: {rc}")
        return y
    def trace(self, index: int) -> np.ndarray:
        n = self.lib.dpdfnet_native_trace(self.handle, index, None, 0)
        y = np.empty(n, dtype=np.float32)
        self.lib.dpdfnet_native_trace(self.handle, index, y.ctypes.data_as(P), n)
        return y
    def process(self, x: np.ndarray, db: float = 100.0, inplace: bool = False) -> np.ndarray:
        x = np.array(x, dtype=np.float32, order="C", copy=True).reshape(-1)
        y = x if inplace else np.empty_like(x)
        rc = self.lib.dpdfnet_native_process(self.handle, x.ctypes.data_as(P), y.ctypes.data_as(P), x.size, db)
        if rc:
            raise RuntimeError(f"Audio processing failed: {rc}")
        return y
    def __enter__(self) -> Self:
        return self
    def __exit__(self, *_: object) -> None:
        self.close()
