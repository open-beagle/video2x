from __future__ import annotations

import inspect

from cuda.bindings import driver as cu
from cuda.bindings import nvrtc


for obj in [
    cu.cuInit,
    cu.cuDeviceGet,
    cu.cuCtxCreate,
    cu.cuModuleLoadData,
    cu.cuModuleGetFunction,
    cu.cuLaunchKernel,
    nvrtc.nvrtcCreateProgram,
    nvrtc.nvrtcCompileProgram,
    nvrtc.nvrtcGetPTXSize,
    nvrtc.nvrtcGetPTX,
]:
    try:
        sig = inspect.signature(obj)
    except Exception as exc:
        sig = f"<no signature: {exc}>"
    print(f"{obj.__name__}: {sig}")
