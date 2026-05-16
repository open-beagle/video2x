from __future__ import annotations

import importlib
import importlib.util


MODULES = [
    "tensorrt",
    "cuda",
    "cuda.bindings",
    "cuda.bindings.runtime",
    "cuda.bindings.driver",
    "cuda.bindings.nvrtc",
    "cuda.cudart",
    "pycuda",
]


for name in MODULES:
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"{name}: MISSING")
        continue
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK {getattr(module, '__file__', '')}")
    except Exception as exc:
        print(f"{name}: ERR {exc!r}")
