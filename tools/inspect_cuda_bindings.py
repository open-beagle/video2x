from __future__ import annotations

from cuda.bindings import runtime as cudart


names = [name for name in dir(cudart) if name.startswith("cuda")]
print("\n".join(names[:200]))
print(f"cudaMalloc={cudart.cudaMalloc}")
print(f"cudaMemcpy={cudart.cudaMemcpy}")
print(f"cudaMemcpyKind={cudart.cudaMemcpyKind}")
