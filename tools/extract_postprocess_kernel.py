from __future__ import annotations

import argparse
import ast
from pathlib import Path


def extract_kernel(worker_path: Path) -> str:
    module = ast.parse(worker_path.read_text(encoding="utf-8"), filename=str(worker_path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "KERNEL":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise TypeError("KERNEL is not a string")
                    return value
    raise ValueError(f"KERNEL not found in {worker_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CUDA postprocess kernel from src/worker.py.")
    parser.add_argument("--worker", type=Path, default=Path("/app/src/worker.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(extract_kernel(args.worker), encoding="utf-8")
    print(f"kernel extracted: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
