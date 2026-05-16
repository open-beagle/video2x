from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to load engine: {args.engine}")

    print(f"num_io_tensors={engine.num_io_tensors}")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        print(
            f"{index}: name={name} mode={engine.get_tensor_mode(name)} "
            f"shape={shape} dtype={engine.get_tensor_dtype(name)}"
        )
        if len(shape) == 4:
            print(f"   nchw={shape[0]}x{shape[1]}x{shape[2]}x{shape[3]} size={shape[3]}x{shape[2]}")
    context = engine.create_execution_context()
    print(f"has_execute_async_v3={hasattr(context, 'execute_async_v3')}")
    print(f"has_set_tensor_address={hasattr(context, 'set_tensor_address')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
