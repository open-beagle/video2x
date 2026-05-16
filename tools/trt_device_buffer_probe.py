from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def check(result, label: str):
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{label} failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def dtype_to_np(dtype: trt.DataType):
    if dtype == trt.DataType.FLOAT:
        return np.float32
    if dtype == trt.DataType.HALF:
        return np.float16
    if dtype == trt.DataType.INT8:
        return np.int8
    if dtype == trt.DataType.INT32:
        return np.int32
    raise ValueError(f"unsupported dtype: {dtype}")


def nbytes(shape: tuple[int, ...], dtype: trt.DataType) -> int:
    return math.prod(shape) * np.dtype(dtype_to_np(dtype)).itemsize


def event_ms(start, end) -> float:
    check(cudart.cudaEventSynchronize(end), "cudaEventSynchronize")
    return float(check(cudart.cudaEventElapsedTime(start, end), "cudaEventElapsedTime"))


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorRT explicit device buffer timing probe.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=153)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--copy-output", action="store_true", help="measure D2H copy every iteration")
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to load engine: {args.engine}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    tensors = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        tensors[name] = {"shape": shape, "dtype": dtype, "mode": mode, "bytes": nbytes(shape, dtype)}

    input_name = next(name for name, meta in tensors.items() if meta["mode"] == trt.TensorIOMode.INPUT)
    output_name = next(name for name, meta in tensors.items() if meta["mode"] == trt.TensorIOMode.OUTPUT)
    input_meta = tensors[input_name]
    output_meta = tensors[output_name]

    host_input = np.random.random(input_meta["shape"]).astype(dtype_to_np(input_meta["dtype"]))
    host_output = np.empty(output_meta["shape"], dtype=dtype_to_np(output_meta["dtype"]))

    stream = check(cudart.cudaStreamCreate(), "cudaStreamCreate")
    d_input = check(cudart.cudaMalloc(input_meta["bytes"]), "cudaMalloc input")
    d_output = check(cudart.cudaMalloc(output_meta["bytes"]), "cudaMalloc output")

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    h2d_start = check(cudart.cudaEventCreate(), "cudaEventCreate")
    h2d_end = check(cudart.cudaEventCreate(), "cudaEventCreate")
    infer_start = check(cudart.cudaEventCreate(), "cudaEventCreate")
    infer_end = check(cudart.cudaEventCreate(), "cudaEventCreate")
    d2h_start = check(cudart.cudaEventCreate(), "cudaEventCreate")
    d2h_end = check(cudart.cudaEventCreate(), "cudaEventCreate")

    h2d_ms = 0.0
    infer_ms = 0.0
    d2h_ms = 0.0
    wall_started = time.time()

    total = args.warmup + args.iterations
    for i in range(total):
        measure = i >= args.warmup
        check(cudart.cudaEventRecord(h2d_start, stream), "cudaEventRecord")
        check(
            cudart.cudaMemcpyAsync(
                d_input,
                host_input.ctypes.data,
                input_meta["bytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )
        check(cudart.cudaEventRecord(h2d_end, stream), "cudaEventRecord")

        check(cudart.cudaEventRecord(infer_start, stream), "cudaEventRecord")
        ok = context.execute_async_v3(int(stream))
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        check(cudart.cudaEventRecord(infer_end, stream), "cudaEventRecord")

        if args.copy_output:
            check(cudart.cudaEventRecord(d2h_start, stream), "cudaEventRecord")
            check(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    d_output,
                    output_meta["bytes"],
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                "cudaMemcpyAsync D2H",
            )
            check(cudart.cudaEventRecord(d2h_end, stream), "cudaEventRecord")

        check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        if measure:
            h2d_ms += event_ms(h2d_start, h2d_end)
            infer_ms += event_ms(infer_start, infer_end)
            if args.copy_output:
                d2h_ms += event_ms(d2h_start, d2h_end)

    wall_elapsed = time.time() - wall_started
    iters = max(args.iterations, 1)
    print(f"input={input_name} shape={input_meta['shape']} dtype={input_meta['dtype']} bytes={input_meta['bytes']}")
    print(f"output={output_name} shape={output_meta['shape']} dtype={output_meta['dtype']} bytes={output_meta['bytes']}")
    print(f"iterations={args.iterations} warmup={args.warmup} copy_output={args.copy_output}")
    print(f"h2d_ms_total={h2d_ms:.3f} h2d_ms_avg={h2d_ms / iters:.3f}")
    print(f"infer_ms_total={infer_ms:.3f} infer_ms_avg={infer_ms / iters:.3f}")
    print(f"d2h_ms_total={d2h_ms:.3f} d2h_ms_avg={d2h_ms / iters:.3f}")
    measured_total = h2d_ms + infer_ms + d2h_ms
    print(f"measured_ms_total={measured_total:.3f} measured_fps={iters / (measured_total / 1000.0):.3f}")
    print(f"wall_elapsed={wall_elapsed:.3f}s")

    check(cudart.cudaFree(d_input), "cudaFree input")
    check(cudart.cudaFree(d_output), "cudaFree output")
    check(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
