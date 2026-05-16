from __future__ import annotations

import argparse
import ctypes
import math
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart


KERNEL = r"""
extern "C" __global__
void chw_float_to_rgb8_resize_pad(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int dst_idx = (y * dst_w + x) * 3;
    if (x < pad_left || x >= pad_left + content_w) {
        dst[dst_idx + 0] = 0;
        dst[dst_idx + 1] = 0;
        dst[dst_idx + 2] = 0;
        return;
    }

    float sx = ((float)(x - pad_left) + 0.5f) * ((float)src_w / (float)content_w) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    if (x0 < 0) { x0 = 0; fx = 0.0f; }
    if (y0 < 0) { y0 = 0; fy = 0.0f; }
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 >= src_w) x1 = src_w - 1;
    if (y1 >= src_h) y1 = src_h - 1;

    int plane = src_w * src_h;
    for (int c = 0; c < 3; ++c) {
        const float* p = src + c * plane;
        float v00 = p[y0 * src_w + x0];
        float v01 = p[y0 * src_w + x1];
        float v10 = p[y1 * src_w + x0];
        float v11 = p[y1 * src_w + x1];
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        float v = v0 + (v1 - v0) * fy;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        dst[dst_idx + c] = (unsigned char)(v * 255.0f + 0.5f);
    }
}
"""


def check_cuda(result, label: str):
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{label} failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def check_driver(result, label: str):
    err = result[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label} failed: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def check_nvrtc(result, label: str):
    err = result[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
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
    raise ValueError(f"unsupported dtype: {dtype}")


def tensor_nbytes(shape: tuple[int, ...], dtype: trt.DataType) -> int:
    return math.prod(shape) * np.dtype(dtype_to_np(dtype)).itemsize


def event_ms(start, end) -> float:
    check_cuda(cudart.cudaEventSynchronize(end), "cudaEventSynchronize")
    return float(check_cuda(cudart.cudaEventElapsedTime(start, end), "cudaEventElapsedTime"))


def compile_kernel() -> cu.CUfunction:
    program = check_nvrtc(
        nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"postprocess.cu", 0, None, None),
        "nvrtcCreateProgram",
    )
    options = [b"--std=c++17", b"--use_fast_math", b"--gpu-architecture=compute_89"]
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if result[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        log_size = check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
        log = bytearray(log_size)
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(log.decode(errors="replace"))
    ptx_size = check_nvrtc(nvrtc.nvrtcGetPTXSize(program), "nvrtcGetPTXSize")
    ptx = bytearray(ptx_size)
    check_nvrtc(nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
    module = check_driver(cu.cuModuleLoadData(bytes(ptx)), "cuModuleLoadData")
    return check_driver(cu.cuModuleGetFunction(module, b"chw_float_to_rgb8_resize_pad"), "cuModuleGetFunction")


def main() -> int:
    parser = argparse.ArgumentParser(description="TRT device output + CUDA postprocess timing probe.")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=153)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--source-content-width", type=int, default=1852, help="aspect-ratio content width at 1080p")
    args = parser.parse_args()

    check_driver(cu.cuInit(0), "cuInit")
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    kernel = compile_kernel()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    context = engine.create_execution_context()

    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    assert input_name and output_name

    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    input_dtype = engine.get_tensor_dtype(input_name)
    output_dtype = engine.get_tensor_dtype(output_name)
    if output_dtype != trt.DataType.FLOAT:
        raise RuntimeError(f"kernel expects FP32 output, got {output_dtype}")

    input_bytes = tensor_nbytes(input_shape, input_dtype)
    output_bytes = tensor_nbytes(output_shape, output_dtype)
    rgb_bytes = args.target_width * args.target_height * 3
    src_h = output_shape[2]
    src_w = output_shape[3]
    pad_left = (args.target_width - args.source_content_width) // 2

    host_input = np.random.random(input_shape).astype(dtype_to_np(input_dtype))
    host_rgb = np.empty((args.target_height, args.target_width, 3), dtype=np.uint8)

    stream = check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
    d_input = check_cuda(cudart.cudaMalloc(input_bytes), "cudaMalloc input")
    d_output = check_cuda(cudart.cudaMalloc(output_bytes), "cudaMalloc output")
    d_rgb = check_cuda(cudart.cudaMalloc(rgb_bytes), "cudaMalloc rgb")
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    h2d_s = check_cuda(cudart.cudaEventCreate(), "event")
    h2d_e = check_cuda(cudart.cudaEventCreate(), "event")
    infer_s = check_cuda(cudart.cudaEventCreate(), "event")
    infer_e = check_cuda(cudart.cudaEventCreate(), "event")
    kernel_s = check_cuda(cudart.cudaEventCreate(), "event")
    kernel_e = check_cuda(cudart.cudaEventCreate(), "event")
    d2h_s = check_cuda(cudart.cudaEventCreate(), "event")
    d2h_e = check_cuda(cudart.cudaEventCreate(), "event")

    h2d_ms = infer_ms = kernel_ms = d2h_ms = 0.0
    total = args.warmup + args.iterations
    wall_start = time.time()
    for i in range(total):
        measure = i >= args.warmup
        check_cuda(cudart.cudaEventRecord(h2d_s, stream), "event record")
        check_cuda(
            cudart.cudaMemcpyAsync(
                d_input,
                host_input.ctypes.data,
                input_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "H2D",
        )
        check_cuda(cudart.cudaEventRecord(h2d_e, stream), "event record")

        check_cuda(cudart.cudaEventRecord(infer_s, stream), "event record")
        if not context.execute_async_v3(int(stream)):
            raise RuntimeError("execute_async_v3 failed")
        check_cuda(cudart.cudaEventRecord(infer_e, stream), "event record")

        params = [
            ctypes.c_void_p(int(d_output)),
            ctypes.c_void_p(int(d_rgb)),
            ctypes.c_int(src_w),
            ctypes.c_int(src_h),
            ctypes.c_int(args.target_width),
            ctypes.c_int(args.target_height),
            ctypes.c_int(args.source_content_width),
            ctypes.c_int(pad_left),
        ]
        param_ptrs = (ctypes.c_void_p * len(params))(*[ctypes.addressof(p) for p in params])

        check_cuda(cudart.cudaEventRecord(kernel_s, stream), "event record")
        check_driver(
            cu.cuLaunchKernel(
                kernel,
                math.ceil(args.target_width / 16),
                math.ceil(args.target_height / 16),
                1,
                16,
                16,
                1,
                0,
                int(stream),
                param_ptrs,
                0,
            ),
            "cuLaunchKernel",
        )
        check_cuda(cudart.cudaEventRecord(kernel_e, stream), "event record")

        check_cuda(cudart.cudaEventRecord(d2h_s, stream), "event record")
        check_cuda(
            cudart.cudaMemcpyAsync(
                host_rgb.ctypes.data,
                d_rgb,
                rgb_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "D2H rgb",
        )
        check_cuda(cudart.cudaEventRecord(d2h_e, stream), "event record")
        check_cuda(cudart.cudaStreamSynchronize(stream), "sync")

        if measure:
            h2d_ms += event_ms(h2d_s, h2d_e)
            infer_ms += event_ms(infer_s, infer_e)
            kernel_ms += event_ms(kernel_s, kernel_e)
            d2h_ms += event_ms(d2h_s, d2h_e)

    wall_elapsed = time.time() - wall_start
    iters = max(args.iterations, 1)
    measured_ms = h2d_ms + infer_ms + kernel_ms + d2h_ms
    print(f"input_bytes={input_bytes} output_bytes={output_bytes} rgb_bytes={rgb_bytes}")
    print(f"src={src_w}x{src_h} target={args.target_width}x{args.target_height} content_w={args.source_content_width} pad_left={pad_left}")
    print(f"iterations={args.iterations} warmup={args.warmup}")
    print(f"h2d_ms_avg={h2d_ms / iters:.3f} infer_ms_avg={infer_ms / iters:.3f}")
    print(f"kernel_ms_avg={kernel_ms / iters:.3f} d2h_rgb_ms_avg={d2h_ms / iters:.3f}")
    print(f"measured_ms_total={measured_ms:.3f} measured_fps={iters / (measured_ms / 1000.0):.3f}")
    print(f"wall_elapsed={wall_elapsed:.3f}s")
    print(f"rgb_checksum={int(host_rgb.sum())}")

    check_cuda(cudart.cudaFree(d_input), "free")
    check_cuda(cudart.cudaFree(d_output), "free")
    check_cuda(cudart.cudaFree(d_rgb), "free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
