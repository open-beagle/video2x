from __future__ import annotations

import argparse
import ctypes
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart


KERNEL = r"""
#include <cuda_fp16.h>

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

extern "C" __global__
void chw_half_to_rgb8_resize_pad(
    const half* __restrict__ src,
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
        const half* p = src + c * plane;
        float v00 = __half2float(p[y0 * src_w + x0]);
        float v01 = __half2float(p[y0 * src_w + x1]);
        float v10 = __half2float(p[y1 * src_w + x0]);
        float v11 = __half2float(p[y1 * src_w + x1]);
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        float v = v0 + (v1 - v0) * fy;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        dst[dst_idx + c] = (unsigned char)(v * 255.0f + 0.5f);
    }
}

static __device__ __forceinline__
float sample_chw_pixel(
    const float* __restrict__ src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    int c
) {
    if (x < pad_left || x >= pad_left + content_w) {
        return 0.0f;
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
    const float* p = src + c * plane;
    float v00 = p[y0 * src_w + x0];
    float v01 = p[y0 * src_w + x1];
    float v10 = p[y1 * src_w + x0];
    float v11 = p[y1 * src_w + x1];
    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    float v = v0 + (v1 - v0) * fy;
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__
float sample_chw_half_pixel(
    const half* __restrict__ src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int content_w,
    int pad_left,
    int x,
    int y,
    int c
) {
    if (x < pad_left || x >= pad_left + content_w) {
        return 0.0f;
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
    const half* p = src + c * plane;
    float v00 = __half2float(p[y0 * src_w + x0]);
    float v01 = __half2float(p[y0 * src_w + x1]);
    float v10 = __half2float(p[y1 * src_w + x0]);
    float v11 = __half2float(p[y1 * src_w + x1]);
    float v0 = v00 + (v01 - v00) * fx;
    float v1 = v10 + (v11 - v10) * fx;
    float v = v0 + (v1 - v0) * fy;
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__
unsigned char clamp_u8(float v) {
    v = fminf(fmaxf(v, 0.0f), 255.0f);
    return (unsigned char)(v + 0.5f);
}

extern "C" __global__
void chw_half_to_nv12_resize_pad(
    const half* __restrict__ src,
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

    float r = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    dst[y * dst_w + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_half_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = dst_w * dst_h + (y / 2) * dst_w + x;
        dst[uv_idx] = clamp_u8(u_sum * 0.25f);
        dst[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
    }
}

extern "C" __global__
void chw_float_to_nv12_resize_pad(
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

    float r = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 0) * 255.0f;
    float g = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 1) * 255.0f;
    float b = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, x, y, 2) * 255.0f;
    dst[y * dst_w + x] = clamp_u8(0.257f * r + 0.504f * g + 0.098f * b + 16.0f);

    if ((x & 1) == 0 && (y & 1) == 0) {
        float u_sum = 0.0f;
        float v_sum = 0.0f;
        for (int oy = 0; oy < 2; ++oy) {
            for (int ox = 0; ox < 2; ++ox) {
                int px = min(x + ox, dst_w - 1);
                int py = min(y + oy, dst_h - 1);
                float sr = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 0) * 255.0f;
                float sg = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 1) * 255.0f;
                float sb = sample_chw_pixel(src, src_w, src_h, dst_w, dst_h, content_w, pad_left, px, py, 2) * 255.0f;
                u_sum += -0.148f * sr - 0.291f * sg + 0.439f * sb + 128.0f;
                v_sum += 0.439f * sr - 0.368f * sg - 0.071f * sb + 128.0f;
            }
        }
        int uv_idx = dst_w * dst_h + (y / 2) * dst_w + x;
        dst[uv_idx] = clamp_u8(u_sum * 0.25f);
        dst[uv_idx + 1] = clamp_u8(v_sum * 0.25f);
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


def compile_kernel() -> dict[tuple[str, str], cu.CUfunction]:
    program = check_nvrtc(nvrtc.nvrtcCreateProgram(KERNEL.encode(), b"postprocess.cu", 0, None, None), "nvrtcCreateProgram")
    options = [
        b"--std=c++17",
        b"--use_fast_math",
        b"--gpu-architecture=compute_89",
        b"--include-path=/usr/local/cuda/include",
    ]
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
    return {
        ("float", "rgb24"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_float_to_rgb8_resize_pad"),
            "cuModuleGetFunction float rgb",
        ),
        ("float", "nv12"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_float_to_nv12_resize_pad"),
            "cuModuleGetFunction float nv12",
        ),
        ("half", "rgb24"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_half_to_rgb8_resize_pad"),
            "cuModuleGetFunction half rgb",
        ),
        ("half", "nv12"): check_driver(
            cu.cuModuleGetFunction(module, b"chw_half_to_nv12_resize_pad"),
            "cuModuleGetFunction half nv12",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real video TRT + CUDA postprocess runner.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=720)
    parser.add_argument("--input-height", type=int, default=420)
    parser.add_argument("--decode-width", type=int, default=0)
    parser.add_argument("--decode-height", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--content-width", type=int, default=1852)
    parser.add_argument("--encoder", default="libx265")
    parser.add_argument("--bitrate", default="5M")
    parser.add_argument("--output-pix-fmt", choices=["rgb24", "nv12"], default="rgb24")
    return parser.parse_args()


def doubled_bitrate(value: str) -> str:
    if len(value) > 1 and value[-1].isalpha() and value[:-1].isdigit():
        return f"{int(value[:-1]) * 2}{value[-1]}"
    if value.isdigit():
        return str(int(value) * 2)
    return value


def merge_audio(input_path: Path, video_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            str(output_path),
        ],
        check=True,
    )


def main() -> int:
    args = parse_args()
    check_driver(cu.cuInit(0), "cuInit")
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    kernels = compile_kernel()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to load engine: {args.engine}")
    context = engine.create_execution_context()

    input_name = output_name = None
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
    decode_width = args.decode_width or args.input_width
    decode_height = args.decode_height or args.input_height
    if input_shape != (1, 3, decode_height, decode_width):
        raise RuntimeError(f"engine input shape {input_shape} does not match decode size {decode_width}x{decode_height}")
    output_kind = "half" if output_dtype == trt.DataType.HALF else "float"

    input_bytes = tensor_nbytes(input_shape, input_dtype)
    output_bytes = tensor_nbytes(output_shape, output_dtype)
    frame_bytes = args.target_width * args.target_height * (3 if args.output_pix_fmt == "rgb24" else 3 // 2)
    if args.output_pix_fmt == "nv12":
        frame_bytes = args.target_width * args.target_height * 3 // 2
    raw_in_bytes = decode_width * decode_height * 3
    src_h = output_shape[2]
    src_w = output_shape[3]
    pad_left = (args.target_width - args.content_width) // 2

    host_input = np.empty(input_shape, dtype=dtype_to_np(input_dtype))
    host_frame = np.empty(frame_bytes, dtype=np.uint8)

    stream = check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")
    d_input = check_cuda(cudart.cudaMalloc(input_bytes), "cudaMalloc input")
    d_output = check_cuda(cudart.cudaMalloc(output_bytes), "cudaMalloc output")
    d_frame = check_cuda(cudart.cudaMalloc(frame_bytes), "cudaMalloc frame")
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        no_audio = tmpdir / "video_no_audio.mp4"
        decode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(args.input),
            "-map",
            "0:v:0",
        ]
        if decode_width != args.input_width or decode_height != args.input_height:
            decode_cmd += ["-vf", f"scale={decode_width}:{decode_height}:flags=bicubic"]
        decode_cmd += [
            "-nostdin",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
        encode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            args.output_pix_fmt,
            "-s",
            f"{args.target_width}x{args.target_height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            args.encoder,
        ]
        if args.bitrate:
            encode_cmd += ["-b:v", args.bitrate, "-maxrate", args.bitrate, "-bufsize", doubled_bitrate(args.bitrate)]
        elif args.encoder in {"libx264", "libx265"}:
            encode_cmd += ["-crf", "18"]
        if args.encoder in {"libx264", "libx265"}:
            encode_cmd += ["-preset", "ultrafast"]
            if args.encoder == "libx265":
                encode_cmd += ["-x265-params", "log-level=error"]
        elif args.encoder in {"h264_nvenc", "hevc_nvenc"}:
            encode_cmd += ["-preset", "p1", "-tune", "ull"]
        encode_cmd += ["-pix_fmt", "yuv420p", str(no_audio)]

        decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)
        started = time.time()
        frames = 0
        decode_time = preprocess_time = h2d_time = infer_time = kernel_time = d2h_time = encode_time = 0.0
        frame_limit = args.frames or args.expected_frames
        try:
            while True:
                if frame_limit and frames >= frame_limit:
                    break
                t0 = time.perf_counter()
                chunk = decoder.stdout.read(raw_in_bytes) if decoder.stdout else b""
                decode_time += time.perf_counter() - t0
                if not chunk:
                    break
                if len(chunk) != raw_in_bytes:
                    raise RuntimeError(f"incomplete input frame: {len(chunk)} != {raw_in_bytes}")

                t0 = time.perf_counter()
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((decode_height, decode_width, 3))
                host_input[0] = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))
                preprocess_time += time.perf_counter() - t0

                t0 = time.perf_counter()
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
                check_cuda(cudart.cudaStreamSynchronize(stream), "sync h2d")
                h2d_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                if not context.execute_async_v3(int(stream)):
                    raise RuntimeError("execute_async_v3 failed")
                check_cuda(cudart.cudaStreamSynchronize(stream), "sync infer")
                infer_time += time.perf_counter() - t0

                params = [
                    ctypes.c_void_p(int(d_output)),
                    ctypes.c_void_p(int(d_frame)),
                    ctypes.c_int(src_w),
                    ctypes.c_int(src_h),
                    ctypes.c_int(args.target_width),
                    ctypes.c_int(args.target_height),
                    ctypes.c_int(args.content_width),
                    ctypes.c_int(pad_left),
                ]
                param_ptrs = (ctypes.c_void_p * len(params))(*[ctypes.addressof(p) for p in params])

                t0 = time.perf_counter()
                check_driver(
                    cu.cuLaunchKernel(
                        kernels[(output_kind, args.output_pix_fmt)],
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
                check_cuda(cudart.cudaStreamSynchronize(stream), "sync kernel")
                kernel_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                check_cuda(
                    cudart.cudaMemcpyAsync(
                        host_frame.ctypes.data,
                        d_frame,
                        frame_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "D2H frame",
                )
                check_cuda(cudart.cudaStreamSynchronize(stream), "sync d2h")
                d2h_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                if not encoder.stdin:
                    raise RuntimeError("encoder stdin closed")
                encoder.stdin.write(host_frame.tobytes())
                encode_time += time.perf_counter() - t0
                frames += 1
        finally:
            if decoder.stdout:
                decoder.stdout.close()
            if encoder.stdin:
                encoder.stdin.close()
            decoder.wait()
            encoder.wait()

        if encoder.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {encoder.returncode}")
        if decoder.returncode != 0 and not (frame_limit and frames >= frame_limit):
            raise RuntimeError(f"ffmpeg decode failed: {decoder.returncode}")

        merge_started = time.perf_counter()
        merge_audio(args.input, no_audio, args.output)
        merge_time = time.perf_counter() - merge_started
        elapsed = max(time.time() - started, 0.001)

    print(f"frames={frames} elapsed={elapsed:.3f}s fps={frames / elapsed:.3f}")
    print(
        f"decode={decode_time:.3f}s preprocess={preprocess_time:.3f}s h2d={h2d_time:.3f}s "
        f"infer={infer_time:.3f}s kernel={kernel_time:.3f}s d2h_frame={d2h_time:.3f}s "
        f"encode_write={encode_time:.3f}s merge_audio={merge_time:.3f}s"
    )

    check_cuda(cudart.cudaFree(d_input), "free input")
    check_cuda(cudart.cudaFree(d_output), "free output")
    check_cuda(cudart.cudaFree(d_frame), "free frame")
    if not args.output.exists():
        shutil.copy2(no_audio, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
