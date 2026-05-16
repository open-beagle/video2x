from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe TRT + GStreamer CUDA postprocess throughput.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--input-width", type=int, default=720)
    parser.add_argument("--input-height", type=int, default=420)
    parser.add_argument("--trt-output-width", type=int, default=2880)
    parser.add_argument("--trt-output-height", type=int, default=1680)
    parser.add_argument("--target-width", type=int, default=1920)
    parser.add_argument("--target-height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    tensor = frame_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...])


def output_to_rgb_bytes(output: np.ndarray) -> bytes:
    image = np.squeeze(output)
    if image.ndim != 3:
        raise RuntimeError(f"unexpected output shape: {output.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8).tobytes()


def main() -> int:
    args = parse_args()
    if not args.engine.exists():
        raise SystemExit(f"engine not found: {args.engine}")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    raw_frame_size = args.input_width * args.input_height * 3
    decode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(args.input),
        "-map",
        "0:v:0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    gst_cmd = [
        "gst-launch-1.0",
        "-q",
        "fdsrc",
        "fd=0",
        f"blocksize={args.trt_output_width * args.trt_output_height * 3}",
        "!",
        f"video/x-raw,format=RGB,width={args.trt_output_width},height={args.trt_output_height},framerate={args.fps}/1",
        "!",
        "queue",
        "max-size-buffers=4",
        "leaky=downstream",
        "!",
        "cudaupload",
        f"cuda-device-id={args.gpu_id}",
        "!",
        "cudascale",
        f"cuda-device-id={args.gpu_id}",
        "add-borders=true",
        "!",
        f"video/x-raw(memory:CUDAMemory),format=RGB,width={args.target_width},height={args.target_height},framerate={args.fps}/1",
        "!",
        "cudadownload",
        "!",
        f"video/x-raw,format=RGB,width={args.target_width},height={args.target_height},framerate={args.fps}/1",
        "!",
        "fakesink",
        "sync=false",
    ]

    started = time.time()
    decode_time = 0.0
    preprocess_time = 0.0
    infer_time = 0.0
    write_time = 0.0
    frames = 0

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    gst = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE)
    try:
        with open(args.engine, "rb") as engine_file, TrtRunner(EngineFromBytes(engine_file.read())) as runner:
            while True:
                if args.frames and frames >= args.frames:
                    break

                t0 = time.perf_counter()
                chunk = decoder.stdout.read(raw_frame_size) if decoder.stdout else b""
                decode_time += time.perf_counter() - t0
                if not chunk:
                    break
                if len(chunk) != raw_frame_size:
                    raise RuntimeError(f"incomplete frame: {len(chunk)} != {raw_frame_size}")

                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((args.input_height, args.input_width, 3))

                t0 = time.perf_counter()
                feed = preprocess(frame)
                preprocess_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                outputs = runner.infer(feed_dict={"input": feed})
                infer_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                raw = output_to_rgb_bytes(next(iter(outputs.values())))
                if not gst.stdin:
                    raise RuntimeError("gstreamer stdin closed")
                gst.stdin.write(raw)
                write_time += time.perf_counter() - t0
                frames += 1
    finally:
        if decoder.stdout:
            decoder.stdout.close()
        if gst.stdin:
            gst.stdin.close()
        decoder.wait()
        gst.wait()

    if decoder.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {decoder.returncode}")
    if gst.returncode != 0:
        raise RuntimeError(f"gstreamer pipeline failed: {gst.returncode}")

    elapsed = max(time.time() - started, 0.001)
    print(
        f"frames={frames} elapsed={elapsed:.3f}s fps={frames / elapsed:.3f} "
        f"decode={decode_time:.3f}s preprocess={preprocess_time:.3f}s "
        f"infer={infer_time:.3f}s trt_to_gst_write={write_time:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
