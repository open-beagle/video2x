from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import gi
import numpy as np
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe TRT + GStreamer appsrc CUDA postprocess throughput.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
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


def make_pipeline(args: argparse.Namespace) -> tuple[Gst.Pipeline, Gst.Element]:
    pipeline_desc = (
        "appsrc name=src is-live=false format=time block=true "
        f"caps=video/x-raw,format=RGB,width={args.trt_output_width},height={args.trt_output_height},framerate={args.fps}/1 "
        f"! queue max-size-buffers=4 "
        f"! cudaupload cuda-device-id={args.gpu_id} "
        f"! cudascale cuda-device-id={args.gpu_id} add-borders=true "
        f"! video/x-raw(memory:CUDAMemory),format=RGB,width={args.target_width},height={args.target_height},framerate={args.fps}/1 "
        "! cudadownload "
        f"! video/x-raw,format=RGB,width={args.target_width},height={args.target_height},framerate={args.fps}/1 "
        "! fakesink name=sink sync=false"
    )
    pipeline = Gst.parse_launch(pipeline_desc)
    appsrc = pipeline.get_by_name("src")
    if appsrc is None:
        raise RuntimeError("appsrc not found")
    return pipeline, appsrc


def watch_bus(pipeline: Gst.Pipeline, errors: list[str], stop: threading.Event) -> None:
    bus = pipeline.get_bus()
    while not stop.is_set():
        msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if not msg:
            continue
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            errors.append(f"{err}: {debug}")
            stop.set()
        elif msg.type == Gst.MessageType.EOS:
            stop.set()


def main() -> int:
    args = parse_args()
    Gst.init([])

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

    pipeline, appsrc = make_pipeline(args)
    errors: list[str] = []
    stop = threading.Event()
    bus_thread = threading.Thread(target=watch_bus, args=(pipeline, errors, stop), daemon=True)

    started = time.time()
    decode_time = 0.0
    preprocess_time = 0.0
    infer_time = 0.0
    push_time = 0.0
    frames = 0

    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    pipeline.set_state(Gst.State.PLAYING)
    bus_thread.start()
    try:
        with open(args.engine, "rb") as engine_file, TrtRunner(EngineFromBytes(engine_file.read())) as runner:
            while not stop.is_set():
                if args.frames and frames >= args.frames:
                    break

                t0 = time.perf_counter()
                chunk = decoder.stdout.read(raw_frame_size) if decoder.stdout else b""
                decode_time += time.perf_counter() - t0
                if not chunk:
                    break
                if len(chunk) != raw_frame_size:
                    raise RuntimeError(f"incomplete input frame: {len(chunk)} != {raw_frame_size}")

                frame = np.frombuffer(chunk, dtype=np.uint8).reshape((args.input_height, args.input_width, 3))

                t0 = time.perf_counter()
                feed = preprocess(frame)
                preprocess_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                outputs = runner.infer(feed_dict={"input": feed})
                infer_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                raw = output_to_rgb_bytes(next(iter(outputs.values())))
                buf = Gst.Buffer.new_allocate(None, len(raw), None)
                buf.fill(0, raw)
                duration = Gst.SECOND // args.fps
                buf.pts = frames * duration
                buf.dts = frames * duration
                buf.duration = duration
                result = appsrc.emit("push-buffer", buf)
                push_time += time.perf_counter() - t0
                if result != Gst.FlowReturn.OK:
                    raise RuntimeError(f"appsrc push failed: {result}")
                frames += 1
    finally:
        appsrc.emit("end-of-stream")
        if decoder.stdout:
            decoder.stdout.close()
        decoder.wait()
        stop.set()
        bus_thread.join(timeout=2)
        pipeline.set_state(Gst.State.NULL)

    elapsed = max(time.time() - started, 0.001)
    print(
        f"frames={frames} elapsed={elapsed:.3f}s fps={frames / elapsed:.3f} "
        f"decode={decode_time:.3f}s preprocess={preprocess_time:.3f}s "
        f"infer={infer_time:.3f}s trt_to_appsrc_push={push_time:.3f}s",
        flush=True,
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    if decoder.returncode != 0 and not (args.frames and frames >= args.frames):
        raise RuntimeError(f"ffmpeg decode failed: {decoder.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
