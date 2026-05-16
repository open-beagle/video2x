from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ffprobe import validate_dimensions
from planner import Task


def _content_width(input_width: int, input_height: int, target_height: int) -> int:
    width = round(input_width * target_height / input_height)
    if width % 2:
        width += 1
    return width


def _decode_size(input_width: int, input_height: int) -> tuple[int, int]:
    if input_width == 1280 and input_height == 720:
        return 960, 540
    return input_width, input_height


def run_trt_cuda_task(
    task: Task,
    engine_path: Path,
    target_width: int,
    target_height: int,
    benchmark_frames: str | None,
    video_encoder: str,
    video_bitrate: str,
    video_pixel_format: str | None,
    tool_path: Path,
) -> None:
    if not engine_path.exists():
        raise RuntimeError(f"TRT engine not found: {engine_path}")
    if not tool_path.exists():
        raise RuntimeError(f"TRT CUDA runner tool not found: {tool_path}")

    frames_arg: list[str] = []
    if benchmark_frames:
        frames_arg = ["--frames", benchmark_frames]
    elif task.info.frames:
        frames_arg = ["--expected-frames", str(task.info.frames)]

    decode_width, decode_height = _decode_size(task.info.width, task.info.height)
    content_width = _content_width(task.info.width, task.info.height, target_height)
    output_pix_fmt = video_pixel_format or ("nv12" if video_encoder in {"h264_nvenc", "hevc_nvenc"} else "rgb24")
    cmd = [
        sys.executable,
        str(tool_path),
        "--input",
        str(task.input),
        "--engine",
        str(engine_path),
        "--output",
        str(task.output),
        "--input-width",
        str(task.info.width),
        "--input-height",
        str(task.info.height),
        "--decode-width",
        str(decode_width),
        "--decode-height",
        str(decode_height),
        "--fps",
        str(int(round(task.info.fps or 30))),
        "--target-width",
        str(target_width),
        "--target-height",
        str(target_height),
        "--content-width",
        str(content_width),
        "--encoder",
        video_encoder,
        "--bitrate",
        video_bitrate,
        "--output-pix-fmt",
        output_pix_fmt,
        *frames_arg,
    ]

    env = os.environ.copy()
    print(
        f"Start TRT-CUDA: input={task.input} output={task.output} "
        f"engine={engine_path} decode={decode_width}x{decode_height} "
        f"content_width={content_width} encoder={video_encoder} bitrate={video_bitrate} pix_fmt={output_pix_fmt}",
        flush=True,
    )
    subprocess.run(cmd, env=env, check=True)
    validate_dimensions(task.output, target_width, target_height)
