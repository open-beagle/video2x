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


def run_trt_cuda_task(
    task: Task,
    engine_path: Path,
    target_width: int,
    target_height: int,
    benchmark_frames: str | None,
    video_encoder: str,
    tool_path: Path,
) -> None:
    if not engine_path.exists():
        raise RuntimeError(f"TRT engine not found: {engine_path}")
    if not tool_path.exists():
        raise RuntimeError(f"TRT CUDA runner tool not found: {tool_path}")

    frames_arg: list[str] = []
    if benchmark_frames:
        frames_arg = ["--frames", benchmark_frames]

    content_width = _content_width(task.info.width, task.info.height, target_height)
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
        *frames_arg,
    ]

    env = os.environ.copy()
    print(
        f"Start TRT-CUDA: input={task.input} output={task.output} "
        f"engine={engine_path} content_width={content_width} encoder={video_encoder}",
        flush=True,
    )
    subprocess.run(cmd, env=env, check=True)
    validate_dimensions(task.output, target_width, target_height)
