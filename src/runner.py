from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ffprobe import validate_dimensions
from planner import Task


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
    pipeline_depth: int,
    video_gop_size: int,
    video_input_mode: str,
    cuda_p010_bridge: Path,
    video_output_mode: str,
    cuda_nvenc_bridge: Path,
    video_postprocess_mode: str,
    srvgg_tail_weights: Path | None,
    srvgg_tail_kernel: str,
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
        str(task.decode_width),
        "--decode-height",
        str(task.decode_height),
        "--fps",
        str(int(round(task.info.fps or 30))),
        "--target-width",
        str(target_width),
        "--target-height",
        str(target_height),
        "--content-width",
        str(task.content_width),
        "--encoder",
        video_encoder,
        "--bitrate",
        video_bitrate,
        "--gop-size",
        str(video_gop_size),
        "--output-pix-fmt",
        output_pix_fmt,
        "--pipeline-depth",
        str(pipeline_depth),
        "--input-mode",
        video_input_mode,
        "--cuda-p010-bridge",
        str(cuda_p010_bridge),
        "--output-mode",
        video_output_mode,
        "--cuda-nvenc-bridge",
        str(cuda_nvenc_bridge),
        "--postprocess-mode",
        video_postprocess_mode,
        "--srvgg-tail-kernel",
        srvgg_tail_kernel,
        *frames_arg,
    ]
    if srvgg_tail_weights:
        cmd += ["--srvgg-tail-weights", str(srvgg_tail_weights)]

    env = os.environ.copy()
    print(
        f"Start TRT-CUDA: input={task.input} output={task.output} "
        f"engine={engine_path} decode={task.decode_width}x{task.decode_height} "
        f"content_width={task.content_width} encoder={video_encoder} bitrate={video_bitrate} "
        f"pix_fmt={output_pix_fmt} gop={video_gop_size} pipeline_depth={pipeline_depth} "
        f"input_mode={video_input_mode} output_mode={video_output_mode} postprocess_mode={video_postprocess_mode}",
        flush=True,
    )
    subprocess.run(cmd, env=env, check=True)
    validate_dimensions(task.output, target_width, target_height)
