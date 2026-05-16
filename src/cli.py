from __future__ import annotations

import os
from pathlib import Path

from ffprobe import probe_video
from gpu import gpu_status
from planner import Task, choose_model, output_path, scale_for_height, should_skip_output
from scanner import find_mp4


def engine_shape(width: int, height: int) -> tuple[int, int]:
    if width == 1280 and height == 720:
        return 960, 540
    return width, height


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main() -> int:
    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    model_dir = Path(os.environ.get("MODEL_DIR", "/models"))
    target_width = int(os.environ.get("TARGET_WIDTH", "1920"))
    target_height = int(os.environ.get("TARGET_HEIGHT", "1080"))
    model_name = os.environ.get("MODEL_NAME", "auto")
    outscale_override = os.environ.get("OUTSCALE") or None
    tile = int(os.environ.get("TILE", "0"))
    gpu_id = os.environ.get("GPU_ID", "0")
    output_suffix = os.environ.get("OUTPUT_SUFFIX", "_1080p")
    skip_existing = env_bool("SKIP_EXISTING", True)
    benchmark_frames = os.environ.get("BENCHMARK_FRAMES") or None
    progress_interval = int(os.environ.get("PROGRESS_INTERVAL", "30"))
    dry_run = env_bool("DRY_RUN", False)
    runner = os.environ.get("RUNNER", "trt-cuda").lower()
    trt_engine_override = os.environ.get("TRT_ENGINE_PATH") or None
    video_encoder = os.environ.get("VIDEO_ENCODER", "libx265")
    video_bitrate = os.environ.get("VIDEO_BITRATE", "5M")
    video_pixel_format = os.environ.get("VIDEO_PIXEL_FORMAT") or None
    trt_cuda_tool = Path(os.environ.get("TRT_CUDA_TOOL", "/app/src/worker.py"))

    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: DATA_DIR does not exist: {data_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print(f"DATA_DIR={data_dir}", flush=True)
    print(f"MODEL_DIR={model_dir}", flush=True)
    print(f"TARGET_WIDTH={target_width}", flush=True)
    print(f"TARGET_HEIGHT={target_height}", flush=True)
    print(f"MODEL_NAME={model_name}", flush=True)
    print(f"GPU_ID={gpu_id}", flush=True)
    print(f"TILE={tile}", flush=True)
    print(f"OUTPUT_SUFFIX={output_suffix}", flush=True)
    print(f"SKIP_EXISTING={str(skip_existing).lower()}", flush=True)
    print(f"PROGRESS_INTERVAL={progress_interval}", flush=True)
    print(f"DRY_RUN={str(dry_run).lower()}", flush=True)
    print(f"RUNNER={runner}", flush=True)
    if runner != "trt-cuda":
        raise SystemExit(f"ERROR: unsupported RUNNER={runner}. This image only supports RUNNER=trt-cuda.")
    print(f"TRT_ENGINE_PATH={trt_engine_override or 'auto'}", flush=True)
    print(f"VIDEO_ENCODER={video_encoder}", flush=True)
    print(f"VIDEO_BITRATE={video_bitrate}", flush=True)
    print(f"VIDEO_PIXEL_FORMAT={video_pixel_format or 'auto'}", flush=True)
    print(f"TRT_CUDA_TOOL={trt_cuda_tool}", flush=True)
    print(f"GPU_STATUS={gpu_status()}", flush=True)

    inputs = find_mp4(data_dir, output_suffix)
    if not inputs:
        print("No input .mp4 files found.", flush=True)
        return 0

    tasks: list[Task] = []
    print("\nScan result:", flush=True)
    for index, input_path in enumerate(inputs, 1):
        output = output_path(input_path, output_suffix)
        if benchmark_frames:
            output = output.with_name(f"{output.stem}_benchmark.mp4")

        print(f"{index}. {input_path}", flush=True)
        if should_skip_output(output, skip_existing):
            print("   action: skip", flush=True)
            print("   reason: output exists", flush=True)
            print(f"   output: {output}", flush=True)
            continue

        try:
            info = probe_video(input_path)
        except Exception as exc:
            print("   action: skip", flush=True)
            print(f"   reason: unreadable video: {exc}", flush=True)
            continue

        fps = f"{info.fps:g}" if info.fps else "unknown"
        frames = str(info.frames) if info.frames else "unknown"
        duration = f"{info.duration:.6f}s" if info.duration else "unknown"
        print(f"   input: {info.width}x{info.height}, {fps}fps, {frames} frames, {duration}", flush=True)

        if info.height >= target_height:
            print("   action: skip", flush=True)
            print(f"   reason: already {target_height}p or higher", flush=True)
            continue

        model = choose_model(info.height, model_name)
        if model_name == "auto" and 360 <= info.height < 540:
            model = "realesr-general-x4v3"
        scale = scale_for_height(info.width, info.height, target_height, outscale_override)
        task = Task(input=input_path, output=output, info=info, model=model, outscale=scale, tile=tile)
        tasks.append(task)

        print(f"   output: {output}", flush=True)
        print("   action: upscale", flush=True)
        print(f"   model: {model}", flush=True)
        print(f"   outscale: {scale:g}", flush=True)
        print(f"   tile: {tile}", flush=True)
        print("   estimated time: after start", flush=True)

    if not tasks:
        print("\nNo videos need processing.", flush=True)
        return 0

    print(f"\nTask plan: {len(tasks)} video(s) will be processed.", flush=True)
    for index, task in enumerate(tasks, 1):
        print(f"{index}. {task.input}", flush=True)
        print(f"   input: {task.info.width}x{task.info.height}, {task.info.frames or 'unknown'} frames", flush=True)
        print(f"   output: {task.output}", flush=True)
        print(f"   model: {task.model}", flush=True)
        print(f"   outscale: {task.outscale:g}", flush=True)

    if dry_run:
        print("\nDRY_RUN=true, stop after scan and task planning.", flush=True)
        return 0

    for index, task in enumerate(tasks, 1):
        print(f"\nProcessing {index}/{len(tasks)}: {task.input}", flush=True)
        from runner import run_trt_cuda_task

        engine_width, engine_height = engine_shape(task.info.width, task.info.height)
        trt_engine_path = (
            Path(trt_engine_override)
            if trt_engine_override
            else model_dir / f"{task.model}-{engine_width}x{engine_height}-fp16.engine"
        )
        run_trt_cuda_task(
            task,
            trt_engine_path,
            target_width,
            target_height,
            benchmark_frames,
            video_encoder,
            video_bitrate,
            video_pixel_format,
            trt_cuda_tool,
        )

    print(f"\nDone. processed tasks: {len(tasks)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
