from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from ffprobe import validate_height
from gpu import gpu_status
from planner import Task


def _monitor(stop: threading.Event, task: Task, started: float, interval: int) -> None:
    while not stop.wait(interval):
        elapsed = int(time.time() - started)
        print(
            f"Progress: input={task.input} elapsed={elapsed}s frames={task.info.frames or 'unknown'} {gpu_status()}",
            flush=True,
        )


def run_task(
    task: Task,
    realesrgan_home: Path,
    model_dir: Path,
    target_height: int,
    output_suffix: str,
    progress_interval: int,
    python_bin: str,
    benchmark_frames: str | None,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        work_input = task.input
        if benchmark_frames:
            work_input = tmpdir / "benchmark.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(task.input),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-frames:v",
                    benchmark_frames,
                    str(work_input),
                ],
                check=True,
            )

        outdir = tmpdir / "out"
        outdir.mkdir()
        suffix_name = output_suffix.removeprefix("_")
        started = time.time()
        stop = threading.Event()
        monitor = threading.Thread(target=_monitor, args=(stop, task, started, progress_interval), daemon=True)

        print(
            f"Start: input={task.input} output={task.output} model={task.model} "
            f"outscale={task.outscale:g} tile={task.tile} {gpu_status()}",
            flush=True,
        )
        monitor.start()
        try:
            env = os.environ.copy()
            env["MODEL_DIR"] = str(model_dir)
            subprocess.run(
                [
                    python_bin,
                    "inference_realesrgan_video.py",
                    "-i",
                    str(work_input),
                    "-o",
                    str(outdir),
                    "-n",
                    task.model,
                    "-s",
                    f"{task.outscale:g}",
                    "--suffix",
                    suffix_name,
                    "--model_dir",
                    str(model_dir),
                    "--tile",
                    str(task.tile),
                ],
                cwd=realesrgan_home,
                env=env,
                check=True,
            )
        finally:
            stop.set()
            monitor.join(timeout=2)

        produced = outdir / f"{work_input.stem}_{suffix_name}.mp4"
        if not produced.exists():
            raise RuntimeError(f"Real-ESRGAN output not found: {produced}")
        shutil.move(str(produced), str(task.output))

        elapsed = max(time.time() - started, 0.001)
        if task.info.frames:
            print(f"Done speed: {task.info.frames / elapsed:.3f} fps ({task.info.frames} frames / {elapsed:.0f}s)", flush=True)
        else:
            print(f"Done elapsed: {elapsed:.0f}s", flush=True)

    validate_height(task.output, target_height)
