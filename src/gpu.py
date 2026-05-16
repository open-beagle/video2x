from __future__ import annotations

import os
import subprocess


def gpu_status() -> str:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "gpu=unknown memory=unknown"

    first = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return "gpu=unknown memory=unknown"
    return f"gpu={parts[0]}% memory={parts[1]}MiB/{parts[2]}MiB"
