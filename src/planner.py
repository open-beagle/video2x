from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ffprobe import VideoInfo


@dataclass(frozen=True)
class Task:
    input: Path
    output: Path
    info: VideoInfo
    model: str
    outscale: float
    tile: int


def output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}.mp4")


def choose_model(height: int, override: str) -> str:
    if override and override != "auto":
        return override
    if 360 <= height < 540:
        return "RealESRGAN_x4plus"
    return "RealESRGAN_x2plus"


def scale_for_height(height: int, target_height: int, override: str | None) -> float:
    if override:
        return float(override)
    return target_height / height


def should_skip_output(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists()
