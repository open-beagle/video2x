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
    engine_width: int
    engine_height: int
    decode_width: int
    decode_height: int
    content_width: int


def output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}.mp4")


def choose_model(height: int, override: str) -> str:
    if override and override != "auto":
        return override
    return "realesr-general-x4v3"


def scale_for_height(width: int, height: int, target_height: int, override: str | None) -> float:
    if override:
        return float(override)

    target_width = round(width * target_height / height)
    if target_width % 2:
        target_width += 1
    return (target_width + 0.01) / width


def should_skip_output(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists()


def content_width_for_height(input_width: int, input_height: int, target_height: int) -> int:
    width = round(input_width * target_height / input_height)
    if width % 2:
        width += 1
    return width


def available_engines(model_dir: Path, model: str) -> set[tuple[int, int]]:
    prefix = f"{model}-"
    suffix = "-fp16.engine"
    engines: set[tuple[int, int]] = set()
    if not model_dir.is_dir():
        return engines
    for path in model_dir.glob(f"{prefix}*{suffix}"):
        size = path.name[len(prefix) : -len(suffix)]
        try:
            width_text, height_text = size.split("x", 1)
            engines.add((int(width_text), int(height_text)))
        except ValueError:
            continue
    return engines


def choose_engine_size(
    input_width: int,
    input_height: int,
    target_width: int,
    target_height: int,
    engines: set[tuple[int, int]],
) -> tuple[int, int] | None:
    aspect_tolerance = 0.02
    max_pre_scale = 1.25

    standard_profiles = [
        (640, 360),
        (720, 480),
        (854, 480),
        (960, 540),
        (1280, 720),
    ]

    if (input_width, input_height) in engines:
        return input_width, input_height

    candidates: list[tuple[float, int, int]] = []
    input_ratio = input_width / input_height
    for width, height in standard_profiles:
        if (width, height) not in engines:
            continue
        if width * 4 < target_width or height * 4 < target_height:
            continue
        ratio = width / height
        ratio_error = abs(ratio - input_ratio) / input_ratio
        if ratio_error > aspect_tolerance:
            continue
        pre_scale = max(width / input_width, height / input_height)
        if pre_scale > max_pre_scale:
            continue
        scale_change = abs(1.0 - min(width / input_width, height / input_height))
        candidates.append((ratio_error * 10.0 + scale_change, width, height))

    if not candidates:
        return None
    _, width, height = min(candidates)
    return width, height
