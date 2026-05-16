from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float | None
    frames: int | None
    duration: float | None


def _run_json(args: list[str]) -> dict:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _parse_fps(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_int(value: object) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: object) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> VideoInfo:
    data = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("no video stream")

    stream = streams[0]
    frames = _parse_int(stream.get("nb_read_frames")) or _parse_int(stream.get("nb_frames"))
    return VideoInfo(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=_parse_fps(stream.get("avg_frame_rate")),
        frames=frames,
        duration=_parse_float((data.get("format") or {}).get("duration")),
    )


def validate_height(path: Path, expected_height: int) -> None:
    info = probe_video(path)
    if info.height != expected_height:
        raise RuntimeError(f"output height {info.height} != {expected_height}: {path}")
