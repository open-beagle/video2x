from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv"})


def find_videos(data_dir: Path, output_suffix: str) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        if "_1080p" not in path.stem.lower()
        and not path.name.lower().endswith(f"{output_suffix.lower()}.mp4")
    )


def find_mp4(data_dir: Path, output_suffix: str) -> list[Path]:
    return find_videos(data_dir, output_suffix)
