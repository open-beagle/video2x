from __future__ import annotations

from pathlib import Path


def find_mp4(data_dir: Path, output_suffix: str) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*.mp4")
        if not path.name.lower().endswith(f"{output_suffix.lower()}.mp4")
    )
