from __future__ import annotations

import subprocess
from pathlib import Path


MODEL_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def ensure_model(model_dir: Path, model: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / f"{model}.pth"
    if target.exists():
        return target

    url = MODEL_URLS.get(model)
    if not url:
        raise RuntimeError(f"model not found: {target}. Mount it at {target}.")

    print(f"Model missing, downloading {target.name} to {model_dir}", flush=True)
    subprocess.run(["wget", "-O", str(target), url], check=True)
    return target
