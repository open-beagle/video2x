from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REALESRGAN_HOME = Path(os.environ.get("REALESRGAN_HOME", ROOT / "vendor" / "realesrgan"))
if not REALESRGAN_HOME.exists() and (ROOT / "realesrgan").exists():
    REALESRGAN_HOME = ROOT / "realesrgan"
sys.path.insert(0, str(REALESRGAN_HOME))


def ensure_torchvision_functional_tensor() -> None:
    import torchvision.transforms

    transforms_dir = Path(torchvision.transforms.__file__).resolve().parent
    shim = transforms_dir / "functional_tensor.py"
    if not shim.exists():
        shim.write_text(
            "from torchvision.transforms.functional import rgb_to_grayscale\n",
            encoding="utf-8",
        )


ensure_torchvision_functional_tensor()

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # noqa: E402


MODEL_CONFIGS = {
    "RealESRGAN_x2plus": {
        "arch": "rrdbnet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 2,
    },
    "RealESRGAN_x4plus": {
        "arch": "rrdbnet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 4,
    },
    "realesr-general-x4v3": {
        "arch": "srvgg",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_conv": 32,
        "upscale": 4,
        "act_type": "prelu",
    },
    "realesr-general-wdn-x4v3": {
        "arch": "srvgg",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_conv": 32,
        "upscale": 4,
        "act_type": "prelu",
    },
}


def build_model(name: str) -> torch.nn.Module:
    config = MODEL_CONFIGS[name].copy()
    arch = config.pop("arch")
    if arch == "rrdbnet":
        return RRDBNet(**config)
    if arch == "srvgg":
        return SRVGGNetCompact(**config)
    raise ValueError(f"unsupported model arch: {arch}")


def parse_shape(value: str) -> tuple[int, int, int, int]:
    parts = [int(part) for part in value.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be N,C,H,W, for example 1,3,420,720")
    if parts[1] != 3:
        raise argparse.ArgumentTypeError("only 3-channel RGB input is supported")
    return tuple(parts)  # type: ignore[return-value]


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f"unsupported checkpoint format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN SRVGG model to ONNX.")
    parser.add_argument("--model", default="realesr-general-x4v3", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--weights", type=Path, default=Path("/models/realesr-general-x4v3.pth"))
    parser.add_argument("--output", type=Path, default=Path("/models/realesr-general-x4v3.onnx"))
    parser.add_argument("--input-shape", type=parse_shape, default=parse_shape("1,3,420,720"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="export with FP16 weights and dummy input")
    parser.add_argument("--dynamic", action="store_true", help="export dynamic H/W axes")
    args = parser.parse_args()

    model = build_model(args.model)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy = torch.randn(*args.input_shape)
    if args.fp16:
        model = model.half()
        dummy = dummy.half()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"ONNX exported: {args.output}", flush=True)
    print(f"model={args.model}", flush=True)
    print(f"weights={args.weights}", flush=True)
    print(f"input_shape={','.join(map(str, args.input_shape))}", flush=True)
    print(f"fp16={args.fp16}", flush=True)
    print(f"dynamic={args.dynamic}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
