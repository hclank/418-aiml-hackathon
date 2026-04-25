from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from src.ghost_fleet.model import build_model, load_checkpoint


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_report(model: torch.nn.Module) -> tuple[int, float]:
    params = sum(parameter.numel() for parameter in model.parameters())
    fp16_mb = params * 2 / 1024 / 1024
    return params, fp16_mb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Ghost Fleet model size and rough batch-1 latency."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--backbone",
        choices=("terramind-small", "sar-cnn"),
        default="terramind-small",
        help="Used only when --checkpoint is omitted.",
    )
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", help="Torch device override, for example cuda or cpu")
    parser.add_argument(
        "--no-terramind-pretrained",
        action="store_true",
        help="Avoid downloading TerraMind weights for local smoke profiling.",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.checkpoint is not None:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        config = checkpoint.get("config", {})
        channels = int(config.get("in_channels", args.channels))
        crop_size = int(config.get("crop_size", args.crop_size))
        backbone = str(config.get("backbone", "sar-cnn"))
    else:
        channels = args.channels
        crop_size = args.crop_size
        backbone = args.backbone
        model = build_model(
            backbone=backbone,
            in_channels=channels,
            terramind_pretrained=not args.no_terramind_pretrained,
        ).to(device)

    model.eval()
    example = torch.rand(1, channels, crop_size, crop_size, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(example)
        _sync(device)

        latencies_ms: list[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            model(example)
            _sync(device)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    params, fp16_mb = _parameter_report(model)
    print(f"backbone={backbone}")
    print(f"device={device}")
    print(f"params={params:,}")
    print(f"estimated_fp16_weights_mb={fp16_mb:.1f}")
    print(f"batch1_latency_ms_median={statistics.median(latencies_ms):.1f}")
    print(f"batch1_latency_ms_mean={statistics.mean(latencies_ms):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
