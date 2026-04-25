from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .datasets import LENGTH_SCALE_M


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return F.gelu(x)


class GhostFleetPatchNet(nn.Module):
    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, stride=2),
            ResidualBlock(192, 192),
        )
        self.embedding = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.vessel_head = nn.Linear(128, 1)
        self.fishing_head = nn.Linear(128, 1)
        self.length_head = nn.Linear(128, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        avg_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        max_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        features = self.embedding(torch.cat([avg_features, max_features], dim=1))
        return {
            "vessel_logits": self.vessel_head(features).squeeze(-1),
            "fishing_logits": self.fishing_head(features).squeeze(-1),
            "length_scaled": F.softplus(self.length_head(features).squeeze(-1)),
        }


class MultiTaskPredictionHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 384),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.vessel_head = nn.Linear(128, 1)
        self.fishing_head = nn.Linear(128, 1)
        self.length_head = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.embedding(features)
        return {
            "vessel_logits": self.vessel_head(features).squeeze(-1),
            "fishing_logits": self.fishing_head(features).squeeze(-1),
            "length_scaled": F.softplus(self.length_head(features).squeeze(-1)),
        }


class TerraMindSmallPatchNet(nn.Module):
    """TerraMind-small Sentinel-1 encoder plus small xView3 task heads.

    The xView3 scene loader keeps VV and VH as the first two channels. TerraMind
    consumes those SAR channels as Sentinel-1 GRD-style input. Extra channels are
    optional, but the recommended TerraMind-only competition run uses just VV/VH.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        modality: str = "untok_sen1grd@224",
        input_size: int = 224,
        sar_input: str = "xview3-normalized",
    ) -> None:
        super().__init__()
        if in_channels < 2:
            raise ValueError("TerraMind-small mode requires VV and VH as the first two channels.")
        if sar_input not in {"xview3-normalized", "db", "standardized", "as-is"}:
            raise ValueError(f"Unsupported TerraMind SAR input mode: {sar_input}")

        self.in_channels = in_channels
        self.modality = modality
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        self.sar_input = sar_input
        self.register_buffer(
            "s1_mean",
            torch.tensor([-12.599, -20.293], dtype=torch.float32).view(1, 2, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "s1_std",
            torch.tensor([5.195, 5.890], dtype=torch.float32).view(1, 2, 1, 1),
            persistent=False,
        )
        self.terramind = _build_terramind_small(
            pretrained=pretrained,
            modality=modality,
        )
        self.terramind_feature_dim = int(getattr(self.terramind, "out_channels", [384])[-1])

        if freeze_backbone:
            for parameter in self.terramind.parameters():
                parameter.requires_grad_(False)
            self.terramind.eval()

        aux_channels = max(0, in_channels - 2)
        self.aux_encoder: nn.Module | None = None
        aux_features = 0
        if aux_channels:
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(aux_channels, 24, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.GELU(),
                ResidualBlock(24, 48, stride=2),
                ResidualBlock(48, 48),
            )
            aux_features = 48 * 2

        self.head = MultiTaskPredictionHead(self.terramind_feature_dim * 2 + aux_features)

    def train(self, mode: bool = True) -> "TerraMindSmallPatchNet":
        super().train(mode)
        if self.freeze_backbone:
            self.terramind.eval()
        return self

    def _prepare_sar_for_terramind(self, sar: torch.Tensor) -> torch.Tensor:
        if self.sar_input in {"standardized", "as-is"}:
            return sar

        if self.sar_input == "xview3-normalized":
            vv_db = sar[:, 0:1] * 45.0 - 40.0
            vh_db = sar[:, 1:2] * 45.0 - 45.0
            sar_db = torch.cat([vv_db, vh_db], dim=1)
        else:
            sar_db = sar

        return (sar_db - self.s1_mean.to(device=sar.device, dtype=sar.dtype)) / self.s1_std.to(
            device=sar.device,
            dtype=sar.dtype,
        )

    def _extract_terramind_features(self, sar: torch.Tensor) -> torch.Tensor:
        sar = self._prepare_sar_for_terramind(sar)
        if sar.shape[-2:] != (self.input_size, self.input_size):
            sar = F.interpolate(
                sar,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.terramind({self.modality: sar})
        else:
            features = self.terramind({self.modality: sar})

        tokens = _last_feature_tensor(features)
        if tokens.ndim == 4:
            avg_features = F.adaptive_avg_pool2d(tokens, 1).flatten(1)
            max_features = F.adaptive_max_pool2d(tokens, 1).flatten(1)
        elif tokens.ndim == 3:
            avg_features = tokens.mean(dim=1)
            max_features = tokens.max(dim=1).values
        else:
            raise RuntimeError(f"Unexpected TerraMind feature shape: {tuple(tokens.shape)}")
        return torch.cat([avg_features, max_features], dim=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        sar = image[:, :2]
        features = [self._extract_terramind_features(sar)]

        if self.aux_encoder is not None:
            aux = self.aux_encoder(image[:, 2:])
            aux_avg = F.adaptive_avg_pool2d(aux, 1).flatten(1)
            aux_max = F.adaptive_max_pool2d(aux, 1).flatten(1)
            features.append(torch.cat([aux_avg, aux_max], dim=1))

        return self.head(torch.cat(features, dim=1))

    def state_dict_for_checkpoint(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        if not self.freeze_backbone:
            return state
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("terramind.")
        }


def _build_terramind_small(*, pretrained: bool, modality: str) -> nn.Module:
    try:
        import terratorch.models.backbones.terramind  # noqa: F401
        from terratorch.registry import BACKBONE_REGISTRY
    except Exception as exc:  # pragma: no cover - depends on optional package install.
        raise RuntimeError(
            "TerraMind-small mode requires terratorch. Install the pinned "
            "requirements or train with --backbone sar-cnn."
        ) from exc

    try:
        return BACKBONE_REGISTRY.build(
            "terratorch_terramind_v1_small",
            pretrained=pretrained,
            modalities=[modality],
            merge_method="mean",
        )
    except Exception as exc:  # pragma: no cover - download/auth errors are environment-specific.
        raise RuntimeError(
            "Could not build TerraMind-small. If the Hugging Face download is "
            "unavailable, retry with --no-terramind-pretrained for a smoke test "
            "or use --backbone sar-cnn for the non-TerraMind baseline."
        ) from exc


def _last_feature_tensor(features: Any) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "output"):
        return _last_feature_tensor(features.output)
    if isinstance(features, dict):
        if "out" in features:
            return _last_feature_tensor(features["out"])
        if "features" in features:
            return _last_feature_tensor(features["features"])
        return _last_feature_tensor(next(reversed(features.values())))
    if isinstance(features, (list, tuple)):
        return _last_feature_tensor(features[-1])
    raise TypeError(f"Unsupported TerraMind feature type: {type(features)!r}")


def build_model(
    *,
    backbone: str,
    in_channels: int,
    terramind_pretrained: bool = True,
    terramind_freeze: bool = True,
    terramind_modality: str = "untok_sen1grd@224",
    terramind_input_size: int = 224,
    terramind_sar_input: str = "xview3-normalized",
) -> nn.Module:
    if backbone == "sar-cnn":
        return GhostFleetPatchNet(in_channels=in_channels)
    if backbone == "terramind-small":
        return TerraMindSmallPatchNet(
            in_channels=in_channels,
            pretrained=terramind_pretrained,
            freeze_backbone=terramind_freeze,
            modality=terramind_modality,
            input_size=terramind_input_size,
            sar_input=terramind_sar_input,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def masked_average(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = loss * mask
    denominator = mask.sum().clamp(min=1.0)
    return weighted.sum() / denominator


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    vessel_mask = batch["vessel_mask"] * batch["weight"] * batch["vessel_task_weight"]
    fishing_mask = batch["fishing_mask"] * batch["weight"] * batch["fishing_task_weight"]
    length_mask = batch["length_mask"] * batch["weight"]

    vessel_loss = F.binary_cross_entropy_with_logits(
        outputs["vessel_logits"],
        batch["vessel_target"],
        reduction="none",
    )
    fishing_loss = F.binary_cross_entropy_with_logits(
        outputs["fishing_logits"],
        batch["fishing_target"],
        reduction="none",
    )
    length_loss = F.smooth_l1_loss(
        outputs["length_scaled"],
        batch["length_target"],
        reduction="none",
    )

    vessel_loss = masked_average(vessel_loss, vessel_mask)
    fishing_loss = masked_average(fishing_loss, fishing_mask)
    length_loss = masked_average(length_loss, length_mask)
    total_loss = 1.5 * vessel_loss + 0.35 * fishing_loss + 0.15 * length_loss

    return total_loss, {
        "vessel_loss": float(vessel_loss.detach().cpu()),
        "fishing_loss": float(fishing_loss.detach().cpu()),
        "length_loss": float(length_loss.detach().cpu()),
    }


def length_scaled_to_meters(length_scaled: torch.Tensor) -> torch.Tensor:
    return length_scaled * LENGTH_SCALE_M


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    config: dict[str, object],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "state_dict_for_checkpoint"):
        model_state = model.state_dict_for_checkpoint()
    else:
        model_state = model.state_dict()
    torch.save(
        {
            "model_state": model_state,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    in_channels = int(config.get("in_channels", len(config.get("channel_names", [])) or 2))
    backbone = str(config.get("backbone", "sar-cnn"))
    model = build_model(
        backbone=backbone,
        in_channels=in_channels,
        terramind_pretrained=bool(config.get("terramind_pretrained", True)),
        terramind_freeze=bool(config.get("terramind_freeze", True)),
        terramind_modality=str(config.get("terramind_modality", "untok_sen1grd@224")),
        terramind_input_size=int(config.get("terramind_input_size", 224)),
        terramind_sar_input=str(config.get("terramind_sar_input", "as-is")),
    )
    strict = backbone == "sar-cnn" or not bool(config.get("terramind_freeze", True))
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    model.to(device)
    model.eval()
    return model, checkpoint
