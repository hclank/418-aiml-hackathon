from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ghost_fleet.datasets import (
    LENGTH_SCALE_M,
    DEFAULT_CHANNEL_NAMES,
    XView3PatchDataset,
    build_weighted_sampler,
)
from src.ghost_fleet.metrics import binary_metrics_at_threshold, sweep_best_threshold
from src.ghost_fleet.metadata import (
    filter_annotations_by_scene,
    filter_labeled_annotations,
    load_annotations,
    split_scene_ids,
)
from src.ghost_fleet.model import (
    build_model,
    compute_multitask_loss,
    save_checkpoint,
)
from src.ghost_fleet.scene_io import find_scene_paths
from src.ghost_fleet.scene_io import is_scene_readable
from src.ghost_fleet.scene_io import list_available_scene_ids


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _prepare_splits(args: argparse.Namespace):
    train_annotations = filter_labeled_annotations(load_annotations(args.train_csv))

    if args.train_on_validation:
        if args.val_csv is None:
            raise SystemExit("--train-on-validation requires --val-csv.")
        validation_annotations = filter_labeled_annotations(load_annotations(args.val_csv))
        combined_annotations = train_annotations + validation_annotations
        print(
            "Merged labeled annotations from train and validation CSVs: "
            f"{len(train_annotations)} + {len(validation_annotations)} = "
            f"{len(combined_annotations)}"
        )
        train_scene_ids, val_scene_ids = split_scene_ids(
            [annotation.scene_id for annotation in combined_annotations],
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        return (
            filter_annotations_by_scene(combined_annotations, train_scene_ids),
            filter_annotations_by_scene(combined_annotations, val_scene_ids),
        )

    if args.val_csv is not None:
        val_annotations = filter_labeled_annotations(load_annotations(args.val_csv))
        return train_annotations, val_annotations

    train_scene_ids, val_scene_ids = split_scene_ids(
        [annotation.scene_id for annotation in train_annotations],
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    return (
        filter_annotations_by_scene(train_annotations, train_scene_ids),
        filter_annotations_by_scene(train_annotations, val_scene_ids),
    )


def _verify_scene_root(scene_root: Path, annotations) -> None:
    sample_scene_ids = sorted({annotation.scene_id for annotation in annotations})[:3]
    for scene_id in sample_scene_ids:
        try:
            find_scene_paths(scene_root, scene_id)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc}\n"
                "Run verify_xview3.py first, then download/extract the scene archives "
                "into data/xview3/full before training."
            ) from exc


def _filter_unreadable_scenes(
    scene_root: Path,
    train_annotations,
    val_annotations,
    *,
    read_probe_size: int,
):
    scene_ids = sorted(
        {annotation.scene_id for annotation in train_annotations + val_annotations}
    )
    readable_scene_ids: set[str] = set()
    unreadable: list[tuple[str, str]] = []

    for scene_id in scene_ids:
        ok, reason = is_scene_readable(
            scene_root,
            scene_id,
            max_dim=read_probe_size,
        )
        if ok:
            readable_scene_ids.add(scene_id)
        else:
            unreadable.append((scene_id, reason or "unknown read error"))

    if not unreadable:
        print(f"Read probe passed for {len(readable_scene_ids)} scenes")
        return train_annotations, val_annotations

    print(f"Skipping {len(unreadable)} unreadable scene(s):")
    for scene_id, reason in unreadable[:20]:
        print(f"- {scene_id}: {reason}")
    if len(unreadable) > 20:
        print(f"- ... {len(unreadable) - 20} more")

    train_before = len(train_annotations)
    val_before = len(val_annotations)
    train_annotations = filter_annotations_by_scene(train_annotations, readable_scene_ids)
    val_annotations = filter_annotations_by_scene(val_annotations, readable_scene_ids)
    print(
        f"Filtered train annotations from {train_before} to {len(train_annotations)} "
        "after read probe"
    )
    print(
        f"Filtered validation annotations from {val_before} to {len(val_annotations)} "
        "after read probe"
    )
    if not train_annotations:
        raise SystemExit("No training annotations remain after removing unreadable scenes.")
    if not val_annotations:
        raise SystemExit("No validation annotations remain after removing unreadable scenes.")
    return train_annotations, val_annotations


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_batches = 0
    vessel_correct = 0.0
    vessel_total = 0.0
    fishing_correct = 0.0
    fishing_total = 0.0
    length_abs_error = 0.0
    length_total = 0.0
    vessel_scores: list[float] = []
    vessel_targets: list[int] = []
    fishing_scores: list[float] = []
    fishing_targets: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(batch["image"])
            loss, _ = compute_multitask_loss(outputs, batch)

            total_loss += float(loss.detach().cpu())
            total_batches += 1

            vessel_mask = batch["vessel_mask"] > 0
            if vessel_mask.any():
                vessel_prob = torch.sigmoid(outputs["vessel_logits"])
                vessel_pred = (vessel_prob >= 0.5).float()
                vessel_correct += float(
                    (vessel_pred[vessel_mask] == batch["vessel_target"][vessel_mask])
                    .float()
                    .sum()
                    .cpu()
                )
                vessel_total += float(vessel_mask.sum().cpu())
                vessel_scores.extend(vessel_prob[vessel_mask].detach().cpu().tolist())
                vessel_targets.extend(
                    batch["vessel_target"][vessel_mask].detach().cpu().int().tolist()
                )

            fishing_mask = batch["fishing_mask"] > 0
            if fishing_mask.any():
                fishing_prob = torch.sigmoid(outputs["fishing_logits"])
                fishing_pred = (fishing_prob >= 0.5).float()
                fishing_correct += float(
                    (fishing_pred[fishing_mask] == batch["fishing_target"][fishing_mask])
                    .float()
                    .sum()
                    .cpu()
                )
                fishing_total += float(fishing_mask.sum().cpu())
                fishing_scores.extend(fishing_prob[fishing_mask].detach().cpu().tolist())
                fishing_targets.extend(
                    batch["fishing_target"][fishing_mask].detach().cpu().int().tolist()
                )

            length_mask = batch["length_mask"] > 0
            if length_mask.any():
                predicted_length = outputs["length_scaled"][length_mask] * LENGTH_SCALE_M
                target_length = batch["length_target"][length_mask] * LENGTH_SCALE_M
                length_abs_error += float(
                    torch.abs(predicted_length - target_length).sum().cpu()
                )
                length_total += float(length_mask.sum().cpu())

    vessel_best = sweep_best_threshold(
        np.asarray(vessel_scores, dtype=np.float32),
        np.asarray(vessel_targets, dtype=np.int64),
    )
    vessel_default = binary_metrics_at_threshold(
        np.asarray(vessel_scores, dtype=np.float32),
        np.asarray(vessel_targets, dtype=np.int64),
        0.5,
    )

    metrics = {
        "loss": total_loss / max(1, total_batches),
        "vessel_accuracy": vessel_correct / max(1.0, vessel_total),
        "fishing_accuracy": fishing_correct / max(1.0, fishing_total),
        "length_mae_m": length_abs_error / max(1.0, length_total),
        "vessel_precision": vessel_default.precision,
        "vessel_recall": vessel_default.recall,
        "vessel_f1": vessel_default.f1,
        "vessel_balanced_accuracy": vessel_default.balanced_accuracy,
        "best_vessel_threshold": vessel_best.threshold,
        "best_vessel_f1": vessel_best.f1,
        "best_vessel_balanced_accuracy": vessel_best.balanced_accuracy,
    }
    if fishing_scores:
        fishing_best = sweep_best_threshold(
            np.asarray(fishing_scores, dtype=np.float32),
            np.asarray(fishing_targets, dtype=np.int64),
        )
        fishing_default = binary_metrics_at_threshold(
            np.asarray(fishing_scores, dtype=np.float32),
            np.asarray(fishing_targets, dtype=np.int64),
            0.5,
        )
        metrics.update(
            {
                "fishing_precision": fishing_default.precision,
                "fishing_recall": fishing_default.recall,
                "fishing_f1": fishing_default.f1,
                "best_fishing_threshold": fishing_best.threshold,
                "best_fishing_f1": fishing_best.f1,
            }
        )
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_batches = 0
    for batch in data_loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["image"])
        loss, _ = compute_multitask_loss(outputs, batch)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return {"loss": total_loss / max(1, total_batches)}


def build_optimizer(
    model: nn.Module,
    *,
    backbone: str,
    finetune_terramind: bool,
    lr: float,
    backbone_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if backbone == "terramind-small" and finetune_terramind and hasattr(model, "terramind"):
        terramind_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("terramind.")
        ]
        head_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("terramind.")
        ]
        parameter_groups = []
        if terramind_params:
            parameter_groups.append(
                {
                    "params": terramind_params,
                    "lr": backbone_lr if backbone_lr is not None else lr * 0.1,
                }
            )
        if head_params:
            parameter_groups.append({"params": head_params, "lr": lr})
        if not parameter_groups:
            raise SystemExit("No trainable parameters found. Check backbone/freeze settings.")
        return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise SystemExit("No trainable parameters found. Check backbone/freeze settings.")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Ghost Fleet SAR patch classifier on xView3 metadata."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/xview3/train.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/xview3/validation.csv"),
        help="Optional validation CSV. Omit to split train scenes by fraction.",
    )
    parser.add_argument(
        "--train-on-validation",
        action="store_true",
        help="Merge --train-csv and --val-csv, then create a fresh scene-level "
        "holdout using --val-fraction. Use this after downloading validation scenes.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("data/xview3/full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete --output-dir before training so metrics/checkpoints are from one clean run.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--crop-size",
        type=int,
        help="Patch size. Defaults to 224 for TerraMind and 128 for sar-cnn.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        help="Optional lower learning rate for TerraMind encoder parameters when fine-tuning.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", help="Torch device override, for example cuda or cpu")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--backbone",
        choices=("terramind-small", "sar-cnn"),
        default="terramind-small",
        help="Model backbone. terramind-small uses a frozen TerraMind-small SAR encoder.",
    )
    parser.add_argument(
        "--no-terramind-pretrained",
        action="store_true",
        help="Build TerraMind-small without downloading pretrained weights. "
        "Useful only for smoke tests; final runs should use pretrained weights.",
    )
    parser.add_argument(
        "--finetune-terramind",
        action="store_true",
        help="Unfreeze TerraMind-small. This is slower and saved checkpoints may exceed 200 MB.",
    )
    parser.add_argument(
        "--terramind-modality",
        default="untok_sen1grd@224",
        help="TerraMind SAR modality key. xView3 VV/VH maps best to untok_sen1grd@224.",
    )
    parser.add_argument(
        "--terramind-input-size",
        type=int,
        default=224,
        help="Spatial size fed into TerraMind-small after crop resizing.",
    )
    parser.add_argument(
        "--terramind-sar-input",
        choices=("xview3-normalized", "db", "standardized", "as-is"),
        default="xview3-normalized",
        help="How VV/VH tensors are encoded before TerraMind standardization.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Feature channels to use. Defaults to vv vh for TerraMind and richer xView3 channels for sar-cnn.",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Train only on scenes currently present under --scene-root. "
        "Use this with the tiny set.",
    )
    parser.add_argument(
        "--skip-corrupt-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe SAR rasters before training and remove scenes that cannot be read.",
    )
    parser.add_argument(
        "--scene-read-probe-size",
        type=int,
        default=512,
        help="Maximum downsampled dimension for the pre-training raster read probe.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.channels is None:
        args.channels = (
            ["vv", "vh"]
            if args.backbone == "terramind-small"
            else list(DEFAULT_CHANNEL_NAMES)
        )
    if args.crop_size is None:
        args.crop_size = 224 if args.backbone == "terramind-small" else 128

    train_annotations, val_annotations = _prepare_splits(args)
    if not train_annotations:
        raise SystemExit("Training annotations are empty after filtering.")
    if not val_annotations:
        raise SystemExit("Validation annotations are empty after filtering.")

    if args.available_only:
        available_scene_ids = list_available_scene_ids(args.scene_root)
        train_before = len(train_annotations)
        val_before = len(val_annotations)
        train_annotations = filter_annotations_by_scene(train_annotations, available_scene_ids)
        val_annotations = filter_annotations_by_scene(val_annotations, available_scene_ids)

        print(
            f"Available scenes under {args.scene_root}: {len(available_scene_ids)}"
        )
        print(
            f"Filtered train annotations from {train_before} to {len(train_annotations)}"
        )
        print(
            f"Filtered validation annotations from {val_before} to {len(val_annotations)}"
        )
        if not train_annotations:
            raise SystemExit(
                "No training annotations matched the currently extracted scenes. "
                "Check your tiny/full dataset extraction."
            )
        if not val_annotations:
            raise SystemExit(
                "No validation annotations matched the currently extracted scenes. "
                "Check your tiny/full dataset extraction."
            )
    else:
        _verify_scene_root(args.scene_root, train_annotations + val_annotations)

    if args.skip_corrupt_scenes:
        train_annotations, val_annotations = _filter_unreadable_scenes(
            args.scene_root,
            train_annotations,
            val_annotations,
            read_probe_size=args.scene_read_probe_size,
        )

    if args.backbone == "terramind-small":
        if len(args.channels) < 2 or tuple(args.channels[:2]) != ("vv", "vh"):
            raise SystemExit(
                "TerraMind-small mode expects --channels to start with 'vv vh' "
                "because those are passed to the Sentinel-1 SAR encoder."
            )

    train_dataset = XView3PatchDataset(
        train_annotations,
        scene_root=args.scene_root,
        crop_size=args.crop_size,
        augment=True,
        channel_names=tuple(args.channels),
    )
    val_dataset = XView3PatchDataset(
        val_annotations,
        scene_root=args.scene_root,
        crop_size=args.crop_size,
        augment=False,
        channel_names=tuple(args.channels),
    )

    train_sampler = build_weighted_sampler(train_annotations)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        backbone=args.backbone,
        in_channels=len(args.channels),
        terramind_pretrained=not args.no_terramind_pretrained,
        terramind_freeze=not args.finetune_terramind,
        terramind_modality=args.terramind_modality,
        terramind_input_size=args.terramind_input_size,
        terramind_sar_input=args.terramind_sar_input,
    ).to(device)
    optimizer = build_optimizer(
        model,
        backbone=args.backbone,
        finetune_terramind=args.finetune_terramind,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    best_val_loss = float("inf")
    best_default_vessel_f1 = float("-inf")
    if args.overwrite_output and args.output_dir.exists():
        resolved_output = args.output_dir.resolve()
        resolved_cwd = Path.cwd().resolve()
        if not resolved_output.is_relative_to(resolved_cwd):
            raise SystemExit(f"Refusing to delete output directory outside cwd: {resolved_output}")
        shutil.rmtree(resolved_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = args.output_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        merged_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_vessel_accuracy": val_metrics["vessel_accuracy"],
            "val_vessel_precision": val_metrics["vessel_precision"],
            "val_vessel_recall": val_metrics["vessel_recall"],
            "val_vessel_f1": val_metrics["vessel_f1"],
            "val_vessel_balanced_accuracy": val_metrics["vessel_balanced_accuracy"],
            "val_best_vessel_threshold": val_metrics["best_vessel_threshold"],
            "val_best_vessel_f1": val_metrics["best_vessel_f1"],
            "val_fishing_accuracy": val_metrics["fishing_accuracy"],
            "val_fishing_f1": val_metrics.get("fishing_f1", 0.0),
            "val_length_mae_m": val_metrics["length_mae_m"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(merged_metrics, indent=2))
        with metrics_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(merged_metrics) + "\n")

        checkpoint_config = {
            "backbone": args.backbone,
            "crop_size": args.crop_size,
            "scene_root": str(args.scene_root),
            "channel_names": list(args.channels),
            "in_channels": len(args.channels),
            "best_vessel_threshold": val_metrics["best_vessel_threshold"],
            "terramind_pretrained": not args.no_terramind_pretrained,
            "terramind_freeze": not args.finetune_terramind,
            "terramind_modality": args.terramind_modality,
            "terramind_input_size": args.terramind_input_size,
            "terramind_sar_input": args.terramind_sar_input,
            "backbone_lr": args.backbone_lr,
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv) if args.val_csv is not None else "",
            "train_on_validation": args.train_on_validation,
            "val_fraction": args.val_fraction,
            "checkpoint_note": (
                "Frozen TerraMind weights are excluded from this checkpoint "
                "and loaded from Hugging Face by terratorch at runtime."
                if args.backbone == "terramind-small" and not args.finetune_terramind
                else ""
            ),
        }
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            epoch=epoch,
            config=checkpoint_config,
            metrics=merged_metrics,
        )
        if val_metrics["vessel_f1"] > best_default_vessel_f1:
            best_default_vessel_f1 = val_metrics["vessel_f1"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                epoch=epoch,
                config=checkpoint_config,
                metrics=merged_metrics,
            )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                args.output_dir / "best_loss.pt",
                model=model,
                epoch=epoch,
                config=checkpoint_config,
                metrics=merged_metrics,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
