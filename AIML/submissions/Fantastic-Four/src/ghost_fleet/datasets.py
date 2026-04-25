from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .metadata import Annotation
from .scene_io import extract_center_crop_from_paths, find_scene_paths


LENGTH_SCALE_M = 200.0
DEFAULT_CHANNEL_NAMES = (
    "vv",
    "vh",
    "vv_minus_vh",
    "depth",
    "wind_speed",
    "owi_mask",
)


def compute_annotation_sampling_weights(
    annotations: list[Annotation],
) -> list[float]:
    labeled_vessel = [annotation for annotation in annotations if annotation.is_vessel is not None]
    positive_count = sum(annotation.is_vessel is True for annotation in labeled_vessel)
    negative_count = sum(annotation.is_vessel is False for annotation in labeled_vessel)

    positive_weight = len(labeled_vessel) / max(1, 2 * positive_count)
    negative_weight = len(labeled_vessel) / max(1, 2 * negative_count)

    weights: list[float] = []
    for annotation in annotations:
        weight = annotation.confidence_weight
        if annotation.is_vessel is True:
            weight *= positive_weight
        elif annotation.is_vessel is False:
            weight *= negative_weight
        else:
            weight *= 0.75
        weights.append(float(weight))
    return weights


def build_weighted_sampler(annotations: list[Annotation]) -> WeightedRandomSampler:
    weights = torch.tensor(compute_annotation_sampling_weights(annotations), dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class XView3PatchDataset(Dataset):
    def __init__(
        self,
        annotations: list[Annotation],
        scene_root: Path,
        crop_size: int,
        *,
        augment: bool,
        channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
    ) -> None:
        if not annotations:
            raise ValueError("No annotations were provided to the dataset.")
        self.annotations = annotations
        self.scene_root = scene_root
        self.crop_size = crop_size
        self.augment = augment
        self.channel_names = tuple(channel_names)
        scene_ids = sorted({annotation.scene_id for annotation in annotations})
        self.scene_paths = {
            scene_id: find_scene_paths(scene_root, scene_id) for scene_id in scene_ids
        }
        labeled_vessel = [annotation for annotation in annotations if annotation.is_vessel is not None]
        positive_vessel = sum(annotation.is_vessel is True for annotation in labeled_vessel)
        negative_vessel = sum(annotation.is_vessel is False for annotation in labeled_vessel)
        self.vessel_positive_weight = len(labeled_vessel) / max(1, 2 * positive_vessel)
        self.vessel_negative_weight = len(labeled_vessel) / max(1, 2 * negative_vessel)

        labeled_fishing = [annotation for annotation in annotations if annotation.is_fishing is not None]
        positive_fishing = sum(annotation.is_fishing is True for annotation in labeled_fishing)
        negative_fishing = sum(annotation.is_fishing is False for annotation in labeled_fishing)
        self.fishing_positive_weight = len(labeled_fishing) / max(1, 2 * positive_fishing)
        self.fishing_negative_weight = len(labeled_fishing) / max(1, 2 * negative_fishing)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, object]:
        annotation = self.annotations[index]
        scene_paths = self.scene_paths[annotation.scene_id]
        crop = extract_center_crop_from_paths(
            scene_paths,
            annotation.detect_scene_row,
            annotation.detect_scene_column,
            self.crop_size,
            channel_names=self.channel_names,
        )
        if self.augment:
            transform_case = int(np.random.randint(0, 8))
            if transform_case & 1:
                crop = crop[:, :, ::-1]
            if transform_case & 2:
                crop = crop[:, ::-1, :]
            if transform_case & 4:
                crop = np.transpose(crop, (0, 2, 1))

            if np.random.rand() < 0.3:
                noise = np.random.normal(loc=0.0, scale=0.01, size=crop.shape).astype(np.float32)
                crop = np.clip(crop + noise, 0.0, 1.0)

        image = torch.from_numpy(np.ascontiguousarray(crop)).float()

        vessel_mask = 1.0 if annotation.is_vessel is not None else 0.0
        fishing_mask = 1.0 if annotation.is_fishing is not None else 0.0
        length_mask = 1.0 if annotation.vessel_length_m is not None else 0.0
        vessel_task_weight = 1.0
        if annotation.is_vessel is True:
            vessel_task_weight = float(self.vessel_positive_weight)
        elif annotation.is_vessel is False:
            vessel_task_weight = float(self.vessel_negative_weight)

        fishing_task_weight = 1.0
        if annotation.is_fishing is True:
            fishing_task_weight = float(self.fishing_positive_weight)
        elif annotation.is_fishing is False:
            fishing_task_weight = float(self.fishing_negative_weight)

        return {
            "image": image,
            "vessel_target": torch.tensor(
                0.0 if annotation.is_vessel is None else float(annotation.is_vessel),
                dtype=torch.float32,
            ),
            "vessel_mask": torch.tensor(vessel_mask, dtype=torch.float32),
            "fishing_target": torch.tensor(
                0.0 if annotation.is_fishing is None else float(annotation.is_fishing),
                dtype=torch.float32,
            ),
            "fishing_mask": torch.tensor(fishing_mask, dtype=torch.float32),
            "length_target": torch.tensor(
                0.0
                if annotation.vessel_length_m is None
                else float(annotation.vessel_length_m) / LENGTH_SCALE_M,
                dtype=torch.float32,
            ),
            "length_mask": torch.tensor(length_mask, dtype=torch.float32),
            "weight": torch.tensor(annotation.confidence_weight, dtype=torch.float32),
            "vessel_task_weight": torch.tensor(vessel_task_weight, dtype=torch.float32),
            "fishing_task_weight": torch.tensor(fishing_task_weight, dtype=torch.float32),
            "scene_id": annotation.scene_id,
            "detect_id": annotation.detect_id,
        }
