from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CONFIDENCE_WEIGHTS = {
    "LOW": 0.4,
    "MEDIUM": 0.7,
    "HIGH": 1.0,
}


@dataclass(frozen=True)
class Annotation:
    detect_id: str
    scene_id: str
    detect_lat: float
    detect_lon: float
    detect_scene_row: int
    detect_scene_column: int
    source: str
    confidence: str
    is_vessel: bool | None
    is_fishing: bool | None
    vessel_length_m: float | None
    distance_from_shore_km: float | None
    top: float | None
    left: float | None
    bottom: float | None
    right: float | None

    @property
    def has_any_label(self) -> bool:
        return (
            self.is_vessel is not None
            or self.is_fishing is not None
            or self.vessel_length_m is not None
        )

    @property
    def confidence_weight(self) -> float:
        return CONFIDENCE_WEIGHTS.get(self.confidence.upper(), 0.5)


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_annotations(path: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            annotations.append(
                Annotation(
                    detect_id=row["detect_id"],
                    scene_id=row["scene_id"],
                    detect_lat=float(row["detect_lat"]),
                    detect_lon=float(row["detect_lon"]),
                    detect_scene_row=int(float(row["detect_scene_row"])),
                    detect_scene_column=int(float(row["detect_scene_column"])),
                    source=row["source"],
                    confidence=row["confidence"],
                    is_vessel=_parse_optional_bool(row.get("is_vessel")),
                    is_fishing=_parse_optional_bool(row.get("is_fishing")),
                    vessel_length_m=_parse_optional_float(row.get("vessel_length_m")),
                    distance_from_shore_km=_parse_optional_float(row.get("distance_from_shore_km")),
                    top=_parse_optional_float(row.get("top")),
                    left=_parse_optional_float(row.get("left")),
                    bottom=_parse_optional_float(row.get("bottom")),
                    right=_parse_optional_float(row.get("right")),
                )
            )
    return annotations


def filter_labeled_annotations(annotations: list[Annotation]) -> list[Annotation]:
    return [annotation for annotation in annotations if annotation.has_any_label]


def split_scene_ids(scene_ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    scene_ids = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    val_count = max(1, int(round(len(scene_ids) * val_fraction)))
    val_ids = set(scene_ids[:val_count])
    train_ids = set(scene_ids[val_count:])
    if not train_ids:
        raise ValueError("Scene split produced an empty training set.")
    return train_ids, val_ids


def filter_annotations_by_scene(
    annotations: list[Annotation],
    scene_ids: set[str],
) -> list[Annotation]:
    return [annotation for annotation in annotations if annotation.scene_id in scene_ids]


def summarize_annotations(annotations: list[Annotation]) -> dict[str, object]:
    return {
        "count": len(annotations),
        "scene_count": len({annotation.scene_id for annotation in annotations}),
        "sources": Counter(annotation.source for annotation in annotations),
        "confidence": Counter(annotation.confidence for annotation in annotations),
        "is_vessel": Counter(str(annotation.is_vessel) for annotation in annotations),
        "is_fishing": Counter(str(annotation.is_fishing) for annotation in annotations),
    }
