from __future__ import annotations

import argparse
from pathlib import Path

from src.ghost_fleet.metadata import load_annotations
from src.ghost_fleet.scene_io import find_scene_paths
from src.ghost_fleet.scene_io import list_available_scene_ids


def check_partition(name: str, csv_path: Path, scene_root: Path) -> tuple[int, int, list[str]]:
    annotations = load_annotations(csv_path)
    scene_ids = sorted({annotation.scene_id for annotation in annotations})

    missing: list[str] = []
    for scene_id in scene_ids:
        try:
            find_scene_paths(scene_root, scene_id)
        except FileNotFoundError:
            missing.append(scene_id)

    present = len(scene_ids) - len(missing)
    print(f"{name}: {present}/{len(scene_ids)} scenes present")
    if missing:
        print(f"{name}: first missing scene ids: {', '.join(missing[:10])}")
    return present, len(scene_ids), missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local xView3 scene completeness.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/xview3/train.csv"),
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=Path("data/xview3/validation.csv"),
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path("data/xview3/full"),
    )
    args = parser.parse_args()

    available_scene_ids = list_available_scene_ids(args.scene_root)
    if available_scene_ids:
        print(
            f"Discovered {len(available_scene_ids)} extracted scenes under {args.scene_root}: "
            f"{', '.join(sorted(available_scene_ids)[:10])}"
        )
        print()

    train_present, train_total, train_missing = check_partition(
        "train",
        args.train_csv,
        args.scene_root,
    )
    val_present, val_total, val_missing = check_partition(
        "validation",
        args.validation_csv,
        args.scene_root,
    )

    print()
    print(f"Total present scenes: {train_present + val_present}")
    print(f"Total referenced scenes: {train_total + val_total}")

    if train_missing or val_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
