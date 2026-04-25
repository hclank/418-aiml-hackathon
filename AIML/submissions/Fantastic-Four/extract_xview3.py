from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def scene_looks_extracted(scene_dir: Path) -> bool:
    return (scene_dir / "VV_dB.tif").exists() and (scene_dir / "VH_dB.tif").exists()


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as handle:
        for member in handle.getmembers():
            target_path = (output_dir / member.name).resolve()
            if not _is_relative_to(target_path, resolved_output):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        handle.extractall(output_dir, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract downloaded xView3 tar.gz archives.")
    parser.add_argument(
        "--archives-dir",
        type=Path,
        default=Path("data/xview3/archives"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xview3/full"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract scenes even when VV_dB.tif and VH_dB.tif already exist.",
    )
    parser.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip corrupt archives and continue. Enabled by default.",
    )
    args = parser.parse_args()

    archives = sorted(args.archives_dir.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"No .tar.gz archives found under {args.archives_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed: list[Path] = []
    extracted = 0
    skipped = 0
    for archive_path in archives:
        scene_id = archive_path.name.removesuffix(".tar.gz")
        scene_dir = args.output_dir / scene_id
        if scene_looks_extracted(scene_dir) and not args.overwrite:
            print(f"Skipping {archive_path.name}; {scene_id} already extracted")
            skipped += 1
            continue
        print(f"Extracting {archive_path.name}")
        try:
            extract_archive(archive_path, args.output_dir)
            extracted += 1
        except (EOFError, tarfile.TarError, gzip.BadGzipFile, OSError) as exc:
            failed.append(archive_path)
            print(f"FAILED {archive_path.name}: {type(exc).__name__}: {exc}")
            if not args.keep_going:
                raise

    print(f"Extracted: {extracted}")
    print(f"Skipped already extracted: {skipped}")
    if failed:
        print("Failed archives:")
        for archive_path in failed:
            print(f"- {archive_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
