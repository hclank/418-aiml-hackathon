from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass(frozen=True)
class ManifestEntry:
    url: str
    sha1: str | None

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


def parse_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    pending_url: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("http"):
            if pending_url is not None:
                entries.append(ManifestEntry(url=pending_url, sha1=None))
            pending_url = line
            continue
        if line.startswith("checksum=sha-1=") and pending_url is not None:
            entries.append(
                ManifestEntry(
                    url=pending_url,
                    sha1=line.split("=", maxsplit=2)[-1],
                )
            )
            pending_url = None

    if pending_url is not None:
        entries.append(ManifestEntry(url=pending_url, sha1=None))
    return entries


def compute_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as temp_handle:
        temp_path = Path(temp_handle.name)

    try:
        with urlopen(url) as response, temp_path.open("wb") as output_handle:
            shutil.copyfileobj(response, output_handle)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as handle:
        handle.extractall(extract_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download xView3 scene archives from a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/xview3/archives"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--extract-dir", type=Path, default=Path("data/xview3/full"))
    args = parser.parse_args()

    entries = parse_manifest(args.manifest)
    if args.limit > 0:
        entries = entries[: args.limit]

    print(f"Found {len(entries)} archive entries in {args.manifest}.")
    for index, entry in enumerate(entries, start=1):
        destination = args.output_dir / entry.filename
        if destination.exists():
            print(f"[{index}/{len(entries)}] Reusing existing {destination.name}")
        else:
            print(f"[{index}/{len(entries)}] Downloading {destination.name}")
            download_file(entry.url, destination)

        if entry.sha1 is not None:
            actual_sha1 = compute_sha1(destination)
            if actual_sha1 != entry.sha1:
                raise SystemExit(
                    f"SHA-1 mismatch for {destination.name}: expected {entry.sha1}, got {actual_sha1}"
                )

        if args.extract:
            print(f"[{index}/{len(entries)}] Extracting {destination.name}")
            extract_archive(destination, args.extract_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
