from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MAX_FILE_MB = 200
MAX_SAMPLE_INPUT_MB = 10
VALID_SLUG = re.compile(r"^[A-Za-z0-9-]+$")
FORBIDDEN_DIRS = {"__pycache__", ".ipynb_checkpoints"}
FORBIDDEN_NAMES = {".env"}
FORBIDDEN_SUFFIXES = {".env"}
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
)


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def check_submission(root: Path) -> list[str]:
    errors: list[str] = []

    if not VALID_SLUG.fullmatch(root.name):
        errors.append(
            f"Folder name '{root.name}' is not a valid slug. Use letters, digits, and dashes only."
        )

    for required in ("README.md", "requirements.txt", "infer.py"):
        if not (root / required).exists():
            errors.append(f"Missing required file: {required}")

    requirements = root / "requirements.txt"
    if requirements.exists():
        for line_number, raw_line in enumerate(requirements.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ">=" in line or "<=" in line or "~=" in line or "<" in line or ">" in line:
                errors.append(f"requirements.txt:{line_number} is not exactly pinned: {line}")
            elif "==" not in line:
                errors.append(f"requirements.txt:{line_number} is missing an exact == pin: {line}")

    sample_input = root / "sample_input"
    if sample_input.exists():
        sample_bytes = sum(path.stat().st_size for path in _iter_files(sample_input))
        if sample_bytes > MAX_SAMPLE_INPUT_MB * 1024 * 1024:
            errors.append(
                f"sample_input is {sample_bytes / 1024 / 1024:.2f} MB; keep it <= {MAX_SAMPLE_INPUT_MB} MB."
            )

    for path in root.rglob("*"):
        relative = _relative(path, root)
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            errors.append(f"Forbidden generated directory present: {relative}")
        if path.is_file():
            size_mb = path.stat().st_size / 1024 / 1024
            if size_mb > MAX_FILE_MB:
                errors.append(f"File exceeds {MAX_FILE_MB} MB: {relative} ({size_mb:.2f} MB)")
            if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
                errors.append(f"Forbidden credential/env file present: {relative}")
            if path.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".ps1"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pattern in SECRET_PATTERNS:
                    if pattern.search(text):
                        errors.append(f"Possible secret-like text in: {relative}")
                        break

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AIML submission folder rules.")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Submission folder to check. Defaults to current directory.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    errors = check_submission(root)
    if errors:
        print("Submission check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Submission check passed for {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
