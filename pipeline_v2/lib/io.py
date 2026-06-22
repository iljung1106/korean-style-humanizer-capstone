"""Small IO helpers for pipeline_v2.

This module is intentionally dependency-light and standalone. It must not import
from the legacy project `train/`, `scripts/`, or `eval/` packages.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PIPELINE_ROOT.parent
REPO_ROOT = TRAINING_ROOT.parent


def resolve_path(path: str | Path, *, base: Path = TRAINING_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected object, got {type(value).__name__}")
            rows.append(value)
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    resolved = Path(path)
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected object, got {type(value).__name__}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_type_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        row_type = str(row.get("row_type") or "unknown")
        counts[row_type] = counts.get(row_type, 0) + 1
    return counts


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as metadata  # type: ignore

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def manifest_base(*, stage: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "time": time.time(),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "training_root": str(TRAINING_ROOT),
        "repo_root": str(REPO_ROOT),
        "args": args,
        "packages": package_versions(
            [
                "torch",
                "transformers",
                "trl",
                "peft",
                "unsloth",
                "unsloth_zoo",
                "bitsandbytes",
                "accelerate",
            ]
        ),
    }

