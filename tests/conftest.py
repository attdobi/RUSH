"""Shared pytest fixtures — RUSH bulk-labeling tests.

Adds the repo root to ``sys.path`` so ``import pipeline`` works when pytest
is invoked from anywhere inside the repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CREATED_MANIFEST: Path | None = None


def _synthetic_manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, prefix in (("dev_golden", "dev_golden"), ("holdout", "holdout")):
        for idx in range(1, 101):
            label_int = idx % 2
            rows.append(
                {
                    "sample_id": f"{prefix}_{idx:04d}",
                    "repo_rel_path": f"data/images/genai-classification/source-datasets/{prefix}_{idx:04d}.jpg",
                    "split": split,
                    "label": "ai_generated" if label_int else "not_ai_generated",
                    "label_int": label_int,
                    "truth_tier": "gold",
                    "dataset": "synthetic-test-fixture",
                    "sha256": f"{idx:064x}"[-64:],
                    "sampling_version": "genai-gold-sampling-v1",
                }
            )
    return rows


def pytest_configure(config) -> None:  # noqa: ARG001
    """Create the small GenAI manifest fixture when a sparse worktree lacks it."""
    global _CREATED_MANIFEST
    manifest = (
        _REPO_ROOT
        / "data"
        / "images"
        / "genai-classification"
        / "manifests"
        / "combined_labels.jsonl"
    )
    if manifest.exists():
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _synthetic_manifest_rows()),
        encoding="utf-8",
    )
    _CREATED_MANIFEST = manifest


def pytest_unconfigure(config) -> None:  # noqa: ARG001
    if _CREATED_MANIFEST is not None and _CREATED_MANIFEST.exists():
        _CREATED_MANIFEST.unlink()
