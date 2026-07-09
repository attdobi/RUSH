"""Thumbnail path resolution — the /api/thumbnail contract.

Regression (Attila 2026-07-09): GenAI anchor-evidence thumbnails rendered
broken because runs on the portable fixture carry ``sample/`` repo paths,
and the validator only allowed ``source-datasets/`` + derived thumbnail
roots — every sample image 400'd.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.thumbnails import (
    thumbnail_rel_path_for_source,
    validate_source_repo_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_genai_sample_paths_resolve_and_serve_as_is() -> None:
    rel = "data/images/genai-classification/sample/midjourney/ai_generated/image_1047.png"
    validated = validate_source_repo_path(_REPO_ROOT, rel)
    assert validated.as_posix() == rel
    # No derived thumbnails exist for the committed fixture: serve the original.
    assert thumbnail_rel_path_for_source(validated).as_posix() == rel


def test_source_dataset_paths_map_to_derived_thumbnails() -> None:
    rel = "data/images/genai-classification/source-datasets/wfir/ai_generated/00745.jpeg"
    validated = validate_source_repo_path(_REPO_ROOT, rel)
    thumb = thumbnail_rel_path_for_source(validated).as_posix()
    assert thumb == (
        "data/images/genai-classification/derived/thumbnails/wfir/ai_generated/00745.jpg"
    )


def test_paths_outside_allowed_roots_rejected() -> None:
    with pytest.raises(ValueError):
        validate_source_repo_path(_REPO_ROOT, "data/runs/x/label_votes.jsonl")
    with pytest.raises(ValueError):
        validate_source_repo_path(_REPO_ROOT, "../../etc/passwd.png")
