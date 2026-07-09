"""Seeded three-way split assignment for the GenAI source datasets.

Attila 2026-07-09: "once you have the images the train / test / validation
split will have to be assigned — use a random seed." The sampler must mint
dev_golden / holdout / validation disjointly from one seeded shuffle, and
adding the validation split later (same seed) must not move a single
existing dev/holdout assignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.sample_genai_gold_sets import (  # noqa: E402
    DATASETS,
    LABELS,
    collect_candidates,
    sample_records,
)


def _fake_source_tree(root: Path, per_group: int = 12) -> Path:
    """A source-datasets tree with unique bytes per file (sha dedupe safe)."""
    source = root / "source-datasets"
    for dataset in DATASETS:
        for label in LABELS:
            class_dir = source / dataset / label
            class_dir.mkdir(parents=True)
            for i in range(per_group):
                (class_dir / f"{i:05d}.jpg").write_bytes(
                    f"{dataset}/{label}/{i}".encode()
                )
    return source


def test_three_way_split_disjoint_balanced_and_seeded(tmp_path):
    source = _fake_source_tree(tmp_path, per_group=12)
    grouped = collect_candidates(tmp_path, source)
    dev, holdout, validation, summary = sample_records(
        grouped, n_dev=12, n_holdout=12, seed=13, n_validation=12
    )
    assert len(dev) == 12 and len(holdout) == 12 and len(validation) == 12
    # Disjoint by path AND content hash across all three splits.
    all_paths = [r["repo_rel_path"] for r in (*dev, *holdout, *validation)]
    all_hashes = [r["sha256"] for r in (*dev, *holdout, *validation)]
    assert len(set(all_paths)) == 36 and len(set(all_hashes)) == 36
    # Class-balanced and source-stratified within each split.
    for rows in (dev, holdout, validation):
        assert sum(1 for r in rows if r["label"] == "ai_generated") == 6
        assert len({(r["dataset"], r["label"]) for r in rows}) == 6
    # Benchmark-split contract mirrors MNIST: bench_* ids, benchmark policy_use.
    assert [r["sample_id"] for r in validation] == [
        f"bench_{i:04d}" for i in range(1, 13)
    ]
    assert all(r["policy_use"] == "benchmark_cross_run" for r in validation)
    assert all(r["split"] == "validation" for r in validation)
    assert summary["n_validation"] == 12
    # Determinism: same seed reproduces the exact assignment.
    dev2, holdout2, validation2, _ = sample_records(
        grouped, n_dev=12, n_holdout=12, seed=13, n_validation=12
    )
    assert (dev, holdout, validation) == (dev2, holdout2, validation2)


def test_adding_validation_later_keeps_dev_and_holdout_stable(tmp_path):
    """The mini's existing dev/holdout labels must survive the validation mint."""
    source = _fake_source_tree(tmp_path, per_group=12)
    grouped = collect_candidates(tmp_path, source)
    dev_before, holdout_before, validation_before, _ = sample_records(
        grouped, n_dev=12, n_holdout=12, seed=20260510
    )
    assert validation_before == []
    dev_after, holdout_after, validation_after, _ = sample_records(
        grouped, n_dev=12, n_holdout=12, seed=20260510, n_validation=6
    )
    assert dev_after == dev_before
    assert holdout_after == holdout_before
    assert len(validation_after) == 6


def test_validation_overdraw_fails_loudly(tmp_path):
    source = _fake_source_tree(tmp_path, per_group=4)
    grouped = collect_candidates(tmp_path, source)
    with pytest.raises(ValueError, match="Not enough unique files"):
        sample_records(grouped, n_dev=12, n_holdout=6, seed=13, n_validation=12)
