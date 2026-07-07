"""The FIXED cross-run validation (benchmark) split: locked + never implicit.

Attila 2026-07-06: per-run test partitions come from dev_golden, so run
numbers are not directly comparable; the ``validation`` split fixes ONE set
of images for every run. These tests pin its safety contract:

  * it is a valid split, but locked like the holdout (explicit flag required);
  * ``--split all`` (the demo's train+test pass) never sweeps it up;
  * the experiment partitioner keeps carving from dev_golden only.
"""
from __future__ import annotations

from pipeline.manifest import (
    ALL_SPLITS,
    HOLDOUT_SPLITS,
    VALID_SPLITS,
    SampleRecord,
    select_samples,
)


def _record(sid: str, split: str, label: str = "3") -> SampleRecord:
    return SampleRecord(
        sample_id=sid,
        repo_rel_path=f"data/images/mnist/{split}/{sid}.png",
        split=split,
        sme_label_raw=label,
        sme_label=label,
        dataset="mnist",
        sha256=f"{abs(hash(sid)):064x}"[-64:],
        sampling_version="mnist-validation-v1",
    )


def _pool() -> list[SampleRecord]:
    rows = [_record(f"train_{i:04d}", "dev_golden") for i in range(4)]
    rows += [_record(f"val_{i:04d}", "holdout") for i in range(3)]
    rows += [_record(f"bench_{i:04d}", "validation") for i in range(5)]
    return rows


def test_validation_is_a_locked_split():
    assert "validation" in VALID_SPLITS
    # Locked like the holdout: bulk passes need the explicit safety flag.
    assert "validation" in HOLDOUT_SPLITS
    assert "validation" not in ALL_SPLITS


def test_split_all_never_includes_validation():
    picked = select_samples(_pool(), split="all")
    splits = {rec.split for rec in picked}
    assert splits == {"dev_golden", "holdout"}
    # ... including the limit-per-portion path.
    picked = select_samples(_pool(), split="all", limit=2)
    assert {rec.split for rec in picked} == {"dev_golden", "holdout"}
    assert all(rec.split != "validation" for rec in picked)


def test_validation_selectable_explicitly():
    picked = select_samples(_pool(), split="validation")
    assert len(picked) == 5
    assert all(rec.sample_id.startswith("bench_") for rec in picked)


def test_explicit_sample_ids_still_reach_validation_rows():
    # The experiment driver addresses benchmark images by id (like holdout).
    picked = select_samples(_pool(), sample_ids=["bench_0001", "train_0000"])
    assert sorted(rec.sample_id for rec in picked) == ["bench_0001", "train_0000"]
