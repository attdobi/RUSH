"""Tests for flip-rate scoring across repeated labeling runs."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring.flip_rate import (  # noqa: E402
    FlipRateRecord,
    build_flip_rate_records,
    cohort_rollups,
)


def _vote(image_id: str, model_id: str, label: str, *, run_id: str, confidence=0.8):
    vote = {
        "run_id": run_id,
        "image_id": image_id,
        "labeler_type": "llm",
        "labeler_id": model_id,
        "model_id": model_id,
        "label": label,
        "node_ids": [],
        "justification": "fixture",
        "policy_graph_version": "Generative_AI.v0.1",
    }
    if confidence != "missing":
        vote["confidence"] = confidence
    return vote


def _run(tmp_path: Path, run_id: str, votes: list[dict]) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    with (run_dir / "label_votes.jsonl").open("w", encoding="utf-8") as f:
        for vote in votes:
            f.write(json.dumps(vote) + "\n")
    return run_dir


def test_one_run_one_vote_stable_flip_count_zero(tmp_path: Path):
    r1 = _run(tmp_path, "run-001", [_vote("img1", "model-a", "gen_ai", run_id="run-001")])

    records = build_flip_rate_records([r1])

    assert len(records) == 1
    record = records[0]
    assert record.image_id == "img1"
    assert record.model_id == "model-a"
    assert record.n_runs == 1
    assert record.labels_observed == ["gen_ai"]
    assert record.label_counts == {"gen_ai": 1}
    assert record.distinct_label_count == 1
    assert record.flip_count == 0
    assert record.flip_rate == 0.0
    assert record.stable_label == "gen_ai"
    assert record.first_seen_run_id == "run-001"
    assert record.last_seen_run_id == "run-001"


def test_two_runs_same_label_stable_flip_count_zero(tmp_path: Path):
    r2 = _run(tmp_path, "run-002", [_vote("img1", "model-a", "not_gen_ai", run_id="run-002")])
    r1 = _run(tmp_path, "run-001", [_vote("img1", "model-a", "not_gen_ai", run_id="run-001")])

    [record] = build_flip_rate_records([r2, r1])

    assert record.n_runs == 2
    assert record.labels_observed == ["not_gen_ai"]
    assert record.label_counts == {"not_gen_ai": 2}
    assert record.flip_count == 0
    assert record.flip_rate == 0.0
    assert record.stable_label == "not_gen_ai"
    assert record.first_seen_run_id == "run-001"
    assert record.last_seen_run_id == "run-002"


def test_two_runs_different_labels_flips_at_rate_one(tmp_path: Path):
    r1 = _run(tmp_path, "run-001", [_vote("img1", "model-a", "gen_ai", run_id="run-001")])
    r2 = _run(tmp_path, "run-002", [_vote("img1", "model-a", "not_gen_ai", run_id="run-002")])

    [record] = build_flip_rate_records([r1, r2])

    assert record.distinct_label_count == 2
    assert record.label_counts == {"gen_ai": 1, "not_gen_ai": 1}
    assert record.flip_count == 1
    assert record.flip_rate == 1.0
    assert record.stable_label is None


def test_three_runs_two_distinct_labels_flip_rate_half(tmp_path: Path):
    run_dirs = [
        _run(tmp_path, "run-001", [_vote("img1", "model-a", "gen_ai", run_id="run-001")]),
        _run(tmp_path, "run-002", [_vote("img1", "model-a", "gen_ai", run_id="run-002")]),
        _run(tmp_path, "run-003", [_vote("img1", "model-a", "not_gen_ai", run_id="run-003")]),
    ]

    [record] = build_flip_rate_records(run_dirs)

    assert record.n_runs == 3
    assert record.distinct_label_count == 2
    assert record.flip_count == 1
    assert record.flip_rate == 0.5
    assert record.stable_label is None


def test_confidence_none_missing_and_abstain_handling(tmp_path: Path):
    run_dirs = [
        _run(tmp_path, "run-001", [_vote("img1", "model-a", "abstain", run_id="run-001", confidence=None)]),
        _run(tmp_path, "run-002", [_vote("img1", "model-a", "abstain", run_id="run-002", confidence="missing")]),
        _run(tmp_path, "run-003", [_vote("img1", "model-a", "abstain", run_id="run-003", confidence=0.25)]),
    ]

    [record] = build_flip_rate_records(run_dirs)

    assert record.abstain_count == 3
    assert record.stable_label == "abstain"
    assert record.confidence_min == 0.25
    assert record.confidence_max == 0.25
    assert record.confidence_mean == 0.25

    no_numeric_runs = [
        _run(tmp_path, "run-004", [_vote("img2", "model-a", "gen_ai", run_id="run-004", confidence=None)]),
        _run(tmp_path, "run-005", [_vote("img2", "model-a", "gen_ai", run_id="run-005", confidence="missing")]),
    ]
    [no_numeric] = build_flip_rate_records(no_numeric_runs)
    assert no_numeric.confidence_min is None
    assert no_numeric.confidence_max is None
    assert no_numeric.confidence_mean is None


def test_cohort_rollups_sanity_per_model_and_per_image_top():
    records = [
        FlipRateRecord(
            image_id="img1",
            model_id="model-a",
            n_runs=2,
            labels_observed=["gen_ai", "not_gen_ai"],
            label_counts={"gen_ai": 1, "not_gen_ai": 1},
            distinct_label_count=2,
            flip_count=1,
            flip_rate=1.0,
            stable_label=None,
            abstain_count=0,
            confidence_min=0.4,
            confidence_max=0.9,
            confidence_mean=0.65,
            first_seen_run_id="run-001",
            last_seen_run_id="run-002",
        ),
        FlipRateRecord(
            image_id="img1",
            model_id="model-b",
            n_runs=2,
            labels_observed=["gen_ai"],
            label_counts={"gen_ai": 2},
            distinct_label_count=1,
            flip_count=0,
            flip_rate=0.0,
            stable_label="gen_ai",
            abstain_count=0,
            confidence_min=0.8,
            confidence_max=0.9,
            confidence_mean=0.85,
            first_seen_run_id="run-001",
            last_seen_run_id="run-002",
        ),
        FlipRateRecord(
            image_id="img2",
            model_id="model-a",
            n_runs=3,
            labels_observed=["abstain", "gen_ai"],
            label_counts={"abstain": 1, "gen_ai": 2},
            distinct_label_count=2,
            flip_count=1,
            flip_rate=0.5,
            stable_label=None,
            abstain_count=1,
            confidence_min=None,
            confidence_max=None,
            confidence_mean=None,
            first_seen_run_id="run-001",
            last_seen_run_id="run-003",
        ),
        FlipRateRecord(
            image_id="img3",
            model_id="model-c",
            n_runs=1,
            labels_observed=["not_gen_ai"],
            label_counts={"not_gen_ai": 1},
            distinct_label_count=1,
            flip_count=0,
            flip_rate=0.0,
            stable_label="not_gen_ai",
            abstain_count=0,
            confidence_min=0.7,
            confidence_max=0.7,
            confidence_mean=0.7,
            first_seen_run_id="run-001",
            last_seen_run_id="run-001",
        ),
    ]

    rollup = cohort_rollups(records)

    assert rollup["n_pairs_total"] == 4
    assert rollup["n_pairs_stable"] == 2
    assert rollup["n_pairs_flipped"] == 2
    assert rollup["mean_flip_rate"] == pytest.approx(0.375)
    assert rollup["per_model_flip_rate"] == {
        "model-a": pytest.approx(0.75),
        "model-b": 0.0,
    }
    assert [r["image_id"] for r in rollup["per_image_flip_rate"][:2]] == ["img1", "img2"]
    top = rollup["per_image_flip_rate"][0]
    assert top == {
        "image_id": "img1",
        "mean_flip_rate": 0.5,
        "max_flip_rate": 1.0,
        "models_that_flipped": ["model-a"],
        "labels_observed": ["gen_ai", "not_gen_ai"],
    }


def test_record_json_shape_is_serializable(tmp_path: Path):
    r1 = _run(tmp_path, "run-001", [_vote("img1", "model-a", "gen_ai", run_id="run-001")])
    [record] = build_flip_rate_records([r1])

    dumped = json.dumps(asdict(record), sort_keys=True)

    assert '"image_id": "img1"' in dumped
