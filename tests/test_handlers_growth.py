"""Tests for the cold-start and grow-batch HTTP handlers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.web import handlers_policy


def _seed_v01(tmp_path: Path) -> Path:
    base = tmp_path / "policy-graph" / "Generative_AI" / "v0.1"
    base.mkdir(parents=True)
    (base / "GA.root.md").write_text("# Root\n", encoding="utf-8")
    (tmp_path / "data").mkdir(exist_ok=True)
    return tmp_path


def _write_misalignment(root: Path, run_id: str) -> None:
    scoring = root / "data" / "runs" / run_id / "scoring"
    scoring.mkdir(parents=True)
    records = [
        {"image_id": f"pos_{i}", "sme_truth": "gen_ai", "votes": []}
        for i in range(3)
    ] + [
        {"image_id": f"neg_{i}", "sme_truth": "not_gen_ai", "votes": []}
        for i in range(3)
    ]
    (scoring / "misalignment.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# handle_cold_start
# ---------------------------------------------------------------------------


def test_handle_cold_start_rejects_missing_task(tmp_path: Path) -> None:
    status, body = handlers_policy.handle_cold_start(tmp_path, {})
    assert status == 400
    assert "task_description" in body["error"]


def test_handle_cold_start_rejects_unknown_domain(tmp_path: Path) -> None:
    status, body = handlers_policy.handle_cold_start(
        tmp_path,
        {"task_description": "x", "domain": "Other_Domain"},
    )
    assert status == 400
    assert "domain" in body["error"]


def test_handle_cold_start_routes_to_seed(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "data").mkdir()
    captured: dict = {}

    def fake_seed(**kwargs):
        captured.update(kwargs)
        return {
            "proposal_id": "p1",
            "kind": "cold_start",
            "domain": "Generative_AI",
            "base_version": None,
            "status": "pending",
            "files_changed": [],
            "files_added": ["GA.root.md"],
            "files_removed": [],
        }

    monkeypatch.setattr(handlers_policy, "seed_cold_start_proposal", fake_seed)

    status, body = handlers_policy.handle_cold_start(
        tmp_path,
        {"task_description": "Classify AI images"},
    )
    assert status == 200
    assert body["proposal_id"] == "p1"
    assert captured["task_description"] == "Classify AI images"
    assert captured["domain"] == "Generative_AI"


def test_handle_cold_start_parse_error_returns_422(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_seed(**kwargs):
        return {
            "proposal_id": "p1",
            "kind": "cold_start",
            "base_version": None,
            "status": "parse_error",
            "files_changed": [],
            "files_added": [],
            "files_removed": [],
        }

    monkeypatch.setattr(handlers_policy, "seed_cold_start_proposal", fake_seed)
    status, body = handlers_policy.handle_cold_start(
        tmp_path,
        {"task_description": "Classify AI images"},
    )
    assert status == 422
    assert body["status"] == "parse_error"


def test_handle_cold_start_500_on_unexpected(tmp_path: Path, monkeypatch) -> None:
    def fake_seed(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(handlers_policy, "seed_cold_start_proposal", fake_seed)
    status, body = handlers_policy.handle_cold_start(
        tmp_path, {"task_description": "x"}
    )
    assert status == 500
    assert body["error"] == "boom"


# ---------------------------------------------------------------------------
# handle_grow_batch
# ---------------------------------------------------------------------------


def test_handle_grow_batch_validates_body(tmp_path: Path) -> None:
    status, body = handlers_policy.handle_grow_batch(tmp_path, {})
    assert status == 400
    assert "run_id" in body["error"]

    status, body = handlers_policy.handle_grow_batch(
        tmp_path, {"run_id": "r"}
    )
    assert status == 400
    assert "base_version" in body["error"]

    status, body = handlers_policy.handle_grow_batch(
        tmp_path, {"run_id": "r", "base_version": "vXX"}
    )
    assert status == 400

    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {"run_id": "r", "base_version": "v0.1", "batch_index": -1, "batch_size": 4},
    )
    assert status == 400
    assert "batch_index" in body["error"]

    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {"run_id": "r", "base_version": "v0.1", "batch_index": 0, "batch_size": 1},
    )
    assert status == 400
    assert "batch_size" in body["error"]


def test_handle_grow_batch_routes_to_proposer(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict = {}

    def fake_proposer(**kwargs):
        captured.update(kwargs)
        return {
            "proposal_id": "p2",
            "kind": "grow_batch",
            "base_version": kwargs["base_version"],
            "batch_index": kwargs["batch_index"],
            "batch": {
                "batch_size_requested": kwargs["batch_size"],
                "batch_size_actual": 4,
                "n_positives": 2,
                "n_negatives": 2,
                "sme_truth_positive_label": "gen_ai",
                "sme_truth_negative_label": "not_gen_ai",
            },
            "status": "pending",
            "files_changed": [],
            "files_added": ["GA.x.md"],
            "files_removed": [],
        }

    monkeypatch.setattr(handlers_policy, "propose_growth_batch", fake_proposer)

    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {
            "run_id": "20260518T180000-abcdef01",
            "base_version": "v0.1",
            "batch_index": 0,
            "batch_size": 4,
        },
    )
    assert status == 200
    assert body["proposal_id"] == "p2"
    assert captured["run_id"] == "20260518T180000-abcdef01"
    assert captured["batch_index"] == 0
    assert captured["batch_size"] == 4


def test_handle_grow_batch_parse_error_returns_422(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_proposer(**kwargs):
        return {
            "proposal_id": "p2",
            "kind": "grow_batch",
            "base_version": "v0.1",
            "batch_index": 0,
            "batch": {
                "batch_size_requested": 4,
                "batch_size_actual": 4,
                "n_positives": 2,
                "n_negatives": 2,
                "sme_truth_positive_label": "gen_ai",
                "sme_truth_negative_label": "not_gen_ai",
            },
            "status": "parse_error",
            "files_changed": [],
            "files_added": [],
            "files_removed": [],
        }

    monkeypatch.setattr(handlers_policy, "propose_growth_batch", fake_proposer)
    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {
            "run_id": "r",
            "base_version": "v0.1",
            "batch_index": 0,
            "batch_size": 4,
        },
    )
    assert status == 422


def test_handle_grow_batch_404_when_run_missing(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_proposer(**kwargs):
        raise FileNotFoundError("missing scoring misalignment file")

    monkeypatch.setattr(handlers_policy, "propose_growth_batch", fake_proposer)
    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {
            "run_id": "r",
            "base_version": "v0.1",
            "batch_index": 0,
            "batch_size": 4,
        },
    )
    assert status == 404
    assert "missing scoring" in body["error"]


def test_handle_grow_batch_value_error_returns_400(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_proposer(**kwargs):
        raise ValueError("bad version")

    monkeypatch.setattr(handlers_policy, "propose_growth_batch", fake_proposer)
    status, body = handlers_policy.handle_grow_batch(
        tmp_path,
        {
            "run_id": "r",
            "base_version": "v0.1",
            "batch_index": 0,
            "batch_size": 4,
        },
    )
    assert status == 400
