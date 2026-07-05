from __future__ import annotations

import json
from pathlib import Path

from pipeline.web import aggregator
from pipeline.web import handlers_dq

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_autoscore_run(runs_root: Path, run_id: str) -> Path:
    run_dir = runs_root / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": "2026-05-11T18:17:35Z",
            "finished_at": "2026-05-11T18:18:35Z",
            "policy_graph_version": "v0.1",
            "sample_ids": ["dev_golden_0001", "dev_golden_0002"],
            "models": [{"model_id": "model-a"}, {"model_id": "model-b"}],
            "totals": {"expected_calls": 4, "completed_calls": 4, "errored_calls": 0},
        },
    )
    rows = [
        {
            "run_id": run_id,
            "image_id": "dev_golden_0001",
            "labeler_type": "llm",
            "labeler_id": "model-a",
            "model_id": "model-a",
            "label": "gen_ai",
            "node_ids": [],
            "confidence": 0.9,
            "justification": "synthetic artifacts",
            "policy_graph_version": "Generative_AI.v0.1",
            "prompt_version": "v0.1",
            "label_tier": "provisional",
            "is_boundary": False,
            "difficulty": "low",
        },
        {
            "run_id": run_id,
            "image_id": "dev_golden_0001",
            "labeler_type": "llm",
            "labeler_id": "model-b",
            "model_id": "model-b",
            "label": "not_gen_ai",
            "node_ids": [],
            "confidence": 0.55,
            "justification": "uncertain",
            "policy_graph_version": "Generative_AI.v0.1",
            "prompt_version": "v0.1",
            "label_tier": "provisional",
            "is_boundary": True,
            "difficulty": "high",
        },
        {
            "run_id": run_id,
            "image_id": "dev_golden_0002",
            "labeler_type": "llm",
            "labeler_id": "model-a",
            "model_id": "model-a",
            "label": "gen_ai",
            "node_ids": [],
            "confidence": 0.8,
            "justification": "generated look",
            "policy_graph_version": "Generative_AI.v0.1",
            "prompt_version": "v0.1",
            "label_tier": "provisional",
            "is_boundary": False,
            "difficulty": "medium",
        },
        {
            "run_id": run_id,
            "image_id": "dev_golden_0002",
            "labeler_type": "llm",
            "labeler_id": "model-b",
            "model_id": "model-b",
            "label": "gen_ai",
            "node_ids": [],
            "confidence": 0.85,
            "justification": "generated look",
            "policy_graph_version": "Generative_AI.v0.1",
            "prompt_version": "v0.1",
            "label_tier": "provisional",
            "is_boundary": False,
            "difficulty": "medium",
        },
    ]
    _write_jsonl(run_dir / "label_votes.jsonl", rows)
    _write_jsonl(
        run_dir / "llm_outputs.jsonl",
        [
            {"run_id": row["run_id"], "image_id": row["image_id"], "model_id": row["model_id"], "output": {}}
            for row in rows
        ],
    )
    return run_dir


def _make_run(
    runs_root: Path,
    run_id: str,
    *,
    started_at: str,
    policy_version: str,
    majority_label: str = "not_gen_ai",
    cost: dict | None = None,
) -> Path:
    run_dir = runs_root / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": started_at,
            "policy_graph_version": policy_version.replace("Generative_AI.", ""),
            "sample_ids": ["img-1", "img-2"],
        },
    )
    dq_payload = {
        "policy_graph_version": policy_version,
        "ground_truth_tier": ["gold"],
        "labelers": [
            {"labeler_id": "model-a", "labeler_type": "llm", "metrics": {"accuracy": 1.0, "n": 2}},
            {"labeler_id": "model-b", "labeler_type": "llm", "metrics": {"accuracy": 0.5, "n": 2}},
            {"labeler_id": "majority_vote", "labeler_type": "ensemble", "metrics": {"accuracy": 0.5, "n": 2}},
        ],
    }
    if cost is not None:
        dq_payload["cost"] = cost
    _write_json(run_dir / "scoring" / "decision_quality.json", dq_payload)
    _write_json(
        run_dir / "scoring" / "consensus.json",
        {
            "run_id": run_id,
            "policy_graph_version": policy_version,
            "ground_truth_tier": ["gold"],
            "summary": {
                "n_images_total": 2,
                "n_images_unanimous": 1,
                "n_images_consensus": 1,
                "n_images_split": 1,
                "n_images_with_tie": 0,
                "n_images_with_boundary_flag": 1,
            },
            "records": [
                {
                    "run_id": run_id,
                    "image_id": "img-1",
                    "majority_label": majority_label,
                    "majority_count": 2,
                    "majority_fraction": 0.666667,
                    "is_split": True,
                    "vote_distribution": {"gen_ai": 1, "not_gen_ai": 2},
                    "voters": [
                        {"labeler_id": "model-a", "model_id": "model-a", "label": "gen_ai", "confidence": 0.9},
                        {"labeler_id": "model-b", "model_id": "model-b", "label": "not_gen_ai", "confidence": 0.8},
                    ],
                },
                {
                    "run_id": run_id,
                    "image_id": "img-2",
                    "majority_label": "gen_ai",
                    "majority_count": 2,
                    "majority_fraction": 1.0,
                    "is_split": False,
                    "vote_distribution": {"gen_ai": 2},
                    "voters": [
                        {"labeler_id": "model-a", "model_id": "model-a", "label": "gen_ai", "confidence": 0.9},
                        {"labeler_id": "model-b", "model_id": "model-b", "label": "gen_ai", "confidence": 0.8},
                    ],
                },
            ],
        },
    )
    _write_json(
        run_dir / "scoring" / "misalignment.json",
        {
            "policy_graph_version": policy_version,
            "summary": {"total_images": 2, "all_agree": 1, "model_vs_sme": 1, "model_vs_model": 0, "consensus_wrong": 0},
            "records": [
                {
                    "image_id": "img-1",
                    "repo_rel_path": "images/img-1.png",
                    "sme_truth": "gen_ai",
                    "misalignment_type": "model_vs_sme",
                    "severity": "medium",
                    "votes": [
                        {"labeler_id": "model-a", "model_id": "model-a", "label": "gen_ai", "l2_label": "GA.synthetic", "confidence": 0.9},
                        {"labeler_id": "model-b", "model_id": "model-b", "label": "not_gen_ai", "l2_label": "GA.photo", "confidence": 0.8},
                    ],
                },
                {
                    "image_id": "img-2",
                    "repo_rel_path": "images/img-2.png",
                    "sme_truth": "gen_ai",
                    "misalignment_type": "all_agree",
                    "severity": "low",
                    "votes": [],
                },
            ],
        },
    )
    _write_json(
        run_dir / "scoring" / "borderline.json",
        {
            "policy_graph_version": policy_version,
            "low_confidence_threshold": 0.6,
            "summary": {"total_images": 2, "borderline_images": 1, "by_l0": {"gen_ai": 1, "not_gen_ai": 0}},
            "groups": {
                "gen_ai": [
                    {
                        "image_id": "img-1",
                        "repo_rel_path": "images/img-1.png",
                        "sme_truth": "gen_ai",
                        "reasons": ["model_disagreement"],
                        "votes": [
                            {"labeler_id": "model-a", "label": "gen_ai", "l2_label": "GA.synthetic"},
                            {"labeler_id": "model-b", "label": "not_gen_ai", "l2_label": "GA.photo"},
                        ],
                    }
                ],
                "not_gen_ai": [],
            },
        },
    )
    return run_dir


def _runs_fixture(tmp_path: Path) -> Path:
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "run-late", started_at="2026-05-10T02:00:00Z", policy_version="Generative_AI.v0.2")
    _make_run(
        runs_root,
        "run-early",
        started_at="2026-05-10T01:00:00Z",
        policy_version="Generative_AI.v0.1",
        cost={"total_cost_usd": 0.0123, "per_model": {"model-a": {"cost_per_1000_labels": 1.23}}},
    )
    return runs_root


def test_aggregate_decision_quality_returns_runs_sorted_by_started_at(tmp_path: Path) -> None:
    runs_root = _runs_fixture(tmp_path)
    payload = aggregator.aggregate_decision_quality(runs_root)

    assert [run["run_id"] for run in payload["runs"]] == ["run-early", "run-late"]
    assert payload["policy_versions"] == ["Generative_AI.v0.1", "Generative_AI.v0.2"]
    assert payload["runs"][0]["majority_vote"] == {"accuracy": 0.5, "n": 2}
    assert payload["runs"][0]["cost"] == {"total_cost_usd": 0.0123, "per_model": {"model-a": {"cost_per_1000_labels": 1.23}}}
    assert payload["runs"][0]["boundary_rate"] == 0.5


def test_aggregate_decision_quality_filters_by_run_id(tmp_path: Path) -> None:
    runs_root = _runs_fixture(tmp_path)
    payload = aggregator.aggregate_decision_quality(runs_root, run_id="run-late")

    assert [run["run_id"] for run in payload["runs"]] == ["run-late"]


def test_aggregate_decision_quality_filters_by_policy_version(tmp_path: Path) -> None:
    runs_root = _runs_fixture(tmp_path)
    payload = aggregator.aggregate_decision_quality(runs_root, policy_version="Generative_AI.v0.1")

    assert [run["run_id"] for run in payload["runs"]] == ["run-early"]


def test_aggregate_decision_quality_surfaces_split_reported_and_update_candidates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = _make_run(
        runs_root,
        "run-split",
        started_at="2026-05-10T03:00:00Z",
        policy_version="Generative_AI.v0.1",
    )
    reported = {
        "labelers": [
            {"labeler_id": "model-a", "labeler_type": "llm", "metrics": {"accuracy": 1.0, "n": 2}},
            {"labeler_id": "majority_vote", "labeler_type": "ensemble", "metrics": {"accuracy": 1.0, "n": 2}},
        ],
        "n_images": 2,
    }
    by_split = {
        "train": {"labelers": [], "n_images": 1},
        "test": reported,
    }
    update_candidates = [
        {
            "image_id": "img-train",
            "sme_truth": "gen_ai",
            "misalignment_type": "consensus_wrong",
            "severity": "high",
            "split": "train",
            "is_boundary": True,
            "repo_rel_path": "images/img-train.png",
        }
    ]
    dq_path = run_dir / "scoring" / "decision_quality.json"
    dq = json.loads(dq_path.read_text(encoding="utf-8"))
    dq.update(
        {
            "reported": reported,
            "by_split": by_split,
            "reported_split": "test",
            "update_candidates": update_candidates,
        }
    )
    _write_json(dq_path, dq)

    payload = aggregator.aggregate_decision_quality(runs_root)
    run = payload["runs"][0]

    assert run["reported"] == reported
    assert run["by_split"] == by_split
    assert run["reported_split"] == "test"
    assert run["update_candidates"] == update_candidates


def test_compute_insights_returns_documented_capped_lists(tmp_path: Path) -> None:
    runs_root = _runs_fixture(tmp_path)
    payload = aggregator.compute_insights(runs_root / "run-early")

    expected_keys = {
        "majority_wrong",
        "model_disagreement",
        "boundary_concentration",
        "consistent_pair_disagreement",
    }
    assert expected_keys <= set(payload)
    for key in expected_keys:
        assert isinstance(payload[key], list)
        assert len(payload[key]) <= 50
    assert payload["majority_wrong"][0]["image_id"] == "img-1"
    assert payload["majority_wrong"][0]["repo_rel_path"] == "images/img-1.png"
    assert payload["model_disagreement"][0]["image_id"] == "img-1"
    assert payload["model_disagreement"][0]["repo_rel_path"] == "images/img-1.png"
    assert payload["consistent_pair_disagreement"][0]["n_disagreements"] == 1


def test_handle_decision_quality_ok_and_not_found(tmp_path: Path, monkeypatch) -> None:
    runs_root = _runs_fixture(tmp_path)
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)

    status, body = handlers_dq.handle_decision_quality({})
    assert status == 200
    assert len(body["runs"]) == 2

    status, body = handlers_dq.handle_decision_quality({"run_id": ["bogus"]})
    assert status == 404
    assert "error" in body


def test_handle_decision_quality_mnist_demo_returns_empty_runs(tmp_path: Path, monkeypatch) -> None:
    runs_root = _runs_fixture(tmp_path)
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)

    status, body = handlers_dq.handle_decision_quality({"demo": ["mnist"]})

    assert status == 200
    assert body == {"runs": [], "policy_versions": []}


def test_aggregate_decision_quality_reads_mnist_multiclass_artifact(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "mnist-run"
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": "mnist-run",
            "started_at": "2026-07-04T12:00:00Z",
            "area": "MNIST_Digits",
            "policy_graph_version": "MNIST_Digits.v0.1",
            "policy_version": "v0.1",
            "sample_ids": ["train_00001"],
        },
    )
    _write_json(
        run_dir / "scoring" / "decision_quality_multiclass.json",
        {
            "policy_graph_version": "MNIST_Digits.v0.1",
            "task": "mnist_multiclass",
            "classes": [str(d) for d in range(10)],
            "labelers": [
                {
                    "labeler_id": "model-a",
                    "labeler_type": "llm",
                    "metrics": {
                        "n": 1,
                        "accuracy": 1.0,
                        "macro_precision": 1.0,
                        "macro_recall": 1.0,
                        "macro_f1": 1.0,
                    },
                }
            ],
        },
    )

    payload = aggregator.aggregate_decision_quality(runs_root, policy_area="MNIST_Digits")

    assert [run["run_id"] for run in payload["runs"]] == ["mnist-run"]
    metrics = payload["runs"][0]["labelers"][0]["metrics"]
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert payload["policy_versions"] == ["MNIST_Digits.v0.1"]


def test_handle_insights_requires_run_id_and_returns_payload(tmp_path: Path, monkeypatch) -> None:
    runs_root = _runs_fixture(tmp_path)
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)

    status, body = handlers_dq.handle_insights({})
    assert status == 400
    assert "error" in body

    status, body = handlers_dq.handle_insights({"run_id": ["run-early"]})
    assert status == 200
    assert body["run_id"] == "run-early"


def test_handle_insights_rejects_run_from_other_demo(tmp_path: Path, monkeypatch) -> None:
    runs_root = _runs_fixture(tmp_path)
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)

    status, body = handlers_dq.handle_insights({"run_id": ["run-early"], "demo": ["mnist"]})

    assert status == 404
    assert "No scored runs matched" in body["error"]


def test_handle_insights_auto_scores_when_artifacts_are_missing(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    run_dir = _make_autoscore_run(runs_root, "run-auto")
    manifest_path = (
        tmp_path
        / "data"
        / "images"
        / "genai-classification"
        / "manifests"
        / "combined_labels.jsonl"
    )
    _write_jsonl(
        manifest_path,
        [
            {
                "sample_id": "dev_golden_0001",
                "repo_rel_path": "images/dev_golden_0001.png",
                "split": "dev_golden",
                "label": "ai_generated",
                "label_int": 1,
                "truth_tier": "gold",
            },
            {
                "sample_id": "dev_golden_0002",
                "repo_rel_path": "images/dev_golden_0002.png",
                "split": "dev_golden",
                "label": "ai_generated",
                "label_int": 1,
                "truth_tier": "gold",
            },
        ],
    )
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(handlers_dq, "REPO_ROOT", tmp_path)

    assert not (run_dir / "scoring").exists()
    status, body = handlers_dq.handle_insights({"run_id": ["run-auto"]})

    assert status == 200
    assert body["run_id"] == "run-auto"
    assert (run_dir / "scoring" / "consensus.json").exists()
    assert (run_dir / "scoring" / "misalignment.json").exists()


def test_handle_insights_does_not_auto_score_without_label_votes(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run-no-votes"
    _write_json(run_dir / "run_manifest.json", {"run_id": "run-no-votes"})
    _write_jsonl(run_dir / "llm_outputs.jsonl", [{"run_id": "run-no-votes"}])
    monkeypatch.setattr(handlers_dq, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(
        handlers_dq,
        "_auto_score_run",
        lambda run_id: (_ for _ in ()).throw(AssertionError("should not auto-score")),
    )

    status, body = handlers_dq.handle_insights({"run_id": ["run-no-votes"]})

    assert status == 404
    assert "missing label_votes.jsonl" in body["error"]
    assert not (run_dir / "scoring").exists()
