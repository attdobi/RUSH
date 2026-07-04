from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import decision_quality as dq_mod  # noqa: E402
from pipeline.scoring import misalignment as mis_mod  # noqa: E402
from pipeline.scoring.run_scoring import run_scoring  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _manifest_row(image_id: str, *, split: str, label_int: int) -> dict:
    return {
        "sample_id": image_id,
        "label": "ai_generated" if label_int else "not_ai_generated",
        "label_int": label_int,
        "truth_tier": "gold",
        "split": split,
        "repo_rel_path": f"data/images/x/{image_id}.png",
    }


def _vote(image_id: str, labeler: str, label: str, *, is_boundary: bool = False) -> dict:
    return {
        "run_id": "split-run",
        "image_id": image_id,
        "labeler_type": "llm",
        "labeler_id": labeler,
        "model_id": labeler,
        "label": label,
        "confidence": 0.8,
        "is_boundary": is_boundary,
        "difficulty": "medium" if is_boundary else "low",
        "justification": "fixture",
        "policy_graph_version": "Generative_AI.v0.1",
    }


def _split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest_rows = [
        _manifest_row("train_wrong", split="dev_golden", label_int=1),
        _manifest_row("train_right", split="dev_golden", label_int=0),
        _manifest_row("test_pos", split="holdout", label_int=1),
        _manifest_row("test_neg", split="holdout", label_int=0),
    ]
    vote_rows = [
        _vote("train_wrong", "claude", "not_gen_ai", is_boundary=True),
        _vote("train_wrong", "gpt", "not_gen_ai"),
        _vote("train_right", "claude", "not_gen_ai"),
        _vote("train_right", "gpt", "not_gen_ai"),
        _vote("test_pos", "claude", "gen_ai"),
        _vote("test_pos", "gpt", "gen_ai"),
        _vote("test_neg", "claude", "not_gen_ai"),
        _vote("test_neg", "gpt", "not_gen_ai"),
    ]
    manifest = tmp_path / "manifest.jsonl"
    votes = tmp_path / "label_votes.jsonl"
    _write_jsonl(manifest, manifest_rows)
    _write_jsonl(votes, vote_rows)
    return votes, manifest


def test_misalignment_records_carry_split(tmp_path: Path) -> None:
    votes, manifest = _split_fixture(tmp_path)

    out = mis_mod.compute_misalignment(
        votes,
        manifest,
        policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )

    by_id = {row["image_id"]: row for row in out["records"]}
    assert by_id["train_wrong"]["split"] == "dev_golden"
    assert by_id["test_pos"]["split"] == "holdout"


def test_decision_quality_by_split_reports_test_only_metrics(tmp_path: Path) -> None:
    votes, manifest = _split_fixture(tmp_path)

    snap = dq_mod.compute_decision_quality(
        votes,
        manifest,
        policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
        schemas_dir=ROOT / "schemas",
    )

    assert snap["by_split"]["train"]["n_images"] == 2
    assert snap["by_split"]["test"]["n_images"] == 2
    assert snap["reported_split"] == "test"
    assert snap["reported"] == snap["by_split"]["test"]

    top_gpt = {row["labeler_id"]: row for row in snap["labelers"]}["gpt"]
    train_gpt = {row["labeler_id"]: row for row in snap["by_split"]["train"]["labelers"]}["gpt"]
    test_gpt = {row["labeler_id"]: row for row in snap["by_split"]["test"]["labelers"]}["gpt"]
    reported_gpt = {row["labeler_id"]: row for row in snap["reported"]["labelers"]}["gpt"]

    assert top_gpt["metrics"]["accuracy"] == 0.75
    assert train_gpt["metrics"]["accuracy"] == 0.5
    assert test_gpt["metrics"]["accuracy"] == 1.0
    assert reported_gpt["metrics"]["accuracy"] == 1.0

    reported_majority = {
        row["labeler_id"]: row for row in snap["reported"]["labelers"]
    }["majority_vote"]
    assert reported_majority["metrics"]["n"] == 2
    assert reported_majority["metrics"]["accuracy"] == 1.0


def test_run_scoring_update_candidates_are_train_only_ordered_and_capped(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "split-run"
    manifest = tmp_path / "manifest.jsonl"
    votes = run_dir / "label_votes.jsonl"

    manifest_rows: list[dict] = []
    vote_rows: list[dict] = []
    for idx in range(30):
        image_id = f"train_high_{idx:02d}"
        manifest_rows.append(_manifest_row(image_id, split="dev_golden", label_int=1))
        vote_rows.extend(
            [
                _vote(image_id, "claude", "not_gen_ai", is_boundary=(idx == 0)),
                _vote(image_id, "gpt", "not_gen_ai"),
            ]
        )
    for idx in range(25):
        image_id = f"train_medium_{idx:02d}"
        manifest_rows.append(_manifest_row(image_id, split="dev_golden", label_int=1))
        vote_rows.extend(
            [
                _vote(image_id, "claude", "gen_ai"),
                _vote(image_id, "gpt", "not_gen_ai"),
            ]
        )
    manifest_rows.append(_manifest_row("train_all_agree", split="dev_golden", label_int=0))
    vote_rows.extend(
        [
            _vote("train_all_agree", "claude", "not_gen_ai"),
            _vote("train_all_agree", "gpt", "not_gen_ai"),
        ]
    )
    manifest_rows.append(_manifest_row("test_wrong", split="holdout", label_int=1))
    vote_rows.extend(
        [
            _vote("test_wrong", "claude", "not_gen_ai"),
            _vote("test_wrong", "gpt", "not_gen_ai"),
        ]
    )
    _write_jsonl(manifest, manifest_rows)
    _write_jsonl(votes, vote_rows)

    result = run_scoring(
        "split-run",
        ROOT,
        runs_root=runs_root,
        manifest=manifest,
        ground_truth_tier=("gold",),
        validate_schemas=True,
    )
    dq = json.loads((run_dir / "scoring" / "decision_quality.json").read_text(encoding="utf-8"))
    candidates = dq["update_candidates"]

    assert result["update_candidates_count"] == 50
    assert len(candidates) == 50
    assert {row["split"] for row in candidates} == {"train"}
    assert "test_wrong" not in {row["image_id"] for row in candidates}
    assert "train_all_agree" not in {row["image_id"] for row in candidates}
    assert [row["severity"] for row in candidates[:30]] == ["high"] * 30
    assert [row["severity"] for row in candidates[30:]] == ["medium"] * 20
    assert candidates[0]["misalignment_type"] == "consensus_wrong"
    assert candidates[0]["is_boundary"] is True
    assert candidates[-1]["misalignment_type"] == "model_vs_sme"
