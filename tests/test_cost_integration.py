from __future__ import annotations

import json
from pathlib import Path

from pipeline.scoring import cost as cost_mod
from pipeline.scoring import decision_quality as dq_mod
from pipeline.scoring._common import try_validate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_cost_attach_to_decision_quality_validates_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "img_1",
                "label": "ai_generated",
                "label_int": 1,
                "truth_tier": "gold",
                "split": "dev_golden",
                "repo_rel_path": "data/images/x/img_1.png",
            },
            {
                "sample_id": "img_2",
                "label": "not_ai_generated",
                "label_int": 0,
                "truth_tier": "gold",
                "split": "dev_golden",
                "repo_rel_path": "data/images/x/img_2.png",
            },
        ],
    )
    votes = [
        {
            "run_id": "r1",
            "image_id": "img_1",
            "labeler_type": "llm",
            "labeler_id": "openai/gpt-5.5",
            "model_id": "openai/gpt-5.5",
            "label": "gen_ai",
            "node_ids": [],
            "confidence": 0.9,
            "justification": "synthetic test vote",
            "policy_graph_version": "Generative_AI.v0.1",
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cost_usd": 2.25,
        },
        {
            "run_id": "r1",
            "image_id": "img_2",
            "labeler_type": "llm",
            "labeler_id": "openai/gpt-5.5",
            "model_id": "openai/gpt-5.5",
            "label": "not_gen_ai",
            "node_ids": [],
            "confidence": 0.8,
            "justification": "synthetic test vote",
            "policy_graph_version": "Generative_AI.v0.1",
            "input_tokens": 500_000,
            "output_tokens": 50_000,
            "cost_usd": 1.125,
        },
    ]
    votes_path = tmp_path / "label_votes.jsonl"
    _write_jsonl(votes_path, votes)

    dq = dq_mod.compute_decision_quality(
        votes_path,
        manifest,
        policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )
    costs = cost_mod.aggregate_per_call_costs(votes)
    out = cost_mod.attach_cost_to_labelers(dq, costs)

    assert out["cost"]["total_cost_usd"] == 3.375
    assert out["cost"]["total_calls"] == 2
    by_id = {row["labeler_id"]: row for row in out["labelers"]}
    assert by_id["openai/gpt-5.5"]["metrics"]["cost_per_1000_labels"] == 1687.5
    assert by_id["majority_vote"]["metrics"]["cost_per_1000_labels"] is None
    assert try_validate(
        out,
        Path(__file__).resolve().parents[1] / "schemas" / "decision-quality.schema.json",
        label="decision-quality",
    ) == []
