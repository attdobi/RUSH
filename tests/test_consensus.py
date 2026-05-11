"""Tests for the per-image consensus / majority-vote layer."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import consensus as consensus_mod  # noqa: E402
from pipeline.scoring._common import load_ground_truth  # noqa: E402


def _vote(image_id, labeler, label, *, confidence=0.8, is_boundary=False, run_id="r1"):
    return {
        "run_id": run_id,
        "image_id": image_id,
        "labeler_type": "llm",
        "labeler_id": labeler,
        "model_id": labeler,
        "label": label,
        "node_ids": [],
        "confidence": confidence,
        "l2_label": "GA.surface_texture.plastic_skin",
        "difficulty": "medium",
        "justification": "x",
        "policy_citations": ["GA.surface_texture.plastic_skin"],
        "policy_quotes": ["Waxy or over-smoothed surfaces can be positive evidence."],
        "justification_too_long": False,
        "input_tokens": 101,
        "output_tokens": 17,
        "cost_usd": 0.0123,
        "policy_graph_version": "Generative_AI.v0.1",
        "is_boundary": is_boundary,
    }


# ---------------------------------------------------------------------------
# build_consensus_records
# ---------------------------------------------------------------------------

def test_consensus_3_of_3_unanimous():
    votes = [
        _vote("img1", "a", "gen_ai"),
        _vote("img1", "b", "gen_ai"),
        _vote("img1", "c", "gen_ai"),
    ]
    out = consensus_mod.build_consensus_records(votes, computed_at="2026-01-01T00:00:00Z")
    assert len(out) == 1
    r = out[0]
    assert r["majority_label"] == "gen_ai"
    assert r["majority_count"] == 3
    assert r["majority_fraction"] == 1.0
    assert r["n_votes_total"] == 3
    assert r["n_votes_decisive"] == 3
    assert r["n_abstain"] == 0
    assert r["is_unanimous"] is True
    assert r["is_consensus"] is True
    assert r["is_split"] is False
    assert r["tie"] is False
    assert r["vote_distribution"] == {"gen_ai": 3}
    assert r["any_boundary_flag"] is False
    assert r["boundary_voter_count"] == 0


def test_consensus_2_of_3_split():
    votes = [
        _vote("img1", "a", "not_gen_ai", confidence=0.82),
        _vote("img1", "b", "not_gen_ai", confidence=0.0),
        _vote("img1", "c", "gen_ai", confidence=0.72, is_boundary=True),
    ]
    [r] = consensus_mod.build_consensus_records(votes, computed_at="2026-01-01T00:00:00Z")
    assert r["majority_label"] == "not_gen_ai"
    assert r["majority_count"] == 2
    assert r["majority_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert r["is_unanimous"] is False
    assert r["is_consensus"] is False
    assert r["is_split"] is True
    assert r["tie"] is False
    assert r["any_boundary_flag"] is True
    assert r["boundary_voter_count"] == 1


def test_consensus_three_way_tie():
    # 1/1/1 across three labels — not a v1 cold-start vocabulary mix in practice
    # but we still exercise the math; gen_ai vs not_gen_ai vs positive.
    votes = [
        _vote("img1", "a", "gen_ai"),
        _vote("img1", "b", "not_gen_ai"),
        _vote("img1", "c", "positive"),
    ]
    [r] = consensus_mod.build_consensus_records(votes)
    assert r["tie"] is True
    assert r["majority_label"] is None
    assert r["majority_count"] == 1
    assert r["majority_fraction"] == pytest.approx(1 / 3, rel=1e-3)
    assert r["is_split"] is True
    assert r["is_unanimous"] is False
    assert r["is_consensus"] is False


def test_consensus_two_decisive_with_one_abstain_unanimous_among_decisive():
    votes = [
        _vote("img1", "a", "gen_ai"),
        _vote("img1", "b", "gen_ai"),
        _vote("img1", "c", "abstain"),
    ]
    [r] = consensus_mod.build_consensus_records(votes)
    assert r["n_votes_total"] == 3
    assert r["n_votes_decisive"] == 2
    assert r["n_abstain"] == 1
    assert r["majority_label"] == "gen_ai"
    assert r["majority_count"] == 2
    assert r["majority_fraction"] == 1.0
    # consensus among decisive voters, but NOT unanimous (one abstained)
    assert r["is_consensus"] is True
    assert r["is_unanimous"] is False
    assert r["is_split"] is False
    assert r["tie"] is False


def test_consensus_all_abstain():
    votes = [
        _vote("img1", "a", "abstain"),
        _vote("img1", "b", "abstain"),
    ]
    [r] = consensus_mod.build_consensus_records(votes)
    assert r["n_votes_total"] == 2
    assert r["n_votes_decisive"] == 0
    assert r["n_abstain"] == 2
    assert r["majority_label"] is None
    assert r["majority_count"] == 0
    assert r["majority_fraction"] is None
    assert r["is_unanimous"] is False
    assert r["is_consensus"] is False
    assert r["is_split"] is False
    assert r["tie"] is False
    assert r["vote_distribution"] == {}


def test_consensus_zero_votes_edge_case():
    out = consensus_mod.build_consensus_records([])
    assert out == []


def test_consensus_voter_audit_shape_and_ordering():
    votes = [
        _vote("img1", "zeta", "gen_ai", confidence=0.5, is_boundary=True),
        _vote("img1", "alpha", "gen_ai", confidence=0.9, is_boundary=False),
    ]
    [r] = consensus_mod.build_consensus_records(votes)
    voters = r["voters"]
    assert [v["labeler_id"] for v in voters] == ["alpha", "zeta"]
    assert voters[0]["confidence"] == 0.9
    assert voters[0]["is_boundary"] is False
    # v2 (X2 wave-2): voter audit carries the per-call evidence inline so the
    # web layer can render justifications/citations/cost without a separate
    # fetch. Pass-through is best-effort: only keys present on the source vote
    # are echoed onto the voter block.
    assert voters[0]["justification"] == "x"
    assert voters[0]["l2_label"] == "GA.surface_texture.plastic_skin"
    assert voters[0]["difficulty"] == "medium"
    assert voters[0]["policy_citations"] == ["GA.surface_texture.plastic_skin"]
    assert voters[0]["policy_quotes"] == ["Waxy or over-smoothed surfaces can be positive evidence."]
    assert voters[0]["justification_too_long"] is False
    assert voters[0]["input_tokens"] == 101
    assert voters[0]["output_tokens"] == 17
    assert voters[0]["cost_usd"] == 0.0123
    assert voters[0]["label"] == "gen_ai"


# ---------------------------------------------------------------------------
# build_cohort_rollups
# ---------------------------------------------------------------------------

def test_cohort_rollups_basic():
    records = [
        # img1: unanimous gen_ai
        {"image_id": "img1", "is_unanimous": True, "is_consensus": True,
         "is_split": False, "tie": False, "any_boundary_flag": False,
         "majority_label": "gen_ai",
         "voters": [
             {"labeler_id": "a", "label": "gen_ai"},
             {"labeler_id": "b", "label": "gen_ai"},
         ]},
        # img2: split, majority not_gen_ai
        {"image_id": "img2", "is_unanimous": False, "is_consensus": False,
         "is_split": True, "tie": False, "any_boundary_flag": True,
         "majority_label": "not_gen_ai",
         "voters": [
             {"labeler_id": "a", "label": "not_gen_ai"},
             {"labeler_id": "b", "label": "gen_ai"},
         ]},
        # img3: tie -> majority_label None, agreement should not credit anyone
        {"image_id": "img3", "is_unanimous": False, "is_consensus": False,
         "is_split": True, "tie": True, "any_boundary_flag": False,
         "majority_label": None,
         "voters": [
             {"labeler_id": "a", "label": "gen_ai"},
             {"labeler_id": "b", "label": "not_gen_ai"},
         ]},
    ]
    out = consensus_mod.build_cohort_rollups(records)
    assert out["n_images_total"] == 3
    assert out["n_images_unanimous"] == 1
    assert out["n_images_split"] == 2
    assert out["n_images_with_tie"] == 1
    assert out["n_images_with_boundary_flag"] == 1
    per = out["per_model_vs_majority_agreement"]
    # 'a' agreed on img1 and img2 (2/3 decisive votes); img3 has no majority.
    assert per["a"]["n_votes"] == 3
    assert per["a"]["n_agreed"] == 2
    assert per["a"]["agreement_with_majority"] == pytest.approx(2 / 3, rel=1e-3)
    # 'b' agreed on img1 only (1/3 decisive).
    assert per["b"]["n_agreed"] == 1


# ---------------------------------------------------------------------------
# End-to-end smoke against the live example run
# ---------------------------------------------------------------------------

def test_smoke_against_live_dev_golden_run():
    votes_path = ROOT / "data" / "runs" / "20260510T184058-0d4479b2" / "label_votes.jsonl"
    if not votes_path.exists():
        pytest.skip(f"live example run not present: {votes_path}")
    rows = [json.loads(line) for line in votes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = consensus_mod.build_consensus_records(rows, computed_at="2026-01-01T00:00:00Z")
    by_id = {r["image_id"]: r for r in out}
    r = by_id["dev_golden_0001"]
    assert r["majority_label"] == "not_gen_ai"
    assert r["majority_count"] == 2
    assert r["majority_fraction"] == pytest.approx(2 / 3, rel=1e-3)
    assert r["n_votes_total"] == 3
    assert r["n_votes_decisive"] == 3
    assert r["any_boundary_flag"] is True
    assert r["boundary_voter_count"] == 1
    assert r["is_unanimous"] is False
    assert r["is_split"] is True
    assert r["tie"] is False


# ---------------------------------------------------------------------------
# Wiring smoke test via score_labels (exporter writes consensus.jsonl + web/)
# ---------------------------------------------------------------------------

def test_score_labels_writes_consensus_artifacts(tmp_path: Path, monkeypatch):
    # Build a minimal run directory + manifest from scratch and call the
    # exporter the same way scripts/score_labels.py does.
    from pipeline.scoring import (
        borderline as borderline_mod,
        decision_quality as dq_mod,
        exporters as exporters_mod,
        misalignment as mis_mod,
    )
    run_dir = tmp_path / "runs" / "rtest"
    run_dir.mkdir(parents=True)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "sample_id": "img1", "label": "ai_generated", "label_int": 1,
        "truth_tier": "gold", "split": "dev_golden",
        "repo_rel_path": "data/images/img1.png",
    }) + "\n", encoding="utf-8")

    votes_path = run_dir / "label_votes.jsonl"
    votes_path.write_text("\n".join(json.dumps(v, sort_keys=True) for v in [
        _vote("img1", "a", "gen_ai"),
        _vote("img1", "b", "gen_ai"),
        _vote("img1", "c", "not_gen_ai"),
    ]) + "\n", encoding="utf-8")

    dq = dq_mod.compute_decision_quality(
        votes_path, manifest, policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )
    mis = mis_mod.compute_misalignment(
        votes_path, manifest, policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )
    bord = borderline_mod.compute_borderline(
        votes_path, manifest, policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
        low_confidence_threshold=0.6,
    )
    raw = [json.loads(line) for line in votes_path.read_text().splitlines() if line.strip()]
    records = consensus_mod.build_consensus_records(raw, run_id="rtest", computed_at="2026-01-01T00:00:00Z")
    truth = load_ground_truth(manifest, truth_tiers=("gold",))
    rollup = consensus_mod.build_cohort_rollups(records, ground_truth=truth)
    consensus_summary = {
        "run_id": "rtest",
        "policy_graph_version": "Generative_AI.v0.1",
        "ground_truth_tier": ["gold"],
        "summary": rollup,
        "records": records,
    }
    written = exporters_mod.write_web_exports(
        run_dir, decision_quality=dq, misalignment=mis,
        borderline=bord, consensus=consensus_summary, run_id="rtest",
    )
    assert "consensus" in written
    web_consensus = json.loads((run_dir / "web" / "consensus.json").read_text())
    assert web_consensus["summary"]["n_images_total"] == 1
    assert web_consensus["records"][0]["majority_label"] == "gen_ai"
    web_summary = json.loads((run_dir / "web" / "summary.json").read_text())
    assert "consensus_summary" in web_summary
    assert web_summary["consensus_summary"]["n_images_total"] == 1
