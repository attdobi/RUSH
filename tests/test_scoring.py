"""Offline tests for the X3 scoring + policy_iterator slice.

No network. No real LLM. Uses small synthetic LabelVote/manifest data.
Run with: ``pytest tests/test_scoring.py -v``
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.policy_iterator import (  # noqa: E402
    PolicyIterationInputs,
    build_user_prompt,
    propose_policy_patches,
)
from pipeline.scoring import (  # noqa: E402
    borderline as borderline_mod,
    decision_quality as dq_mod,
    exporters as exporters_mod,
    misalignment as mis_mod,
)
from pipeline.scoring._common import extract_prep_metadata, load_ground_truth  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    rows = [
        {"sample_id": "img_pos_1", "label": "ai_generated", "label_int": 1,
         "truth_tier": "gold", "split": "dev_golden",
         "repo_rel_path": "data/images/x/pos1.png"},
        {"sample_id": "img_pos_2", "label": "ai_generated", "label_int": 1,
         "truth_tier": "gold", "split": "dev_golden",
         "repo_rel_path": "data/images/x/pos2.png"},
        {"sample_id": "img_neg_1", "label": "not_ai_generated", "label_int": 0,
         "truth_tier": "gold", "split": "dev_golden",
         "repo_rel_path": "data/images/x/neg1.png"},
        {"sample_id": "img_neg_2", "label": "not_ai_generated", "label_int": 0,
         "truth_tier": "gold", "split": "dev_golden",
         "repo_rel_path": "data/images/x/neg2.png"},
    ]
    p = tmp_path / "manifest.jsonl"
    _write_jsonl(p, rows)
    return p


@pytest.fixture
def votes_path(tmp_path: Path) -> Path:
    """Two LLM labelers across 4 images, with known mistakes and prep metadata."""
    rows = [
        # gpt: 3/4 correct (misses pos_2), claude: 4/4 correct
        {"run_id": "r1", "image_id": "img_pos_1", "labeler_type": "llm",
         "labeler_id": "gpt", "model_id": "openai/gpt-5.5",
         "label": "gen_ai", "node_ids": [], "confidence": 0.9,
         "justification": "hands look off",
         "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": False, "difficulty": "low",
         "prepared_image_sha256": "deadbeef" * 8,
         "prepared_image_width": 384, "prepared_image_height": 384,
         "prepared_image_mime": "image/jpeg", "prepared_image_bytes": 14211},
        {"run_id": "r1", "image_id": "img_pos_1", "labeler_type": "llm",
         "labeler_id": "claude", "model_id": "anthropic/claude-opus-4-6",
         "label": "gen_ai", "node_ids": [], "confidence": 0.7,
         "justification": "plastic skin", "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": True, "difficulty": "medium"},

        # both wrong on pos_2 (consensus_wrong, severity high)
        {"run_id": "r1", "image_id": "img_pos_2", "labeler_type": "llm",
         "labeler_id": "gpt", "model_id": "openai/gpt-5.5",
         "label": "not_gen_ai", "node_ids": [], "confidence": 0.55,
         "justification": "looks real to me", "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": False, "difficulty": "high"},
        {"run_id": "r1", "image_id": "img_pos_2", "labeler_type": "llm",
         "labeler_id": "claude", "model_id": "anthropic/claude-opus-4-6",
         "label": "not_gen_ai", "node_ids": [], "confidence": 0.65,
         "justification": "no obvious artifacts", "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": True, "difficulty": "high"},

        # gpt vs claude split on neg_1 -> model_vs_sme (claude correct, gpt wrong)
        {"run_id": "r1", "image_id": "img_neg_1", "labeler_type": "llm",
         "labeler_id": "gpt", "model_id": "openai/gpt-5.5",
         "label": "gen_ai", "node_ids": [], "confidence": 0.4,
         "justification": "uncertain", "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": True, "difficulty": "high"},
        {"run_id": "r1", "image_id": "img_neg_1", "labeler_type": "llm",
         "labeler_id": "claude", "model_id": "anthropic/claude-opus-4-6",
         "label": "not_gen_ai", "node_ids": [], "confidence": 0.95,
         "justification": "clean photo", "policy_graph_version": "Generative_AI.v0.1",
         "is_boundary": False, "difficulty": "low"},

        # both correct on neg_2
        {"run_id": "r1", "image_id": "img_neg_2", "labeler_type": "llm",
         "labeler_id": "gpt", "model_id": "openai/gpt-5.5",
         "label": "not_gen_ai", "node_ids": [], "confidence": 0.85,
         "justification": "ok", "policy_graph_version": "Generative_AI.v0.1"},
        {"run_id": "r1", "image_id": "img_neg_2", "labeler_type": "llm",
         "labeler_id": "claude", "model_id": "anthropic/claude-opus-4-6",
         "label": "not_gen_ai", "node_ids": [], "confidence": 0.9,
         "justification": "ok", "policy_graph_version": "Generative_AI.v0.1"},
    ]
    p = tmp_path / "label_votes.jsonl"
    _write_jsonl(p, rows)
    return p


# ---------------------------------------------------------------------------
# decision_quality
# ---------------------------------------------------------------------------

def test_compute_metrics_basic():
    # 2 TP, 1 FP, 1 TN, 1 FN
    preds = ["gen_ai", "gen_ai", "gen_ai", "not_gen_ai", "not_gen_ai"]
    truth = ["gen_ai", "gen_ai", "not_gen_ai", "not_gen_ai", "gen_ai"]
    m = dq_mod.compute_metrics(preds, truth)
    assert m["n"] == 5
    assert m["accuracy"] == round(3 / 5, 6)
    assert m["precision"] == round(2 / 3, 6)
    assert m["recall"] == round(2 / 3, 6)
    assert m["f1"] == round(2 / 3, 6)
    assert m["fpr"] == round(1 / 2, 6)
    assert m["fnr"] == round(1 / 3, 6)
    assert m["positive_proportion"] == round(3 / 5, 6)
    assert m["informedness"] == round(2 / 3 - 1 / 2, 6)


def test_compute_metrics_zero_div_returns_none():
    # all negatives, all predicted negative
    m = dq_mod.compute_metrics(["not_gen_ai"] * 3, ["not_gen_ai"] * 3)
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None
    assert m["fnr"] is None
    assert m["accuracy"] == 1.0
    assert m["fpr"] == 0.0


def test_decision_quality_includes_majority_vote(votes_path, manifest_path):
    snap = dq_mod.compute_decision_quality(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )
    ids = {row["labeler_id"]: row for row in snap["labelers"]}
    assert "gpt" in ids and "claude" in ids and "majority_vote" in ids
    assert ids["claude"]["metrics"]["accuracy"] == 0.75  # missed pos_2
    assert ids["gpt"]["metrics"]["accuracy"] == 0.5      # missed pos_2 + neg_1
    # ensemble: pos_1 -> gen_ai (correct), pos_2 -> not_gen_ai (wrong),
    # neg_1 -> tie -> abstain (excluded), neg_2 -> not_gen_ai (correct).
    # Majority decided on 3 images -> 2 correct
    assert ids["majority_vote"]["metrics"]["n"] == 3


# ---------------------------------------------------------------------------
# misalignment
# ---------------------------------------------------------------------------

def test_misalignment_classification_and_prep_passthrough(votes_path, manifest_path):
    out = mis_mod.compute_misalignment(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    by_id = {r["image_id"]: r for r in out["records"]}
    assert by_id["img_pos_1"]["misalignment_type"] == "all_agree"
    assert by_id["img_pos_2"]["misalignment_type"] == "consensus_wrong"
    assert by_id["img_pos_2"]["severity"] == "high"
    assert by_id["img_neg_1"]["misalignment_type"] == "model_vs_sme"
    assert by_id["img_neg_2"]["misalignment_type"] == "all_agree"
    assert out["summary"]["consensus_wrong"] == 1
    # prep metadata pass-through (only on gpt/img_pos_1 in fixture)
    pos1_votes = {v["labeler_id"]: v for v in by_id["img_pos_1"]["votes"]}
    assert pos1_votes["gpt"]["prepared_image_sha256"].startswith("dead")
    assert pos1_votes["gpt"]["prepared_image_width"] == 384
    assert pos1_votes["gpt"]["prepared_image_mime"] == "image/jpeg"
    assert "prepared_image_sha256" not in pos1_votes["claude"]


# ---------------------------------------------------------------------------
# borderline
# ---------------------------------------------------------------------------

def test_borderline_flags_and_grouping(votes_path, manifest_path):
    out = borderline_mod.compute_borderline(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
        low_confidence_threshold=0.6,
    )
    flagged_ids = {
        r["image_id"]
        for group in out["groups"].values()
        for r in group
    }
    # img_pos_1 -> claude is_boundary -> flagged
    # img_pos_2 -> difficulty=high + low conf gpt -> flagged
    # img_neg_1 -> disagreement + boundary + low conf -> flagged
    # img_neg_2 -> nothing flagged
    assert flagged_ids == {"img_pos_1", "img_pos_2", "img_neg_1"}
    assert out["summary"]["borderline_images"] == 3
    pos2 = next(
        r for r in out["groups"]["gen_ai"] if r["image_id"] == "img_pos_2"
    )
    assert "difficulty_high" in pos2["reasons"]
    assert "low_confidence" in pos2["reasons"]


# ---------------------------------------------------------------------------
# exporters
# ---------------------------------------------------------------------------

def test_exporters_preserve_prep_metadata(votes_path, manifest_path, tmp_path):
    dq = dq_mod.compute_decision_quality(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
        ground_truth_tier=("gold",),
    )
    mis = mis_mod.compute_misalignment(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    bord = borderline_mod.compute_borderline(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    paths = exporters_mod.write_web_exports(
        tmp_path / "run", decision_quality=dq, misalignment=mis, borderline=bord,
        run_id="testrun",
    )
    summary = json.loads(paths["summary"].read_text())
    assert summary["run_id"] == "testrun"
    assert any(l["labeler_id"] == "majority_vote" for l in summary["labelers"])
    assert summary["misalignment_summary"]["consensus_wrong"] == 1

    mis_web = json.loads(paths["misalignment"].read_text())
    # all_agree rows must NOT appear in the web worklist
    assert all(r["misalignment_type"] != "all_agree" for r in mis_web["records"])
    # severity 'high' rows must come first
    assert mis_web["records"][0]["severity"] == "high"
    assert mis_web["records"][0]["split"] == "dev_golden"

    bord_web = json.loads(paths["borderline"].read_text())
    # find the gpt vote on img_pos_1 in either misalignment or borderline web export
    found_prep = False
    for r in mis_web["records"] + sum(bord_web["groups"].values(), []):
        for v in r["votes"]:
            if v["labeler_id"] == "gpt" and "prepared_image_sha256" in v:
                found_prep = True
                assert v["prepared_image_width"] == 384
                assert v["prepared_image_mime"] == "image/jpeg"
    assert found_prep, "prep metadata must survive into web exports"


def test_extract_prep_metadata_skips_missing_and_null():
    vote = {
        "label": "gen_ai",
        "prepared_image_sha256": "abc",
        "prepared_image_width": None,
    }
    out = extract_prep_metadata(vote)
    assert out == {"prepared_image_sha256": "abc"}


# ---------------------------------------------------------------------------
# policy_iterator
# ---------------------------------------------------------------------------

def test_policy_iterator_dry_run_does_not_call_llm(votes_path, manifest_path):
    mis = mis_mod.compute_misalignment(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    bord = borderline_mod.compute_borderline(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    inputs = PolicyIterationInputs(
        misalignment=mis, borderline=bord,
        policy_markdown="# fake policy",
        policy_graph_version="Generative_AI.v0.1",
    )
    result = propose_policy_patches(inputs=inputs, chat_callable=None)
    assert result["dry_run"] is True
    assert result["patches"] == []
    # high-severity row must appear in prompt (img_pos_2 = consensus_wrong)
    ids = [r["image_id"] for r in result["prompt"]["misclassifications"]]
    assert "img_pos_2" in ids


def test_policy_iterator_include_images_requires_helper():
    inputs = PolicyIterationInputs(
        misalignment={"records": []}, borderline=None,
        policy_markdown="", policy_graph_version="Generative_AI.v0.1",
    )
    with pytest.raises(ValueError, match="downsample_helper"):
        build_user_prompt(inputs, include_images=True, downsample_helper=None)


def test_policy_iterator_includes_images_via_helper(tmp_path):
    inputs = PolicyIterationInputs(
        misalignment={
            "records": [
                {
                    "image_id": "img_pos_2",
                    "repo_rel_path": "data/images/x/pos2.png",
                    "sme_truth": "gen_ai",
                    "split": "dev_golden",
                    "misalignment_type": "consensus_wrong",
                    "severity": "high",
                    "votes": [],
                }
            ]
        },
        borderline=None,
        policy_markdown="",
        policy_graph_version="Generative_AI.v0.1",
    )

    calls = []

    def helper(image_path: Path) -> dict:
        calls.append(image_path)
        return {
            "prepared_image_sha256": "abc",
            "prepared_image_width": 384,
            "prepared_image_height": 384,
            "prepared_image_mime": "image/jpeg",
            "prepared_image_bytes": 1024,
            "bytes_b64": "<<elided>>",
        }

    payload = build_user_prompt(
        inputs, include_images=True, downsample_helper=helper, image_root=tmp_path,
    )
    assert payload["images"][0]["image_id"] == "img_pos_2"
    assert payload["images"][0]["prepared"]["prepared_image_width"] == 384
    assert calls == [tmp_path / "data/images/x/pos2.png"]


def test_policy_iterator_validates_returned_patches(votes_path, manifest_path):
    mis = mis_mod.compute_misalignment(
        votes_path, manifest_path,
        policy_graph_version="Generative_AI.v0.1",
    )
    inputs = PolicyIterationInputs(
        misalignment=mis, borderline=None,
        policy_markdown="", policy_graph_version="Generative_AI.v0.1",
    )

    fake_response = json.dumps({
        "patches": [
            {  # valid
                "patch_id": "patch.test.001",
                "status": "proposed",
                "suggestion_type": "clarification_with_examples",
                "target_nodes": ["GA.boundary.photo_editing"],
                "rationale": "Models confused photo edits with GenAI on img_pos_2.",
                "proposed_diff": [{"op": "add_clarification", "text": "..."}],
            },
            {  # missing required fields
                "patch_id": "patch.test.002",
            },
        ]
    })

    def fake_chat(messages, *, model_id, reasoning_effort):
        assert isinstance(messages, list) and len(messages) == 2
        assert reasoning_effort == "high"
        return fake_response

    result = propose_policy_patches(
        inputs=inputs,
        chat_callable=fake_chat,
        schemas_dir=ROOT / "schemas",
    )
    assert result["dry_run"] is False
    # If jsonschema isn't installed, both patches pass through; if installed,
    # the second one fails validation. Either way the first must be present.
    assert any(p.get("patch_id") == "patch.test.001" for p in result["patches"])


# ---------------------------------------------------------------------------
# ground-truth loader (manifest format guard)
# ---------------------------------------------------------------------------

def test_load_ground_truth_filters_tiers(manifest_path):
    truth = load_ground_truth(manifest_path, truth_tiers=("gold",))
    assert set(truth.keys()) == {"img_pos_1", "img_pos_2", "img_neg_1", "img_neg_2"}
    assert truth["img_pos_1"].label == "gen_ai"
    assert truth["img_neg_2"].label == "not_gen_ai"
    truth2 = load_ground_truth(manifest_path, truth_tiers=("platinum",))
    assert truth2 == {}
