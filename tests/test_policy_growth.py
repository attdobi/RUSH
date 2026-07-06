"""Tests for cold-start seed + stratified batch growth policy proposals."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.policy_diff import (
    _stratified_batch_rows,
    accept_proposal,
    get_proposal,
    propose_growth_batch,
    seed_cold_start_proposal,
)
from pipeline.policy_iterator import _select_priority_rows


COLD_ROOT_MD = (
    "---\n"
    "id: GA.root\n"
    "version: v0.1\n"
    "title: Root\n"
    "area: Generative_AI\n"
    "node_type: root\n"
    "polarity: mixed\n"
    "parent: null\n"
    "status: draft\n"
    "edges: []\n"
    "---\n"
    "# Root\n\nSeed root.\n"
)


def _seed_policy_graph(tmp_path: Path) -> Path:
    """Seed a v0.1 graph used by grow-batch tests."""
    root = tmp_path
    base = root / "policy-graph" / "Generative_AI" / "v0.1"
    base.mkdir(parents=True)
    (base / "GA.root.md").write_text(COLD_ROOT_MD, encoding="utf-8")
    (base / "GA.visual_artifacts.md").write_text(
        "---\nid: GA.visual_artifacts\nversion: v0.1\ntitle: Visual artifacts\n"
        "area: Generative_AI\nnode_type: category\npolarity: positive\n"
        "parent: GA.root\nstatus: draft\nedges: []\n---\n# Visual artifacts\n",
        encoding="utf-8",
    )
    (root / "data").mkdir()
    return root


def _write_misalignment(
    root: Path,
    run_id: str,
    *,
    n_pos: int,
    n_neg: int,
    split: str = "dev_golden",
) -> Path:
    run_dir = root / "data" / "runs" / run_id
    scoring = run_dir / "scoring"
    scoring.mkdir(parents=True)
    records = []
    for i in range(n_pos):
        records.append(
            {
                "image_id": f"pos_{i:03d}",
                "sme_truth": "gen_ai",
                "split": split,
                "misalignment_type": "model_wrong",
                "severity": "high",
                "votes": [
                    {
                        "labeler_id": "openai/gpt-5.5",
                        "label": "not_gen_ai",
                        "l2_label": None,
                        "confidence": 0.7,
                        "is_boundary": False,
                        "difficulty": "medium",
                        "justification": "looked authentic",
                    }
                ],
            }
        )
    for i in range(n_neg):
        records.append(
            {
                "image_id": f"neg_{i:03d}",
                "sme_truth": "not_gen_ai",
                "split": split,
                "misalignment_type": "model_wrong",
                "severity": "high",
                "votes": [
                    {
                        "labeler_id": "openai/gpt-5.5",
                        "label": "gen_ai",
                        "l2_label": None,
                        "confidence": 0.7,
                        "is_boundary": False,
                        "difficulty": "medium",
                        "justification": "looked synthetic",
                    }
                ],
            }
        )
    (scoring / "misalignment.json").write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# cold-start
# ---------------------------------------------------------------------------


def test_seed_cold_start_writes_proposal_and_accept_creates_v01(tmp_path: Path) -> None:
    root = tmp_path
    (root / "data").mkdir()
    # Note: no policy-graph dir at all yet — this is the very first version.

    proposal = seed_cold_start_proposal(
        repo_root=root,
        task_description="Classify whether an image is AI-generated.",
        proposed_files={
            "GA.root.md": COLD_ROOT_MD,
            "GA.visual_artifacts.eyes.md": (
                "---\nid: GA.visual_artifacts.eyes\nversion: v0.1\n"
                "title: Eyes\narea: Generative_AI\nnode_type: leaf\n"
                "polarity: positive\nparent: GA.root\nstatus: draft\nedges: []\n"
                "---\n# Eyes\n"
            ),
        },
    )

    assert proposal["kind"] == "cold_start"
    assert proposal["base_version"] is None
    assert proposal["status"] == "pending"
    assert proposal["domain"] == "Generative_AI"
    assert proposal["files_changed"] == []
    assert sorted(proposal["files_added"]) == [
        "GA.root.md",
        "GA.visual_artifacts.eyes.md",
    ]
    assert proposal["files_removed"] == []
    assert proposal["task_description"].startswith("Classify")

    prop_dir = root / "data" / "policy_proposals" / proposal["proposal_id"]
    assert (prop_dir / "proposal.json").is_file()
    assert (prop_dir / "proposed" / "GA.root.md").is_file()

    # Accept the cold-start proposal — should create v0.1 from scratch.
    accepted = accept_proposal(
        repo_root=root, proposal_id=proposal["proposal_id"]
    )
    assert accepted["new_version"] == "v0.1"
    v01 = root / "policy-graph" / "Generative_AI" / "v0.1"
    assert v01.is_dir()
    assert (v01 / "GA.root.md").read_text(encoding="utf-8") == COLD_ROOT_MD
    assert (v01 / "GA.visual_artifacts.eyes.md").is_file()

    # get_proposal must not crash on base_version=None
    detail = get_proposal(repo_root=root, proposal_id=proposal["proposal_id"])
    assert all(diff["change"] == "added" for diff in detail["diffs"])


def test_seed_cold_start_rejects_empty_task_description(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError):
        seed_cold_start_proposal(
            repo_root=tmp_path,
            task_description="   ",
            proposed_files={"GA.root.md": COLD_ROOT_MD},
        )


def test_seed_cold_start_parse_error(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()

    def bad_chat(messages, *, model_id, reasoning_effort):  # noqa: ARG001
        return "this is not json {"

    proposal = seed_cold_start_proposal(
        repo_root=tmp_path,
        task_description="Classify AI-generated images.",
        chat_callable=bad_chat,
    )
    assert proposal["status"] == "parse_error"
    assert proposal["kind"] == "cold_start"
    assert proposal["base_version"] is None
    prop_dir = tmp_path / "data" / "policy_proposals" / proposal["proposal_id"]
    assert prop_dir.is_dir()
    raw = (prop_dir / "raw_response.txt").read_text(encoding="utf-8")
    assert "not json" in raw


# ---------------------------------------------------------------------------
# grow-batch stratification
# ---------------------------------------------------------------------------


def test_grow_batch_balanced_first_batch(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    run_id = "20260518T180000-abcdef01"
    _write_misalignment(root, run_id, n_pos=3, n_neg=3)

    proposed = {
        "GA.visual_artifacts.eyes.md": (
            "---\nid: GA.visual_artifacts.eyes\nversion: v0.1\n"
            "title: Eyes\narea: Generative_AI\nnode_type: leaf\n"
            "polarity: positive\nparent: GA.visual_artifacts\nstatus: draft\n"
            "edges: []\n---\n# Eyes\n"
        ),
    }
    proposal = propose_growth_batch(
        repo_root=root,
        run_id=run_id,
        base_version="v0.1",
        batch_index=0,
        batch_size=4,
        proposed_files=proposed,
    )
    assert proposal["kind"] == "grow_batch"
    assert proposal["batch_index"] == 0
    assert proposal["batch"]["batch_size_requested"] == 4
    assert proposal["batch"]["n_positives"] == 2
    assert proposal["batch"]["n_negatives"] == 2
    assert proposal["batch"]["batch_size_actual"] == 4
    assert proposal["batch"]["sme_truth_positive_label"] == "gen_ai"
    assert proposal["batch"]["sme_truth_negative_label"] == "not_gen_ai"
    assert "GA.visual_artifacts.eyes.md" in proposal["files_added"]
    assert proposal["files_removed"] == []


def test_grow_batch_second_batch_partial(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    run_id = "20260518T180001-abcdef02"
    _write_misalignment(root, run_id, n_pos=3, n_neg=3)

    proposal = propose_growth_batch(
        repo_root=root,
        run_id=run_id,
        base_version="v0.1",
        batch_index=1,
        batch_size=4,
        proposed_files={
            "GA.visual_artifacts.eyes.md": (
                "---\nid: GA.visual_artifacts.eyes\nversion: v0.1\n"
                "title: Eyes\narea: Generative_AI\nnode_type: leaf\n"
                "polarity: positive\nparent: GA.visual_artifacts\nstatus: draft\n"
                "edges: []\n---\n# Eyes\n"
            ),
        },
    )
    # half=2; positives[2:4]=1 row, negatives[2:4]=1 row → 1+1=2 actual.
    assert proposal["batch"]["n_positives"] == 1
    assert proposal["batch"]["n_negatives"] == 1
    assert proposal["batch"]["batch_size_actual"] == 2
    assert proposal["batch"]["batch_size_requested"] == 4


def test_grow_batch_fallback_fills_from_other_class(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    run_id = "20260518T180002-abcdef03"
    _write_misalignment(root, run_id, n_pos=5, n_neg=1)

    proposal = propose_growth_batch(
        repo_root=root,
        run_id=run_id,
        base_version="v0.1",
        batch_index=0,
        batch_size=8,
        proposed_files={
            "GA.visual_artifacts.eyes.md": (
                "---\nid: GA.visual_artifacts.eyes\nversion: v0.1\n"
                "title: Eyes\narea: Generative_AI\nnode_type: leaf\n"
                "polarity: positive\nparent: GA.visual_artifacts\nstatus: draft\n"
                "edges: []\n---\n# Eyes\n"
            ),
        },
    )
    # half=4; pos[0:4]=4 rows, neg[0:4]=1 row → need 3 more, fill from pos[4:7]
    # → 1 leftover pos. Final: pos=5, neg=1, actual=6.
    assert proposal["batch"]["n_positives"] >= 4
    assert proposal["batch"]["n_negatives"] == 1
    assert proposal["batch"]["batch_size_actual"] <= 8
    assert (
        proposal["batch"]["batch_size_actual"]
        == proposal["batch"]["n_positives"] + proposal["batch"]["n_negatives"]
    )


def test_grow_batch_parse_error_persists_raw(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    run_id = "20260518T180003-abcdef04"
    _write_misalignment(root, run_id, n_pos=2, n_neg=2)

    def bad_chat(messages, *, model_id, reasoning_effort):  # noqa: ARG001
        return "definitely not valid json"

    proposal = propose_growth_batch(
        repo_root=root,
        run_id=run_id,
        base_version="v0.1",
        batch_index=0,
        batch_size=4,
        chat_callable=bad_chat,
    )
    assert proposal["status"] == "parse_error"
    assert proposal["kind"] == "grow_batch"
    assert proposal["base_version"] == "v0.1"
    prop_dir = root / "data" / "policy_proposals" / proposal["proposal_id"]
    assert prop_dir.is_dir()
    raw = (prop_dir / "raw_response.txt").read_text(encoding="utf-8")
    assert "not valid json" in raw


def test_grow_batch_missing_misalignment_raises(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    with pytest.raises(FileNotFoundError):
        propose_growth_batch(
            repo_root=root,
            run_id="no-such-run",
            base_version="v0.1",
            batch_index=0,
            batch_size=4,
            proposed_files={
                "GA.visual_artifacts.eyes.md": "---\nid: x\n---\n# x\n",
            },
        )


def test_grow_batch_validates_inputs(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    with pytest.raises(ValueError):
        propose_growth_batch(
            repo_root=root,
            run_id="",
            base_version="v0.1",
            batch_index=0,
            batch_size=4,
        )
    with pytest.raises(ValueError):
        propose_growth_batch(
            repo_root=root,
            run_id="r",
            base_version="v0.1",
            batch_index=-1,
            batch_size=4,
        )
    with pytest.raises(ValueError):
        propose_growth_batch(
            repo_root=root,
            run_id="r",
            base_version="v0.1",
            batch_index=0,
            batch_size=1,
        )


def _mis(image_id, sme_truth, split, mtype="model_wrong", severity="high"):
    return {
        "image_id": image_id,
        "sme_truth": sme_truth,
        "split": split,
        "misalignment_type": mtype,
        "severity": severity,
    }


def test_stratified_batch_excludes_holdout_and_all_agree():
    # Only TRAIN misalignments may seed a growth batch; holdout (reported-only)
    # and all_agree rows must never reach the prompt.
    records = [
        _mis("t_pos", "gen_ai", "dev_golden"),
        _mis("t_neg", "not_gen_ai", "train"),
        _mis("h_pos", "gen_ai", "holdout"),               # test split -> excluded
        _mis("h_neg", "not_gen_ai", "locked_holdout"),    # test split -> excluded
        _mis("agree", "gen_ai", "dev_golden", mtype="all_agree"),  # excluded
    ]
    rows, n_pos, n_neg = _stratified_batch_rows(records, batch_index=0, batch_size=10)
    ids = {r["image_id"] for r in rows}
    assert ids == {"t_pos", "t_neg"}
    assert (n_pos, n_neg) == (1, 1)


def test_select_priority_rows_excludes_holdout():
    mis = {"records": [
        _mis("t1", "gen_ai", "dev_golden"),
        _mis("h1", "gen_ai", "holdout"),
        _mis("t2", "not_gen_ai", "training"),
    ]}
    rows = _select_priority_rows(mis)
    assert {r["image_id"] for r in rows} == {"t1", "t2"}
