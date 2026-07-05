from __future__ import annotations

import json
from pathlib import Path

from pipeline.policy_diff import (
    accept_proposal,
    get_proposal,
    list_proposals,
    propose_diff,
    reject_proposal,
)


def _seed_policy_graph(tmp_path: Path) -> Path:
    root = tmp_path
    base = root / "policy-graph" / "Generative_AI" / "v0.1"
    base.mkdir(parents=True)
    (base / "GA.root.md").write_text("# Root\n\nOld root text.\n", encoding="utf-8")
    (base / "GA.boundary.low_quality_uncertain.md").write_text(
        "# Low quality\n\nExisting boundary.\n", encoding="utf-8"
    )
    (base / "GA.negative.authentic_photo.md").write_text(
        "# Authentic\n\nNot generated.\n", encoding="utf-8"
    )
    (root / "data").mkdir()
    return root


def test_propose_accept_and_reject_lifecycle(tmp_path: Path) -> None:
    root = _seed_policy_graph(tmp_path)
    base = root / "policy-graph" / "Generative_AI" / "v0.1"
    original = {p.name: p.read_text(encoding="utf-8") for p in base.glob("*.md")}

    proposal = propose_diff(
        repo_root=root,
        run_id="run-123",
        base_version="v0.1",
        proposed_files={
            "GA.root.md": "# Root\n\nNew root text.\n",
            "GA.boundary.over_smoothed_skin.md": "# Over-smoothed skin\n\nNew node.\n",
        },
    )

    assert proposal["status"] == "pending"
    assert proposal["model_id"] == "openai/gpt-5.5"
    assert proposal["files_changed"] == ["GA.root.md"]
    assert proposal["files_added"] == ["GA.boundary.over_smoothed_skin.md"]
    assert proposal["files_removed"] == []

    proposal_dir = root / "data" / "policy_proposals" / proposal["proposal_id"]
    assert (proposal_dir / "proposal.json").is_file()
    assert (proposal_dir / "prompt.json").is_file()
    assert (proposal_dir / "raw_response.txt").is_file()
    assert (proposal_dir / "proposed" / "GA.root.md").read_text(encoding="utf-8") == (
        "# Root\n\nNew root text.\n"
    )
    meta = json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    assert meta["proposal_id"] == proposal["proposal_id"]

    # propose_diff must not touch the bound policy graph.
    assert {p.name: p.read_text(encoding="utf-8") for p in base.glob("*.md")} == original
    assert not (root / "policy-graph" / "Generative_AI" / "v0.2").exists()

    detail = get_proposal(repo_root=root, proposal_id=proposal["proposal_id"])
    assert detail["diffs"][0]["change"] == "modified"
    assert "--- a/GA.root.md" in detail["diffs"][0]["unified_diff"]
    assert "+++ b/GA.root.md" in detail["diffs"][0]["unified_diff"]

    accepted = accept_proposal(repo_root=root, proposal_id=proposal["proposal_id"])
    assert accepted == {
        "new_version": "v0.2",
        "path": "policy-graph/Generative_AI/v0.2",
    }
    v02 = root / "policy-graph" / "Generative_AI" / "v0.2"
    assert v02.is_dir()
    assert (v02 / "GA.root.md").read_text(encoding="utf-8") == "# Root\n\nNew root text.\n"
    assert (v02 / "GA.boundary.over_smoothed_skin.md").read_text(encoding="utf-8") == (
        "# Over-smoothed skin\n\nNew node.\n"
    )
    assert (v02 / "GA.boundary.low_quality_uncertain.md").read_text(encoding="utf-8") == original[
        "GA.boundary.low_quality_uncertain.md"
    ]
    assert (v02 / "GA.negative.authentic_photo.md").read_text(encoding="utf-8") == original[
        "GA.negative.authentic_photo.md"
    ]
    assert {p.name: p.read_text(encoding="utf-8") for p in base.glob("*.md")} == original

    reject_me = propose_diff(
        repo_root=root,
        run_id="run-456",
        base_version="v0.1",
        proposed_files={"GA.boundary.low_quality_uncertain.md": "# Low quality\n\nRejected edit.\n"},
    )
    rejected = reject_proposal(repo_root=root, proposal_id=reject_me["proposal_id"])
    assert rejected["status"] == "rejected"
    assert not (root / "data" / "policy_proposals" / reject_me["proposal_id"]).exists()
    archived = root / "data" / "policy_proposals" / "_archive" / reject_me["proposal_id"]
    assert archived.is_dir()
    archived_meta = json.loads((archived / "proposal.json").read_text(encoding="utf-8"))
    assert archived_meta["status"] == "rejected"


def _seed_mnist_graph(tmp_path: Path) -> Path:
    """Seed an MNIST_Digits domain with only v0.1 (mirrors on-disk state)."""
    root = tmp_path
    base = root / "policy-graph" / "MNIST_Digits" / "v0.1"
    base.mkdir(parents=True)
    (base / "MNIST.root.md").write_text("# MNIST root\n\nDigit policy.\n", encoding="utf-8")
    (base / "MNIST.confused.3_8.md").write_text(
        "# 3 vs 8\n\nExisting boundary.\n", encoding="utf-8"
    )
    (root / "data").mkdir()
    return root


def test_propose_accept_area_aware_mnist_next_version(tmp_path: Path) -> None:
    """Accepting an MNIST proposal must materialize MNIST_Digits/v0.2, not GenAI."""
    root = _seed_mnist_graph(tmp_path)

    proposal = propose_diff(
        repo_root=root,
        run_id="mnist-run-1",
        base_version="v0.1",
        domain="MNIST_Digits",
        proposed_files={
            "MNIST.root.md": "# MNIST root\n\nUpdated digit policy.\n",
            "MNIST.confused.5_6.md": "# 5 vs 6\n\nNew boundary node.\n",
        },
    )
    assert proposal["status"] == "pending"
    assert proposal["domain"] == "MNIST_Digits"
    assert proposal["files_changed"] == ["MNIST.root.md"]
    assert proposal["files_added"] == ["MNIST.confused.5_6.md"]

    listed = list_proposals(repo_root=root, include_errors=True)
    assert [item["proposal_id"] for item in listed["proposals"]] == [proposal["proposal_id"]]
    assert listed["proposals"][0]["domain"] == "MNIST_Digits"

    detail = get_proposal(repo_root=root, proposal_id=proposal["proposal_id"])
    assert detail["domain"] == "MNIST_Digits"
    assert detail["diffs"][0]["change"] == "modified"
    assert "--- a/MNIST.root.md" in detail["diffs"][0]["unified_diff"]
    assert "+++ b/MNIST.root.md" in detail["diffs"][0]["unified_diff"]

    # propose must not touch GenAI or create MNIST v0.2 yet
    assert not (root / "policy-graph" / "Generative_AI").exists()
    assert not (root / "policy-graph" / "MNIST_Digits" / "v0.2").exists()

    accepted = accept_proposal(repo_root=root, proposal_id=proposal["proposal_id"])
    assert accepted == {
        "new_version": "v0.2",
        "path": "policy-graph/MNIST_Digits/v0.2",
    }
    v02 = root / "policy-graph" / "MNIST_Digits" / "v0.2"
    assert v02.is_dir()
    # base file copied + changed content applied
    assert (v02 / "MNIST.root.md").read_text(encoding="utf-8") == (
        "# MNIST root\n\nUpdated digit policy.\n"
    )
    # added file present
    assert (v02 / "MNIST.confused.5_6.md").is_file()
    # untouched base node carried forward
    assert (v02 / "MNIST.confused.3_8.md").is_file()
    # GenAI domain never created as a side effect
    assert not (root / "policy-graph" / "Generative_AI").exists()
