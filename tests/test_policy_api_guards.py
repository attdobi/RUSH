from __future__ import annotations

import json
from pathlib import Path

from pipeline.policy_diff import list_policy_versions, list_proposals
from pipeline.web.handlers_policy import handle_policy_graph


def _write_policy_node(version_dir: Path, node_id: str = "GA.root") -> None:
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / f"{node_id}.md").write_text(
        "---\n"
        f"id: {node_id}\n"
        "title: Root\n"
        "node_type: root\n"
        "polarity: mixed\n"
        "parent: null\n"
        "status: draft\n"
        "---\n"
        "# Root\n",
        encoding="utf-8",
    )


def _write_proposal(root: Path, proposal_id: str, *, status: str, created_at: str) -> None:
    proposal_dir = root / "data" / "policy_proposals" / proposal_id
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / "proposal.json").write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "status": status,
                "created_at": created_at,
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
            }
        ),
        encoding="utf-8",
    )


def test_handle_policy_graph_returns_500_for_empty_version_dir(tmp_path: Path) -> None:
    complete = tmp_path / "policy-graph" / "Generative_AI" / "v0.1"
    _write_policy_node(complete)
    incomplete = tmp_path / "policy-graph" / "Generative_AI" / "v0.2"
    incomplete.mkdir(parents=True)

    status, body = handle_policy_graph(tmp_path, "v0.2")

    assert status == 500
    assert body == {
        "error": "policy version v0.2 is incomplete (no .md files)",
        "error_type": "IncompletePolicyVersion",
    }


def test_list_proposals_filters_parse_errors_and_reports_hidden_count(tmp_path: Path) -> None:
    created_at = "2026-05-19T12:00:00+00:00"
    _write_proposal(tmp_path, "accepted-1", status="accepted", created_at=created_at)
    _write_proposal(tmp_path, "parse-1", status="parse_error", created_at=created_at)
    _write_proposal(tmp_path, "pending-1", status="pending", created_at=created_at)
    _write_proposal(tmp_path, "rejected-1", status="rejected", created_at=created_at)

    body = list_proposals(repo_root=tmp_path)

    assert body["include_errors"] is False
    assert body["hidden_error_count"] == 1
    assert [p["proposal_id"] for p in body["proposals"]] == [
        "pending-1",
        "accepted-1",
        "rejected-1",
    ]


def test_list_proposals_include_errors_returns_all_proposals(tmp_path: Path) -> None:
    created_at = "2026-05-19T12:00:00+00:00"
    _write_proposal(tmp_path, "accepted-1", status="accepted", created_at=created_at)
    _write_proposal(tmp_path, "parse-1", status="parse_error", created_at=created_at)
    _write_proposal(tmp_path, "pending-1", status="pending", created_at=created_at)
    _write_proposal(tmp_path, "rejected-1", status="rejected", created_at=created_at)

    body = list_proposals(repo_root=tmp_path, include_errors=True)

    assert body["include_errors"] is True
    assert body["hidden_error_count"] == 0
    assert [p["proposal_id"] for p in body["proposals"]] == [
        "pending-1",
        "accepted-1",
        "rejected-1",
        "parse-1",
    ]


def test_list_policy_versions_marks_complete_and_current_skips_empty(tmp_path: Path) -> None:
    domain = tmp_path / "policy-graph" / "Generative_AI"
    _write_policy_node(domain / "v0.1")
    _write_policy_node(domain / "v0.2")
    (domain / "v0.3").mkdir(parents=True)

    body = list_policy_versions(repo_root=tmp_path)
    versions = {version["version"]: version for version in body["versions"]}

    assert versions["v0.1"]["complete"] is True
    assert versions["v0.2"]["complete"] is True
    assert versions["v0.3"]["complete"] is False
    assert versions["v0.3"]["files"] == 0
    assert body["current"] == "v0.2"
