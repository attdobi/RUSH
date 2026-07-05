from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any

from pipeline.web import handlers_policy
from pipeline.web.server import create_server


class FakeRegistry:
    def __init__(self) -> None:
        self.computed: list[str] = []
        self.canceled: list[str] = []

    def list_runs(self) -> list[dict[str, Any]]:
        return [{"run_id": "run-1", "status": "completed"}]

    def compute_now(self, token: str) -> dict[str, Any]:
        self.computed.append(token)
        return {"run_id": token, "status": "scored"}

    def status(self, token: str) -> dict[str, Any]:
        return {"run_id": token, "status": "completed"}

    def score(self, token: str) -> dict[str, Any]:
        return self.compute_now(token)

    def cancel_run(self, token: str) -> dict[str, Any]:
        self.canceled.append(token)
        return {"run_id": token, "job_id": token, "running": False, "status": "canceled"}

    def start_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"job_id": "run-1", "payload": payload}


def _serve(tmp_path: Path, registry: FakeRegistry | None = None):
    (tmp_path / "web").mkdir(exist_ok=True)
    server = create_server(
        host="127.0.0.1",
        port=0,
        repo_root=tmp_path,
        registry=registry or FakeRegistry(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request_json(server, method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    body = json.dumps(payload or {}).encode("utf-8") if method == "POST" else None
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body or b""))} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return response.status, json.loads(raw.decode("utf-8"))


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def _write_policy_proposal(
    root: Path,
    proposal_id: str,
    *,
    base_version: str,
    created_at: str = "2026-05-19T12:00:00+00:00",
) -> None:
    proposal_dir = root / "data" / "policy_proposals" / proposal_id
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / "proposal.json").write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "status": "pending",
                "created_at": created_at,
                "base_version": base_version,
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
            }
        ),
        encoding="utf-8",
    )


def test_compute_alias_covers_insights_score_button(tmp_path: Path) -> None:
    registry = FakeRegistry()
    server, thread = _serve(tmp_path, registry)
    try:
        for action in ("compute-now", "compute"):
            status, body = _request_json(server, "POST", f"/api/runs/run-1/{action}", {})
            assert status == 200
            assert body == {"run_id": "run-1", "status": "scored"}
        assert registry.computed == ["run-1", "run-1"]
    finally:
        _stop(server, thread)


def test_cancel_and_stop_routes_are_wired(tmp_path: Path) -> None:
    registry = FakeRegistry()
    server, thread = _serve(tmp_path, registry)
    try:
        for action in ("cancel", "stop"):
            status, body = _request_json(server, "POST", f"/api/runs/run-1/{action}", {})
            assert status == 200
            assert body == {
                "run_id": "run-1",
                "job_id": "run-1",
                "running": False,
                "status": "canceled",
            }
        assert registry.canceled == ["run-1", "run-1"]
    finally:
        _stop(server, thread)


def test_policy_growth_and_review_routes_are_wired(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_cold_start(repo_root, body):
        calls.append(("cold", body))
        return 200, {"proposal_id": "cold-1", "kind": "cold_start", "status": "pending"}

    def fake_grow_batch(repo_root, body):
        calls.append(("grow", body))
        return 200, {
            "proposal_id": "grow-1",
            "kind": "grow_batch",
            "status": "pending",
            "batch": {"batch_size_actual": body["batch_size"], "n_positives": 2, "n_negatives": 2},
        }

    def fake_get(repo_root, proposal_id):
        calls.append(("get", proposal_id))
        return 200, {"proposal_id": proposal_id, "status": "pending", "diffs": []}

    def fake_accept(repo_root, proposal_id):
        calls.append(("accept", proposal_id))
        return 200, {"proposal_id": proposal_id, "status": "accepted", "new_version": "v0.2"}

    def fake_reject(repo_root, proposal_id):
        calls.append(("reject", proposal_id))
        return 200, {"proposal_id": proposal_id, "status": "rejected"}

    monkeypatch.setattr(handlers_policy, "handle_cold_start", fake_cold_start)
    monkeypatch.setattr(handlers_policy, "handle_grow_batch", fake_grow_batch)
    monkeypatch.setattr(handlers_policy, "handle_get_proposal", fake_get)
    monkeypatch.setattr(handlers_policy, "handle_accept_proposal", fake_accept)
    monkeypatch.setattr(handlers_policy, "handle_reject_proposal", fake_reject)

    server, thread = _serve(tmp_path)
    try:
        status, body = _request_json(server, "POST", "/api/policy/cold-start", {"task_description": "Classify GenAI images"})
        assert status == 200
        assert body["proposal_id"] == "cold-1"

        status, body = _request_json(
            server,
            "POST",
            "/api/policy/grow-batch",
            {"run_id": "run-1", "base_version": "v0.1", "batch_index": 0, "batch_size": 4},
        )
        assert status == 200
        assert body["batch"]["batch_size_actual"] == 4

        status, body = _request_json(server, "GET", "/api/policy/proposals/grow-1")
        assert status == 200
        assert body["status"] == "pending"

        status, body = _request_json(server, "POST", "/api/policy/proposals/grow-1/accept", {})
        assert status == 200
        assert body["new_version"] == "v0.2"

        status, body = _request_json(server, "POST", "/api/policy/proposals/cold-1/reject", {})
        assert status == 200
        assert body["status"] == "rejected"
    finally:
        _stop(server, thread)

    assert ("cold", {"task_description": "Classify GenAI images"}) in calls
    assert ("accept", "grow-1") in calls
    assert ("reject", "cold-1") in calls


def test_policy_proposals_route_forwards_include_errors_query(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[bool, str | None]] = []

    def fake_list(repo_root, *, include_errors=False, area=None):  # noqa: ARG001
        calls.append((include_errors, area))
        return 200, {
            "proposals": [],
            "hidden_error_count": 0,
            "include_errors": include_errors,
        }

    monkeypatch.setattr(handlers_policy, "handle_list_proposals", fake_list)

    server, thread = _serve(tmp_path)
    try:
        status, body = _request_json(server, "GET", "/api/policy/proposals")
        assert status == 200
        assert body["include_errors"] is False

        status, body = _request_json(
            server, "GET", "/api/policy/proposals?include_errors=true"
        )
        assert status == 200
        assert body["include_errors"] is True
    finally:
        _stop(server, thread)

    assert calls == [(False, None), (True, None)]


def test_policy_proposals_route_filters_by_area(tmp_path: Path) -> None:
    _write_policy_proposal(tmp_path, "GA.bare-1", base_version="v0.1")
    _write_policy_proposal(tmp_path, "GA.prefixed-1", base_version="Generative_AI.v0.2")
    _write_policy_proposal(tmp_path, "MNIST_Digits.proposal-1", base_version="MNIST_Digits.v0.1")

    server, thread = _serve(tmp_path)
    try:
        status, body = _request_json(server, "GET", "/api/policy/proposals?area=MNIST_Digits")
        assert status == 200
        assert [p["proposal_id"] for p in body["proposals"]] == ["MNIST_Digits.proposal-1"]
        assert all(
            p["base_version"].startswith("MNIST_Digits.")
            for p in body["proposals"]
        )

        status, body = _request_json(server, "GET", "/api/policy/proposals?area=Generative_AI")
        assert status == 200
        assert sorted(p["proposal_id"] for p in body["proposals"]) == [
            "GA.bare-1",
            "GA.prefixed-1",
        ]
    finally:
        _stop(server, thread)


def test_runs_route_filters_by_demo_query(tmp_path: Path) -> None:
    class MixedRegistry(FakeRegistry):
        def list_runs(self) -> list[dict[str, Any]]:
            return [
                {"run_id": "genai-1", "policy_graph_version": "Generative_AI.v0.3"},
                {"run_id": "old-genai-1", "policy_graph_version": "v0.1"},
                {"run_id": "mnist-1", "policy_graph_version": "MNIST_Digits.v0.1"},
            ]

    server, thread = _serve(tmp_path, MixedRegistry())
    try:
        status, body = _request_json(server, "GET", "/api/runs?demo=mnist")
        assert status == 200
        assert [run["run_id"] for run in body["runs"]] == ["mnist-1"]

        status, body = _request_json(server, "GET", "/api/runs")
        assert status == 200
        assert [run["run_id"] for run in body["runs"]] == ["genai-1", "old-genai-1"]
    finally:
        _stop(server, thread)


def test_policy_routes_forward_area_query(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_versions(repo_root, area=None):  # noqa: ARG001
        calls.append(("versions", area))
        return 200, {"versions": [{"version": "v0.1"}], "current": "v0.1"}

    def fake_graph(repo_root, version, area=None):  # noqa: ARG001
        calls.append(("graph", area))
        return 200, {"version": version, "area": area, "nodes": [], "edges": []}

    monkeypatch.setattr(handlers_policy, "handle_policy_versions", fake_versions)
    monkeypatch.setattr(handlers_policy, "handle_policy_graph", fake_graph)

    server, thread = _serve(tmp_path)
    try:
        status, body = _request_json(server, "GET", "/api/policy/versions?area=MNIST_Digits")
        assert status == 200
        assert body["current"] == "v0.1"

        status, body = _request_json(server, "GET", "/api/policy/graph?version=v0.1&area=MNIST_Digits")
        assert status == 200
        assert body["area"] == "MNIST_Digits"
    finally:
        _stop(server, thread)

    assert calls == [("versions", "MNIST_Digits"), ("graph", "MNIST_Digits")]


def test_web_propose_accept_new_version_mnist_area_aware(tmp_path: Path) -> None:
    """Full web path: propose from an MNIST run -> accept -> MNIST_Digits/v0.2."""
    base = tmp_path / "policy-graph" / "MNIST_Digits" / "v0.1"
    base.mkdir(parents=True)
    (base / "MNIST.root.md").write_text("# MNIST root\n\nDigit policy.\n", encoding="utf-8")
    # Run manifest carries the area prefix so the server derives MNIST_Digits.
    run_dir = tmp_path / "data" / "runs" / "mnist-web-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": "mnist-web-1", "policy_graph_version": "MNIST_Digits.v0.1"}),
        encoding="utf-8",
    )

    server, thread = _serve(tmp_path)
    try:
        status, body = _request_json(
            server,
            "POST",
            "/api/policy/propose-diff",
            {
                "run_id": "mnist-web-1",
                "base_version": "v0.1",
                "proposed_files": {"MNIST.root.md": "# MNIST root\n\nStronger digit policy.\n"},
            },
        )
        assert status == 200, body
        assert body["domain"] == "MNIST_Digits"
        proposal_id = body["proposal_id"]

        status, body = _request_json(
            server, "POST", f"/api/policy/proposals/{proposal_id}/accept", {}
        )
        assert status == 200, body
        assert body["new_version"] == "v0.2"
        assert body["path"] == "policy-graph/MNIST_Digits/v0.2"

        status, body = _request_json(
            server, "GET", "/api/policy/versions?area=MNIST_Digits"
        )
        assert status == 200, body
        versions = [v.get("version", v) if isinstance(v, dict) else v for v in body["versions"]]
        assert "v0.2" in versions
        assert body["current"] == "v0.2"
    finally:
        _stop(server, thread)

    assert (tmp_path / "policy-graph" / "MNIST_Digits" / "v0.2" / "MNIST.root.md").is_file()
    # GenAI domain must never be created as a side effect.
    assert not (tmp_path / "policy-graph" / "Generative_AI").exists()
