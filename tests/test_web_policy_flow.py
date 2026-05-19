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

    def list_runs(self) -> list[dict[str, Any]]:
        return [{"run_id": "run-1", "status": "completed"}]

    def compute_now(self, token: str) -> dict[str, Any]:
        self.computed.append(token)
        return {"run_id": token, "status": "scored"}

    def status(self, token: str) -> dict[str, Any]:
        return {"run_id": token, "status": "completed"}

    def score(self, token: str) -> dict[str, Any]:
        return self.compute_now(token)

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
