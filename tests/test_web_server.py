from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from pipeline.web._safety import APIError, safe_static_path, validate_start_payload
from pipeline.web.server import create_server


_VALID = {
    "models": ["openai/gpt-5.5"],
    "split": "dev_golden",
    "limit": 1,
    "policy_version": "v0.1",
    "mode": "cold_start",
    "allow_spend": True,
    "concurrency": 1,
}


@pytest.fixture
def web_server(tmp_path: Path):
    (tmp_path / "data" / "runs").mkdir(parents=True)
    server = create_server(host="127.0.0.1", port=0, repo_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _post_json(server, path: str, payload: dict) -> tuple[int, dict]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    body = json.dumps(payload).encode("utf-8")
    conn.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return response.status, json.loads(raw.decode("utf-8"))


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p: p.update({"models": ["unknown/model"]}), "unknown_model_id"),
        (lambda p: p.pop("allow_spend"), "validation_error"),
        (lambda p: p.update({"split": "holdout"}), "validation_error"),
        (lambda p: p.update({"policy_version": "version-1"}), "validation_error"),
        (lambda p: p.update({"concurrency": 5}), "validation_error"),
        (lambda p: p.pop("limit"), "validation_error"),
    ],
)
def test_post_start_rejects_invalid_payloads(web_server, mutate, code: str) -> None:
    payload = dict(_VALID)
    mutate(payload)
    status, data = _post_json(web_server, "/api/runs/start", payload)
    assert status == 400
    assert data["error"]["code"] == code


def test_validate_start_payload_allows_missing_reasoning_effort_with_variants() -> None:
    payload = dict(_VALID)
    payload["models"] = ["openai/gpt-5.5-xhigh", "openai/gpt-5.4-mini-high"]

    normalized = validate_start_payload(payload)

    assert normalized["models"] == ["openai/gpt-5.5-xhigh", "openai/gpt-5.4-mini-high"]
    assert normalized["reasoning_effort"] is None


def test_validate_start_payload_accepts_high_reasoning_effort() -> None:
    payload = dict(_VALID)
    payload["reasoning_effort"] = "high"

    normalized = validate_start_payload(payload)

    assert normalized["reasoning_effort"] == "high"


def test_validate_start_payload_rejects_invalid_reasoning_effort() -> None:
    payload = dict(_VALID)
    payload["reasoning_effort"] = "medium"

    with pytest.raises(APIError) as excinfo:
        validate_start_payload(payload)

    assert excinfo.value.status == 400
    assert excinfo.value.code == "validation_error"
    assert excinfo.value.details == {"field": "reasoning_effort"}


def test_static_path_validation_rejects_dotdot(tmp_path: Path) -> None:
    with pytest.raises(APIError):
        safe_static_path(tmp_path, "/web/%2e%2e/secret.txt")


def test_server_refuses_non_localhost_bind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(host="0.0.0.0", port=0, repo_root=tmp_path)
