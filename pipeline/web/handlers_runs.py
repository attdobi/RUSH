"""JSON handlers for RUSH local run lifecycle endpoints."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from ._safety import APIError, read_json_body, validate_start_payload
from .run_registry import RunRegistry


def send_json(handler, status: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, sort_keys=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def send_api_error(handler, exc: APIError) -> None:
    send_json(handler, exc.status, exc.to_payload())


def _not_found(path: str) -> APIError:
    return APIError(404, "not_found", f"unknown endpoint: {path}")


def handle_api(handler, registry: RunRegistry, *, method: str) -> None:
    path = urlsplit(handler.path).path.rstrip("/") or "/"
    try:
        if method == "GET" and path == "/api/health":
            send_json(
                handler,
                200,
                {
                    "status": "ok",
                    "server_version": handler.server_version_name,
                    "started_at": handler.started_at,
                    "repo_root": str(handler.repo_root),
                },
            )
            return

        if method == "GET" and path == "/api/runs":
            send_json(handler, 200, {"runs": registry.list_runs()})
            return

        if method == "POST" and path == "/api/runs/start":
            payload = validate_start_payload(read_json_body(handler))
            state = registry.start_job(payload)
            run_token = state["job_id"]
            send_json(
                handler,
                202,
                {
                    "run_id": run_token,
                    "job_id": run_token,
                    "status_url": f"/api/runs/{run_token}/status",
                    "log_url": f"/api/runs/{run_token}/log",
                },
            )
            return

        parts = path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "runs"]:
            token = parts[3]
            action = parts[4]
            if method == "GET" and action == "status":
                send_json(handler, 200, registry.status(token))
                return
            if method == "POST" and action == "score":
                send_json(handler, 200, registry.score(token))
                return

        raise _not_found(path)
    except APIError as exc:
        send_api_error(handler, exc)
