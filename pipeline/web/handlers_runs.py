"""JSON dispatcher for RUSH local API endpoints.

Dispatches:
* ``/api/health`` and ``/api/runs*`` (X1 — this module)
* ``/api/decision-quality`` and ``/api/insights`` (X2 — ``handlers_dq``)
* ``/api/policy/*`` (X3 — ``handlers_policy``)

Keeping a single dispatcher avoids duplicating the JSON envelope/error
plumbing per slice; per-feature handlers stay in their own modules.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from pipeline.thumbnails import thumbnail_rel_path_for_source, validate_source_repo_path

from . import handlers_dq, handlers_policy
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


def _redirect(handler, location: str) -> None:
    handler.send_response(302)
    handler.send_header("Location", location)
    handler.send_header("Content-Length", "0")
    handler.send_header("Cache-Control", "public, max-age=3600")
    handler.end_headers()


def handle_thumbnail(handler) -> None:
    query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
    requested = (query.get("path") or [""])[0]
    if not requested:
        raise APIError(400, "bad_request", "path query parameter is required")
    try:
        source_rel = validate_source_repo_path(handler.repo_root, requested)
    except ValueError as exc:
        raise APIError(400, "bad_request", str(exc)) from exc

    thumbnail_rel = thumbnail_rel_path_for_source(source_rel)
    target_rel = thumbnail_rel if (handler.repo_root / thumbnail_rel).is_file() else source_rel
    location = "/" + quote(target_rel.as_posix(), safe="/")
    _redirect(handler, location)


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

        if method == "GET" and path == "/api/thumbnail":
            handle_thumbnail(handler)
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

        # ----- X2: decision-quality / insights ---------------------------
        if method == "GET" and path == "/api/decision-quality":
            query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
            status, body = handlers_dq.handle_decision_quality(query)
            send_json(handler, status, body)
            return
        if method == "GET" and path == "/api/insights":
            query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
            status, body = handlers_dq.handle_insights(query)
            send_json(handler, status, body)
            return

        # ----- X3: policy versions / proposals ---------------------------
        if method == "GET" and path == "/api/policy/versions":
            status, body = handlers_policy.handle_policy_versions(handler.repo_root)
            send_json(handler, status, body)
            return
        if method == "GET" and path == "/api/policy/graph":
            query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
            version = (query.get("version") or [None])[0]
            status, body = handlers_policy.handle_policy_graph(
                handler.repo_root, version
            )
            send_json(handler, status, body)
            return
        if method == "GET" and path == "/api/policy/proposals":
            status, body = handlers_policy.handle_list_proposals(handler.repo_root)
            send_json(handler, status, body)
            return
        if method == "POST" and path == "/api/policy/propose-diff":
            body_in = read_json_body(handler) or {}
            status, body = handlers_policy.handle_propose_diff(
                handler.repo_root, body_in
            )
            send_json(handler, status, body)
            return
        if method == "POST" and path == "/api/policy/build-pdf":
            body_in = read_json_body(handler) or {}
            status, body = handlers_policy.handle_build_pdf(
                handler.repo_root, body_in
            )
            send_json(handler, status, body)
            return
        # /api/policy/proposals/<id>(/accept|/reject)?
        if len(parts) >= 5 and parts[:4] == ["", "api", "policy", "proposals"]:
            proposal_id = parts[4]
            if not proposal_id:
                raise _not_found(path)
            if len(parts) == 5 and method == "GET":
                status, body = handlers_policy.handle_get_proposal(
                    handler.repo_root, proposal_id
                )
                send_json(handler, status, body)
                return
            if len(parts) == 6 and method == "POST":
                action = parts[5]
                if action == "accept":
                    status, body = handlers_policy.handle_accept_proposal(
                        handler.repo_root, proposal_id
                    )
                    send_json(handler, status, body)
                    return
                if action == "reject":
                    status, body = handlers_policy.handle_reject_proposal(
                        handler.repo_root, proposal_id
                    )
                    send_json(handler, status, body)
                    return

        raise _not_found(path)
    except APIError as exc:
        send_api_error(handler, exc)
