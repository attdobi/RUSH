"""API handlers for the experiment crank (seeded PPO policy-iteration runs).

File-based like the rest of the web layer: everything is served from the
portable ``data/experiments/<id>/experiment.json`` the driver rewrites
atomically — a fresh clone demos with no database. SME gate reviews are the
one write path; they update the JSON and best-effort mirror to Postgres
(``rush.gate_review``) as future RLHF data for the critic agent.

Endpoints (dispatched from handlers_runs.handle_api):
    GET  /api/experiments                     — newest-first summaries
    GET  /api/experiments/{id}                — full state (the UI poll target)
    POST /api/experiments/start               — spawn run_experiment.py (registry job)
    POST /api/experiments/{id}/review         — {k, verdict, reviewer?, comment?}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline import experiment as exp

from ._safety import APIError


def handle_list_experiments(repo_root: Path | str) -> tuple[int, dict[str, Any]]:
    return 200, {"experiments": exp.list_experiments(repo_root)}


def handle_get_experiment(
    repo_root: Path | str, experiment_id: str
) -> tuple[int, dict[str, Any]]:
    try:
        exp.validate_experiment_id(experiment_id)
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    try:
        state = exp.load_state(repo_root, experiment_id)
    except FileNotFoundError as excinfo:
        raise APIError(
            404, "not_found", f"unknown experiment: {experiment_id}"
        ) from excinfo
    return 200, state


def handle_gate_review(
    repo_root: Path | str, experiment_id: str, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    body = body or {}
    try:
        exp.validate_experiment_id(experiment_id)
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    k = body.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise APIError(
            400, "validation_error", "k must be a positive integer",
            details={"field": "k"},
        )
    verdict = str(body.get("verdict") or "")
    reviewer = str(body.get("reviewer") or "sme")[:80]
    comment = str(body.get("comment") or "")[:2000]
    try:
        review = exp.record_gate_review(
            repo_root, experiment_id, k,
            verdict=verdict, reviewer=reviewer, comment=comment,
        )
    except FileNotFoundError as excinfo:
        raise APIError(
            404, "not_found", f"unknown experiment: {experiment_id}"
        ) from excinfo
    except KeyError as excinfo:
        raise APIError(404, "not_found", str(excinfo)) from excinfo
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    return 200, {"experiment_id": experiment_id, "k": k, "review": review}
