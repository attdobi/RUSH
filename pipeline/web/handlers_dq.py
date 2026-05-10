"""HTTP handler adapters for decision-quality and insights endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.web.aggregator import aggregate_decision_quality, compute_insights

RUNS_ROOT = Path("data/runs")


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0]
    return value if value != "" else None


def handle_decision_quality(query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    run_id = _first(query, "run_id")
    policy_version = _first(query, "policy_version")
    try:
        payload = aggregate_decision_quality(
            RUNS_ROOT, run_id=run_id, policy_version=policy_version
        )
    except ValueError as exc:
        return 400, {"error": str(exc)}
    if (run_id or policy_version) and not payload.get("runs"):
        return 404, {"error": "No scored runs matched the requested filters."}
    return 200, payload


def handle_insights(query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    run_id = _first(query, "run_id")
    if not run_id:
        return 400, {"error": "run_id is required"}
    runs_root = RUNS_ROOT.resolve()
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        return 400, {"error": "run_id must resolve under the runs root"}
    if not run_dir.exists() or not (run_dir / "scoring" / "consensus.json").exists():
        return 404, {"error": f"Run not found or not scored: {run_id}"}
    try:
        return 200, compute_insights(run_dir)
    except FileNotFoundError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
