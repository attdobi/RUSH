"""HTTP handler adapters for decision-quality and insights endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.web.aggregator import aggregate_decision_quality, compute_insights

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _run_has_score_inputs(run_dir: Path) -> bool:
    return (run_dir / "label_votes.jsonl").exists() and (run_dir / "llm_outputs.jsonl").exists()


def _auto_score_run(run_id: str) -> dict[str, Any]:
    from pipeline.scoring.run_scoring import run_scoring  # noqa: PLC0415

    return run_scoring(run_id, REPO_ROOT, runs_root=RUNS_ROOT)


def handle_insights(query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    run_id = _first(query, "run_id")
    if not run_id:
        return 400, {"error": "run_id is required"}
    runs_root = RUNS_ROOT.resolve()
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        return 400, {"error": "run_id must resolve under the runs root"}
    consensus_path = run_dir / "scoring" / "consensus.json"
    if not run_dir.exists():
        return 404, {"error": f"Run not found: {run_id}"}
    if not consensus_path.exists():
        if _run_has_score_inputs(run_dir):
            try:
                _auto_score_run(run_id)
            except Exception as exc:  # noqa: BLE001 - API should surface actionable local failure
                return 404, {
                    "error": (
                        f"Run exists but scoring failed: {type(exc).__name__}: {exc}. "
                        "Use 'Score now' to retry."
                    )
                }
        else:
            missing = [
                name
                for name in ("label_votes.jsonl", "llm_outputs.jsonl")
                if not (run_dir / name).exists()
            ]
            return 404, {
                "error": (
                    f"Run exists but is not ready for scoring: {run_id} "
                    f"(missing {', '.join(missing)})."
                )
            }
    if not consensus_path.exists():
        return 404, {
            "error": (
                f"Run exists but scoring did not produce consensus.json: {run_id}. "
                "Use 'Score now' to retry."
            )
        }
    try:
        return 200, compute_insights(run_dir)
    except FileNotFoundError as exc:
        return 404, {"error": str(exc)}
    except ValueError as exc:
        return 400, {"error": str(exc)}
