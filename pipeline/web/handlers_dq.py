"""HTTP handler adapters for decision-quality and insights endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.web.aggregator import aggregate_decision_quality, compute_insights
from pipeline.web.demo_area import area_from_query, first_query_value, policy_version_matches_area

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("data/runs")


def _empty_insights() -> dict[str, Any]:
    return {
        "run_id": None,
        "majority_wrong": [],
        "model_disagreement": [],
        "boundary_concentration": [],
        "consistent_pair_disagreement": [],
    }


def _query_has_area_filter(query: dict[str, list[str]]) -> bool:
    return bool(first_query_value(query, "area") or first_query_value(query, "demo"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _run_policy_graph_version(run_dir: Path) -> str | None:
    manifest = _read_json_if_exists(run_dir / "run_manifest.json")
    dq = _read_json_if_exists(run_dir / "scoring" / "decision_quality.json")
    return (
        str(dq.get("policy_graph_version"))
        if dq.get("policy_graph_version") is not None
        else (
            str(manifest.get("policy_graph_version"))
            if manifest.get("policy_graph_version") is not None
            else None
        )
    )


def handle_decision_quality(query: dict[str, list[str]]) -> tuple[int, dict[str, Any]]:
    run_id = first_query_value(query, "run_id")
    policy_version = first_query_value(query, "policy_version")
    try:
        area = area_from_query(query)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    try:
        payload = aggregate_decision_quality(
            RUNS_ROOT,
            run_id=run_id,
            policy_version=policy_version,
            policy_area=area,
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
    run_id = first_query_value(query, "run_id")
    if not run_id:
        if _query_has_area_filter(query):
            try:
                area_from_query(query)
            except ValueError as exc:
                return 400, {"error": str(exc)}
            return 200, _empty_insights()
        return 400, {"error": "run_id is required"}
    try:
        area = area_from_query(query)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    runs_root = RUNS_ROOT.resolve()
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not run_dir.is_relative_to(runs_root):
        return 400, {"error": "run_id must resolve under the runs root"}
    consensus_path = run_dir / "scoring" / "consensus.json"
    if not run_dir.exists():
        return 404, {"error": f"Run not found: {run_id}"}
    version = _run_policy_graph_version(run_dir)
    if not policy_version_matches_area(version, area):
        return 404, {"error": "No scored runs matched the requested filters."}
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
