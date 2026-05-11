"""Cost aggregation helpers for label-vote records."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from pipeline.providers.pricing import compute_call_cost


def _coerce_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _empty_model_bucket() -> dict[str, Any]:
    return {
        "total_cost_usd": 0.0,
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "cost_per_1000_labels": None,
        "usage_unknown_calls": 0,
    }


def _finalize_bucket(bucket: dict[str, Any]) -> None:
    if bucket["total_calls"] > 0 and bucket["total_cost_usd"] > 0:
        bucket["cost_per_1000_labels"] = bucket["total_cost_usd"] / bucket["total_calls"] * 1000
    else:
        bucket["cost_per_1000_labels"] = None


def aggregate_per_call_costs(votes: list[dict]) -> dict:
    """Aggregate per-call usage/cost fields from LabelVote rows."""
    per_model: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    total_calls = 0

    for vote in votes:
        model_id = str(vote.get("model_id") or vote.get("labeler_id") or "unknown")
        bucket = per_model.setdefault(model_id, _empty_model_bucket())
        bucket["total_calls"] += 1
        total_calls += 1

        input_tokens_raw = vote.get("input_tokens")
        output_tokens_raw = vote.get("output_tokens")
        input_tokens = _coerce_int(input_tokens_raw)
        output_tokens = _coerce_int(output_tokens_raw)
        bucket["total_input_tokens"] += input_tokens
        bucket["total_output_tokens"] += output_tokens
        if input_tokens_raw is None or output_tokens_raw is None:
            bucket["usage_unknown_calls"] += 1

        cost = _coerce_float(vote.get("cost_usd"))
        if cost is None:
            cost = compute_call_cost(model_id, input_tokens_raw, output_tokens_raw, image_count=1)
        if cost is not None:
            bucket["total_cost_usd"] += cost
            total_cost += cost

    for bucket in per_model.values():
        _finalize_bucket(bucket)

    overall_cost_per_1000 = (total_cost / total_calls * 1000) if total_calls > 0 and total_cost > 0 else None
    return {
        "per_model": dict(sorted(per_model.items())),
        "total_cost_usd": total_cost,
        "total_calls": total_calls,
        "cost_per_1000_labels": overall_cost_per_1000,
    }


def attach_cost_to_labelers(decision_quality_snapshot: dict, per_model_cost: dict) -> dict:
    """Return a copy of a DQ snapshot with cost metrics attached."""
    snapshot = deepcopy(decision_quality_snapshot)
    per_model = per_model_cost.get("per_model", {}) if isinstance(per_model_cost, dict) else {}

    for labeler in snapshot.get("labelers", []):
        metrics = labeler.setdefault("metrics", {})
        labeler_id = labeler.get("labeler_id")
        model_cost = per_model.get(labeler_id) if labeler_id is not None else None
        metrics["cost_per_1000_labels"] = (
            model_cost.get("cost_per_1000_labels") if isinstance(model_cost, dict) else None
        )

    snapshot["cost"] = {
        "total_cost_usd": float(per_model_cost.get("total_cost_usd", 0.0)),
        "total_calls": int(per_model_cost.get("total_calls", 0)),
        "cost_per_1000_labels": per_model_cost.get("cost_per_1000_labels"),
        "per_model": per_model,
    }
    return snapshot


__all__ = ["aggregate_per_call_costs", "attach_cost_to_labelers"]
