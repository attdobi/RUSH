"""Durable per-image / per-model / per-batch cost ledger (X1 backend).

Attila: "keep tabs on all pricing of each batch run per LLM, per image — record
and store the data for future data analysis."

This module builds analysis-ready rows persisted to
``data/runs/<run_id>/costs.jsonl``. Every row is self-describing: it carries the
raw usage tokens, the CURRENT registry rates (input/output per Mtok), the
computed cost, and a ``pricing_version`` stamp so downstream analysis can filter
by pricing epoch.

IMPORTANT (legacy pricing): older runs stored ``cost_usd`` computed with STALE
rates (e.g. Opus $15/$75, which is why some historical opus rows show ~$150/1k).
Going forward the ledger recomputes/records cost with the LIVE registry and
stamps ``pricing_version``. We do NOT rewrite history: historical llm_outputs
rows keep their legacy cost; new ledger rows are the source of truth for
current-pricing analysis. Rows with ``pricing_version != current`` (or absent)
should be treated as legacy.
"""
from __future__ import annotations

from typing import Any

from pipeline.providers.pricing import (
    PRICING_VERSION,
    compute_call_cost,
    price_for,
)


def build_cost_row(
    *,
    run_id: str,
    batch_index: int,
    image_id: str,
    model_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    recorded_at: str,
    image_count: int = 1,
    cost_usd: float | None = None,
    latency_ms: int | float | None = None,
) -> dict[str, Any]:
    """Build one analysis-ready per-image-per-model cost ledger row.

    Cost is (re)computed with the LIVE pricing registry unless an explicit
    ``cost_usd`` is provided. Rates and ``pricing_version`` are always stamped
    from the current registry so the row is self-describing for later analysis.
    """
    pricing = price_for(model_id)
    input_rate = float(pricing["input_per_mtok"]) if pricing else None
    output_rate = float(pricing["output_per_mtok"]) if pricing else None
    image_rate = float(pricing["image_per_image"]) if pricing else None

    if cost_usd is None:
        cost_usd = compute_call_cost(
            model_id, input_tokens, output_tokens, image_count=image_count
        )

    latency_value: int | None = None
    if latency_ms is not None:
        try:
            latency_value = int(max(0, float(latency_ms)))
        except (TypeError, ValueError):
            latency_value = None
    output_value = None if output_tokens is None else int(output_tokens)
    tokens_per_sec = None
    if output_value is not None and latency_value is not None and latency_value > 0:
        tokens_per_sec = float(output_value) / float(latency_value) * 1000.0

    return {
        "run_id": run_id,
        "batch_index": int(batch_index),
        "batch_id": f"{run_id}:{int(batch_index)}",
        "image_id": image_id,
        "model_id": model_id,
        "input_tokens": None if input_tokens is None else int(input_tokens),
        "output_tokens": output_value,
        "latency_ms": latency_value,
        "tokens_per_sec": tokens_per_sec,
        "input_rate_per_mtok": input_rate,
        "output_rate_per_mtok": output_rate,
        "image_rate_per_image": image_rate,
        "image_count": int(image_count),
        "cost_usd": None if cost_usd is None else float(cost_usd),
        "pricing_version": PRICING_VERSION,
        "recorded_at": recorded_at,
    }


def _blank_rollup() -> dict[str, Any]:
    return {
        "images": 0,
        "priced_images": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
        "cost_per_image_usd": None,
        "cost_per_1000_labels": None,
    }


def _finalize(bucket: dict[str, Any]) -> None:
    priced = bucket["priced_images"]
    if priced > 0 and bucket["total_cost_usd"] > 0:
        per_image = bucket["total_cost_usd"] / priced
        bucket["cost_per_image_usd"] = per_image
        bucket["cost_per_1000_labels"] = per_image * 1000
    else:
        bucket["cost_per_image_usd"] = None
        bucket["cost_per_1000_labels"] = None


def rollup_cost_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate ledger rows into per-run + per-model + per-batch rollups.

    Returns a dict with ``total`` (run-level), ``per_model`` (per-LLM
    breakdown), ``per_batch`` (per-batch, incl. model_id), and
    ``pricing_versions`` (the set of pricing epochs present, so analysis can
    detect mixed/legacy pricing).
    """
    total = _blank_rollup()
    per_model: dict[str, dict[str, Any]] = {}
    per_batch: dict[int, dict[str, Any]] = {}
    pricing_versions: set[str] = set()

    for row in rows:
        model_id = str(row.get("model_id") or "unknown")
        batch_index = int(row.get("batch_index", 0) or 0)
        cost = row.get("cost_usd")
        in_tok = int(row.get("input_tokens") or 0)
        out_tok = int(row.get("output_tokens") or 0)
        pv = row.get("pricing_version")
        if pv:
            pricing_versions.add(str(pv))

        mb = per_model.setdefault(model_id, _blank_rollup())
        bb = per_batch.setdefault(
            batch_index, {**_blank_rollup(), "batch_index": batch_index, "model_id": model_id}
        )

        for bucket in (total, mb, bb):
            bucket["images"] += 1
            bucket["total_input_tokens"] += in_tok
            bucket["total_output_tokens"] += out_tok
            if cost is not None:
                bucket["priced_images"] += 1
                bucket["total_cost_usd"] += float(cost)

    for bucket in (total, *per_model.values(), *per_batch.values()):
        _finalize(bucket)

    return {
        "total": total,
        "per_model": dict(sorted(per_model.items())),
        "per_batch": [per_batch[i] for i in sorted(per_batch)],
        "pricing_versions": sorted(pricing_versions),
    }


def build_model_speed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-call speed/cost rows by model.

    Field names are intentionally frontend-friendly and stable:
    ``model``, ``avg_s_per_call``, ``tokens_per_sec``, ``images_per_min``,
    ``total_output_tokens``, ``total_cost``, and ``n_calls``.

    Works on partial/in-progress rows too (each row is one completed call), so
    the same helper backs both finalize-time and LIVE (mid-run) summaries.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model_id") or row.get("model") or "unknown")
        bucket = buckets.setdefault(
            model,
            {
                "model": model,
                "n_calls": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "_latency_ms": 0.0,
                "_latency_n": 0,
            },
        )
        bucket["n_calls"] += 1
        try:
            bucket["total_output_tokens"] += int(row.get("output_tokens") or 0)
        except (TypeError, ValueError):
            pass
        cost = row.get("cost_usd")
        if cost is not None:
            try:
                bucket["total_cost"] += float(cost)
            except (TypeError, ValueError):
                pass
        latency = row.get("latency_ms")
        if latency is not None:
            try:
                latency_f = float(latency)
            except (TypeError, ValueError):
                latency_f = 0.0
            if latency_f > 0:
                bucket["_latency_ms"] += latency_f
                bucket["_latency_n"] += 1

    out: list[dict[str, Any]] = []
    for model in sorted(buckets):
        bucket = buckets[model]
        latency_n = int(bucket.pop("_latency_n"))
        latency_ms = float(bucket.pop("_latency_ms"))
        avg_s = (latency_ms / latency_n / 1000.0) if latency_n else None
        tokens_per_sec = (
            (float(bucket["total_output_tokens"]) / (latency_ms / 1000.0))
            if latency_ms > 0
            else None
        )
        images_per_min = (60.0 / avg_s) if (avg_s and avg_s > 0) else None
        bucket["avg_s_per_call"] = avg_s
        bucket["tokens_per_sec"] = tokens_per_sec
        bucket["images_per_min"] = images_per_min
        out.append(bucket)
    return out


__all__ = ["build_cost_row", "rollup_cost_rows", "build_model_speed_summary"]
