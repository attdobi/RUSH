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

from datetime import datetime, timedelta, timezone
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
    cached_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> dict[str, Any]:
    """Build one analysis-ready per-image-per-model cost ledger row.

    Cost is (re)computed with the LIVE pricing registry unless an explicit
    ``cost_usd`` is provided. Rates and ``pricing_version`` are always stamped
    from the current registry so the row is self-describing for later analysis.

    Cache fields matter: without them the recompute bills the full input rate,
    which OVERSTATES OpenAI/Gemini calls (cached tokens are a discounted
    subset of input) and UNDERSTATES Anthropic calls (cache reads/writes are
    billed on top of input_tokens). Callers with cache usage must pass both.
    """
    pricing = price_for(model_id)
    input_rate = float(pricing["input_per_mtok"]) if pricing else None
    output_rate = float(pricing["output_per_mtok"]) if pricing else None
    image_rate = float(pricing["image_per_image"]) if pricing else None

    if cost_usd is None:
        cost_usd = compute_call_cost(
            model_id, input_tokens, output_tokens, image_count=image_count,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
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
        "cached_input_tokens": (
            None if cached_input_tokens is None else int(cached_input_tokens)
        ),
        "cache_creation_input_tokens": (
            None if cache_creation_input_tokens is None
            else int(cache_creation_input_tokens)
        ),
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
        "total_cached_input_tokens": 0,
        "total_cache_write_tokens": 0,
        "total_cost_usd": 0.0,
        "cost_per_image_usd": None,
        "cost_per_1000_labels": None,
    }


def _total_tokens_for(
    model_id: str, input_tokens: int, output_tokens: int,
    cached_tokens: int, write_tokens: int,
) -> int:
    """Provider-consistent "tokens the model processed" total.

    Anthropic reports cache reads/writes OUTSIDE input_tokens, so they must be
    added back; OpenAI/Gemini report cached tokens INSIDE their prompt count,
    so input + output already covers them. Without this, an Anthropic row
    shows only the uncached remainder (e.g. 120 input tokens against 118k
    served from cache) and cross-provider token columns are apples-to-oranges.
    """
    provider = model_id.split("/", 1)[0] if "/" in model_id else model_id
    if provider == "anthropic":
        return input_tokens + cached_tokens + write_tokens + output_tokens
    return input_tokens + output_tokens


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
        cached_tok = int(row.get("cached_input_tokens") or 0)
        write_tok = int(row.get("cache_creation_input_tokens") or 0)
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
            bucket["total_cached_input_tokens"] += cached_tok
            bucket["total_cache_write_tokens"] += write_tok
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


def _parse_iso(ts: object) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iso_z(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_model_speed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-call speed/cost/timing rows by model.

    Each row is one completed model call. When ``recorded_at`` and
    ``latency_ms`` are available, throughput is computed from the model's own
    active window: earliest ``recorded_at - latency_ms`` through latest
    ``recorded_at``. That window stops advancing when a model finishes, so
    per-model ``images_per_min`` naturally freezes while other models continue.

    The output keeps the legacy render aliases (``model``, ``n_calls``,
    ``total_cost``) while adding the status-table contract fields used by the
    frontend (``model_id``, ``calls_done``, token totals, cost totals, timing
    window, and rates).
    """
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row.get("model_id") or row.get("model") or "unknown")
        bucket = buckets.setdefault(
            model,
            {
                "model": model,
                "model_id": model,
                "n_calls": 0,
                "calls_done": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cached_input_tokens": 0,
                "total_cache_write_tokens": 0,
                "total_cost_usd": 0.0,
                "total_cost": 0.0,
                "_latency_ms": 0.0,
                "_latency_n": 0,
                "_first_started_at": None,
                "_last_finished_at": None,
            },
        )
        bucket["n_calls"] += 1
        bucket["calls_done"] += 1
        try:
            bucket["total_input_tokens"] += int(row.get("input_tokens") or 0)
        except (TypeError, ValueError):
            pass
        try:
            bucket["total_output_tokens"] += int(row.get("output_tokens") or 0)
        except (TypeError, ValueError):
            pass
        try:
            bucket["total_cached_input_tokens"] += int(row.get("cached_input_tokens") or 0)
        except (TypeError, ValueError):
            pass
        try:
            bucket["total_cache_write_tokens"] += int(
                row.get("cache_creation_input_tokens") or 0
            )
        except (TypeError, ValueError):
            pass
        cost = row.get("cost_usd")
        if cost is not None:
            try:
                cost_f = float(cost)
            except (TypeError, ValueError):
                pass
            else:
                bucket["total_cost"] += cost_f
                bucket["total_cost_usd"] += cost_f
        latency = row.get("latency_ms")
        latency_f: float | None = None
        if latency is not None:
            try:
                latency_f = float(latency)
            except (TypeError, ValueError):
                latency_f = 0.0
            if latency_f > 0:
                bucket["_latency_ms"] += latency_f
                bucket["_latency_n"] += 1
        finished_at = _parse_iso(row.get("recorded_at"))
        if finished_at is not None:
            latency_for_start = latency_f if latency_f is not None and latency_f >= 0 else 0.0
            started_at = finished_at - timedelta(milliseconds=latency_for_start)
            first_started = bucket["_first_started_at"]
            last_finished = bucket["_last_finished_at"]
            if first_started is None or started_at < first_started:
                bucket["_first_started_at"] = started_at
            if last_finished is None or finished_at > last_finished:
                bucket["_last_finished_at"] = finished_at

    out: list[dict[str, Any]] = []
    for model in sorted(buckets):
        bucket = buckets[model]
        latency_n = int(bucket.pop("_latency_n"))
        latency_ms = float(bucket.pop("_latency_ms"))
        first_started_at = bucket.pop("_first_started_at")
        last_finished_at = bucket.pop("_last_finished_at")
        avg_s = (latency_ms / latency_n / 1000.0) if latency_n else None
        active_elapsed_s = None
        if first_started_at is not None and last_finished_at is not None:
            active_elapsed_s = max(0.0, (last_finished_at - first_started_at).total_seconds())
        elif latency_ms > 0:
            active_elapsed_s = latency_ms / 1000.0
        tokens_per_sec = (
            (float(bucket["total_output_tokens"]) / active_elapsed_s)
            if active_elapsed_s and active_elapsed_s > 0
            else None
        )
        images_per_min = (
            (float(bucket["calls_done"]) / (active_elapsed_s / 60.0))
            if active_elapsed_s and active_elapsed_s > 0
            else None
        )
        bucket["first_started_at"] = _iso_z(first_started_at)
        bucket["last_finished_at"] = _iso_z(last_finished_at)
        bucket["active_elapsed_s"] = active_elapsed_s
        bucket["avg_s_per_call"] = avg_s
        bucket["avg_latency_ms"] = (avg_s * 1000.0) if avg_s is not None else None
        bucket["tokens_per_sec"] = tokens_per_sec
        bucket["images_per_min"] = images_per_min
        bucket["throughput_imgs_per_min"] = images_per_min
        bucket["total_tokens"] = _total_tokens_for(
            model,
            int(bucket["total_input_tokens"]),
            int(bucket["total_output_tokens"]),
            int(bucket["total_cached_input_tokens"]),
            int(bucket["total_cache_write_tokens"]),
        )
        out.append(bucket)
    return out


def build_per_model_timing_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the persisted per-model timing block plus run-level totals."""
    per_model = build_model_speed_summary(rows)
    firsts = [_parse_iso(row.get("first_started_at")) for row in per_model]
    lasts = [_parse_iso(row.get("last_finished_at")) for row in per_model]
    firsts = [dt for dt in firsts if dt is not None]
    lasts = [dt for dt in lasts if dt is not None]
    total_calls = sum(int(row.get("calls_done") or row.get("n_calls") or 0) for row in per_model)
    total_input = sum(int(row.get("total_input_tokens") or 0) for row in per_model)
    total_output = sum(int(row.get("total_output_tokens") or 0) for row in per_model)
    total_cached = sum(int(row.get("total_cached_input_tokens") or 0) for row in per_model)
    total_write = sum(int(row.get("total_cache_write_tokens") or 0) for row in per_model)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in per_model)
    total_cost = sum(float(row.get("total_cost_usd") or 0.0) for row in per_model)
    first_started = min(firsts) if firsts else None
    last_finished = max(lasts) if lasts else None
    active_elapsed_s = (
        max(0.0, (last_finished - first_started).total_seconds())
        if first_started is not None and last_finished is not None
        else None
    )
    total = {
        "calls_done": total_calls,
        "n_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cached_input_tokens": total_cached,
        "total_cache_write_tokens": total_write,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "total_cost": total_cost,
        "first_started_at": _iso_z(first_started),
        "last_finished_at": _iso_z(last_finished),
        "active_elapsed_s": active_elapsed_s,
        "images_per_min": (
            (float(total_calls) / (active_elapsed_s / 60.0))
            if active_elapsed_s and active_elapsed_s > 0
            else None
        ),
        "tokens_per_sec": (
            (float(total_output) / active_elapsed_s)
            if active_elapsed_s and active_elapsed_s > 0
            else None
        ),
    }
    return {"per_model": per_model, "total": total}


__all__ = [
    "build_cost_row",
    "rollup_cost_rows",
    "build_model_speed_summary",
    "build_per_model_timing_block",
]
