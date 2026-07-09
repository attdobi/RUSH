"""Durable per-image cost ledger tests (X1 backend).

Covers the ledger row builder (live-registry rates + pricing_version), the
rollup helper (per-run/per-model/per-batch), the runner integration (costs.jsonl
+ manifest per_model), and the aggregate_costs.py roll-up script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pipeline.manifest import SampleRecord
from pipeline.providers import pricing as P
from pipeline.providers.base import LabelRequest, LabelResponse
from pipeline.runner import DeterministicFakeClient, run_labeling
from pipeline.scoring.cost_ledger import (
    build_cost_row,
    build_model_speed_summary,
    rollup_cost_rows,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

SAMPLES = [
    SampleRecord(
        sample_id=f"dev_golden_{idx:04d}",
        repo_rel_path=f"data/images/test/dev_golden_{idx:04d}.jpg",
        split="dev_golden",
        sme_label_raw="ai_generated" if idx % 2 else "not_ai_generated",
        sme_label="gen_ai" if idx % 2 else "not_gen_ai",
        dataset="test",
        sha256=f"{idx:064x}"[-64:],
        sampling_version="test-sampling-v1",
    )
    for idx in range(1, 7)
]


class UsageClient(DeterministicFakeClient):
    def _response_for(self, request: LabelRequest) -> LabelResponse:
        resp = super()._response_for(request)
        resp.input_tokens = 1000
        resp.output_tokens = 200
        return resp


# --- build_cost_row ------------------------------------------------------------

def test_build_cost_row_uses_live_registry_and_stamps_version() -> None:
    row = build_cost_row(
        run_id="20260704T000000-abcd1234",
        batch_index=2,
        image_id="img_1",
        model_id="anthropic/claude-opus-4-6",
        input_tokens=1_000_000,
        output_tokens=100_000,
        latency_ms=2000,
        recorded_at="2026-07-04T00:00:00Z",
    )
    # LIVE opus-4-6 rate is 5 / 25 (NOT the stale $15/$75 that inflated history).
    assert row["input_rate_per_mtok"] == 5.0
    assert row["output_rate_per_mtok"] == 25.0
    assert row["cost_usd"] == 5.0 * 1 + 25.0 * 0.1  # 7.5
    assert row["pricing_version"] == P.PRICING_VERSION
    assert row["batch_id"] == "20260704T000000-abcd1234:2"
    assert row["run_id"] == "20260704T000000-abcd1234"
    assert row["latency_ms"] == 2000
    assert row["tokens_per_sec"] == 50000.0


def test_build_cost_row_cache_fields_change_cost_and_persist() -> None:
    """The ledger must bill cache-aware, not full-rate.

    Regression: the 2026-07-09 live run's costs.jsonl billed gemini/openai at
    full input rate (over) and dropped haiku's cache-read charges (under) —
    the live card's per-model rows disagreed with the header by ~8%.
    """
    # Anthropic: reads/writes are ADDITIVE on top of input_tokens.
    row = build_cost_row(
        run_id="r", batch_index=0, image_id="a",
        model_id="anthropic/claude-haiku-4-5-low",
        input_tokens=100, output_tokens=1000, recorded_at="t",
        cached_input_tokens=100_000, cache_creation_input_tokens=10_000,
    )
    expected = (
        1.0 * 100 + 1.0 * 100_000 * 0.10 + 1.0 * 10_000 * 1.25 + 5.0 * 1000
    ) / 1_000_000
    assert row["cost_usd"] == pytest.approx(expected)
    assert row["cached_input_tokens"] == 100_000
    assert row["cache_creation_input_tokens"] == 10_000

    # OpenAI: cached tokens are a DISCOUNTED SUBSET of input_tokens.
    row = build_cost_row(
        run_id="r", batch_index=0, image_id="a",
        model_id="openai/gpt-5.4-mini-low",
        input_tokens=100_000, output_tokens=1000, recorded_at="t",
        cached_input_tokens=80_000,
    )
    expected = (0.15 * 20_000 + 0.15 * 80_000 * 0.50 + 0.60 * 1000) / 1_000_000
    assert row["cost_usd"] == pytest.approx(expected)


def test_build_cost_row_unknown_model_null_rates_and_cost() -> None:
    row = build_cost_row(
        run_id="r",
        batch_index=0,
        image_id="i",
        model_id="unknown/model",
        input_tokens=100,
        output_tokens=10,
        recorded_at="t",
    )
    assert row["input_rate_per_mtok"] is None
    assert row["cost_usd"] is None
    assert row["pricing_version"] == P.PRICING_VERSION


# --- rollup --------------------------------------------------------------------

def test_rollup_per_model_and_per_batch() -> None:
    rows = [
        build_cost_row(run_id="r", batch_index=0, image_id="a",
                       model_id="openai/gpt-5.5", input_tokens=1_000_000,
                       output_tokens=100_000, recorded_at="t"),
        build_cost_row(run_id="r", batch_index=0, image_id="b",
                       model_id="openai/gpt-5.5", input_tokens=1_000_000,
                       output_tokens=100_000, recorded_at="t"),
        build_cost_row(run_id="r", batch_index=1, image_id="c",
                       model_id="local/qwen3.6-27b", input_tokens=500,
                       output_tokens=50, recorded_at="t"),
    ]
    out = rollup_cost_rows(rows)
    assert out["total"]["images"] == 3
    assert set(out["per_model"]) == {"openai/gpt-5.5", "local/qwen3.6-27b"}
    gpt = out["per_model"]["openai/gpt-5.5"]
    assert gpt["images"] == 2
    per_call = 1.25 + 1.0  # 1.25/Mtok*1M + 10/Mtok*100k
    assert gpt["total_cost_usd"] == per_call * 2
    assert gpt["cost_per_1000_labels"] == per_call * 1000
    assert len(out["per_batch"]) == 2
    assert out["pricing_versions"] == [P.PRICING_VERSION]


def test_rollup_flags_mixed_pricing_versions() -> None:
    rows = [
        {"model_id": "m", "batch_index": 0, "input_tokens": 1, "output_tokens": 1,
         "cost_usd": 1.0, "pricing_version": "legacy"},
        {"model_id": "m", "batch_index": 0, "input_tokens": 1, "output_tokens": 1,
         "cost_usd": 1.0, "pricing_version": P.PRICING_VERSION},
    ]
    out = rollup_cost_rows(rows)
    assert out["pricing_versions"] == sorted({"legacy", P.PRICING_VERSION})


def test_model_speed_summary_fields() -> None:
    rows = [
        build_cost_row(
            run_id="r",
            batch_index=0,
            image_id="a",
            model_id="m",
            input_tokens=100,
            output_tokens=10,
            latency_ms=1000,
            recorded_at="2026-07-04T00:00:01Z",
            cost_usd=0.25,
        ),
        build_cost_row(
            run_id="r",
            batch_index=1,
            image_id="b",
            model_id="m",
            input_tokens=100,
            output_tokens=30,
            latency_ms=3000,
            recorded_at="2026-07-04T00:00:03Z",
            cost_usd=0.75,
        ),
    ]

    out = build_model_speed_summary(rows)

    assert len(out) == 1
    row = out[0]
    assert row["model"] == "m"
    assert row["model_id"] == "m"
    assert row["n_calls"] == 2
    assert row["calls_done"] == 2
    assert row["total_input_tokens"] == 200
    assert row["total_output_tokens"] == 40
    assert row["total_cost"] == 1.0
    assert row["total_cost_usd"] == 1.0
    assert row["first_started_at"] == "2026-07-04T00:00:00Z"
    assert row["last_finished_at"] == "2026-07-04T00:00:03Z"
    assert row["active_elapsed_s"] == 3.0
    assert row["avg_s_per_call"] == 2.0
    assert row["avg_latency_ms"] == 2000.0
    assert row["tokens_per_sec"] == pytest.approx(40.0 / 3.0)
    assert row["images_per_min"] == 40.0
    assert row["throughput_imgs_per_min"] == 40.0
    # No cache activity: the canonical total is just in + out.
    assert row["total_cached_input_tokens"] == 0
    assert row["total_cache_write_tokens"] == 0
    assert row["total_tokens"] == 240


def test_model_speed_summary_total_tokens_is_provider_consistent() -> None:
    """Anthropic reports cache reads OUTSIDE input_tokens; the display total
    must add them back or a warm-cache haiku pass shows ~120 input tokens
    while gemini (cached INSIDE the prompt count) shows six figures."""
    rows = [
        build_cost_row(
            run_id="r", batch_index=0, image_id="a",
            model_id="anthropic/claude-haiku-4-5-low",
            input_tokens=6, output_tokens=390, latency_ms=1000,
            recorded_at="2026-07-09T00:00:01Z",
            cached_input_tokens=5900, cache_creation_input_tokens=100,
        ),
        build_cost_row(
            run_id="r", batch_index=0, image_id="a",
            model_id="openai/gpt-5.4-mini-low",
            input_tokens=6000, output_tokens=300, latency_ms=1000,
            recorded_at="2026-07-09T00:00:01Z",
            cached_input_tokens=5000,
        ),
    ]
    out = {row["model_id"]: row for row in build_model_speed_summary(rows)}
    haiku = out["anthropic/claude-haiku-4-5-low"]
    assert haiku["total_cached_input_tokens"] == 5900
    assert haiku["total_cache_write_tokens"] == 100
    assert haiku["total_tokens"] == 6 + 5900 + 100 + 390
    mini = out["openai/gpt-5.4-mini-low"]
    # Cached already sits INSIDE prompt_tokens: no double count.
    assert mini["total_tokens"] == 6000 + 300
    assert mini["total_cached_input_tokens"] == 5000


# --- runner integration --------------------------------------------------------

def test_runner_writes_costs_jsonl_and_manifest_per_model() -> None:
    with TemporaryDirectory() as tmp:
        summary = run_labeling(
            models=["openai/gpt-5.5"],
            samples=SAMPLES,
            split="dev_golden",
            runs_root=Path(tmp),
            client_factory=lambda spec: UsageClient(spec.model_id),
            batch_size=3,
            dry_run=True,
        )
        run_dir = summary.paths.root
        ledger_path = run_dir / "costs.jsonl"
        assert ledger_path.exists()
        rows = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
        assert len(rows) == 6
        for row in rows:
            assert row["model_id"] == "openai/gpt-5.5"
            assert row["input_rate_per_mtok"] == 1.25
            assert row["output_rate_per_mtok"] == 10.0
            assert row["pricing_version"] == P.PRICING_VERSION
            assert row["input_tokens"] == 1000
            assert row["output_tokens"] == 200
            assert row["latency_ms"] == 0
            assert row["tokens_per_sec"] is None
            assert row["batch_index"] in (0, 1)
            per_image = 1.25 * 1000 / 1e6 + 10.0 * 200 / 1e6
            assert abs(row["cost_usd"] - per_image) < 1e-12

        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        cost = manifest["cost"]
        assert cost["pricing_version"] == P.PRICING_VERSION
        assert "openai/gpt-5.5" in cost["per_model"]
        assert cost["per_model"]["openai/gpt-5.5"]["images"] == 6
        speed = json.loads((run_dir / "model_speed_summary.json").read_text())
        assert speed["run_id"] == summary.run_id
        assert speed["models"][0]["model"] == "openai/gpt-5.5"
        assert set(speed["models"][0]) == {
            "model",
            "model_id",
            "avg_s_per_call",
            "avg_latency_ms",
            "tokens_per_sec",
            "images_per_min",
            "throughput_imgs_per_min",
            "first_started_at",
            "last_finished_at",
            "active_elapsed_s",
            "total_input_tokens",
            "total_output_tokens",
            "total_cached_input_tokens",
            "total_cache_write_tokens",
            "total_tokens",
            "total_cost_usd",
            "total_cost",
            "n_calls",
            "calls_done",
        }
        assert "per_model_timing" in speed
        assert speed["per_model_timing"]["per_model"] == speed["models"]
        assert speed["per_model_timing"]["total"]["total_input_tokens"] == 6000
        assert manifest["per_model_timing"]["per_model"] == speed["models"]
        assert manifest["per_model_timing"]["total"]["total_output_tokens"] == 1200


# --- aggregate script ----------------------------------------------------------

def test_aggregate_costs_script_combines_ledger_and_legacy() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from scripts import aggregate_costs as agg

    with TemporaryDirectory() as tmp:
        runs_root = Path(tmp)
        # Ledger run.
        r1 = runs_root / "20260704T000000-aaaaaaaa"
        r1.mkdir()
        (r1 / "costs.jsonl").write_text(json.dumps(build_cost_row(
            run_id=r1.name, batch_index=0, image_id="x",
            model_id="openai/gpt-5.5", input_tokens=100, output_tokens=10,
            recorded_at="t")) + "\n")
        # Legacy run (only llm_outputs.jsonl, stale cost).
        r2 = runs_root / "20260101T000000-bbbbbbbb"
        r2.mkdir()
        (r2 / "llm_outputs.jsonl").write_text(json.dumps({
            "image_id": "y", "model_id": "anthropic/claude-opus-4-6",
            "recorded_at": "t",
            "output": {"input_tokens": 8000, "output_tokens": 300, "cost_usd": 0.15},
        }) + "\n")

        out_csv = runs_root / "combined.csv"
        rc = agg.main(["--runs-root", str(runs_root), "--out-csv", str(out_csv),
                       "--include-legacy"])
        assert rc == 0
        text = out_csv.read_text()
        assert "ledger" in text and "legacy_llm_outputs" in text
        assert "legacy" in text  # pricing_version flag for stale rows

        rows = agg.collect_rows(runs_root, include_legacy=True)
        assert len(rows) == 2
        legacy = [r for r in rows if r["source"] == "legacy_llm_outputs"][0]
        assert legacy["pricing_version"] == "legacy"
