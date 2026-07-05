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
            recorded_at="t",
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
            recorded_at="t",
            cost_usd=0.75,
        ),
    ]

    assert build_model_speed_summary(rows) == [
        {
            "model": "m",
            "n_calls": 2,
            "total_output_tokens": 40,
            "total_cost": 1.0,
            "avg_s_per_call": 2.0,
            "tokens_per_sec": 10.0,
        }
    ]


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
            "avg_s_per_call",
            "tokens_per_sec",
            "total_output_tokens",
            "total_cost",
            "n_calls",
        }


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
