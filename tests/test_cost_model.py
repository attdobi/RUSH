"""Measured-token cost-model tests (X4 — pricing/cost-model specialist).

Recalibration: the old REASONING_TOKEN_APPETITE model was fantasy (it made
gpt-5.4-mini-xhigh ~$10.6/1k, ≈ Opus). This suite pins the estimate to the REAL
measured-token model:
  * input ~constant (prompt-driven) ~7,500 tokens (ontology dominates),
  * output grows modestly by effort tier (none<low<medium<high<xhigh),
  * cost is INPUT-DOMINATED,
and validates against Attila's corrected target prices + sanity anchors, plus
Python <-> JS constant sync.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.providers import pricing as P

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JS = (_REPO_ROOT / "web" / "run-trigger.js").read_text(encoding="utf-8")


# --- Tier parsing + monotonicity ----------------------------------------------

def test_reasoning_tier_parsing() -> None:
    assert P.reasoning_tier_for("openai/gpt-5.5-xhigh") == "xhigh"
    assert P.reasoning_tier_for("openai/gpt-5.5-high") == "high"
    assert P.reasoning_tier_for("openai/gpt-5.5-medium") == "medium"
    assert P.reasoning_tier_for("openai/gpt-5.5-low") == "low"
    # No recognized suffix -> baseline "none".
    assert P.reasoning_tier_for("anthropic/claude-opus-4-6") == "none"
    assert P.reasoning_tier_for("google/gemini-3.5-flash") == "none"


def test_output_tokens_monotonic_by_tier() -> None:
    t = P.OUTPUT_TOKENS_BY_TIER
    assert t["none"] < t["low"] < t["medium"] < t["high"] < t["xhigh"]


def test_same_base_model_tiers_are_distinct_and_monotonic() -> None:
    base = "openai/gpt-5.5"
    xhigh = P.estimate_per_thousand_labels(f"{base}-xhigh")
    high = P.estimate_per_thousand_labels(f"{base}-high")
    medium = P.estimate_per_thousand_labels(f"{base}-medium")
    low = P.estimate_per_thousand_labels(f"{base}-low")
    assert low < medium < high < xhigh
    assert len({round(v, 6) for v in (xhigh, high, medium, low)}) == 4


def test_input_is_measured_not_2200() -> None:
    # The old fantasy used 2200; real median is ~7,500 and prompt-driven.
    assert P.INPUT_TOKENS_PER_LABEL >= 6000
    assert P.estimate_input_tokens_for("openai/gpt-5.5-high") == P.INPUT_TOKENS_PER_LABEL


def test_cost_is_input_dominated() -> None:
    # For a big-input model the input term should dominate the output term.
    mid = "anthropic/claude-opus-4-6"
    pr = P.price_for(mid)
    input_term = pr["input_per_mtok"] * P.estimate_input_tokens_for(mid)
    output_term = pr["output_per_mtok"] * P.estimate_output_tokens_for(mid)
    assert input_term > output_term


def test_appetite_model_is_gone() -> None:
    assert not hasattr(P, "REASONING_TOKEN_APPETITE")
    assert not hasattr(P, "reasoning_tokens_for")
    assert "REASONING_TOKEN_APPETITE" not in _JS


# --- Validation against Attila's corrected target prices ----------------------

# Corrected target $/1k at current rates (measured-token model).
_TARGETS = {
    "openai/gpt-5.4-mini-low": 1.40,
    "openai/gpt-5.4-mini-high": 1.82,
    "google/gemini-3.1-flash-lite": 2.32,
    "anthropic/claude-haiku-4-5-low": 9.75,
    "anthropic/claude-haiku-4-5-medium": 12.25,
    "openai/gpt-5.5-low": 13.88,
    "google/gemini-3.5-flash": 13.95,
    "anthropic/claude-sonnet-5-low": 19.50,
    "openai/gpt-5.5-high": 20.98,
    "openai/gpt-5.5-xhigh": 26.07,
    "anthropic/claude-opus-4-7": 45.00,
}


@pytest.mark.parametrize("model_id,target", sorted(_TARGETS.items()))
def test_matches_corrected_target_table(model_id: str, target: float) -> None:
    est = P.estimate_per_thousand_labels(model_id)
    assert est == pytest.approx(target, abs=0.05), f"{model_id}: {est} vs {target}"


def test_gpt55_high_within_10pct_of_measured() -> None:
    # Sanity anchor Attila wants: within ~10% of measured $21.4/1k.
    est = P.estimate_per_thousand_labels("openai/gpt-5.5-high")
    assert abs(est - 21.4) / 21.4 < 0.10


def test_gpt55_xhigh_within_10pct_of_measured() -> None:
    est = P.estimate_per_thousand_labels("openai/gpt-5.5-xhigh")
    assert abs(est - 26.5) / 26.5 < 0.10


def test_mini_dramatically_cheaper_than_opus() -> None:
    mini = P.estimate_per_thousand_labels("openai/gpt-5.4-mini-low")
    opus = P.estimate_per_thousand_labels("anthropic/claude-opus-4-7")
    assert opus / mini > 10, f"opus/mini ratio only {opus / mini:.1f}x"


def test_no_cheap_model_is_absurdly_expensive() -> None:
    # The old bug: gpt-5.4-mini-xhigh ~$10.6/1k. Must now be a few dollars.
    assert P.estimate_per_thousand_labels("openai/gpt-5.4-mini-xhigh") < 5.0


def test_local_models_estimate_zero() -> None:
    for model_id in ("local/qwen3.6-27b", "local/gemma-4-26b-a4b-qat"):
        assert P.estimate_per_thousand_labels(model_id) == 0.0


def test_unknown_model_estimate_none() -> None:
    assert P.estimate_per_thousand_labels("unknown/model") is None


def test_measured_output_override_used_when_present(monkeypatch) -> None:
    monkeypatch.setitem(P.MEASURED_OUTPUT_TOKENS, "openai/gpt-5.5-high", 786)
    assert P.estimate_output_tokens_for("openai/gpt-5.5-high") == 786


# --- Tier buckets on the new scale --------------------------------------------

def test_cost_tier_matches_thresholds() -> None:
    assert P.cost_tier_for("local/qwen3.6-27b") == "LOCAL"
    assert P.cost_tier_for("local/gemma-4-26b-a4b-qat") == "LOCAL"
    # Premium land HIGH.
    assert P.cost_tier_for("anthropic/claude-opus-4-6") == "HIGH"
    assert P.cost_tier_for("openai/gpt-5.5-xhigh") == "HIGH"
    assert P.cost_tier_for("openai/gpt-5.5-high") == "HIGH"
    # Cheap hosted models land LOW (mini/flash-lite).
    assert P.cost_tier_for("google/gemini-3.1-flash-lite") == "LOW"
    assert P.cost_tier_for("openai/gpt-5.4-mini-low") == "LOW"
    assert P.cost_tier_for("openai/gpt-5.4-mini-xhigh") == "LOW"
    # Mid land MEDIUM.
    assert P.cost_tier_for("anthropic/claude-haiku-4-5-low") == "MEDIUM"
    assert P.cost_tier_for("openai/gpt-5.5-low") == "MEDIUM"


def test_bucket_always_consistent_with_price() -> None:
    for model_id in P.PRICING:
        tier = P.cost_tier_for(model_id)
        if tier == "LOCAL":
            continue
        est = P.estimate_per_thousand_labels(model_id)
        if tier == "HIGH":
            assert est >= P.COST_TIER_THRESHOLDS["high"]
        elif tier == "MEDIUM":
            assert P.COST_TIER_THRESHOLDS["medium"] <= est < P.COST_TIER_THRESHOLDS["high"]
        else:
            assert est < P.COST_TIER_THRESHOLDS["medium"]


# --- Python <-> JS constant sync ----------------------------------------------

def _js_number(name: str) -> float:
    m = re.search(rf"const {name} = ([0-9.]+);", _JS)
    assert m, f"{name} not found in run-trigger.js"
    return float(m.group(1))


def _js_object(name: str) -> dict[str, float]:
    m = re.search(rf"const {name} = \{{([^}}]*)\}};", _JS)
    assert m, f"{name} not found in run-trigger.js"
    return {
        k: float(v)
        for k, v in re.findall(r"(\w+):\s*([0-9.]+)", m.group(1))
    }


def test_js_input_tokens_match_python() -> None:
    assert _js_number("INPUT_TOKENS_PER_LABEL") == P.INPUT_TOKENS_PER_LABEL


def test_js_output_tokens_by_tier_match_python() -> None:
    js = _js_object("OUTPUT_TOKENS_BY_TIER")
    assert js == {k: float(v) for k, v in P.OUTPUT_TOKENS_BY_TIER.items()}


def test_js_thresholds_match_python() -> None:
    assert _js_object("COST_TIER_THRESHOLDS") == P.COST_TIER_THRESHOLDS
