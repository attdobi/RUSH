"""Reasoning-aware cost-model tests (X4 — pricing/cost-model specialist).

Covers Bug 1 (estimate must respect reasoning tier), Bug 2 (bucket derived from
the computed estimate), and Bug 4 (Python <-> JS cost-model constants in sync).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.providers import pricing as P

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JS = (_REPO_ROOT / "web" / "run-trigger.js").read_text(encoding="utf-8")


# --- Bug 1: reasoning tier changes the estimate --------------------------------

def test_reasoning_tier_parsing() -> None:
    assert P.reasoning_tier_for("openai/gpt-5.5-xhigh") == "xhigh"
    assert P.reasoning_tier_for("openai/gpt-5.5-high") == "high"
    assert P.reasoning_tier_for("openai/gpt-5.5-medium") == "medium"
    assert P.reasoning_tier_for("openai/gpt-5.5-low") == "low"
    # No recognized suffix -> baseline "none".
    assert P.reasoning_tier_for("anthropic/claude-opus-4-6") == "none"
    assert P.reasoning_tier_for("google/gemini-3.5-flash") == "none"


def test_same_base_model_tiers_are_distinct_and_monotonic() -> None:
    base = "openai/gpt-5.5"
    xhigh = P.estimate_per_thousand_labels(f"{base}-xhigh")
    high = P.estimate_per_thousand_labels(f"{base}-high")
    medium = P.estimate_per_thousand_labels(f"{base}-medium")
    low = P.estimate_per_thousand_labels(f"{base}-low")
    assert low < medium < high < xhigh
    # No longer all-identical (the original bug).
    assert len({round(v, 6) for v in (xhigh, high, medium, low)}) == 4


@pytest.mark.parametrize("base", ["openai/gpt-5.5", "openai/gpt-5.4-mini"])
def test_calibration_ratios_hold_on_reasoning_component(base: str) -> None:
    """Attila's calibration holds on the REASONING-token component.

    Under the realistic model the total estimate includes a ~constant input +
    visible-output floor, so the 0.5/0.7 ratios live on the reasoning-token
    portion (not the total $/1k).
    """
    high = P.reasoning_tokens_for(f"{base}-high")
    medium = P.reasoning_tokens_for(f"{base}-medium")
    low = P.reasoning_tokens_for(f"{base}-low")
    assert low / high == pytest.approx(0.5, abs=0.02)
    # low is ~-30% vs medium (0.5/0.7 ≈ 0.714).
    assert low / medium == pytest.approx(0.7, abs=0.02)


def test_cross_family_realism_haiku_vs_gpt_mini() -> None:
    """Bug fix: Haiku-medium ≈ gpt-5.4-mini-low (not ~11x apart).

    Real pricing calculators put these ~equal (~$0.0042 vs ~$0.0039 / img).
    """
    haiku = P.estimate_per_thousand_labels("anthropic/claude-haiku-4-5-medium")
    mini = P.estimate_per_thousand_labels("openai/gpt-5.4-mini-low")
    # Both land in the low single-digit $/1k range and within ~25% of each other.
    assert haiku == pytest.approx(4.2, abs=0.4)
    assert mini == pytest.approx(3.9, abs=0.4)
    ratio = max(haiku, mini) / min(haiku, mini)
    assert ratio < 1.3, f"Haiku vs gpt-5.4-mini-low too far apart: {ratio:.2f}x"


def test_efficient_family_emits_little_reasoning() -> None:
    """Efficient models (Haiku/flash/local) emit far fewer reasoning tokens."""
    haiku = P.reasoning_tokens_for("anthropic/claude-haiku-4-5-medium")
    mini = P.reasoning_tokens_for("openai/gpt-5.4-mini-low")
    assert P.reasoning_family_for("anthropic/claude-haiku-4-5-medium") == "efficient"
    assert P.reasoning_family_for("google/gemini-3.5-flash") == "efficient"
    assert P.reasoning_family_for("local/qwen3.6-27b") == "efficient"
    assert P.reasoning_family_for("openai/gpt-5.4-mini-low") == "heavy"
    assert P.reasoning_family_for("anthropic/claude-opus-4-6") == "heavy"
    assert haiku < 100 < mini


def test_multipliers_anchor_high_at_one() -> None:
    assert P.REASONING_TIER_MULTIPLIERS["high"] == 1.0
    m = P.REASONING_TIER_MULTIPLIERS
    assert m["low"] < m["none"] < m["medium"] < m["high"] < m["xhigh"]


def test_local_models_estimate_zero() -> None:
    for model_id in ("local/qwen3.6-27b", "local/gemma-4-26b-a4b-qat"):
        assert P.estimate_per_thousand_labels(model_id) == 0.0


def test_unknown_model_estimate_none() -> None:
    assert P.estimate_per_thousand_labels("unknown/model") is None


# --- Bug 2 / Bug 3: tier bucket derived from computed estimate -----------------

def test_cost_tier_matches_thresholds() -> None:
    # Locals always get their own dedicated tier.
    assert P.cost_tier_for("local/qwen3.6-27b") == "LOCAL"
    assert P.cost_tier_for("local/gemma-4-26b-a4b-qat") == "LOCAL"
    # Expensive reasoning + big base models land HIGH.
    assert P.cost_tier_for("anthropic/claude-opus-4-6") == "HIGH"
    assert P.cost_tier_for("openai/gpt-5.5-xhigh") == "HIGH"
    # Cheap hosted models land LOW.
    assert P.cost_tier_for("google/gemini-3.1-flash-lite") == "LOW"
    assert P.cost_tier_for("openai/gpt-5.4-mini-low") == "LOW"


def test_bucket_always_consistent_with_price() -> None:
    """Every hosted model's bucket must agree with its computed estimate."""
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


# --- Bug 4: Python <-> JS cost-model constants in exact sync -------------------

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


def test_js_multipliers_match_python() -> None:
    assert _js_object("REASONING_TIER_MULTIPLIERS") == P.REASONING_TIER_MULTIPLIERS


def test_js_token_assumptions_match_python() -> None:
    assert _js_number("ESTIMATE_INPUT_TOKENS_PER_LABEL") == P.ESTIMATE_INPUT_TOKENS_PER_LABEL
    assert (
        _js_number("ESTIMATE_VISIBLE_OUTPUT_TOKENS_PER_LABEL")
        == P.ESTIMATE_VISIBLE_OUTPUT_TOKENS_PER_LABEL
    )


def test_js_reasoning_appetite_matches_python() -> None:
    assert _js_object("REASONING_TOKEN_APPETITE") == P.REASONING_TOKEN_APPETITE


def test_js_thresholds_match_python() -> None:
    assert _js_object("COST_TIER_THRESHOLDS") == P.COST_TIER_THRESHOLDS
