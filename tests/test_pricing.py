from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.providers.pricing import PRICING, compute_call_cost, price_for

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_price_for_known_and_unknown_model() -> None:
    assert price_for("openai/gpt-5.5") == PRICING["openai/gpt-5.5"]
    assert price_for("unknown/model") is None


def test_compute_call_cost_known_model() -> None:
    cost = compute_call_cost("openai/gpt-5.5", 1_000_000, 500_000)
    assert cost == 1.25 + 5.0


def test_compute_call_cost_unknown_model_returns_none() -> None:
    assert compute_call_cost("unknown/model", 1_000, 1_000) is None


def test_compute_call_cost_both_tokens_null_without_image_price_returns_none() -> None:
    assert compute_call_cost("openai/gpt-5.5", None, None) is None


def test_compute_call_cost_only_output_null_treats_as_zero() -> None:
    assert compute_call_cost("openai/gpt-5.5", 1_000_000, None) == 1.25


def test_compute_call_cost_image_price_path(monkeypatch) -> None:
    monkeypatch.setitem(
        PRICING,
        "test/image-priced",
        {"input_per_mtok": 0.0, "output_per_mtok": 0.0, "image_per_image": 0.02},
    )
    assert compute_call_cost("test/image-priced", None, None, image_count=3) == 0.06


def test_reasoning_variant_prices_match_base_models() -> None:
    for variant in ("openai/gpt-5.5-xhigh", "openai/gpt-5.5-high"):
        assert price_for(variant) == price_for("openai/gpt-5.5")
        assert compute_call_cost(variant, 1_000_000, 500_000) == compute_call_cost(
            "openai/gpt-5.5", 1_000_000, 500_000
        )

    # gpt-5.5-low mirrors gpt-5.5 base pricing too.
    assert price_for("openai/gpt-5.5-low") == price_for("openai/gpt-5.5")

    for variant in ("openai/gpt-5.4-mini-xhigh", "openai/gpt-5.4-mini-high", "openai/gpt-5.4-mini-low"):
        assert price_for(variant) == price_for("openai/gpt-5.4-mini")
        assert compute_call_cost(variant, 1_000_000, 500_000) == compute_call_cost(
            "openai/gpt-5.4-mini", 1_000_000, 500_000
        )


def test_openai_medium_variant_prices_match_base_models() -> None:
    assert price_for("openai/gpt-5.5-medium") == price_for("openai/gpt-5.5")
    assert price_for("openai/gpt-5.4-mini-medium") == price_for("openai/gpt-5.4-mini")


def test_opus_pricing_corrected_2026() -> None:
    # Verified 2026 rates: Opus 4.6 (dated but kept) and 4.7 both list at 5 / 25.
    for model_id in ("anthropic/claude-opus-4-6", "anthropic/claude-opus-4-7"):
        assert price_for(model_id) == {
            "input_per_mtok": 5.0,
            "output_per_mtok": 25.0,
            "image_per_image": 0.0,
        }


def test_local_models_are_free() -> None:
    for model_id in ("local/qwen3.6-27b", "local/gemma-4-26b-a4b-qat"):
        pricing = price_for(model_id)
        assert pricing == {"input_per_mtok": 0.0, "output_per_mtok": 0.0, "image_per_image": 0.0}
        # Free even with large token counts.
        assert compute_call_cost(model_id, 1_000_000, 1_000_000) == 0.0


def test_new_hosted_models_priced() -> None:
    assert price_for("anthropic/claude-sonnet-4-6") == {
        "input_per_mtok": 3.0,
        "output_per_mtok": 15.0,
        "image_per_image": 0.0,
    }
    assert price_for("google/gemini-3.5-flash") is not None
    assert price_for("google/gemini-3-flash-preview") is not None
    assert price_for("google/gemini-3.1-flash-lite") == {
        "input_per_mtok": 0.25,
        "output_per_mtok": 1.50,
        "image_per_image": 0.0,
    }


def test_new_models_priced() -> None:
    # Sonnet 5 intro pricing (2.0/10.0) and Haiku 4.5 (1.0/5.0).
    for model_id in ("anthropic/claude-sonnet-5-low", "anthropic/claude-sonnet-5-medium"):
        assert price_for(model_id) == {
            "input_per_mtok": 2.0,
            "output_per_mtok": 10.0,
            "image_per_image": 0.0,
        }
    for model_id in ("anthropic/claude-haiku-4-5-low", "anthropic/claude-haiku-4-5-medium"):
        assert price_for(model_id) == {
            "input_per_mtok": 1.0,
            "output_per_mtok": 5.0,
            "image_per_image": 0.0,
        }


def _parse_js_pricing() -> dict[str, dict[str, float]]:
    """Parse the PRICING_PER_MTOK object literal from web/run-trigger.js."""
    text = (_REPO_ROOT / "web" / "run-trigger.js").read_text(encoding="utf-8")
    start = text.index("const PRICING_PER_MTOK = {")
    body = text[start:]
    depth = 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                body = body[body.index("{"): i + 1]
                break
    out: dict[str, dict[str, float]] = {}
    row_re = re.compile(
        r"'([^']+)'\s*:\s*\{\s*input:\s*([0-9.]+)\s*,\s*output:\s*([0-9.]+)\s*\}"
    )
    for m in row_re.finditer(body):
        out[m.group(1)] = {"input": float(m.group(2)), "output": float(m.group(3))}
    return out


def test_py_js_pricing_in_exact_sync() -> None:
    """pipeline/providers/pricing.py and web/run-trigger.js must match exactly."""
    js = _parse_js_pricing()
    py = {
        model: {"input": spec["input_per_mtok"], "output": spec["output_per_mtok"]}
        for model, spec in PRICING.items()
    }
    assert js == py, f"pricing drift: py-only={set(py) - set(js)}, js-only={set(js) - set(py)}"
