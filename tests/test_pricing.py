from __future__ import annotations

from pipeline.providers.pricing import PRICING, compute_call_cost, price_for


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

    for variant in ("openai/gpt-5.4-mini-xhigh", "openai/gpt-5.4-mini-high", "openai/gpt-5.4-mini-low"):
        assert price_for(variant) == price_for("openai/gpt-5.4-mini")
        assert compute_call_cost(variant, 1_000_000, 500_000) == compute_call_cost(
            "openai/gpt-5.4-mini", 1_000_000, 500_000
        )


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
