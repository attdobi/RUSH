from __future__ import annotations

import pytest

from pipeline.scoring.cost import aggregate_per_call_costs


def test_aggregate_per_call_costs_mixed_models_and_missing_tokens() -> None:
    votes = [
        {
            "model_id": "openai/gpt-5.5",
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cost_usd": 2.25,
        },
        {
            "model_id": "openai/gpt-5.5",
            "input_tokens": 500_000,
            "output_tokens": None,
            # no explicit cost: computed as 0.625 from input tokens only
        },
        {
            "model_id": "anthropic/claude-opus-4-6",
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
        },
    ]

    out = aggregate_per_call_costs(votes)

    assert out["total_calls"] == 3
    assert out["total_cost_usd"] == pytest.approx(2.875)
    assert out["cost_per_1000_labels"] == pytest.approx(2.875 / 3 * 1000)

    openai = out["per_model"]["openai/gpt-5.5"]
    assert openai["total_calls"] == 2
    assert openai["total_input_tokens"] == 1_500_000
    assert openai["total_output_tokens"] == 100_000
    assert openai["total_cost_usd"] == pytest.approx(2.875)
    assert openai["cost_per_1000_labels"] == pytest.approx(2.875 / 2 * 1000)
    assert openai["usage_unknown_calls"] == 1

    anthropic = out["per_model"]["anthropic/claude-opus-4-6"]
    assert anthropic["total_calls"] == 1
    assert anthropic["total_input_tokens"] == 0
    assert anthropic["total_output_tokens"] == 0
    assert anthropic["total_cost_usd"] == 0.0
    assert anthropic["cost_per_1000_labels"] is None
    assert anthropic["usage_unknown_calls"] == 1
