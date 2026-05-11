"""Provider pricing helpers for cost tracking."""
from __future__ import annotations

# Update when provider pricing changes; placeholder if unknown.
PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-5.5": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "google/gemini-3.1-pro-preview": {"input_per_mtok": 1.25, "output_per_mtok": 5.0, "image_per_image": 0.0},
    "anthropic/claude-opus-4-6": {"input_per_mtok": 15.0, "output_per_mtok": 75.0, "image_per_image": 0.0},
    "anthropic/claude-opus-4-7": {"input_per_mtok": 15.0, "output_per_mtok": 75.0, "image_per_image": 0.0},
    "openai/gpt-5.4-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    "google/gemini-3.1-flash-lite-preview": {"input_per_mtok": 0.10, "output_per_mtok": 0.40, "image_per_image": 0.0},
}


def price_for(model_id: str) -> dict[str, float] | None:
    """Return pricing for ``model_id``, or None when unknown."""
    return PRICING.get(model_id)


def compute_call_cost(
    model_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
    image_count: int = 1,
) -> float | None:
    """Compute one provider call's USD cost from usage tokens.

    Unknown model pricing returns None. If both token counts are unknown, we
    only return a cost when the model has a non-zero per-image price.
    """
    pricing = price_for(model_id)
    if pricing is None:
        return None

    image_cost = pricing["image_per_image"] * max(0, int(image_count or 0))
    tokens_known = input_tokens is not None or output_tokens is not None
    if not tokens_known and image_cost <= 0:
        return None

    input_count = 0 if input_tokens is None else max(0, int(input_tokens))
    output_count = 0 if output_tokens is None else max(0, int(output_tokens))
    return (
        pricing["input_per_mtok"] * input_count / 1_000_000
        + pricing["output_per_mtok"] * output_count / 1_000_000
        + image_cost
    )


__all__ = ["PRICING", "price_for", "compute_call_cost"]
