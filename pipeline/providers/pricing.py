"""Provider pricing helpers for cost tracking."""
from __future__ import annotations

# Update when provider pricing changes; placeholder if unknown.
PRICING: dict[str, dict[str, float]] = {
    # Soft note: gpt-5.5 input 1.25 looks like the cached-input rate if a standard rate is ever needed.
    "openai/gpt-5.5": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "openai/gpt-5.5-xhigh": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "openai/gpt-5.5-high": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "openai/gpt-5.5-medium": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    # gpt-5.5 low-reasoning variant: mirrors gpt-5.5 base pricing.
    "openai/gpt-5.5-low": {"input_per_mtok": 1.25, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "google/gemini-3.1-pro-preview": {"input_per_mtok": 2.0, "output_per_mtok": 12.0, "image_per_image": 0.0},
    # Opus 4.6 (dated but kept). Verified 2026 rate: 5 / 25 per Mtok.
    "anthropic/claude-opus-4-6": {"input_per_mtok": 5.0, "output_per_mtok": 25.0, "image_per_image": 0.0},
    # Opus 4.7: same list price (5 / 25), BUT Opus 4.7+ uses a newer tokenizer that
    # emits ~30% more tokens per image/prompt, so effective cost is ~1.3x the
    # list rate versus 4.6. HIGH cost tier, unchecked by default in the panel.
    "anthropic/claude-opus-4-7": {"input_per_mtok": 5.0, "output_per_mtok": 25.0, "image_per_image": 0.0},
    "openai/gpt-5.4-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    "openai/gpt-5.4-mini-xhigh": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    "openai/gpt-5.4-mini-high": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    "openai/gpt-5.4-mini-medium": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    "openai/gpt-5.4-mini-low": {"input_per_mtok": 0.15, "output_per_mtok": 0.60, "image_per_image": 0.0},
    # Sonnet 4.6: verified 3.0 / 15.0 per Mtok.
    "anthropic/claude-sonnet-4-6": {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "image_per_image": 0.0},
    # Sonnet 5: INTRODUCTORY pricing 2.0 / 10.0 through 2026-08-31.
    # standard 3.0/15.0 after 2026-08-31; +30% tokenizer.
    "anthropic/claude-sonnet-5-low": {"input_per_mtok": 2.0, "output_per_mtok": 10.0, "image_per_image": 0.0},
    "anthropic/claude-sonnet-5-medium": {"input_per_mtok": 2.0, "output_per_mtok": 10.0, "image_per_image": 0.0},
    # Haiku 4.5: cheap/fast vision model. Verified 1.0 / 5.0 per Mtok.
    "anthropic/claude-haiku-4-5-low": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "image_per_image": 0.0},
    "anthropic/claude-haiku-4-5-medium": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "image_per_image": 0.0},
    # Latest Gemini flash (GA).
    "google/gemini-3.5-flash": {"input_per_mtok": 1.50, "output_per_mtok": 9.0, "image_per_image": 0.0},
    # Gemini 3 Flash Preview.
    "google/gemini-3-flash-preview": {"input_per_mtok": 0.50, "output_per_mtok": 3.0, "image_per_image": 0.0},
    # Cheapest Gemini: Flash-Lite GA.
    "google/gemini-3.1-flash-lite": {"input_per_mtok": 0.25, "output_per_mtok": 1.50, "image_per_image": 0.0},
    # Local GPU (LM Studio) — free.
    "local/qwen3.6-27b": {"input_per_mtok": 0.0, "output_per_mtok": 0.0, "image_per_image": 0.0},
    "local/gemma-4-26b-a4b-qat": {"input_per_mtok": 0.0, "output_per_mtok": 0.0, "image_per_image": 0.0},
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
