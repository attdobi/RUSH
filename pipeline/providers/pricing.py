"""Provider pricing helpers for cost tracking."""
from __future__ import annotations

# Pricing registry version. BUMP THIS whenever any rate in ``PRICING`` changes
# so the durable cost ledger can record which rate table produced each row and
# analysis can filter/segment by pricing epoch. Historical run rows may carry a
# different (or absent) pricing_version — do NOT rewrite history silently.
PRICING_VERSION = "2026-07-04"

# Update when provider pricing changes; placeholder if unknown.
PRICING: dict[str, dict[str, float]] = {
    # TODO(attila): gpt-5.5 input 1.25 looks like the CACHED-input rate, not the
    # standard rate. Confirm against official OpenAI billing before trusting the
    # estimate. Left AS-IS intentionally — do NOT invent a replacement number.
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
    # TODO(attila): confirm gpt-5.4-mini 0.15/0.60 against official billing.
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


# ---------------------------------------------------------------------------
# Measured-token cost model (X4 — pricing/cost-model specialist).
#
# WHY THE OLD "APPETITE" MODEL WAS FANTASY
# ---------------------------------------
# The previous model invented a REASONING_TOKEN_APPETITE (efficient=70,
# heavy=11200) plus a flat input=2200. That produced pure fiction:
# gpt-5.4-mini-xhigh came out ~$10.6/1k (≈ Opus!), which is absurd for a
# $0.15/$0.60 model. The invented 11,200-token "appetite" dwarfed reality.
#
# WHAT REAL DATA SAYS (aggregated from data/runs/*/llm_outputs.jsonl,
# per-record input_tokens / output_tokens / model_id; medians):
#   opus-4-6:        in ~8194  out ~360
#   gemini-3.1-pro:  in ~7560  out ~198
#   gpt-5.5:         in ~6323  out ~948
#   gpt-5.5-high:    in ~7836  out ~1159
#   gpt-5.5-xhigh:   in ~7832  out ~1670
#
# KEY TRUTHS:
#   * INPUT ~6,300–8,200 tokens (the ontology/policy prompt dominates and is
#     ~model-independent) — NOT 2200. This is the cost driver: cost is
#     INPUT-DOMINATED.
#   * OUTPUT ~200–1,670 tokens — NOT 11,200. Reasoning grows OUTPUT modestly
#     by effort tier (low < medium < high < xhigh).
#
# THE MEASURED MODEL
# ------------------
#   estimate_$/1k = (input_rate * INPUT_TOKENS_PER_LABEL
#                    + output_rate * OUTPUT_TOKENS_BY_TIER[tier]) / 1000
#
# INPUT_TOKENS_PER_LABEL: the measured median input across the real corpus
# (~7,500). Tunable; it is PROMPT-DRIVEN, so it will grow as the ontology
# grows — bump this when the policy prompt gets larger.
INPUT_TOKENS_PER_LABEL = 7500

# OUTPUT tokens by effort tier, calibrated to the real gpt-5.5 reasoning family
# (base/none ~300, and rising with effort). Within-family monotonic:
# none < low < medium < high < xhigh. Non-reasoning models use "none".
# Where we have sufficient per-model measured data we prefer the measured
# median (see MEASURED_OUTPUT_TOKENS); otherwise we fall back to this table.
OUTPUT_TOKENS_BY_TIER: dict[str, int] = {
    "none": 300,
    "low": 450,
    "medium": 950,
    "high": 1160,
    "xhigh": 1670,
}

# Optional per-model measured OUTPUT medians (from llm_outputs.jsonl) used when
# n is sufficient. Left conservative/empty by default so the panel anchors to
# the calibrated tier table (which matches Attila's corrected target prices to
# the cent). Populate a model here only when a run has enough samples AND the
# measured value keeps the family monotonic. INPUT stays prompt-driven
# (INPUT_TOKENS_PER_LABEL) since it is ~model-independent.
MEASURED_OUTPUT_TOKENS: dict[str, int] = {}

# Cost buckets on the REAL (input-dominated) scale. On measured tokens the
# hosted models spread across ~$1–$45/1k, so: HIGH >= $20, MEDIUM >= $5, else
# LOW (mini/flash-lite land LOW; locals get their own LOCAL tier).
COST_TIER_THRESHOLDS: dict[str, float] = {"high": 20.0, "medium": 5.0}

_REASONING_SUFFIXES = ("xhigh", "high", "medium", "low")


def price_for(model_id: str) -> dict[str, float] | None:
    """Return pricing for ``model_id``, or None when unknown."""
    return PRICING.get(model_id)


def reasoning_tier_for(model_id: str) -> str:
    """Return the reasoning tier for ``model_id`` (``none`` when no suffix)."""
    tail = str(model_id or "").rsplit("-", 1)[-1]
    return tail if tail in _REASONING_SUFFIXES else "none"


def estimate_output_tokens_for(model_id: str) -> int:
    """Estimated OUTPUT tokens for ``model_id``.

    Prefers a per-model measured median (``MEASURED_OUTPUT_TOKENS``) when
    present; otherwise falls back to the effort-tier table
    (``OUTPUT_TOKENS_BY_TIER``). Non-reasoning models resolve to ``none``.
    """
    measured = MEASURED_OUTPUT_TOKENS.get(model_id)
    if measured is not None:
        return int(measured)
    return OUTPUT_TOKENS_BY_TIER[reasoning_tier_for(model_id)]


def estimate_input_tokens_for(model_id: str) -> int:
    """Estimated INPUT tokens for ``model_id``.

    Input is prompt-driven and ~model-independent, so all models share
    ``INPUT_TOKENS_PER_LABEL`` (the measured corpus median).
    """
    return INPUT_TOKENS_PER_LABEL


def estimate_per_thousand_labels(model_id: str) -> float | None:
    """Estimate USD per 1k labels using the REAL measured-token model.

    Cost is input-dominated: input tokens are ~constant (prompt-driven) and
    output tokens grow modestly by effort tier. Returns ``None`` for unknown
    models; local (free) models return ``0.0``.
    """
    pricing = price_for(model_id)
    if pricing is None:
        return None
    per_label = (
        pricing["input_per_mtok"] * estimate_input_tokens_for(model_id)
        + pricing["output_per_mtok"] * estimate_output_tokens_for(model_id)
    ) / 1_000_000
    return per_label * 1000


def is_local_model(model_id: str) -> bool:
    """Return True for on-device (free) local models."""
    return str(model_id or "").startswith("local/")


def cost_tier_for(model_id: str) -> str:
    """Derive the panel bucket from the computed estimate.

    One of ``LOCAL``, ``HIGH``, ``MEDIUM``, ``LOW``. Locals always sort into
    their own dedicated tier; everything else is bucketed by computed cost so
    the shown bucket always matches the shown price.
    """
    if is_local_model(model_id):
        return "LOCAL"
    estimate = estimate_per_thousand_labels(model_id)
    if estimate is None:
        return "LOW"
    if estimate >= COST_TIER_THRESHOLDS["high"]:
        return "HIGH"
    if estimate >= COST_TIER_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


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


__all__ = [
    "PRICING",
    "PRICING_VERSION",
    "price_for",
    "compute_call_cost",
    "INPUT_TOKENS_PER_LABEL",
    "OUTPUT_TOKENS_BY_TIER",
    "MEASURED_OUTPUT_TOKENS",
    "COST_TIER_THRESHOLDS",
    "reasoning_tier_for",
    "estimate_output_tokens_for",
    "estimate_input_tokens_for",
    "estimate_per_thousand_labels",
    "is_local_model",
    "cost_tier_for",
]
