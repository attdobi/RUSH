"""Provider pricing helpers for cost tracking."""
from __future__ import annotations

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
# Reasoning-aware cost model (X4 — pricing/cost-model specialist).
#
# WHY THE OLD UNIFORM MODEL WAS WRONG
# -----------------------------------
# The previous model assumed EVERY model emits the same token counts
# (input=800, output=400), scaled only by a per-tier multiplier. That cannot
# capture the real driver of cross-family cost: how many *reasoning* tokens a
# model burns. It made anthropic/claude-haiku-4-5-medium look ~11x pricier
# ($1.96/1k) than openai/gpt-5.4-mini-low ($0.18/1k) when real pricing
# calculators put them ~EQUAL (~$0.0042 vs ~$0.0039 per image). The per-token
# RATES were correct; the token-COUNT model was the bug.
#
# THE REALISTIC MODEL
# -------------------
# total_output_tokens = VISIBLE_output (~constant JSON + short justification)
#                       + reasoning_tokens(model_family, effort_tier)
#
#   * Efficient / near-linear models (Haiku 4.5, Gemini flash / flash-lite,
#     local gemma/qwen) emit mostly VISIBLE tokens with a TINY reasoning tail.
#   * Heavy reasoners (gpt-5.4-mini, gpt-5.5, opus/sonnet reasoning, gemini
#     pro) burn THOUSANDS of hidden reasoning tokens. This is exactly why a
#     cheap-per-token heavy reasoner (gpt-5.4-mini) converges in $/img with a
#     pricier-per-token near-linear model (Haiku).
#
# reasoning_tokens = REASONING_TOKEN_APPETITE[family] × REASONING_TIER_MULTIPLIERS[tier]
#
# CALIBRATION ANCHORS (Attila's real-world calculator, ~2200 input tokens):
#   * Haiku-4.5 (efficient)      ≈ $0.0042/img (~$4.2/1k) → ~400 output tokens.
#   * gpt-5.4-mini-low (heavy)   ≈ $0.0039/img (~$3.9/1k) → ~5,950 output tokens.
#
# All constants below are named, documented and tunable, and mirrored in
# web/run-trigger.js (sync-tested).
#
# Within-family reasoning monotonicity + Attila's ratios are preserved on the
# REASONING-token component (anchor high = 1.0):
#   low  ≈ 0.5 × high     (0.5 / 1.0)
#   low  ≈ 0.7 × medium   (0.5 / 0.7 ≈ 0.714 → low is ~-29% vs medium)
REASONING_TIER_MULTIPLIERS: dict[str, float] = {
    "xhigh": 1.5,
    "high": 1.0,   # anchor
    "medium": 0.7,
    "low": 0.5,
    # Base / non-reasoning models: a documented "none" baseline between low and
    # medium (a plain call still spends some hidden tokens).
    "none": 0.6,
}

# Baseline per-label token assumptions used only for the panel *estimate*
# (not for actual billing, which uses real usage via compute_call_cost).
#
# INPUT: one image + labeling prompt. A single ~512px vision tile plus the
# policy/justification prompt lands around ~2200 tokens across providers. This
# is the a-priori assumption the calculator anchors on; tune here.
ESTIMATE_INPUT_TOKENS_PER_LABEL = 2200
# VISIBLE output: the JSON verdict + <=300-word justification (~350 tokens),
# near-constant across models/tiers. Reasoning tokens are added on TOP of this.
ESTIMATE_VISIBLE_OUTPUT_TOKENS_PER_LABEL = 350

# Per-FAMILY reasoning-token appetite at the anchor tier (high = 1.0). This is
# the knob that captures cross-family realism.
#   * "efficient": near-linear vision models with a tiny reasoning tail. Anchored
#     so Haiku-medium (0.7×) lands ~400 total output tokens → ~$4.2/1k.
#   * "heavy": deliberate reasoners that burn thousands of hidden tokens.
#     Anchored so gpt-5.4-mini-low (0.5×) lands ~5,950 total output tokens
#     (~5,600 reasoning) → ~$3.9/1k.
REASONING_TOKEN_APPETITE: dict[str, float] = {
    "efficient": 70.0,
    "heavy": 11200.0,
}

# Cost buckets derived from the computed reasoning-adjusted $/1k-label estimate.
# Under the realistic model, cheap models cluster in the low single digits, so
# thresholds are set to keep meaningful separation: cheap near-linear + cheap
# reasoners (Haiku, gpt-5.4-mini-low, flash-lite) land LOW; mid land MEDIUM;
# premium/heavy-reasoning models land HIGH. Locals get their own tier.
COST_TIER_THRESHOLDS: dict[str, float] = {"high": 8.0, "medium": 5.0}

_REASONING_SUFFIXES = ("xhigh", "high", "medium", "low")

# Substrings that mark a model as belonging to the EFFICIENT (near-linear)
# family. Everything else is treated as a HEAVY reasoner.
_EFFICIENT_FAMILY_MARKERS = ("haiku", "flash", "local/")


def price_for(model_id: str) -> dict[str, float] | None:
    """Return pricing for ``model_id``, or None when unknown."""
    return PRICING.get(model_id)


def reasoning_tier_for(model_id: str) -> str:
    """Return the reasoning tier for ``model_id`` (``none`` when no suffix)."""
    tail = str(model_id or "").rsplit("-", 1)[-1]
    return tail if tail in _REASONING_SUFFIXES else "none"


def reasoning_multiplier_for(model_id: str) -> float:
    """Return the reasoning-tier cost multiplier for ``model_id``."""
    return REASONING_TIER_MULTIPLIERS[reasoning_tier_for(model_id)]


def reasoning_family_for(model_id: str) -> str:
    """Classify ``model_id`` into a reasoning-appetite family.

    Returns ``"efficient"`` for near-linear vision models (Haiku, Gemini flash
    variants, local models) and ``"heavy"`` for deliberate reasoners.
    """
    mid = str(model_id or "").lower()
    if any(marker in mid for marker in _EFFICIENT_FAMILY_MARKERS):
        return "efficient"
    return "heavy"


def reasoning_tokens_for(model_id: str) -> float:
    """Estimated hidden REASONING tokens for ``model_id``.

    ``REASONING_TOKEN_APPETITE[family] * REASONING_TIER_MULTIPLIERS[tier]``.
    Preserves Attila's within-family ratios on the reasoning component
    (low = 0.5x high, low = 0.7x medium).
    """
    appetite = REASONING_TOKEN_APPETITE[reasoning_family_for(model_id)]
    return appetite * reasoning_multiplier_for(model_id)


def estimate_output_tokens_for(model_id: str) -> float:
    """Estimated TOTAL output tokens = visible output + reasoning tokens."""
    return ESTIMATE_VISIBLE_OUTPUT_TOKENS_PER_LABEL + reasoning_tokens_for(model_id)


def estimate_per_thousand_labels(model_id: str) -> float | None:
    """Estimate USD per 1k labels using the reasoning-aware token model.

    Input tokens are ~constant; output tokens = visible + per-family reasoning
    tokens scaled by the effort tier. Returns ``None`` for unknown models.
    Local (free) models return ``0.0``.
    """
    pricing = price_for(model_id)
    if pricing is None:
        return None
    per_label = (
        pricing["input_per_mtok"] * ESTIMATE_INPUT_TOKENS_PER_LABEL
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
    "price_for",
    "compute_call_cost",
    "REASONING_TIER_MULTIPLIERS",
    "REASONING_TOKEN_APPETITE",
    "ESTIMATE_INPUT_TOKENS_PER_LABEL",
    "ESTIMATE_VISIBLE_OUTPUT_TOKENS_PER_LABEL",
    "COST_TIER_THRESHOLDS",
    "reasoning_tier_for",
    "reasoning_multiplier_for",
    "reasoning_family_for",
    "reasoning_tokens_for",
    "estimate_output_tokens_for",
    "estimate_per_thousand_labels",
    "is_local_model",
    "cost_tier_for",
]
