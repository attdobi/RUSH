"""Shared provider request configuration.

This module is the single source of truth for default labeling generation
temperature across provider clients and runner manifests.
"""

from __future__ import annotations

LABELING_TEMPERATURE: float = 0.1

# Target VISIBLE-output ceiling for provider APIs that expose a clean output cap
# separate from hidden reasoning (Anthropic max_tokens, Gemini max_output_tokens).
# The justification prompt caps at <=300 words (~400 tokens), so 1000 visible
# tokens comfortably fits the JSON + justification without truncation. Anthropic
# max_tokens / Gemini max_output_tokens visible backstops are raised to 1000
# (Attila's call, reasoning-safe).
#
# NOTE: OpenAI Chat Completions only exposes max_completion_tokens, which is a
# COMBINED budget (hidden reasoning + visible JSON). Those registry entries set
# their own higher max_completion_tokens and do NOT use this visible-only cap,
# so lowering this value never truncates OpenAI reasoning.
LABELING_VISIBLE_OUTPUT_TOKENS: int = 1000


def resolve_temperature(model_id: str, override: float | None = None) -> float | None:
    """Return the temperature to use for a labeling request.

    Explicit overrides are honored first. Otherwise, normal labeling models use
    :data:`LABELING_TEMPERATURE`. GPT-5.5 reasoning models return ``None`` so
    callers can omit the request key entirely, because custom temperature is
    not supported for that model family.
    """
    if override is not None:
        return override

    bare_model_id = model_id.split("/", 1)[-1]
    if bare_model_id.startswith("gpt-5.5"):
        # Reasoning model: custom temperature is unsupported; omit the key.
        return None
    return LABELING_TEMPERATURE


__all__ = ["LABELING_TEMPERATURE", "LABELING_VISIBLE_OUTPUT_TOKENS", "resolve_temperature"]
