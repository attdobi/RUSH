"""Shared provider request configuration.

This module is the single source of truth for default labeling generation
temperature across provider clients and runner manifests.
"""

from __future__ import annotations

LABELING_TEMPERATURE: float = 0.1


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


__all__ = ["LABELING_TEMPERATURE", "resolve_temperature"]
