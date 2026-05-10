"""Model registry: ``model_id -> (provider, params, phase)``.

This is the single place new models get wired into the bulk-labeling
pipeline. Adding a new model means:

1. Append a :class:`ModelSpec` to :data:`MODEL_REGISTRY`.
2. (If it requires a new provider) add a ``LabelClient`` subclass and
   wire it into :func:`build_client`.

The five entries below cover the v1 plan:

* phase 1 (canonical pass): ``openai/gpt-5.5`` (reasoning=high),
  ``google/gemini-3.1-pro-preview``, ``anthropic/claude-opus-4-6``.
* phase 2 (cheaper sweep / fanout): ``openai/gpt-5.4-mini``,
  ``google/gemini-3.1-flash-lite-preview``.

We keep the model_id string intentionally shaped as ``<provider>/<model>``
so it doubles as the persistence key and the registry key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from pipeline.providers.base import LabelClient


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a registered model.

    Attributes:
        model_id: Canonical ``provider/model`` slug used everywhere.
        provider: Short provider tag (``openai``/``anthropic``/``gemini``).
        provider_model_name: Vendor-side model name passed to the SDK.
        phase: ``1`` for the canonical labeling pass, ``2`` for cheaper
            fanout / consensus expansion runs.
        params: Provider-specific knobs (``reasoning_effort``,
            ``max_tokens``, etc.) forwarded into the client config.
    """

    model_id: str
    provider: str
    provider_model_name: str
    phase: int
    params: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: Final[dict[str, ModelSpec]] = {
    # --- Phase 1: canonical labeling pass --------------------------------
    "openai/gpt-5.5": ModelSpec(
        model_id="openai/gpt-5.5",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "high",
            "max_completion_tokens": 6000,
        },
    ),
    "google/gemini-3.1-pro-preview": ModelSpec(
        model_id="google/gemini-3.1-pro-preview",
        provider="gemini",
        provider_model_name="gemini-3.1-pro-preview",
        phase=1,
        params={},
    ),
    "anthropic/claude-opus-4-6": ModelSpec(
        model_id="anthropic/claude-opus-4-6",
        provider="anthropic",
        provider_model_name="claude-opus-4-6",
        phase=1,
        params={
            "max_tokens": 2048,
        },
    ),
    # --- Phase 2: cheaper sweep / consensus fanout -----------------------
    "openai/gpt-5.4-mini": ModelSpec(
        model_id="openai/gpt-5.4-mini",
        provider="openai",
        provider_model_name="gpt-5.4-mini",
        phase=2,
        params={
            "max_completion_tokens": 2000,
        },
    ),
    "google/gemini-3.1-flash-lite-preview": ModelSpec(
        model_id="google/gemini-3.1-flash-lite-preview",
        provider="gemini",
        provider_model_name="gemini-3.1-flash-lite-preview",
        phase=2,
        params={},
    ),
}


def list_models(*, phase: int | None = None) -> list[ModelSpec]:
    """Return registry entries, optionally filtered by phase."""
    specs = list(MODEL_REGISTRY.values())
    if phase is not None:
        specs = [s for s in specs if s.phase == phase]
    return sorted(specs, key=lambda s: (s.phase, s.model_id))


def build_client(model_id: str, *, client: Any | None = None) -> LabelClient:
    """Construct a configured :class:`LabelClient` for ``model_id``.

    Imports are lazy and provider-scoped so importing the registry never
    pulls in every SDK. Pass ``client`` to inject a pre-built SDK client
    (used by tests).

    Args:
        model_id: A key from :data:`MODEL_REGISTRY`.
        client: Optional pre-built SDK client, forwarded to the provider
            constructor. ``None`` lets the client lazy-initialize.

    Returns:
        A ready-to-call :class:`LabelClient` instance.

    Raises:
        KeyError: If ``model_id`` isn't registered.
        ValueError: If the provider tag isn't recognized.
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"unknown model_id: {model_id}")
    spec = MODEL_REGISTRY[model_id]
    params = dict(spec.params)

    if spec.provider == "openai":
        from pipeline.providers.openai_client import (
            OpenAIClient,
            OpenAIClientConfig,
        )

        reasoning_effort = params.pop("reasoning_effort", None)
        max_completion_tokens = params.pop("max_completion_tokens", 6000)
        config = OpenAIClientConfig(
            model_name=spec.provider_model_name,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            extra_params=params,
        )
        return OpenAIClient(config=config, client=client)

    if spec.provider == "anthropic":
        from pipeline.providers.anthropic_client import (
            AnthropicClient,
            AnthropicClientConfig,
        )

        max_tokens = params.pop("max_tokens", 2048)
        config = AnthropicClientConfig(
            model_name=spec.provider_model_name,
            max_tokens=max_tokens,
            extra_params=params,
        )
        return AnthropicClient(config=config, client=client)

    if spec.provider == "gemini":
        from pipeline.providers.gemini_client import (
            GeminiClient,
            GeminiClientConfig,
        )

        config = GeminiClientConfig(
            model_name=spec.provider_model_name,
            extra_params=params,
        )
        return GeminiClient(config=config, client=client)

    raise ValueError(f"unsupported provider: {spec.provider}")


__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "list_models",
    "build_client",
]
