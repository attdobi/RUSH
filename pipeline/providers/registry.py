"""Model registry: ``model_id -> (provider, params, phase)``.

This is the single place new models get wired into the bulk-labeling
pipeline. Adding a new model means:

1. Append a :class:`ModelSpec` to :data:`MODEL_REGISTRY`.
2. (If it requires a new provider) add a ``LabelClient`` subclass and
   wire it into :func:`build_client`.

The five entries below cover the v1 plan:

* phase 1 (canonical pass): ``openai/gpt-5.5`` (reasoning=xhigh),
  ``google/gemini-3.1-pro-preview``, ``anthropic/claude-opus-4-6``,
  ``anthropic/claude-opus-4-7``.
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
            "reasoning_effort": "xhigh",
            # High reasoning can consume well over 6k internal tokens before
            # emitting JSON; keep the cap roomy enough to avoid empty outputs.
            "max_completion_tokens": 24000,
        },
    ),
    "openai/gpt-5.5-xhigh": ModelSpec(
        model_id="openai/gpt-5.5-xhigh",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "xhigh",
            "max_completion_tokens": 24000,
        },
    ),
    "openai/gpt-5.5-high": ModelSpec(
        model_id="openai/gpt-5.5-high",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "high",
            "max_completion_tokens": 24000,
        },
    ),
    "google/gemini-3.1-pro-preview": ModelSpec(
        model_id="google/gemini-3.1-pro-preview",
        provider="gemini",
        provider_model_name="gemini-3.1-pro-preview",
        phase=1,
        params={"thinking_budget_tokens": -1},
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
    "anthropic/claude-opus-4-7": ModelSpec(
        model_id="anthropic/claude-opus-4-7",
        provider="anthropic",
        provider_model_name="claude-opus-4-7",
        phase=1,
        params={
            "max_tokens": 4096,
            "thinking_budget_tokens": 32000,
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
    "openai/gpt-5.4-mini-xhigh": ModelSpec(
        model_id="openai/gpt-5.4-mini-xhigh",
        provider="openai",
        provider_model_name="gpt-5.4-mini",
        phase=2,
        params={
            "reasoning_effort": "xhigh",
            "max_completion_tokens": 2000,
        },
    ),
    "openai/gpt-5.4-mini-high": ModelSpec(
        model_id="openai/gpt-5.4-mini-high",
        provider="openai",
        provider_model_name="gpt-5.4-mini",
        phase=2,
        params={
            "reasoning_effort": "high",
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


def build_client(
    model_id: str,
    *,
    client: Any | None = None,
    reasoning_effort: str | None = None,
) -> LabelClient:
    """Construct a configured :class:`LabelClient` for ``model_id``.

    Imports are lazy and provider-scoped so importing the registry never
    pulls in every SDK. Pass ``client`` to inject a pre-built SDK client
    (used by tests).

    Args:
        model_id: A key from :data:`MODEL_REGISTRY`.
        client: Optional pre-built SDK client, forwarded to the provider
            constructor. ``None`` lets the client lazy-initialize.
        reasoning_effort: Optional per-run OpenAI reasoning override for
            historical ``openai/gpt-5.5``. Variant ids such as
            ``openai/gpt-5.5-high`` encode their reasoning level in the
            registry and should not be combined with this override.

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

        configured_reasoning_effort = params.pop("reasoning_effort", None)
        if model_id == "openai/gpt-5.5" and reasoning_effort is not None:
            configured_reasoning_effort = reasoning_effort
        max_completion_tokens = params.pop("max_completion_tokens", 6000)
        config = OpenAIClientConfig(
            model_name=spec.provider_model_name,
            reasoning_effort=configured_reasoning_effort,
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
        thinking_budget_tokens = params.pop("thinking_budget_tokens", None)
        config = AnthropicClientConfig(
            model_name=spec.provider_model_name,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_params=params,
        )
        return AnthropicClient(config=config, client=client)

    if spec.provider == "gemini":
        from pipeline.providers.gemini_client import (
            GeminiClient,
            GeminiClientConfig,
        )

        thinking_budget_tokens = params.pop("thinking_budget_tokens", None)
        config = GeminiClientConfig(
            model_name=spec.provider_model_name,
            thinking_budget_tokens=thinking_budget_tokens,
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
