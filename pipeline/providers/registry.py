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
  ``google/gemini-3.1-flash-lite``.

We keep the model_id string intentionally shaped as ``<provider>/<model>``
so it doubles as the persistence key and the registry key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Final

from pipeline.providers._config import LABELING_VISIBLE_OUTPUT_TOKENS
from pipeline.providers.base import LabelClient

# Local OpenAI-compatible endpoint (LM Studio). Overridable via env so the
# same code runs against a remote box if Attila moves the server.
DEFAULT_LOCAL_BASE_URL: Final[str] = "http://127.0.0.1:1234/v1"


def local_base_url() -> str:
    """Resolve the local OpenAI-compatible base URL (env-overridable)."""
    return os.environ.get("RUSH_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL)


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
            # Chat Completions exposes only max_completion_tokens, which is a
            # COMBINED budget (hidden reasoning + visible JSON). Do NOT cap it
            # at the ~768 visible budget or reasoning gets truncated. Bound it
            # at reasoning headroom + ~768 visible: ~4000 for high/xhigh.
            "max_completion_tokens": 4000,
        },
    ),
    "openai/gpt-5.5-xhigh": ModelSpec(
        model_id="openai/gpt-5.5-xhigh",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "xhigh",
            "max_completion_tokens": 4000,
        },
    ),
    "openai/gpt-5.5-high": ModelSpec(
        model_id="openai/gpt-5.5-high",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "high",
            "max_completion_tokens": 4000,
        },
    ),
    # Low-reasoning gpt-5.5 variant: cheapest reasoning tier of the 5.5 family.
    # Combined budget = low-reasoning headroom + ~768 visible ~= 2000.
    "openai/gpt-5.5-low": ModelSpec(
        model_id="openai/gpt-5.5-low",
        provider="openai",
        provider_model_name="gpt-5.5",
        phase=1,
        params={
            "reasoning_effort": "low",
            "max_completion_tokens": 2000,
        },
    ),
    "google/gemini-3.1-pro-preview": ModelSpec(
        model_id="google/gemini-3.1-pro-preview",
        provider="gemini",
        provider_model_name="gemini-3.1-pro-preview",
        phase=1,
        params={
            # Gemini SEPARATES thinking from visible output: thinking_budget is
            # its own pool, and max_output_tokens caps only the visible JSON.
            # Keep the thinking pool generous; cap visible output at ~768.
            "thinking_budget_tokens": 8000,
            "max_output_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
        },
    ),
    "anthropic/claude-opus-4-6": ModelSpec(
        model_id="anthropic/claude-opus-4-6",
        provider="anthropic",
        provider_model_name="claude-opus-4-6",
        phase=1,
        params={
            "max_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
        },
    ),
    "anthropic/claude-opus-4-7": ModelSpec(
        model_id="anthropic/claude-opus-4-7",
        provider="anthropic",
        provider_model_name="claude-opus-4-7",
        phase=1,
        params={
            "max_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
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
            # Combined budget: reasoning headroom + ~768 visible.
            "max_completion_tokens": 4000,
        },
    ),
    "openai/gpt-5.4-mini-high": ModelSpec(
        model_id="openai/gpt-5.4-mini-high",
        provider="openai",
        provider_model_name="gpt-5.4-mini",
        phase=2,
        params={
            "reasoning_effort": "high",
            "max_completion_tokens": 4000,
        },
    ),
    # Cheap mini at low reasoning — sane cheapest default for a fast sweep.
    "openai/gpt-5.4-mini-low": ModelSpec(
        model_id="openai/gpt-5.4-mini-low",
        provider="openai",
        provider_model_name="gpt-5.4-mini",
        phase=2,
        params={
            "reasoning_effort": "low",
            "max_completion_tokens": 2000,
        },
    ),
    # Cheap Sonnet 4.6 — low/standard reasoning (NO extended thinking budget).
    "anthropic/claude-sonnet-4-6": ModelSpec(
        model_id="anthropic/claude-sonnet-4-6",
        provider="anthropic",
        provider_model_name="claude-sonnet-4-6",
        phase=2,
        params={
            "max_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
            # No thinking_budget_tokens: standard (non-extended) reasoning.
        },
    ),
    # Cheap Sonnet 5 — LOW reasoning (small/no thinking budget). Anthropic
    # max_tokens is a clean VISIBLE cap (separate from thinking), so ~768 is
    # safe here. INTRO pricing 2.0/10.0 through 2026-08-31 (see pricing.py).
    "anthropic/claude-sonnet-5": ModelSpec(
        model_id="anthropic/claude-sonnet-5",
        provider="anthropic",
        provider_model_name="claude-sonnet-5",
        phase=2,
        params={
            "max_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
            # No thinking_budget_tokens: low/standard (non-extended) reasoning.
        },
    ),
    # Anthropic's cheap/fast VISION model (200K ctx, vision-capable) — the
    # recommended gpt-5.4-mini equivalent for image labeling. Pricing 1.0/5.0.
    "anthropic/claude-haiku-4-5": ModelSpec(
        model_id="anthropic/claude-haiku-4-5",
        provider="anthropic",
        provider_model_name="claude-haiku-4-5",
        phase=2,
        params={
            "max_tokens": LABELING_VISIBLE_OUTPUT_TOKENS,
            # No thinking_budget_tokens: fast, non-extended reasoning.
        },
    ),
    # Latest Gemini flash (GA). Mid-tier: output $9/Mtok.
    "google/gemini-3.5-flash": ModelSpec(
        model_id="google/gemini-3.5-flash",
        provider="gemini",
        provider_model_name="gemini-3.5-flash",
        phase=2,
        params={},
    ),
    # Gemini 3 Flash Preview — cheaper than 3.5 flash ($0.50/$3.00).
    "google/gemini-3-flash-preview": ModelSpec(
        model_id="google/gemini-3-flash-preview",
        provider="gemini",
        provider_model_name="gemini-3-flash-preview",
        phase=2,
        params={},
    ),
    # Cheapest Gemini: Flash-Lite GA ($0.25/$1.50).
    "google/gemini-3.1-flash-lite": ModelSpec(
        model_id="google/gemini-3.1-flash-lite",
        provider="gemini",
        provider_model_name="gemini-3.1-flash-lite",
        phase=2,
        params={},
    ),
    # --- Local GPU (LM Studio, free) -------------------------------------
    "local/qwen3.6-27b": ModelSpec(
        model_id="local/qwen3.6-27b",
        provider="local",
        provider_model_name="qwen/qwen3.6-27b",
        phase=2,
        params={
            # Local reasoning model: hidden reasoning eats completion tokens
            # before the visible JSON. qwen needs more headroom than gemma so
            # the JSON is not truncated (finish_reason=length -> parse_failed).
            "max_completion_tokens": 6000,
        },
    ),
    "local/gemma-4-26b-a4b-qat": ModelSpec(
        model_id="local/gemma-4-26b-a4b-qat",
        provider="local",
        provider_model_name="google/gemma-4-26b-a4b-qat",
        phase=2,
        params={
            # Reasoning model, but gemma is compact: a ~4000 combined budget
            # keeps it snappy while leaving room for reasoning + visible JSON.
            "max_completion_tokens": 4000,
        },
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
        max_completion_tokens = params.pop("max_completion_tokens", 10000)
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

        max_tokens = params.pop("max_tokens", LABELING_VISIBLE_OUTPUT_TOKENS)
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
        max_output_tokens = params.pop("max_output_tokens", None)
        config = GeminiClientConfig(
            model_name=spec.provider_model_name,
            thinking_budget_tokens=thinking_budget_tokens,
            max_output_tokens=max_output_tokens,
            extra_params=params,
        )
        return GeminiClient(config=config, client=client)

    if spec.provider == "local":
        # Local OpenAI-compatible endpoint (LM Studio). Reuse OpenAIClient with
        # a base_url override; auth is optional for local servers.
        from pipeline.providers.openai_client import (
            OpenAIClient,
            OpenAIClientConfig,
        )

        configured_reasoning_effort = params.pop("reasoning_effort", None)
        max_completion_tokens = params.pop("max_completion_tokens", 4000)
        config = OpenAIClientConfig(
            model_name=spec.provider_model_name,
            reasoning_effort=configured_reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            base_url=local_base_url(),
            # LM Studio rejects response_format={"type": "json_object"} with
            # HTTP 400 (only "json_schema" or "text" are accepted). Use "text";
            # the shared parse_label_json handles the fenced JSON these local
            # reasoning models emit.
            response_format={"type": "text"},
            extra_params=params,
        )
        return OpenAIClient(config=config, client=client)

    raise ValueError(f"unsupported provider: {spec.provider}")


__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "list_models",
    "build_client",
    "local_base_url",
    "DEFAULT_LOCAL_BASE_URL",
]
