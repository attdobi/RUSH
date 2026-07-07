"""Chat-callable factory for policy proposal drafting with OpenAI.

The OpenAI SDK is imported and instantiated lazily by the returned callable, so
importing this module never performs authentication or network setup.
"""
from __future__ import annotations

import logging
from typing import Any

from pipeline.policy_iterator import ChatCallable
from pipeline.providers import auth
from pipeline.providers.registry import MODEL_REGISTRY


# Policy DRAFTING is a different workload than image LABELING: proposals can be
# long, so this path keeps a generous combined budget and intentionally does NOT
# inherit the (much lower) per-model labeling max_completion_tokens caps.
_DEFAULT_MAX_COMPLETION_TOKENS = 10000
logger = logging.getLogger(__name__)


def _spec_for(model_id: str) -> tuple[str, int]:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise KeyError(f"unknown model_id: {model_id}")
    if spec.provider != "openai":
        raise ValueError(f"model_id is not OpenAI-backed: {model_id}")
    # Floor the labeling cap up to the policy-drafting budget so shrinking the
    # labeling caps never truncates policy generation.
    labeling_cap = int(spec.params.get("max_completion_tokens", _DEFAULT_MAX_COMPLETION_TOKENS))
    return spec.provider_model_name, max(labeling_cap, _DEFAULT_MAX_COMPLETION_TOKENS)


_EFFORT_SUFFIXES = ("low", "medium", "high", "xhigh")


def _effective_effort(model_id: str, requested: str) -> str:
    """Honor the effort encoded in an effort-suffixed model id.

    ``openai/gpt-5.5-low`` MEANS low reasoning effort — callers that pass a
    generic default (e.g. policy drafting's ``reasoning_effort="high"``)
    must not silently promote it. Non-suffixed ids keep the caller's choice.
    """
    tail = model_id.rsplit("-", 1)[-1]
    if tail in _EFFORT_SUFFIXES:
        spec = MODEL_REGISTRY.get(model_id)
        if spec is not None and spec.params.get("reasoning_effort"):
            return str(spec.params["reasoning_effort"])
        return tail
    return requested


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", prompt)
        completion = usage.get("completion_tokens", completion)
    try:
        input_tokens = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        input_tokens = None
    try:
        output_tokens = int(completion) if completion is not None else None
    except (TypeError, ValueError):
        output_tokens = None
    return input_tokens, output_tokens


def _extract_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except AttributeError:
        content = response["choices"][0].get("message", {}).get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return "" if content is None else str(content)


def policy_chat_callable(
    model_id: str,
    *,
    usage_sink: list[dict[str, Any]] | None = None,
) -> ChatCallable:
    """Return ``chat(messages, *, model_id, reasoning_effort='high', **_) -> str``.

    Messages are forwarded directly to OpenAI's chat-completions endpoint with
    ``response_format={"type": "json_object"}`` to keep policy proposal parsing
    deterministic.

    ``usage_sink``, when given, receives one ``{"model_id", "input_tokens",
    "output_tokens"}`` dict per call — the hook cost-accounted callers (the
    experiment crank's drafter/gate agents) use to ledger text-only spend,
    which is otherwise only logged.
    """
    default_model_name, default_max_completion_tokens = _spec_for(model_id)
    client_holder: dict[str, Any] = {}

    def _ensure_client() -> Any:
        if "client" not in client_holder:
            from openai import OpenAI  # type: ignore[import-not-found]

            client_holder["client"] = OpenAI(api_key=auth.get_secret(auth.OPENAI_API_KEY_VAR))
        return client_holder["client"]

    def chat(
        messages: list[dict[str, Any]],
        *,
        model_id: str = model_id,
        reasoning_effort: str = "high",
        **_: Any,
    ) -> str:
        provider_model_name = default_model_name
        max_completion_tokens = default_max_completion_tokens
        if model_id != policy_chat_callable_model_id:
            provider_model_name, max_completion_tokens = _spec_for(model_id)

        params: dict[str, Any] = {
            "model": provider_model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "reasoning_effort": _effective_effort(model_id, reasoning_effort),
            "max_completion_tokens": max_completion_tokens,
        }
        response = _ensure_client().chat.completions.create(**params)
        input_tokens, output_tokens = _extract_usage_tokens(response)
        if usage_sink is not None:
            usage_sink.append(
                {
                    "model_id": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
        if input_tokens is None and output_tokens is None:
            logger.info("usage_unknown for %s", model_id)
        else:
            logger.info(
                "openai_policy_usage model_id=%s input_tokens=%s output_tokens=%s",
                model_id,
                input_tokens,
                output_tokens,
            )
        return _extract_text(response)

    policy_chat_callable_model_id = model_id
    return chat


__all__ = ["policy_chat_callable"]
