"""Chat-callable factory for policy proposal drafting with Claude.

The Anthropic SDK is imported and instantiated lazily by the returned callable,
so importing this module never performs authentication or network setup.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from pipeline.policy_iterator import ChatCallable
from pipeline.providers._config import LABELING_VISIBLE_OUTPUT_TOKENS
from pipeline.providers.registry import MODEL_REGISTRY


_DEFAULT_MAX_TOKENS = LABELING_VISIBLE_OUTPUT_TOKENS
# Generous budget for policy DRAFTING (distinct from image labeling output caps).
_POLICY_MAX_TOKENS = 8000
logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block["text"]))
                elif block.get("content") is not None:
                    parts.append(str(block["content"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None
    input_raw = getattr(usage, "input_tokens", None)
    output_raw = getattr(usage, "output_tokens", None)
    if isinstance(usage, dict):
        input_raw = usage.get("input_tokens", input_raw)
        output_raw = usage.get("output_tokens", output_raw)
    try:
        input_tokens = int(input_raw) if input_raw is not None else None
    except (TypeError, ValueError):
        input_tokens = None
    try:
        output_tokens = int(output_raw) if output_raw is not None else None
    except (TypeError, ValueError):
        output_tokens = None
    return input_tokens, output_tokens


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype != "text":
            continue
        text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
        if text:
            parts.append(str(text))
    return "".join(parts)


def _spec_for(model_id: str) -> tuple[str, int]:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise KeyError(f"unknown model_id: {model_id}")
    if spec.provider != "anthropic":
        raise ValueError(f"model_id is not Anthropic-backed: {model_id}")
    # Policy DRAFTING can be long: floor the (small) labeling visible-output cap
    # up to a generous policy-generation budget so shrinking labeling max_tokens
    # never truncates policy proposals.
    labeling_cap = int(spec.params.get("max_tokens", _DEFAULT_MAX_TOKENS))
    return spec.provider_model_name, max(labeling_cap, _POLICY_MAX_TOKENS)


def policy_chat_callable(
    model_id: str,
    *,
    usage_sink: list[dict[str, Any]] | None = None,
) -> ChatCallable:
    """Return ``chat(messages, *, model_id, reasoning_effort='high', **_) -> str``.

    The callable maps OpenAI-style chat messages into Anthropic Messages API
    params: the first system message becomes ``system=...`` and subsequent
    user/assistant messages are forwarded via ``messages=[...]``.

    ``usage_sink``, when given, receives one ``{"model_id", "input_tokens",
    "output_tokens"}`` dict per call (see the OpenAI twin for rationale).
    """
    default_model_name, default_max_tokens = _spec_for(model_id)
    client_holder: dict[str, Any] = {}

    def _ensure_client() -> Any:
        if "client" not in client_holder:
            import anthropic  # type: ignore[import-not-found]

            client_holder["client"] = anthropic.Anthropic()
        return client_holder["client"]

    def chat(
        messages: list[dict[str, Any]],
        *,
        model_id: str = model_id,
        reasoning_effort: str = "high",
        **_: Any,
    ) -> str:
        provider_model_name = default_model_name
        max_tokens = default_max_tokens
        if model_id != policy_chat_callable_model_id:
            provider_model_name, max_tokens = _spec_for(model_id)

        system: str | None = None
        rest: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")
            if idx == 0 and role == "system":
                system = _content_to_text(content)
                continue
            if role == "system":
                # Anthropic accepts only a top-level system prompt; preserve
                # any later system text as user-visible context instead of
                # dropping it silently.
                rest.append({"role": "user", "content": _content_to_text(content)})
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"unsupported chat role for Anthropic: {role!r}")
            rest.append({"role": role, "content": content})

        params: dict[str, Any] = {
            "model": provider_model_name,
            "max_tokens": max_tokens,
            "messages": rest,
        }
        if system is not None:
            # Block form with an ephemeral breakpoint: the system prompt is
            # identical across a run's drafter/gate calls, so repeats within
            # the cache TTL read it at ~0.1x. Below the model's minimum
            # cacheable prefix this is a silent no-op (no extra cost). Callers
            # marking their own stable user blocks (e.g. the drafter's policy
            # bundle) extend the cached prefix further — user content is
            # forwarded verbatim, cache_control included.
            params["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        response = _ensure_client().messages.create(**params)
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
        return _extract_text(response)

    policy_chat_callable_model_id = model_id
    return chat


__all__ = ["policy_chat_callable"]
