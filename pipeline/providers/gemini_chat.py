"""Chat-callable factory for policy drafting / gate review with Gemini.

Text-only twin of :mod:`pipeline.providers.openai_chat` /
:mod:`pipeline.providers.anthropic_chat` so Gemini models can serve as the
experiment crank's drafter or gate agent (Attila 2026-07-09: gemini-3.1-flash
as a gate option). The ``google-genai`` SDK is imported and the client built
lazily by the returned callable, so importing this module never authenticates.

Message mapping: OpenAI-style chat messages -> ``models.generate_content``.
The leading system message becomes ``system_instruction``; user/assistant
turns map to Gemini's ``user``/``model`` roles.
"""
from __future__ import annotations

import logging
from typing import Any

from pipeline.policy_iterator import ChatCallable
from pipeline.providers import auth
from pipeline.providers._config import LABELING_VISIBLE_OUTPUT_TOKENS
from pipeline.providers.registry import MODEL_REGISTRY

_DEFAULT_MAX_TOKENS = LABELING_VISIBLE_OUTPUT_TOKENS
# Generous budget for policy drafting / gate verdicts (JSON, not labels).
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


def _usage_field(usage: Any, *names: str) -> int | None:
    for name in names:
        raw = getattr(usage, name, None)
        if isinstance(usage, dict):
            raw = usage.get(name, raw)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return None


def _extract_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata") or response.get("usage")
    if usage is None:
        return None, None, None
    # Gemini's prompt_token_count INCLUDES cached tokens; the cached count is
    # reported separately for the 0.25x discount (same convention the labeling
    # client + cost ledger already use).
    return (
        _usage_field(usage, "prompt_token_count", "input_token_count"),
        _usage_field(usage, "candidates_token_count", "output_token_count"),
        _usage_field(usage, "cached_content_token_count"),
    )


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None) or (
        response.get("candidates") if isinstance(response, dict) else None
    )
    parts_out: list[str] = []
    for candidate in candidates or []:
        content = getattr(candidate, "content", None) or (
            candidate.get("content") if isinstance(candidate, dict) else None
        )
        parts = getattr(content, "parts", None) or (
            content.get("parts") if isinstance(content, dict) else None
        )
        for part in parts or []:
            part_text = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if part_text:
                parts_out.append(str(part_text))
    return "".join(parts_out)


def _spec_for(model_id: str) -> tuple[str, int]:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise KeyError(f"unknown model_id: {model_id}")
    if spec.provider == "local":
        raise ValueError(f"model_id is not Gemini-backed: {model_id}")
    if spec.provider != "gemini" and not spec.model_id.startswith("google/"):
        raise ValueError(f"model_id is not Gemini-backed: {model_id}")
    labeling_cap = int(spec.params.get("max_output_tokens", _DEFAULT_MAX_TOKENS))
    return spec.provider_model_name, max(labeling_cap, _POLICY_MAX_TOKENS)


def policy_chat_callable(
    model_id: str,
    *,
    usage_sink: list[dict[str, Any]] | None = None,
) -> ChatCallable:
    """Return ``chat(messages, *, model_id, reasoning_effort='high', **_) -> str``.

    ``usage_sink``, when given, receives one ``{"model_id", "input_tokens",
    "output_tokens", "cached_input_tokens"}`` dict per call — same contract as
    the OpenAI/Anthropic twins, so the crank's cost ledger works unchanged.
    """
    default_model_name, default_max_tokens = _spec_for(model_id)
    client_holder: dict[str, Any] = {}

    def _ensure_client() -> Any:
        if "client" not in client_holder:
            from google import genai  # type: ignore[import-not-found]

            client_holder["client"] = genai.Client(
                api_key=auth.get_secret(auth.GEMINI_API_KEY_VAR)
            )
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
        if model_id != bound_model_id:
            provider_model_name, max_tokens = _spec_for(model_id)

        system: str | None = None
        contents: list[dict[str, Any]] = []
        for idx, message in enumerate(messages):
            role = message.get("role")
            text = _content_to_text(message.get("content", ""))
            if idx == 0 and role == "system":
                system = text
                continue
            if role == "system":
                # Gemini takes exactly one system instruction; keep later
                # system text visible as user context instead of dropping it.
                contents.append({"role": "user", "parts": [{"text": text}]})
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(f"unsupported chat role for Gemini: {role!r}")
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": text}],
            })

        config: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system is not None:
            config["system_instruction"] = system
        response = _ensure_client().models.generate_content(
            model=provider_model_name, contents=contents, config=config
        )
        input_tokens, output_tokens, cached_tokens = _extract_usage(response)
        if usage_sink is not None:
            usage_sink.append(
                {
                    "model_id": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_input_tokens": cached_tokens,
                }
            )
        if input_tokens is None and output_tokens is None:
            logger.info("usage_unknown for %s", model_id)
        return _extract_text(response)

    bound_model_id = model_id
    return chat


__all__ = ["policy_chat_callable"]
