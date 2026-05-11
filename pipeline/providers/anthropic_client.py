"""Anthropic Claude vision labeler.

Uses the ``anthropic`` SDK Messages API with a single user message
containing an ``image`` block (base64 source) and a ``text`` block. The
prompt instructs Claude to emit the same six-field JSON object every
provider returns; we parse the assistant's text reply.

Image bytes come exclusively from
:func:`pipeline.labeling.image_prep.prepare_image_for_labeling`. The
client never touches raw on-disk image bytes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from pipeline.labeling.image_prep import (
    PreparedImage,
    prepare_image_for_labeling,
)
from pipeline.providers import auth
from pipeline.providers._config import resolve_temperature
from pipeline.providers.base import (
    ClientConfig,
    LabelClient,
    LabelRequest,
    LabelResponse,
    ProviderError,
    ProviderRateLimitError,
    abstain_response,
    coerce_label_fields,
    parse_label_json,
    strip_image_bytes,
)
from pipeline.providers.pricing import compute_call_cost
from pipeline.providers.retries import retry_call

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a policy-graph image labeler. Use ONLY the supplied policy "
    "document to classify the image. Reply with a single JSON object "
    "carrying the six fields: label, l2_label, justification, confidence, "
    "difficulty, is_boundary. No prose, no markdown fences."
)

USER_INSTRUCTIONS = (
    "Classify the attached image against the policy below. Return only the "
    "six-field JSON object. label must be one of: gen_ai, not_gen_ai, "
    "abstain (cold-start) or violative, non_violative, abstain (warm-start). "
    "justification must be at least 10 characters and cite specific policy "
    "text. If evidence is insufficient, abstain."
)


@dataclass(frozen=True)
class AnthropicClientConfig(ClientConfig):
    """Anthropic-specific config."""

    api_key_env_var: str = auth.ANTHROPIC_API_KEY_VAR
    max_tokens: int = 2048
    thinking_budget_tokens: int | None = None


class AnthropicClient(LabelClient):
    """Vision-capable Claude client.

    SDK is imported lazily so this module can be imported in environments
    without ``anthropic`` installed. Tests inject a fake ``client``.
    """

    provider_id = "anthropic"

    def __init__(
        self,
        *,
        config: AnthropicClientConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(config=config)
        self.config: AnthropicClientConfig = config
        self._client = client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("anthropic SDK not installed") from exc
        api_key = auth.get_secret(self.config.api_key_env_var)
        self._client = Anthropic(api_key=api_key, timeout=self.config.request_timeout_s)
        return self._client

    def _build_messages(
        self,
        *,
        prepared: PreparedImage,
        policy_markdown: str,
    ) -> list[dict[str, Any]]:
        user_text = (
            f"{USER_INSTRUCTIONS}\n\n"
            f"[POLICY DOCUMENT]\n{policy_markdown}\n"
        )
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": prepared.mime_type,
                            "data": prepared.to_base64(),
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ]

    def _build_api_params(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "system": DEFAULT_SYSTEM_PROMPT,
            "messages": messages,
        }
        temperature = resolve_temperature(self.config.model_name)
        if self.config.thinking_budget_tokens is not None and self.config.thinking_budget_tokens > 0:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(self.config.thinking_budget_tokens),
            }
            # Anthropic requires temperature=1 when extended thinking is enabled.
            params["temperature"] = 1
        elif temperature is not None:
            params["temperature"] = temperature
        for k, v in self.config.extra_params.items():
            if k == "temperature":
                continue
            params.setdefault(k, v)
        return params

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        try:
            from anthropic import (  # type: ignore[import-not-found]
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:  # pragma: no cover
            APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = ()  # type: ignore[assignment]
        if isinstance(exc, ProviderRateLimitError):
            return True
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
        return False

    @staticmethod
    def _retry_after(exc: BaseException) -> float | None:
        if isinstance(exc, ProviderRateLimitError):
            return exc.retry_after_s
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers is not None:
                try:
                    val = headers.get("retry-after") or headers.get("Retry-After")
                except Exception:  # pragma: no cover
                    val = None
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, request: LabelRequest) -> LabelResponse:
        prepared = prepare_image_for_labeling(
            request.image_path,
            max_size=request.max_image_size,
            jpeg_quality=request.jpeg_quality,
        )
        messages = self._build_messages(
            prepared=prepared,
            policy_markdown=request.policy_markdown,
        )
        api_params = self._build_api_params(messages=messages)

        attempts_holder = {"n": 0}

        def _do_call() -> Any:
            attempts_holder["n"] += 1
            client = self._ensure_client()
            return client.messages.create(**api_params)

        start = time.monotonic()
        try:
            response = retry_call(
                _do_call,
                is_retryable=self._is_retryable,
                extract_retry_after=self._retry_after,
                label="anthropic.messages",
            )
        except BaseException as exc:  # noqa: BLE001
            elapsed = int((time.monotonic() - start) * 1000)
            return abstain_response(
                image_id=request.image_id,
                model_id=request.model_id,
                error=f"provider_error:{type(exc).__name__}",
                latency_ms=elapsed,
                attempts=attempts_holder["n"],
                prepared=prepared,
                raw_payload={
                    "request_model": api_params.get("model"),
                    "messages": strip_image_bytes(api_params.get("messages")),
                },
                justification=(
                    "Anthropic provider error; abstaining without label so "
                    "downstream consensus can ignore this vote."
                ),
            )

        elapsed = int((time.monotonic() - start) * 1000)
        text = self._extract_text(response)
        raw_payload = self._serialize_response(response, api_params)
        input_tokens, output_tokens = self._extract_usage_tokens(response)
        if input_tokens is None and output_tokens is None:
            logger.info("usage_unknown for %s", request.model_id)
        cost_usd = compute_call_cost(request.model_id, input_tokens, output_tokens, image_count=1)

        try:
            parsed = parse_label_json(text)
        except ValueError:
            return abstain_response(
                image_id=request.image_id,
                model_id=request.model_id,
                error="parse_failed",
                latency_ms=elapsed,
                attempts=attempts_holder["n"],
                prepared=prepared,
                raw_payload=raw_payload,
                justification=(
                    "Anthropic returned non-JSON content; abstaining to keep "
                    "the vote out of consensus until the prompt is fixed."
                ),
            )

        fields = coerce_label_fields(parsed)
        return LabelResponse(
            image_id=request.image_id,
            model_id=request.model_id,
            label=fields["label"],
            l2_label=fields["l2_label"],
            justification=fields["justification"] or "no justification provided",
            confidence=fields["confidence"],
            difficulty=fields["difficulty"],
            is_boundary=fields["is_boundary"],
            raw_provider_payload=raw_payload,
            error=None,
            latency_ms=elapsed,
            attempts=attempts_holder["n"],
            prepared_image_sha256=prepared.sha256,
            prepared_image_width=prepared.width,
            prepared_image_height=prepared.height,
            prepared_image_mime_type=prepared.mime_type,
            prepared_image_byte_size=prepared.byte_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            policy_citations=fields["policy_citations"],
            policy_quotes=fields["policy_quotes"],
            justification_too_long=fields["justification_too_long"],
        )

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate text blocks from a Claude messages response."""
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if not content:
            return ""
        parts: list[str] = []
        for block in content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "text":
                text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                if text:
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _serialize_response(response: Any, api_params: dict[str, Any]) -> dict[str, Any]:
        body: Any
        if hasattr(response, "model_dump"):
            try:
                body = response.model_dump()
            except Exception:  # pragma: no cover
                body = {"_repr": repr(response)}
        elif isinstance(response, dict):
            body = response
        else:
            body = {"_repr": repr(response)}
        return {
            "provider": "anthropic",
            "request_model": api_params.get("model"),
            "request_params": {
                k: v for k, v in api_params.items() if k not in {"messages", "system"}
            },
            "response": strip_image_bytes(body),
        }


__all__ = [
    "AnthropicClient",
    "AnthropicClientConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "USER_INSTRUCTIONS",
]
