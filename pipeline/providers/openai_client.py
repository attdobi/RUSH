"""OpenAI vision labeler.

Uses the ``openai`` SDK's chat-completions API in the GPT-5 reasoning
shape we know works (see ``/Users/sacsimoto/GitHub/d-ai-trader/config.py::
ask_openai`` for the reference pattern). We do NOT import or couple to
d-ai-trader; we only mirror the request shape:

* system message + user content array (text + ``image_url``);
* ``image_url`` carries a ``data:image/jpeg;base64,...`` URL with
  ``detail: "high"`` for the prepared image;
* ``response_format={"type": "json_object"}`` for deterministic parsing;
* ``reasoning_effort=...`` honored when the model accepts it.

Image bytes are produced exclusively by
:func:`pipeline.labeling.image_prep.prepare_image_for_labeling`. The
client never reads the original file directly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pipeline.labeling.image_prep import (
    PreparedImage,
    prepare_image_for_labeling,
)
from pipeline.providers import auth
from pipeline.providers._prompts import (
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_INSTRUCTIONS,
)
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

DEFAULT_SYSTEM_PROMPT = LABELING_SYSTEM_PROMPT
DEFAULT_USER_PROMPT = LABELING_USER_INSTRUCTIONS
USER_INSTRUCTIONS = DEFAULT_USER_PROMPT


@dataclass(frozen=True)
class OpenAIClientConfig(ClientConfig):
    """OpenAI-specific config — adds reasoning_effort knob."""

    reasoning_effort: str | None = None
    api_key_env_var: str = auth.OPENAI_API_KEY_VAR
    max_completion_tokens: int = 10000
    image_detail: str = "high"


class OpenAIClient(LabelClient):
    """Vision-capable OpenAI chat client.

    SDK is imported lazily so this module can be imported in environments
    without ``openai`` installed (e.g. CI jobs that only run schema checks).
    Tests inject a fake ``client`` via the ``client`` constructor argument.
    """

    provider_id = "openai"

    def __init__(
        self,
        *,
        config: OpenAIClientConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(config=config)
        self.config: OpenAIClientConfig = config  # narrow type for IDEs
        self._client = client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ProviderError("openai SDK not installed") from exc
        api_key = auth.get_secret(self.config.api_key_env_var)
        self._client = OpenAI(api_key=api_key, timeout=self.config.request_timeout_s)
        return self._client

    def _build_messages(
        self,
        *,
        prepared: PreparedImage,
        policy_markdown: str,
    ) -> list[dict[str, Any]]:
        """Build the system+user message array."""
        user_text = (
            f"{USER_INSTRUCTIONS}\n\n"
            f"[POLICY DOCUMENT]\n{policy_markdown}\n"
        )
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": prepared.to_data_url(),
                    # Hard requirement: pass detail="high" explicitly so the
                    # token budget aligns with the 1024² downsample.
                    "detail": self.config.image_detail,
                },
            },
        ]
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_api_params(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.config.reasoning_effort and self.config.reasoning_effort not in {"high", "xhigh"}:
            raise ProviderError(
                "OpenAI reasoning_effort must be one of: high, xhigh "
                f"(got {self.config.reasoning_effort!r})"
            )
        params: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "max_completion_tokens": self.config.max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort
        # GPT-5.5 reasoning models do not accept custom temperature; never
        # forward it for OpenAI while preserving reasoning behavior.
        for k, v in self.config.extra_params.items():
            if k == "temperature":
                continue
            params.setdefault(k, v)
        return params

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        # Lazy-import the SDK error types to keep this module light.
        try:
            from openai import (  # type: ignore[import-not-found]
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:  # pragma: no cover - SDK absent
            APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = ()  # type: ignore[assignment]

        if isinstance(exc, ProviderRateLimitError):
            return True
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
            return True
        # Generic 5xx fallback if the SDK surfaces an HTTP-style status.
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if isinstance(status, int) and status >= 500:
            return True
        if isinstance(status, int) and status == 429:
            return True
        return False

    @staticmethod
    def _retry_after(exc: BaseException) -> float | None:
        if isinstance(exc, ProviderRateLimitError):
            return exc.retry_after_s
        # OpenAI SDK exposes response headers via ``response`` on the error.
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
            return client.chat.completions.create(**api_params)

        start = time.monotonic()
        try:
            response = retry_call(
                _do_call,
                is_retryable=self._is_retryable,
                extract_retry_after=self._retry_after,
                label="openai.chat.completions",
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
                    "image_detail": self.config.image_detail,
                    # Always sanitize messages — they include the data URL.
                    "messages": strip_image_bytes(api_params.get("messages")),
                },
                justification=(
                    "OpenAI provider error; abstaining without label so "
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
                    "OpenAI returned non-JSON content; abstaining to keep the "
                    "vote out of consensus until the prompt is fixed."
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

    def batch_label(self, requests: list[LabelRequest]) -> list[LabelResponse]:
        """Label multiple images in one OpenAI multimodal request.

        The current transport uses Chat Completions, but the provider call is
        genuinely batched: one request contains the shared policy text plus all
        prepared image blocks, and the model returns one JSON ``items`` array.
        """
        if not requests:
            return []
        if len(requests) == 1:
            return [self.label(requests[0])]

        prepared_images = [
            prepare_image_for_labeling(
                request.image_path,
                max_size=request.max_image_size,
                jpeg_quality=request.jpeg_quality,
            )
            for request in requests
        ]
        policy_markdown = requests[0].policy_markdown
        user_text = (
            f"{USER_INSTRUCTIONS}\n\n"
            "BATCH MODE. Classify each image below independently against the "
            "same policy document. Return EXACTLY one JSON object with this "
            "shape: {\"items\":[{\"image_id\":\"...\", \"label\":..., "
            "\"l2_label\":..., \"justification\":..., \"policy_citations\":..., "
            "\"policy_quotes\":..., \"confidence\":..., \"difficulty\":..., "
            "\"is_boundary\":...}]}. Preserve the input order and echo each "
            "image_id exactly.\n\n"
            f"[POLICY DOCUMENT]\n{policy_markdown}\n"
        )
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for idx, (request, prepared) in enumerate(zip(requests, prepared_images, strict=True), start=1):
            user_content.append(
                {"type": "text", "text": f"IMAGE {idx} image_id={request.image_id}"}
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": prepared.to_data_url(),
                        "detail": self.config.image_detail,
                    },
                }
            )
        batch_system_prompt = DEFAULT_SYSTEM_PROMPT.replace(
            "classify a single image",
            "classify each supplied image independently",
        ).replace(
            "Return EXACTLY one JSON object. No prose, no markdown fences.",
            "Return EXACTLY one JSON object with an items array. No prose, no markdown fences.",
        )
        messages = [
            {"role": "system", "content": batch_system_prompt},
            {"role": "user", "content": user_content},
        ]
        api_params = self._build_api_params(messages=messages)

        attempts_holder = {"n": 0}

        def _do_call() -> Any:
            attempts_holder["n"] += 1
            client = self._ensure_client()
            return client.chat.completions.create(**api_params)

        start = time.monotonic()
        try:
            response = retry_call(
                _do_call,
                is_retryable=self._is_retryable,
                extract_retry_after=self._retry_after,
                label="openai.chat.completions.batch",
            )
        except BaseException as exc:  # noqa: BLE001
            elapsed = int((time.monotonic() - start) * 1000)
            return [
                abstain_response(
                    image_id=request.image_id,
                    model_id=request.model_id,
                    error=f"provider_error:{type(exc).__name__}",
                    latency_ms=elapsed,
                    attempts=attempts_holder["n"],
                    prepared=prepared,
                    raw_payload={
                        "request_model": api_params.get("model"),
                        "image_detail": self.config.image_detail,
                        "batch_size": len(requests),
                        "messages": strip_image_bytes(api_params.get("messages")),
                    },
                    justification=(
                        "OpenAI batch provider error; abstaining without label "
                        "so downstream consensus can ignore this vote."
                    ),
                )
                for request, prepared in zip(requests, prepared_images, strict=True)
            ]

        elapsed = int((time.monotonic() - start) * 1000)
        text = self._extract_text(response)
        raw_payload = self._serialize_response(response, api_params)
        input_tokens, output_tokens = self._extract_usage_tokens(response)
        if input_tokens is None and output_tokens is None:
            logger.info("usage_unknown for %s batch_size=%s", requests[0].model_id, len(requests))
        total_cost = compute_call_cost(
            requests[0].model_id,
            input_tokens,
            output_tokens,
            image_count=len(requests),
        )
        per_image_cost = None if total_cost is None else total_cost / len(requests)

        try:
            parsed = parse_label_json(text)
            items = parsed.get("items")
            if not isinstance(items, list):
                raise ValueError("missing items array")
        except (ValueError, json.JSONDecodeError):
            return [
                abstain_response(
                    image_id=request.image_id,
                    model_id=request.model_id,
                    error="parse_failed",
                    latency_ms=elapsed,
                    attempts=attempts_holder["n"],
                    prepared=prepared,
                    raw_payload=raw_payload,
                    justification=(
                        "OpenAI returned non-JSON batch content; abstaining "
                        "to keep the vote out of consensus until the prompt is fixed."
                    ),
                )
                for request, prepared in zip(requests, prepared_images, strict=True)
            ]

        by_id = {
            str(item.get("image_id")): item
            for item in items
            if isinstance(item, dict) and item.get("image_id") is not None
        }
        responses: list[LabelResponse] = []
        for idx, (request, prepared) in enumerate(zip(requests, prepared_images, strict=True)):
            item = by_id.get(request.image_id)
            if item is None and idx < len(items) and isinstance(items[idx], dict):
                item = items[idx]
            if item is None:
                item = {}
            fields = coerce_label_fields(item)
            responses.append(
                LabelResponse(
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
                    cost_usd=per_image_cost,
                    policy_citations=fields["policy_citations"],
                    policy_quotes=fields["policy_quotes"],
                    justification_too_long=fields["justification_too_long"],
                )
            )
        return responses

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

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant text out of a chat completions response."""
        try:
            choice = response.choices[0]
            message = choice.message
            content = message.content
        except AttributeError:
            # Dict-shaped response (some SDK versions / fakes).
            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
        if isinstance(content, list):
            # Newer SDKs may return a list of content parts.
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return content or ""

    @staticmethod
    def _serialize_response(response: Any, api_params: dict[str, Any]) -> dict[str, Any]:
        """Build a sanitized payload dict for persistence."""
        # Try the SDK's model_dump / dict conversion; fall back to repr.
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
            "provider": "openai",
            "request_model": api_params.get("model"),
            "request_params": {
                k: v for k, v in api_params.items() if k != "messages"
            },
            # ``messages`` are deliberately omitted from the persisted record;
            # the prompt template is reproducible from prompt_version + policy.
            "response": strip_image_bytes(body),
        }


__all__ = [
    "OpenAIClient",
    "OpenAIClientConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT",
    "USER_INSTRUCTIONS",
]
