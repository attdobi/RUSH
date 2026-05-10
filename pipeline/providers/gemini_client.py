"""Google Gemini multimodal labeler.

Uses the ``google-genai`` SDK's ``models.generate_content`` API. We feed
in two parts:

* an inline image part (``inline_data`` with the prepared JPEG bytes), and
* a text part (system prompt + user instructions + policy markdown).

We ask for ``application/json`` output via ``GenerationConfig`` so the
response text can be JSON-parsed deterministically.

Image bytes come exclusively from
:func:`pipeline.labeling.image_prep.prepare_image_for_labeling`.
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
from pipeline.providers.retries import retry_call

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a policy-graph image labeler. Use ONLY the supplied policy "
    "document to classify the image. Reply with a single JSON object "
    "carrying exactly six fields: label, l2_label, justification, "
    "confidence, difficulty, is_boundary. No prose, no markdown."
)

USER_INSTRUCTIONS = (
    "Classify the attached image against the policy. Return only the "
    "six-field JSON object. label must be one of: gen_ai, not_gen_ai, "
    "abstain (cold-start) or violative, non_violative, abstain (warm-start). "
    "justification must be at least 10 characters and cite specific policy "
    "text. If evidence is insufficient, abstain."
)


@dataclass(frozen=True)
class GeminiClientConfig(ClientConfig):
    """Gemini-specific config."""

    api_key_env_var: str = auth.GEMINI_API_KEY_VAR
    response_mime_type: str = "application/json"


class GeminiClient(LabelClient):
    """Vision-capable Gemini client.

    SDK is imported lazily so this module is safe to import without
    ``google-genai`` installed. Tests inject a fake ``client``.
    """

    provider_id = "gemini"

    def __init__(
        self,
        *,
        config: GeminiClientConfig,
        client: Any | None = None,
    ) -> None:
        super().__init__(config=config)
        self.config: GeminiClientConfig = config
        self._client = client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("google-genai SDK not installed") from exc
        api_key = auth.get_secret(self.config.api_key_env_var)
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_contents(
        self,
        *,
        prepared: PreparedImage,
        policy_markdown: str,
    ) -> list[dict[str, Any]]:
        text = (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"{USER_INSTRUCTIONS}\n\n"
            f"[POLICY DOCUMENT]\n{policy_markdown}\n"
        )
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": prepared.mime_type,
                            "data": prepared.to_base64(),
                        }
                    },
                    {"text": text},
                ],
            }
        ]

    def _build_api_params(
        self,
        *,
        contents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "response_mime_type": self.config.response_mime_type,
        }
        temperature = resolve_temperature(self.config.model_name)
        if temperature is not None:
            config["temperature"] = temperature
        for k, v in self.config.extra_params.items():
            if k == "temperature":
                continue
            config.setdefault(k, v)
        return {
            "model": self.config.model_name,
            "contents": contents,
            "config": config,
        }

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        if isinstance(exc, ProviderRateLimitError):
            return True
        # google-genai raises ``google.genai.errors.APIError`` with a code.
        code = (
            getattr(exc, "code", None)
            or getattr(exc, "status_code", None)
            or getattr(exc, "http_status", None)
        )
        if isinstance(code, int) and (code == 429 or code >= 500):
            return True
        # Connection / timeout style names.
        name = type(exc).__name__.lower()
        if any(t in name for t in ("timeout", "connection", "unavailable")):
            return True
        return False

    @staticmethod
    def _retry_after(exc: BaseException) -> float | None:
        if isinstance(exc, ProviderRateLimitError):
            return exc.retry_after_s
        # google-genai surfaces details on ``response`` / ``response_json``.
        for attr in ("retry_after", "retry_after_s"):
            val = getattr(exc, attr, None)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
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
        contents = self._build_contents(
            prepared=prepared,
            policy_markdown=request.policy_markdown,
        )
        api_params = self._build_api_params(contents=contents)

        attempts_holder = {"n": 0}

        def _do_call() -> Any:
            attempts_holder["n"] += 1
            client = self._ensure_client()
            return client.models.generate_content(**api_params)

        start = time.monotonic()
        try:
            response = retry_call(
                _do_call,
                is_retryable=self._is_retryable,
                extract_retry_after=self._retry_after,
                label="gemini.generate_content",
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
                    "contents": strip_image_bytes(api_params.get("contents")),
                },
                justification=(
                    "Gemini provider error; abstaining without label so "
                    "downstream consensus can ignore this vote."
                ),
            )

        elapsed = int((time.monotonic() - start) * 1000)
        text = self._extract_text(response)
        raw_payload = self._serialize_response(response, api_params)

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
                    "Gemini returned non-JSON content; abstaining to keep "
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
        )

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the assistant text from a Gemini generate_content response."""
        # Preferred: SDK exposes a top-level ``text`` accessor.
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text
        # Fallback: walk candidates -> content.parts[].text.
        candidates = getattr(response, "candidates", None) or (
            response.get("candidates") if isinstance(response, dict) else None
        )
        if not candidates:
            return ""
        first = candidates[0]
        content = getattr(first, "content", None) or (
            first.get("content") if isinstance(first, dict) else None
        )
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or (
            content.get("parts") if isinstance(content, dict) else None
        )
        if not parts:
            return ""
        out: list[str] = []
        for part in parts:
            ptext = getattr(part, "text", None) or (
                part.get("text") if isinstance(part, dict) else None
            )
            if ptext:
                out.append(ptext)
        return "".join(out)

    @staticmethod
    def _serialize_response(response: Any, api_params: dict[str, Any]) -> dict[str, Any]:
        body: Any
        if hasattr(response, "model_dump"):
            try:
                body = response.model_dump()
            except Exception:  # pragma: no cover
                body = {"_repr": repr(response)}
        elif hasattr(response, "to_json_dict"):
            try:
                body = response.to_json_dict()
            except Exception:  # pragma: no cover
                body = {"_repr": repr(response)}
        elif isinstance(response, dict):
            body = response
        else:
            body = {"_repr": repr(response)}
        return {
            "provider": "gemini",
            "request_model": api_params.get("model"),
            "request_params": {
                "config": api_params.get("config"),
            },
            "response": strip_image_bytes(body),
        }


__all__ = [
    "GeminiClient",
    "GeminiClientConfig",
    "DEFAULT_SYSTEM_PROMPT",
    "USER_INSTRUCTIONS",
]
