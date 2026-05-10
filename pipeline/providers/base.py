"""Provider-agnostic types for LLM labeling.

This module defines the in-process contracts that flow between the runner
(X2) and provider clients (X1). The data classes here are intentionally
plain dataclasses — no Pydantic — so they're cheap to construct in tight
loops and trivial to serialize for persistence.

Three things every concrete client owes us:

1. Subclass :class:`LabelClient` and implement :meth:`LabelClient.label`.
2. Route every image through
   :func:`pipeline.labeling.image_prep.prepare_image_for_labeling` and
   propagate the resulting metadata into the :class:`LabelResponse` so X2
   can persist it per vote.
3. Strip image bytes from any persisted ``raw_provider_payload`` (use
   :func:`strip_image_bytes`); we never want base64 blobs landing in JSONL.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

# Sentinel used in place of base64 image content in persisted payloads.
IMAGE_OMITTED_SENTINEL: Final[str] = "<image-bytes-omitted>"

# Heuristic: any string that looks like a long base64 blob. We don't want to
# false-positive on short identifiers, but we also don't want to miss image
# payloads that providers nest under arbitrary keys. 256 chars is comfortably
# below even the smallest 1024² JPEG (~30 KB base64).
_BASE64ISH_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9+/=_\-\s]{256,}$")
_DATA_URL_RE: Final[re.Pattern[str]] = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


class ProviderError(RuntimeError):
    """Permanent (non-retryable) provider failure.

    Raised by clients when they decide an error is terminal. Messages must
    NEVER include API keys or full base64 image bytes.
    """


class ProviderRateLimitError(ProviderError):
    """Retryable rate-limit / 429 error.

    The optional ``retry_after_s`` mirrors the provider's ``Retry-After``
    header in seconds; the retry helper honors it when present.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s: float | None = retry_after_s


@dataclass(frozen=True)
class LabelRequest:
    """Single (image, model, policy) labeling request.

    See ``docs/EXECUTION-PLAN-bulk-labeling-v1.md`` §3.1 for field semantics.
    """

    image_path: Path
    image_id: str
    policy_markdown: str
    policy_graph_version: str
    prompt_version: str
    model_id: str
    # Optional override for downsampler knobs — defaults match the spec.
    max_image_size: tuple[int, int] = (1024, 1024)
    jpeg_quality: int = 85


@dataclass
class LabelResponse:
    """Single labeling result.

    Mirrors ``docs/EXECUTION-PLAN-bulk-labeling-v1.md`` §3.2. Persisted by
    X2 after schema validation. ``raw_provider_payload`` MUST already have
    image bytes stripped via :func:`strip_image_bytes` before construction.
    """

    image_id: str
    model_id: str
    label: str
    l2_label: str
    justification: str
    confidence: float
    difficulty: str
    is_boundary: bool
    raw_provider_payload: dict[str, Any]
    error: str | None
    latency_ms: int
    attempts: int
    # Prepared-image audit fields — set by every provider client from the
    # PreparedImage returned by image_prep.prepare_image_for_labeling().
    prepared_image_sha256: str = ""
    prepared_image_width: int = 0
    prepared_image_height: int = 0
    prepared_image_mime_type: str = ""
    prepared_image_byte_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serializable shape for downstream persistence."""
        d = asdict(self)
        # ``raw_provider_payload`` should already be sanitized; do a final
        # defensive pass so we never leak image bytes even if a provider
        # forgets.
        d["raw_provider_payload"] = strip_image_bytes(d.get("raw_provider_payload", {}))
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_image_bytes(value: Any) -> Any:
    """Recursively replace base64-ish strings in ``value`` with a sentinel.

    Used to sanitize ``raw_provider_payload`` before persistence. Handles:

    * Top-level / nested ``data:image/...;base64,...`` URLs.
    * Bare base64 blobs nested under provider-specific keys (OpenAI's
      ``image_url.url``, Anthropic's ``source.data``, Gemini's
      ``inline_data.data``).
    * Arbitrary dicts/lists/tuples — recursion is structural, not key-based.

    Non-JSON-y values (numbers, bools, ``None``) pass through unchanged.
    """
    if isinstance(value, dict):
        return {k: strip_image_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_image_bytes(v) for v in value]
    if isinstance(value, tuple):
        return tuple(strip_image_bytes(v) for v in value)
    if isinstance(value, str):
        if _DATA_URL_RE.match(value):
            return IMAGE_OMITTED_SENTINEL
        # Bare base64 blob heuristic: long, alphabet-restricted string.
        if _BASE64ISH_RE.match(value):
            return IMAGE_OMITTED_SENTINEL
        return value
    return value


def parse_label_json(text: str) -> dict[str, Any]:
    """Best-effort parse of a six-field LLM label JSON object.

    Tolerates leading/trailing prose and ```json fences (some models add
    them despite instructions). Raises :class:`ValueError` if no JSON
    object can be located.
    """
    if not text:
        raise ValueError("empty response")

    stripped = text.strip()
    # Strip ```json fences if present.
    if stripped.startswith("```"):
        # Drop opening fence (with optional language tag) and closing fence.
        stripped = re.sub(r"^```[a-zA-Z0-9]*\s*", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()

    # Fast path: the whole thing is valid JSON.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Slow path: find the first balanced {...} block.
    depth = 0
    start = -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = stripped[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = -1  # keep scanning
                    continue
    raise ValueError("no JSON object found in response")


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def coerce_label_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalize a parsed label dict to the six canonical fields.

    Missing fields fall back to safe defaults (``label="abstain"``,
    ``confidence=0.0``, etc.). Callers should still validate against
    ``schemas/llm-output.schema.json`` before persistence.
    """
    label = str(parsed.get("label", "abstain")).strip().lower()
    l2_label = str(parsed.get("l2_label", "")).strip()
    justification = str(parsed.get("justification", "")).strip()
    confidence = max(0.0, min(1.0, _coerce_float(parsed.get("confidence"), default=0.0)))
    difficulty = str(parsed.get("difficulty", "medium")).strip().lower()
    if difficulty not in {"high", "medium", "low"}:
        difficulty = "medium"
    is_boundary = _coerce_bool(parsed.get("is_boundary"), default=False)
    return {
        "label": label,
        "l2_label": l2_label,
        "justification": justification,
        "confidence": confidence,
        "difficulty": difficulty,
        "is_boundary": is_boundary,
    }


def abstain_response(
    *,
    image_id: str,
    model_id: str,
    error: str,
    latency_ms: int,
    attempts: int,
    prepared: Any | None = None,
    raw_payload: dict[str, Any] | None = None,
    justification: str = "Provider call failed; no label produced.",
) -> LabelResponse:
    """Construct a safe abstain :class:`LabelResponse` for failure paths.

    ``prepared`` may be a :class:`PreparedImage`-like object exposing the
    audit fields (``sha256``, ``width``, ``height``, ``mime_type``,
    ``byte_size``). When ``None``, audit fields are left at their zero
    defaults.
    """
    payload = strip_image_bytes(raw_payload or {})
    sha = getattr(prepared, "sha256", "") or ""
    width = int(getattr(prepared, "width", 0) or 0)
    height = int(getattr(prepared, "height", 0) or 0)
    mime = getattr(prepared, "mime_type", "") or ""
    byte_size = int(getattr(prepared, "byte_size", 0) or 0)
    return LabelResponse(
        image_id=image_id,
        model_id=model_id,
        label="abstain",
        l2_label="",
        justification=justification,
        confidence=0.0,
        difficulty="high",
        is_boundary=False,
        raw_provider_payload=payload,
        error=error,
        latency_ms=latency_ms,
        attempts=attempts,
        prepared_image_sha256=sha,
        prepared_image_width=width,
        prepared_image_height=height,
        prepared_image_mime_type=mime,
        prepared_image_byte_size=byte_size,
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClientConfig:
    """Per-client configuration knobs surfaced to all subclasses."""

    model_name: str
    extra_params: dict[str, Any] = field(default_factory=dict)
    request_timeout_s: float = 120.0


class LabelClient(ABC):
    """Abstract base for every provider client.

    Concrete subclasses set ``provider_id`` and implement :meth:`label`.
    They must:

    * Call :func:`pipeline.labeling.image_prep.prepare_image_for_labeling`
      for every request and propagate the metadata into
      :class:`LabelResponse`.
    * Use :func:`pipeline.providers.retries.retry_call` for transient
      errors (429/5xx), with provider-appropriate ``is_retryable`` and
      ``extract_retry_after`` predicates.
    * Sanitize ``raw_provider_payload`` via :func:`strip_image_bytes`
      before constructing the response.
    """

    provider_id: str = "unknown"

    def __init__(self, *, config: ClientConfig) -> None:
        self.config = config

    @abstractmethod
    def label(self, request: LabelRequest) -> LabelResponse:
        """Run a single labeling request end-to-end."""


__all__ = [
    "IMAGE_OMITTED_SENTINEL",
    "ProviderError",
    "ProviderRateLimitError",
    "LabelRequest",
    "LabelResponse",
    "ClientConfig",
    "LabelClient",
    "strip_image_bytes",
    "parse_label_json",
    "coerce_label_fields",
    "abstain_response",
]
