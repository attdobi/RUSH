"""Offline smoke tests for ``pipeline.providers`` + ``pipeline.labeling``.

No network, no real LLM SDKs imported. Each provider client is exercised
through a recording fake transport that mimics the SDK's response shape.

Coverage targets (per docs/EXECUTION-PLAN-bulk-labeling-v1.md and Pista's
image_prep correction):

* ``image_prep.prepare_image_for_labeling`` downsamples >1024 px images,
  re-encodes JPEG quality 85, computes a stable sha256, and exposes
  metadata without leaking bytes.
* ``base.parse_label_json`` tolerates fences/narrative; rejects nonsense.
* ``base.strip_image_bytes`` scrubs both ``data:image/...`` URLs and bare
  base64 blobs out of nested dicts/lists/tuples.
* ``base.coerce_label_fields`` clamps + defaults sanely.
* ``base.abstain_response`` carries prepared-image metadata even on the
  failure path.
* Each provider client (OpenAI, Anthropic, Gemini):
    - calls ``prepare_image_for_labeling`` (no original-byte access);
    - sends the prepared base64 JPEG (sha256 round-trips);
    - OpenAI passes ``detail: "high"`` explicitly;
    - populates the prepared_image_* audit fields on ``LabelResponse``;
    - scrubs image bytes out of the persisted ``raw_provider_payload``;
    - returns an abstain response (not raises) on provider failures.
* ``retries.retry_call`` retries retryables, gives up after max_attempts,
  honors ``Retry-After`` extraction.
* ``auth.get_secret`` raises ``MissingSecretError`` without leaking the
  secret value (and without logging anything).
* ``registry.build_client`` builds the correct subclass per ``model_id``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from pipeline.labeling.image_prep import (
    JPEG_QUALITY,
    MAX_LONGEST_EDGE_PX,
    OUTPUT_MIME_TYPE,
    PreparedImage,
    prepare_image,
    prepare_image_bytes,
    prepare_image_for_labeling,
)
from pipeline.providers import build_client
from pipeline.providers.anthropic_client import (
    AnthropicClient,
    AnthropicClientConfig,
)
from pipeline.providers.auth import (
    OPENAI_API_KEY_VAR,
    MissingSecretError,
    get_secret,
    reset_for_tests,
)
from pipeline.providers.base import (
    IMAGE_OMITTED_SENTINEL,
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
from pipeline.providers.gemini_client import GeminiClient, GeminiClientConfig
from pipeline.providers.openai_client import OpenAIClient, OpenAIClientConfig
from pipeline.providers.registry import MODEL_REGISTRY, build_client, list_models
from pipeline.providers.retries import RetryPolicy, retry_call


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def big_image_path(tmp_path: Path) -> Path:
    """A 2048x1536 PNG to verify downsampling actually shrinks the image."""
    img = Image.new("RGB", (2048, 1536), color=(200, 50, 50))
    # Add a vertical stripe so re-encoding has some content to chew on.
    for x in range(0, 2048, 128):
        for y in range(1536):
            img.putpixel((x, y), (10, 10, 10))
    out = tmp_path / "big.png"
    img.save(out, format="PNG")
    return out


@pytest.fixture()
def label_request(big_image_path: Path) -> LabelRequest:
    return LabelRequest(
        image_path=big_image_path,
        image_id="img-001",
        policy_markdown="# Cold-start GenAI policy\nViolation if AI-generated.",
        policy_graph_version="v0.1",
        prompt_version="2026-05-10",
        model_id="openai/gpt-5.5",
    )


VALID_JSON_RESPONSE = (
    '{"label":"gen_ai","l2_label":"GA.visual_artifacts.anatomy.hands",'
    '"justification":"Six fingers on left hand violates anatomy policy.",'
    '"confidence":0.81,"difficulty":"high","is_boundary":true}'
)


# --------------------------------------------------------------------------- #
# image_prep                                                                  #
# --------------------------------------------------------------------------- #


class TestImagePrep:
    def test_downsamples_to_max_longest_edge(self, big_image_path: Path) -> None:
        prepared = prepare_image_for_labeling(big_image_path)
        assert max(prepared.width, prepared.height) == MAX_LONGEST_EDGE_PX
        ratio_in = 2048 / 1536
        ratio_out = prepared.width / prepared.height
        assert abs(ratio_in - ratio_out) < 0.01

    def test_output_is_jpeg_quality_85(self, big_image_path: Path) -> None:
        prepared = prepare_image_for_labeling(big_image_path)
        assert prepared.mime_type == OUTPUT_MIME_TYPE == "image/jpeg"
        with Image.open(io.BytesIO(prepared.bytes_)) as img:
            assert img.format == "JPEG"
            assert img.size == (prepared.width, prepared.height)

    def test_sha256_matches_bytes(self, big_image_path: Path) -> None:
        prepared = prepare_image_for_labeling(big_image_path)
        assert prepared.sha256 == hashlib.sha256(prepared.bytes_).hexdigest()
        assert prepared.byte_size == len(prepared.bytes_)

    def test_deterministic_across_calls(self, big_image_path: Path) -> None:
        a = prepare_image_for_labeling(big_image_path)
        b = prepare_image_for_labeling(big_image_path)
        assert a.sha256 == b.sha256
        assert a.bytes_ == b.bytes_

    def test_metadata_omits_bytes(self, big_image_path: Path) -> None:
        meta = prepare_image_for_labeling(big_image_path).metadata()
        assert "bytes_" not in meta
        for required in ("sha256", "width", "height", "byte_size", "mime_type"):
            assert required in meta

    def test_data_url_round_trips(self, big_image_path: Path) -> None:
        prepared = prepare_image_for_labeling(big_image_path)
        url = prepared.to_data_url()
        assert url.startswith("data:image/jpeg;base64,")
        body = url.split(",", 1)[1]
        assert base64.b64decode(body) == prepared.bytes_

    def test_rejects_empty_bytes(self) -> None:
        with pytest.raises(ValueError):
            prepare_image_bytes(b"")

    def test_rejects_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            prepare_image_for_labeling(tmp_path / "nope.png")

    def test_jpeg_quality_constant_is_85(self) -> None:
        assert JPEG_QUALITY == 85

    def test_max_longest_edge_is_1024(self) -> None:
        assert MAX_LONGEST_EDGE_PX == 1024

    def test_legacy_prepare_image_alias_still_works(
        self, big_image_path: Path
    ) -> None:
        # ``prepare_image`` is the older name kept for back-compat with any
        # code that imported it before Pista's correction.
        prepared = prepare_image(big_image_path)
        assert isinstance(prepared, PreparedImage)
        assert prepared.mime_type == "image/jpeg"


# --------------------------------------------------------------------------- #
# base.parse_label_json / coerce_label_fields / strip_image_bytes             #
# --------------------------------------------------------------------------- #


class TestParseLabelJson:
    def test_happy_path(self) -> None:
        obj = parse_label_json(VALID_JSON_RESPONSE)
        assert obj["label"] == "gen_ai"
        assert obj["confidence"] == pytest.approx(0.81)

    def test_strips_code_fences(self) -> None:
        wrapped = f"```json\n{VALID_JSON_RESPONSE}\n```"
        assert parse_label_json(wrapped)["label"] == "gen_ai"

    def test_extracts_object_from_narrative(self) -> None:
        wrapped = f"Sure thing!\n{VALID_JSON_RESPONSE}\n--end--"
        assert parse_label_json(wrapped)["label"] == "gen_ai"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_label_json("")

    def test_rejects_no_object(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            parse_label_json("not json at all")


class TestCoerceLabelFields:
    def test_normalizes_case_and_clamps_confidence(self) -> None:
        out = coerce_label_fields(
            {
                "label": " GEN_AI ",
                "l2_label": " ga.something ",
                "justification": " short ",
                "confidence": "1.5",
                "difficulty": "EXTREME",
                "is_boundary": "true",
            }
        )
        assert out["label"] == "gen_ai"
        assert out["l2_label"] == "ga.something"
        assert out["confidence"] == 1.0
        assert out["difficulty"] == "medium"  # invalid -> default
        assert out["is_boundary"] is True

    def test_missing_fields_defaults(self) -> None:
        out = coerce_label_fields({})
        assert out["label"] == "abstain"
        assert out["confidence"] is None
        assert out["difficulty"] == "medium"
        assert out["is_boundary"] is False


class TestStripImageBytes:
    def test_replaces_data_url(self) -> None:
        scrubbed = strip_image_bytes(
            {"image_url": {"url": "data:image/jpeg;base64,YWJj"}}
        )
        assert scrubbed["image_url"]["url"] == IMAGE_OMITTED_SENTINEL

    def test_replaces_long_base64_blob(self) -> None:
        blob = base64.b64encode(b"x" * 2048).decode("ascii")
        scrubbed = strip_image_bytes({"source": {"data": blob}})
        assert scrubbed["source"]["data"] == IMAGE_OMITTED_SENTINEL

    def test_recurses_through_lists_and_tuples(self) -> None:
        url = "data:image/png;base64,QUJD"
        scrubbed = strip_image_bytes(["a", url, ("b", url)])
        assert scrubbed[1] == IMAGE_OMITTED_SENTINEL
        assert scrubbed[2][1] == IMAGE_OMITTED_SENTINEL

    def test_passthrough_for_short_strings(self) -> None:
        assert strip_image_bytes("hello") == "hello"
        assert strip_image_bytes({"k": 42, "ok": True}) == {"k": 42, "ok": True}


# --------------------------------------------------------------------------- #
# Auth                                                                        #
# --------------------------------------------------------------------------- #


class TestAuth:
    def test_missing_secret_raises_without_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        reset_for_tests()
        monkeypatch.delenv(OPENAI_API_KEY_VAR, raising=False)
        monkeypatch.setenv("RUSH_DOTENV_PATH", "/nonexistent/.env")
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(MissingSecretError) as exc_info:
                get_secret(OPENAI_API_KEY_VAR)
        msg = str(exc_info.value)
        assert OPENAI_API_KEY_VAR in msg
        # Crude leak check: no '=' -> no "VAR=value" leakage.
        assert "=" not in msg
        # And we must not have logged anything about secrets.
        assert not [r for r in caplog.records if "API_KEY" in r.getMessage()]

    def test_returns_value_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_for_tests()
        monkeypatch.setenv(OPENAI_API_KEY_VAR, "sk-test-not-a-real-key")
        monkeypatch.setenv("RUSH_DOTENV_PATH", "/nonexistent/.env")
        assert get_secret(OPENAI_API_KEY_VAR) == "sk-test-not-a-real-key"

    def test_returns_none_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_for_tests()
        monkeypatch.delenv(OPENAI_API_KEY_VAR, raising=False)
        monkeypatch.setenv("RUSH_DOTENV_PATH", "/nonexistent/.env")
        assert get_secret(OPENAI_API_KEY_VAR, required=False) is None


# --------------------------------------------------------------------------- #
# Retries                                                                     #
# --------------------------------------------------------------------------- #


class _Boom(Exception):
    pass


class _Final(Exception):
    pass


class TestRetries:
    def test_retries_until_success(self) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise _Boom()
            return "ok"

        result = retry_call(
            fn,
            is_retryable=lambda e: isinstance(e, _Boom),
            policy=RetryPolicy(max_attempts=5, base_delay_s=0.01, jitter_s=0),
            sleep=sleeps.append,
        )
        assert result == "ok"
        assert calls["n"] == 3
        assert len(sleeps) == 2

    def test_non_retryable_raises_immediately(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1
            raise _Final()

        with pytest.raises(_Final):
            retry_call(
                fn,
                is_retryable=lambda e: isinstance(e, _Boom),
                policy=RetryPolicy(max_attempts=4, base_delay_s=0.01, jitter_s=0),
                sleep=lambda _s: None,
            )
        assert calls["n"] == 1

    def test_gives_up_after_max_attempts(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1
            raise _Boom()

        with pytest.raises(_Boom):
            retry_call(
                fn,
                is_retryable=lambda e: isinstance(e, _Boom),
                policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, jitter_s=0),
                sleep=lambda _s: None,
            )
        assert calls["n"] == 3

    def test_honors_retry_after(self) -> None:
        sleeps: list[float] = []
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Boom()
            return "ok"

        retry_call(
            fn,
            is_retryable=lambda e: True,
            extract_retry_after=lambda e: 7.5,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, jitter_s=0),
            sleep=sleeps.append,
        )
        assert sleeps == [7.5]


# --------------------------------------------------------------------------- #
# Provider client fakes                                                       #
# --------------------------------------------------------------------------- #


def _fake_openai_response(text: str = VALID_JSON_RESPONSE) -> SimpleNamespace:
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(finish_reason="stop", message=msg)
    return SimpleNamespace(
        id="resp_123",
        model="gpt-5.5",
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=200, completion_tokens=80, total_tokens=280
        ),
        # No model_dump on purpose — test the repr-fallback path.
    )


class _RecordingOpenAIClient:
    """Stand-in for ``openai.OpenAI`` chat.completions.create."""

    def __init__(
        self,
        response: Any,
        *,
        raise_first_n: int = 0,
        exc_factory: Any = None,
    ) -> None:
        self._response = response
        self._raise_first_n = raise_first_n
        self._exc_factory = exc_factory or (lambda: _FakeOpenAIRateLimit())
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                if outer._raise_first_n > 0:
                    outer._raise_first_n -= 1
                    raise outer._exc_factory()
                return outer._response

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()


class _FakeOpenAIRateLimit(Exception):
    """Mimics openai.RateLimitError by class name + status_code+headers."""

    status_code = 429

    def __init__(self) -> None:
        super().__init__("rate limited")
        self.response = SimpleNamespace(headers={"retry-after": "0.01"})


_FakeOpenAIRateLimit.__name__ = "RateLimitError"


def _fake_anthropic_response(text: str = VALID_JSON_RESPONSE) -> SimpleNamespace:
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        id="msg_123",
        model="claude-opus-4-6",
        stop_reason="end_turn",
        content=[block],
        usage=SimpleNamespace(input_tokens=300, output_tokens=70),
    )


class _RecordingAnthropicClient:
    def __init__(self, response: Any, *, raise_first_n: int = 0) -> None:
        self._response = response
        self._raise_first_n = raise_first_n
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                if outer._raise_first_n > 0:
                    outer._raise_first_n -= 1
                    raise RuntimeError("transient anthropic blip")
                return outer._response

        self.messages = _Messages()


def _fake_gemini_response(text: str = VALID_JSON_RESPONSE) -> SimpleNamespace:
    part = SimpleNamespace(text=text)
    cand = SimpleNamespace(
        content=SimpleNamespace(parts=[part]),
        finish_reason="STOP",
    )
    return SimpleNamespace(
        text=text,
        candidates=[cand],
        usage_metadata=SimpleNamespace(
            prompt_token_count=250,
            candidates_token_count=70,
            total_token_count=320,
        ),
    )


class _RecordingGeminiClient:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> Any:
                outer.calls.append(kwargs)
                return outer._response

        self.models = _Models()


# --------------------------------------------------------------------------- #
# OpenAI client                                                               #
# --------------------------------------------------------------------------- #


class TestOpenAIClient:
    def test_builds_messages_with_high_detail_image(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingOpenAIClient(_fake_openai_response())
        config = OpenAIClientConfig(
            model_name="gpt-5.5",
            reasoning_effort="xhigh",
            max_completion_tokens=1024,
        )
        client = OpenAIClient(config=config, client=fake)
        resp = client.label(label_request)

        assert isinstance(resp, LabelResponse)
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256
        assert resp.prepared_image_mime_type == "image/jpeg"
        assert resp.prepared_image_byte_size > 0
        assert max(resp.prepared_image_width, resp.prepared_image_height) <= MAX_LONGEST_EDGE_PX
        assert resp.attempts == 1
        assert resp.latency_ms >= 0

        # Wire shape assertions.
        assert len(fake.calls) == 1
        kwargs = fake.calls[0]
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["reasoning"] == {"effort": "xhigh"}
        assert "reasoning_effort" not in kwargs
        # GPT-5 family: no temperature in the standard shape.
        assert "temperature" not in kwargs
        assert kwargs["max_completion_tokens"] == 1024

        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        user = messages[1]
        assert user["role"] == "user"
        # First content block: text. Second: image_url with detail=high.
        assert user["content"][0]["type"] == "text"
        assert "POLICY DOCUMENT" in user["content"][0]["text"]
        image_block = user["content"][1]
        assert image_block["type"] == "image_url"
        assert image_block["image_url"]["detail"] == "high"
        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        body = url.split(",", 1)[1]
        decoded = base64.b64decode(body)
        # Sha256 of the bytes that went on the wire matches the audit field.
        assert hashlib.sha256(decoded).hexdigest() == resp.prepared_image_sha256

    def test_invalid_reasoning_effort_raises_provider_error(self) -> None:
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.5", reasoning_effort="medium"),
            client=object(),
        )

        with pytest.raises(ProviderError):
            client._build_api_params(messages=[])

    def test_raw_payload_omits_image_messages(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingOpenAIClient(_fake_openai_response())
        config = OpenAIClientConfig(model_name="gpt-5.5")
        client = OpenAIClient(config=config, client=fake)
        resp = client.label(label_request)
        # The persisted payload deliberately drops messages (they're huge);
        # only the request_params dict is kept, plus the response body.
        rp = resp.raw_provider_payload
        assert rp["provider"] == "openai"
        assert "messages" not in rp.get("request_params", {})
        # to_dict() runs another defensive scrub.
        d = resp.to_dict()
        assert "data:image/jpeg;base64," not in repr(d)

    def test_retries_then_succeeds(self, label_request: LabelRequest) -> None:
        fake = _RecordingOpenAIClient(
            _fake_openai_response(),
            raise_first_n=2,
        )
        config = OpenAIClientConfig(model_name="gpt-5.5")
        client = OpenAIClient(config=config, client=fake)
        # Patch the module-level retry_call so it doesn't sleep for real.
        # The client uses default RetryPolicy; speed it up via monkeypatch
        # of time.sleep at the retries module level.
        import pipeline.providers.retries as retries_mod

        orig_sleep = retries_mod.time.sleep
        retries_mod.time.sleep = lambda _s: None  # type: ignore[assignment]
        try:
            resp = client.label(label_request)
        finally:
            retries_mod.time.sleep = orig_sleep  # type: ignore[assignment]

        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.attempts == 3

    def test_provider_failure_returns_abstain(
        self, label_request: LabelRequest
    ) -> None:
        # Always raise -> retries exhausted -> abstain (NOT raise).
        class _Permanent(Exception):
            status_code = 400

        fake = _RecordingOpenAIClient(
            _fake_openai_response(),
            raise_first_n=99,
            exc_factory=_Permanent,
        )
        config = OpenAIClientConfig(model_name="gpt-5.5")
        client = OpenAIClient(config=config, client=fake)
        resp = client.label(label_request)
        assert resp.label == "abstain"
        assert resp.error and "provider_error" in resp.error
        # Audit metadata MUST still be populated even on the failure path.
        assert resp.prepared_image_sha256
        assert resp.prepared_image_byte_size > 0


# --------------------------------------------------------------------------- #
# Anthropic client                                                            #
# --------------------------------------------------------------------------- #


class TestAnthropicClient:
    def test_builds_image_block_and_metadata(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingAnthropicClient(_fake_anthropic_response())
        config = AnthropicClientConfig(
            model_name="claude-opus-4-6",
            max_tokens=800,
        )
        client = AnthropicClient(config=config, client=fake)
        resp = client.label(label_request)
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256
        assert resp.prepared_image_mime_type == "image/jpeg"

        kw = fake.calls[0]
        assert kw["model"] == "claude-opus-4-6"
        assert kw["max_tokens"] == 800
        msg = kw["messages"][0]
        assert msg["role"] == "user"
        # Inspect the user content for an image block + decode the base64.
        image_blocks = [
            b
            for b in msg["content"]
            if isinstance(b, dict) and b.get("type") == "image"
        ]
        assert image_blocks, "expected at least one image block"
        src = image_blocks[0]["source"]
        assert src["type"] == "base64"
        assert src["media_type"] == "image/jpeg"
        decoded = base64.b64decode(src["data"])
        assert hashlib.sha256(decoded).hexdigest() == resp.prepared_image_sha256

    def test_thinking_budget_enables_extended_thinking(self) -> None:
        client = AnthropicClient(
            config=AnthropicClientConfig(
                model_name="claude-opus-4-7",
                max_tokens=4096,
                thinking_budget_tokens=32000,
            ),
            client=object(),
        )

        params = client._build_api_params(messages=[])

        assert params["thinking"] == {"type": "enabled", "budget_tokens": 32000}
        assert params["temperature"] == 1

    def test_raw_payload_scrubs_image_data(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingAnthropicClient(_fake_anthropic_response())
        config = AnthropicClientConfig(model_name="claude-opus-4-6")
        client = AnthropicClient(config=config, client=fake)
        resp = client.label(label_request)
        # No raw base64 data URL or huge blob should survive into the
        # persisted payload (or its serialized form).
        d = resp.to_dict()
        assert "data:image/jpeg;base64," not in repr(d)
        # And bytes_/data fields, if present, are sanitized.
        text_repr = repr(d)
        # The prepared image's first base64 chars should NOT show up anywhere.
        # We can't easily extract them here, but check that the sentinel is
        # the only image-shaped value remaining.
        assert IMAGE_OMITTED_SENTINEL in text_repr or "data" not in text_repr

    def test_provider_failure_returns_abstain(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingAnthropicClient(
            _fake_anthropic_response(),
            raise_first_n=99,
        )
        config = AnthropicClientConfig(model_name="claude-opus-4-6")
        client = AnthropicClient(config=config, client=fake)
        # Speed up retries.
        import pipeline.providers.retries as retries_mod

        orig_sleep = retries_mod.time.sleep
        retries_mod.time.sleep = lambda _s: None  # type: ignore[assignment]
        try:
            resp = client.label(label_request)
        finally:
            retries_mod.time.sleep = orig_sleep  # type: ignore[assignment]
        assert resp.label == "abstain"
        assert resp.prepared_image_sha256


# --------------------------------------------------------------------------- #
# Gemini client                                                               #
# --------------------------------------------------------------------------- #


class TestGeminiClient:
    def test_builds_inline_data_image_part(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingGeminiClient(_fake_gemini_response())
        config = GeminiClientConfig(model_name="gemini-3.1-pro-preview")
        client = GeminiClient(config=config, client=fake)
        resp = client.label(label_request)
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256
        assert resp.prepared_image_mime_type == "image/jpeg"

        kw = fake.calls[0]
        assert kw["model"] == "gemini-3.1-pro-preview"
        # Walk the contents and find the inline_data block.
        contents = kw["contents"]
        # Contents may be a list of dicts with parts.
        parts: list[dict[str, Any]] = []
        for entry in contents:
            if isinstance(entry, dict) and "parts" in entry:
                parts.extend(p for p in entry["parts"] if isinstance(p, dict))
        inline = [p for p in parts if "inline_data" in p]
        assert inline, "expected at least one inline_data part"
        idata = inline[0]["inline_data"]
        assert idata["mime_type"] == "image/jpeg"
        decoded = base64.b64decode(idata["data"])
        assert hashlib.sha256(decoded).hexdigest() == resp.prepared_image_sha256

    def test_thinking_budget_is_added_to_generation_config(self) -> None:
        client = GeminiClient(
            config=GeminiClientConfig(
                model_name="gemini-3.1-pro-preview",
                thinking_budget_tokens=-1,
            ),
            client=object(),
        )

        params = client._build_api_params(contents=[])

        assert params["config"]["thinking_config"] == {"thinking_budget": -1}

    def test_raw_payload_scrubs_inline_data(
        self, label_request: LabelRequest
    ) -> None:
        fake = _RecordingGeminiClient(_fake_gemini_response())
        config = GeminiClientConfig(model_name="gemini-3.1-pro-preview")
        client = GeminiClient(config=config, client=fake)
        resp = client.label(label_request)
        d = resp.to_dict()
        assert "data:image/jpeg;base64," not in repr(d)


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_all_specs_have_known_provider(self) -> None:
        for spec in MODEL_REGISTRY.values():
            assert spec.provider in {"openai", "anthropic", "gemini"}
            assert spec.phase in {1, 2}

    def test_list_models_filter_by_phase(self) -> None:
        cold = list_models(phase=1)
        assert cold and all(s.phase == 1 for s in cold)

    def test_build_client_returns_correct_subclass(self) -> None:
        # Pre-build fake transports so no SDK is touched.
        c1 = build_client(
            "openai/gpt-5.5",
            client=_RecordingOpenAIClient(_fake_openai_response()),
        )
        c2 = build_client(
            "anthropic/claude-opus-4-6",
            client=_RecordingAnthropicClient(_fake_anthropic_response()),
        )
        c3 = build_client(
            "google/gemini-3.1-pro-preview",
            client=_RecordingGeminiClient(_fake_gemini_response()),
        )
        assert isinstance(c1, OpenAIClient)
        assert isinstance(c2, AnthropicClient)
        assert isinstance(c3, GeminiClient)

    def test_registry_runtime_defaults_and_overrides(self) -> None:
        openai = build_client(
            "openai/gpt-5.5",
            client=_RecordingOpenAIClient(_fake_openai_response()),
        )
        assert isinstance(openai, OpenAIClient)
        assert openai.config.reasoning_effort == "xhigh"

        openai_override = build_client(
            "openai/gpt-5.5",
            client=_RecordingOpenAIClient(_fake_openai_response()),
            reasoning_effort="high",
        )
        assert isinstance(openai_override, OpenAIClient)
        assert openai_override.config.reasoning_effort == "high"

        anthropic = build_client(
            "anthropic/claude-opus-4-7",
            client=_RecordingAnthropicClient(_fake_anthropic_response()),
        )
        assert isinstance(anthropic, AnthropicClient)
        assert anthropic.config.thinking_budget_tokens == 32000

        gemini = build_client(
            "google/gemini-3.1-pro-preview",
            client=_RecordingGeminiClient(_fake_gemini_response()),
        )
        assert isinstance(gemini, GeminiClient)
        assert gemini.config.thinking_budget_tokens == -1

    @pytest.mark.parametrize(
        ("model_id", "vendor_model", "reasoning_effort", "max_completion_tokens"),
        [
            ("openai/gpt-5.5-xhigh", "gpt-5.5", "xhigh", 24000),
            ("openai/gpt-5.5-high", "gpt-5.5", "high", 24000),
            ("openai/gpt-5.4-mini-xhigh", "gpt-5.4-mini", "xhigh", 2000),
            ("openai/gpt-5.4-mini-high", "gpt-5.4-mini", "high", 2000),
        ],
    )
    def test_openai_reasoning_variants_build_client_config(
        self,
        model_id: str,
        vendor_model: str,
        reasoning_effort: str,
        max_completion_tokens: int,
    ) -> None:
        client = build_client(
            model_id,
            client=_RecordingOpenAIClient(_fake_openai_response()),
        )

        assert isinstance(client, OpenAIClient)
        assert client.config.model_name == vendor_model
        assert client.config.reasoning_effort == reasoning_effort
        assert client.config.max_completion_tokens == max_completion_tokens

    def test_build_client_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            build_client("nope/nope")


# --------------------------------------------------------------------------- #
# Cross-provider invariants (the X1↔X2 contract)                              #
# --------------------------------------------------------------------------- #


class TestProviderInvariants:
    @pytest.mark.parametrize(
        "make_client",
        [
            lambda: OpenAIClient(
                config=OpenAIClientConfig(model_name="gpt-5.5"),
                client=_RecordingOpenAIClient(_fake_openai_response()),
            ),
            lambda: AnthropicClient(
                config=AnthropicClientConfig(model_name="claude-opus-4-6"),
                client=_RecordingAnthropicClient(_fake_anthropic_response()),
            ),
            lambda: GeminiClient(
                config=GeminiClientConfig(model_name="gemini-3.1-pro-preview"),
                client=_RecordingGeminiClient(_fake_gemini_response()),
            ),
        ],
        ids=["openai", "anthropic", "gemini"],
    )
    def test_every_client_populates_prepared_image_metadata(
        self,
        make_client: Any,
        label_request: LabelRequest,
    ) -> None:
        client = make_client()
        resp = client.label(label_request)
        # The X2 contract: every label record has these audit fields set.
        assert resp.prepared_image_sha256, f"{type(client).__name__} missing sha256"
        assert resp.prepared_image_width > 0
        assert resp.prepared_image_height > 0
        assert resp.prepared_image_byte_size > 0
        assert resp.prepared_image_mime_type == "image/jpeg"
