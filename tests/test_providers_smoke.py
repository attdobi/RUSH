"""Offline smoke tests for the bulk-labeling provider slice (X1).

All tests are network-free. Provider SDK calls are intercepted by injecting
fake clients into the provider classes; the shared image_prep helper is
exercised on synthetic PIL images so we never depend on dataset files.

What we cover:

* image_prep downsamples to longest edge ≤ 1024, JPEG output, metadata sha
  / size / dims are populated and stable.
* Every provider routes through ``prepare_image_for_labeling`` (we
  monkeypatch the helper and assert it was called).
* OpenAI payloads include ``detail: "high"``, a ``data:`` image URL,
  ``response_format`` JSON, and ``reasoning_effort`` when configured.
* Anthropic payloads include a base64 image source block + the system
  prompt.
* Gemini payloads include an ``inline_data`` part and a JSON response
  mime-type.
* ``raw_provider_payload`` has image bytes stripped before persistence.
* Auth: missing env vars raise ``MissingSecretError`` with the var name
  only; ``mask`` (via ``get_secret``) never echoes values.
* Retry helper retries on a 429-style exception, honors ``Retry-After``,
  and gives up after ``max_attempts``.
* All five registry models are present and ``build_client`` returns the
  expected provider for each.
"""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.labeling import image_prep  # noqa: E402
from pipeline.labeling.image_prep import (  # noqa: E402
    PreparedImage,
    prepare_image_for_labeling,
)
from pipeline.providers import auth, registry, retries  # noqa: E402
from pipeline.providers.anthropic_client import (  # noqa: E402
    AnthropicClient,
    AnthropicClientConfig,
)
from pipeline.providers.base import (  # noqa: E402
    IMAGE_OMITTED_SENTINEL,
    LabelRequest,
    LabelResponse,
    ProviderRateLimitError,
    coerce_label_fields,
    parse_label_json,
    strip_image_bytes,
)
from pipeline.providers.gemini_client import (  # noqa: E402
    GeminiClient,
    GeminiClientConfig,
)
from pipeline.providers.openai_client import (  # noqa: E402
    OpenAIClient,
    OpenAIClientConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_png(path: Path, *, size: tuple[int, int]) -> Path:
    """Generate a synthetic deterministic PNG and write it to ``path``."""
    img = Image.new("RGB", size, color=(34, 139, 230))
    # Draw a simple gradient so JPEG re-encoding produces non-trivial bytes.
    pixels = img.load()
    w, h = size
    for y in range(h):
        for x in range(0, w, 16):
            pixels[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256)
    img.save(path, format="PNG")
    return path


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """A 2048×1024 PNG that should be downsampled to 1024×512 by image_prep."""
    p = tmp_path / "sample.png"
    return _write_png(p, size=(2048, 1024))


@pytest.fixture()
def small_image(tmp_path: Path) -> Path:
    """A 400×300 PNG — already under the cap, dims must be preserved."""
    p = tmp_path / "small.png"
    return _write_png(p, size=(400, 300))


@pytest.fixture()
def label_request(sample_image: Path) -> LabelRequest:
    return LabelRequest(
        image_path=sample_image,
        image_id="sample-001",
        policy_markdown="# Policy\n\nDo not violate.",
        policy_graph_version="v0.1",
        prompt_version="2026-05-10",
        model_id="openai/gpt-5.5",
    )


# ---------------------------------------------------------------------------
# image_prep tests
# ---------------------------------------------------------------------------


class TestImagePrep:
    def test_downsample_longest_edge_capped_to_1024(self, sample_image: Path) -> None:
        prepared = prepare_image_for_labeling(sample_image)
        assert max(prepared.width, prepared.height) <= 1024
        # 2048x1024 source -> longest edge 1024 means dims (1024, 512).
        assert prepared.width == 1024
        assert prepared.height == 512

    def test_output_is_jpeg(self, sample_image: Path) -> None:
        prepared = prepare_image_for_labeling(sample_image)
        assert prepared.mime_type == "image/jpeg"
        # JPEGs start with the SOI marker FFD8.
        assert prepared.bytes_[:2] == b"\xff\xd8"
        with Image.open(io.BytesIO(prepared.bytes_)) as decoded:
            assert decoded.format == "JPEG"

    def test_metadata_fields_populated(self, sample_image: Path) -> None:
        prepared = prepare_image_for_labeling(sample_image)
        assert len(prepared.sha256) == 64
        assert prepared.byte_size == len(prepared.bytes_)
        assert prepared.byte_size > 0
        meta = prepared.metadata()
        assert meta["sha256"] == prepared.sha256
        assert "bytes_" not in meta  # never leak raw bytes in metadata.
        assert meta["mime_type"] == "image/jpeg"

    def test_small_image_not_upscaled(self, small_image: Path) -> None:
        prepared = prepare_image_for_labeling(small_image)
        # Thumbnail never upscales.
        assert prepared.width == 400
        assert prepared.height == 300

    def test_deterministic_sha(self, sample_image: Path) -> None:
        a = prepare_image_for_labeling(sample_image)
        b = prepare_image_for_labeling(sample_image)
        assert a.sha256 == b.sha256
        assert a.byte_size == b.byte_size

    def test_data_url_roundtrip(self, sample_image: Path) -> None:
        prepared = prepare_image_for_labeling(sample_image)
        url = prepared.to_data_url()
        assert url.startswith("data:image/jpeg;base64,")
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == prepared.bytes_


# ---------------------------------------------------------------------------
# Fake SDK clients
# ---------------------------------------------------------------------------


class _FakeOpenAIResponse:
    """Stand-in for the openai SDK chat completions response object."""

    def __init__(self, text: str) -> None:
        @dataclass
        class _Msg:
            content: str

        @dataclass
        class _Choice:
            message: _Msg
            finish_reason: str = "stop"

        self.choices = [_Choice(message=_Msg(content=text))]
        self.id = "chatcmpl-fake"
        self.model = "gpt-5.5"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": self.choices[0].message.content},
                }
            ],
        }


class _FakeOpenAIClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None

        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> _FakeOpenAIResponse:
                outer.last_kwargs = kwargs
                return _FakeOpenAIResponse(outer._text)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        @dataclass
        class _Block:
            type: str
            text: str

        self.content = [_Block(type="text", text=text)]
        self.id = "msg-fake"
        self.model = "claude-opus-4-6"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "content": [{"type": "text", "text": self.content[0].text}],
        }


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None

        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
                outer.last_kwargs = kwargs
                return _FakeAnthropicResponse(outer._text)

        self.messages = _Messages()


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def model_dump(self) -> dict[str, Any]:
        return {"text": self.text}


class _FakeGeminiClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None

        outer = self

        class _Models:
            def generate_content(self, **kwargs: Any) -> _FakeGeminiResponse:
                outer.last_kwargs = kwargs
                return _FakeGeminiResponse(outer._text)

        self.models = _Models()


VALID_LABEL_JSON = (
    '{"label":"gen_ai","l2_label":"GA.root","justification":"Clear synthetic '
    'gradient artifacts per policy.","confidence":0.81,"difficulty":"medium",'
    '"is_boundary":false}'
)


# ---------------------------------------------------------------------------
# OpenAI client tests
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    def test_label_uses_image_prep_helper(
        self, label_request: LabelRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, int] = {"n": 0}
        real = image_prep.prepare_image_for_labeling

        def _spy(*args: Any, **kwargs: Any) -> PreparedImage:
            called["n"] += 1
            return real(*args, **kwargs)

        # Provider client imports the symbol by name; patch in its module too.
        monkeypatch.setattr(image_prep, "prepare_image_for_labeling", _spy)
        from pipeline.providers import openai_client as oc

        monkeypatch.setattr(oc, "prepare_image_for_labeling", _spy)

        fake = _FakeOpenAIClient(VALID_LABEL_JSON)
        client = OpenAIClient(
            config=OpenAIClientConfig(
                model_name="gpt-5.5",
                reasoning_effort="high",
            ),
            client=fake,
        )
        resp = client.label(label_request)
        assert called["n"] == 1, "image_prep helper must be called exactly once"
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256 and len(resp.prepared_image_sha256) == 64
        assert resp.prepared_image_mime_type == "image/jpeg"
        assert resp.prepared_image_byte_size > 0

    def test_payload_includes_detail_high_and_data_url(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeOpenAIClient(VALID_LABEL_JSON)
        client = OpenAIClient(
            config=OpenAIClientConfig(
                model_name="gpt-5.5",
                reasoning_effort="high",
            ),
            client=fake,
        )
        client.label(label_request)
        kwargs = fake.last_kwargs
        assert kwargs is not None
        # Response format JSON object.
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert kwargs.get("reasoning_effort") == "high"
        # Find image_url block and check detail high + data URL.
        messages = kwargs["messages"]
        user = next(m for m in messages if m["role"] == "user")
        image_blocks = [b for b in user["content"] if b.get("type") == "image_url"]
        assert image_blocks, "expected an image_url block"
        block = image_blocks[0]
        assert block["image_url"]["detail"] == "high"
        assert block["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_raw_payload_has_image_bytes_stripped(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeOpenAIClient(VALID_LABEL_JSON)
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.5"),
            client=fake,
        )
        resp = client.label(label_request)
        # ``raw_provider_payload`` should never contain a data: URL.
        flat = repr(resp.raw_provider_payload)
        assert "data:image/jpeg;base64," not in flat
        # And running the dict through to_dict (which re-sanitizes) must be safe.
        d = resp.to_dict()
        assert "data:image/jpeg;base64," not in repr(d)

    def test_parse_failure_returns_safe_abstain(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeOpenAIClient("Sorry, I cannot help with that.")
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.5"),
            client=fake,
        )
        resp = client.label(label_request)
        assert resp.error == "parse_failed"
        assert resp.label == "abstain"
        assert len(resp.justification) >= 10

    def test_no_reasoning_effort_when_unset(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeOpenAIClient(VALID_LABEL_JSON)
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.4-mini"),
            client=fake,
        )
        client.label(label_request)
        assert fake.last_kwargs is not None
        assert "reasoning_effort" not in fake.last_kwargs


# ---------------------------------------------------------------------------
# Anthropic client tests
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    def test_label_uses_image_prep_helper(
        self, label_request: LabelRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, int] = {"n": 0}
        real = image_prep.prepare_image_for_labeling

        def _spy(*args: Any, **kwargs: Any) -> PreparedImage:
            called["n"] += 1
            return real(*args, **kwargs)

        from pipeline.providers import anthropic_client as ac

        monkeypatch.setattr(ac, "prepare_image_for_labeling", _spy)

        fake = _FakeAnthropicClient(VALID_LABEL_JSON)
        client = AnthropicClient(
            config=AnthropicClientConfig(model_name="claude-opus-4-6"),
            client=fake,
        )
        resp = client.label(label_request)
        assert called["n"] == 1
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256 and resp.prepared_image_byte_size > 0

    def test_payload_includes_image_block_and_system(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeAnthropicClient(VALID_LABEL_JSON)
        client = AnthropicClient(
            config=AnthropicClientConfig(model_name="claude-opus-4-6"),
            client=fake,
        )
        client.label(label_request)
        kwargs = fake.last_kwargs
        assert kwargs is not None
        assert "system" in kwargs and kwargs["system"]
        msgs = kwargs["messages"]
        blocks = msgs[0]["content"]
        image_blocks = [b for b in blocks if b.get("type") == "image"]
        assert image_blocks
        src = image_blocks[0]["source"]
        assert src["type"] == "base64"
        assert src["media_type"] == "image/jpeg"
        assert src["data"]  # non-empty base64 string

    def test_raw_payload_has_image_bytes_stripped(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeAnthropicClient(VALID_LABEL_JSON)
        client = AnthropicClient(
            config=AnthropicClientConfig(model_name="claude-opus-4-6"),
            client=fake,
        )
        resp = client.label(label_request)
        d = resp.to_dict()
        # No long base64-ish payload should survive in the persisted dict.
        flat = repr(d["raw_provider_payload"])
        # The text block can carry the policy markdown, which is short, so we
        # check there's no big base64 blob lingering.
        assert "data:image/jpeg;base64," not in flat
        assert IMAGE_OMITTED_SENTINEL in flat or "messages" not in d["raw_provider_payload"].get("request_params", {})


# ---------------------------------------------------------------------------
# Gemini client tests
# ---------------------------------------------------------------------------


class TestGeminiClient:
    def test_label_uses_image_prep_helper(
        self, label_request: LabelRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: dict[str, int] = {"n": 0}
        real = image_prep.prepare_image_for_labeling

        def _spy(*args: Any, **kwargs: Any) -> PreparedImage:
            called["n"] += 1
            return real(*args, **kwargs)

        from pipeline.providers import gemini_client as gc

        monkeypatch.setattr(gc, "prepare_image_for_labeling", _spy)

        fake = _FakeGeminiClient(VALID_LABEL_JSON)
        client = GeminiClient(
            config=GeminiClientConfig(model_name="gemini-3.1-pro-preview"),
            client=fake,
        )
        resp = client.label(label_request)
        assert called["n"] == 1
        assert resp.error is None
        assert resp.label == "gen_ai"
        assert resp.prepared_image_sha256 and resp.prepared_image_byte_size > 0

    def test_payload_includes_inline_data_and_json_mime(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeGeminiClient(VALID_LABEL_JSON)
        client = GeminiClient(
            config=GeminiClientConfig(model_name="gemini-3.1-pro-preview"),
            client=fake,
        )
        client.label(label_request)
        kwargs = fake.last_kwargs
        assert kwargs is not None
        cfg = kwargs["config"]
        assert cfg.get("response_mime_type") == "application/json"
        contents = kwargs["contents"]
        parts = contents[0]["parts"]
        inline = [p for p in parts if "inline_data" in p]
        assert inline
        assert inline[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert inline[0]["inline_data"]["data"]

    def test_raw_payload_has_image_bytes_stripped(
        self, label_request: LabelRequest
    ) -> None:
        fake = _FakeGeminiClient(VALID_LABEL_JSON)
        client = GeminiClient(
            config=GeminiClientConfig(model_name="gemini-3.1-pro-preview"),
            client=fake,
        )
        resp = client.label(label_request)
        d = resp.to_dict()
        flat = repr(d["raw_provider_payload"])
        assert "data:image/jpeg;base64," not in flat


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_var_raises_with_var_name_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Don't load the project's real .env, which carries values.
        monkeypatch.setenv(auth.DOTENV_PATH_ENV_VAR, "/nonexistent/.env")
        auth.reset_for_tests()
        with pytest.raises(auth.MissingSecretError) as exc_info:
            auth.get_secret("OPENAI_API_KEY")
        msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in msg
        # Don't echo any candidate secret value substring.
        assert "sk-" not in msg
        assert "Bearer" not in msg

    def test_get_secret_returns_value_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-xyz")
        monkeypatch.setenv(auth.DOTENV_PATH_ENV_VAR, "/nonexistent/.env")
        auth.reset_for_tests()
        val = auth.get_secret("OPENAI_API_KEY")
        assert val == "sk-fake-xyz"

    def test_get_secret_optional_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(auth.DOTENV_PATH_ENV_VAR, "/nonexistent/.env")
        auth.reset_for_tests()
        assert auth.get_secret("ANTHROPIC_API_KEY", required=False) is None


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestRetries:
    def test_retries_then_succeeds(self) -> None:
        sleeps: list[float] = []
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ProviderRateLimitError("429", retry_after_s=0.0)
            return "ok"

        result = retries.retry_call(
            fn,
            is_retryable=lambda e: isinstance(e, ProviderRateLimitError),
            extract_retry_after=lambda e: getattr(e, "retry_after_s", None),
            sleep=sleeps.append,
            policy=retries.RetryPolicy(max_attempts=4, base_delay_s=0.01, jitter_s=0.0),
            label="test",
        )
        assert result == "ok"
        assert attempts["n"] == 3
        assert len(sleeps) == 2  # two retries before success

    def test_honors_retry_after(self) -> None:
        sleeps: list[float] = []
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ProviderRateLimitError("429", retry_after_s=2.5)
            return "ok"

        retries.retry_call(
            fn,
            is_retryable=lambda e: isinstance(e, ProviderRateLimitError),
            extract_retry_after=lambda e: getattr(e, "retry_after_s", None),
            sleep=sleeps.append,
            policy=retries.RetryPolicy(max_attempts=3, base_delay_s=0.01, jitter_s=0.0),
            label="test",
        )
        assert sleeps and abs(sleeps[0] - 2.5) < 0.01

    def test_gives_up_after_max_attempts(self) -> None:
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            raise ProviderRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            retries.retry_call(
                fn,
                is_retryable=lambda e: isinstance(e, ProviderRateLimitError),
                sleep=lambda _s: None,
                policy=retries.RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter_s=0.0),
                label="test",
            )
        assert attempts["n"] == 3

    def test_non_retryable_propagates_immediately(self) -> None:
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            raise ValueError("boom")

        with pytest.raises(ValueError):
            retries.retry_call(
                fn,
                is_retryable=lambda e: isinstance(e, ProviderRateLimitError),
                sleep=lambda _s: None,
                policy=retries.RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter_s=0.0),
                label="test",
            )
        assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_five_models_present(self) -> None:
        expected = {
            "openai/gpt-5.5",
            "google/gemini-3.1-pro-preview",
            "anthropic/claude-opus-4-6",
            "openai/gpt-5.4-mini",
            "google/gemini-3.1-flash-lite-preview",
        }
        assert expected == set(registry.MODEL_REGISTRY.keys())

    def test_phase_split(self) -> None:
        phase1 = {s.model_id for s in registry.list_models(phase=1)}
        phase2 = {s.model_id for s in registry.list_models(phase=2)}
        assert phase1 == {
            "openai/gpt-5.5",
            "google/gemini-3.1-pro-preview",
            "anthropic/claude-opus-4-6",
        }
        assert phase2 == {
            "openai/gpt-5.4-mini",
            "google/gemini-3.1-flash-lite-preview",
        }

    def test_build_client_returns_correct_provider(self) -> None:
        oc = registry.build_client("openai/gpt-5.5", client=_FakeOpenAIClient(VALID_LABEL_JSON))
        assert oc.provider_id == "openai"
        ac = registry.build_client(
            "anthropic/claude-opus-4-6", client=_FakeAnthropicClient(VALID_LABEL_JSON)
        )
        assert ac.provider_id == "anthropic"
        gc = registry.build_client(
            "google/gemini-3.1-pro-preview", client=_FakeGeminiClient(VALID_LABEL_JSON)
        )
        assert gc.provider_id == "gemini"

    def test_build_client_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            registry.build_client("openai/does-not-exist")

    def test_gpt55_carries_reasoning_high(self) -> None:
        spec = registry.MODEL_REGISTRY["openai/gpt-5.5"]
        assert spec.params.get("reasoning_effort") == "high"


# ---------------------------------------------------------------------------
# Helper / parser tests
# ---------------------------------------------------------------------------


class TestParserAndStripper:
    def test_parse_label_json_handles_fenced_block(self) -> None:
        text = "Sure!\n```json\n" + VALID_LABEL_JSON + "\n```\n"
        parsed = parse_label_json(text)
        assert parsed["label"] == "gen_ai"

    def test_parse_label_json_handles_prose_around_object(self) -> None:
        text = "Here is my answer: " + VALID_LABEL_JSON + " thanks."
        parsed = parse_label_json(text)
        assert parsed["confidence"] == 0.81

    def test_coerce_label_fields_clamps(self) -> None:
        out = coerce_label_fields(
            {
                "label": "GEN_AI",
                "l2_label": " GA.x ",
                "justification": "ok",
                "confidence": "1.5",
                "difficulty": "WAT",
                "is_boundary": "true",
            }
        )
        assert out["label"] == "gen_ai"
        assert out["l2_label"] == "GA.x"
        assert out["confidence"] == 1.0
        assert out["difficulty"] == "medium"
        assert out["is_boundary"] is True

    def test_strip_image_bytes_replaces_data_urls(self) -> None:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                        },
                    ],
                }
            ]
        }
        cleaned = strip_image_bytes(payload)
        url = cleaned["messages"][0]["content"][1]["image_url"]["url"]
        assert url == IMAGE_OMITTED_SENTINEL

    def test_strip_image_bytes_replaces_long_base64(self) -> None:
        big = "A" * 1024
        cleaned = strip_image_bytes({"data": big})
        assert cleaned["data"] == IMAGE_OMITTED_SENTINEL

    def test_strip_image_bytes_preserves_short_strings(self) -> None:
        cleaned = strip_image_bytes({"id": "msg-123", "n": 1})
        assert cleaned == {"id": "msg-123", "n": 1}
