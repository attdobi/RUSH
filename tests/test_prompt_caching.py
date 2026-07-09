"""Prompt-caching contract tests for the three hosted judge clients.

Caching is a PREFIX match on every provider, so the shared bytes
(instructions + policy document) must precede the per-image bytes:

* Anthropic — text block first WITH an ephemeral ``cache_control``
  breakpoint (nothing caches without one), image block second.
* Gemini — text part first (implicit caching engages automatically once
  the shared prefix leads), image part second.
* OpenAI — already text-first; a stable ``prompt_cache_key`` routes all
  calls of a pass to the same cache shard so concurrent judges hit.

Pricing must mirror each provider's usage semantics: Anthropic reports
cache reads/writes SEPARATELY from ``input_tokens``; OpenAI and Gemini
report cached tokens INSIDE the prompt-token count.
"""

from __future__ import annotations

from types import SimpleNamespace

from pipeline.labeling.image_prep import PreparedImage
from pipeline.providers.anthropic_client import AnthropicClient, AnthropicClientConfig
from pipeline.providers.base import LabelRequest
from pipeline.providers.gemini_client import GeminiClient, GeminiClientConfig
from pipeline.providers.openai_client import OpenAIClient, OpenAIClientConfig
from pipeline.providers.pricing import PRICING, compute_call_cost


def _prepared() -> PreparedImage:
    payload = b"\xff\xd8\xff\xdbfakejpegbytes"
    import hashlib

    return PreparedImage(
        bytes_=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        width=28,
        height=28,
        byte_size=len(payload),
        mime_type="image/jpeg",
        source_path=None,
    )


def _request(model_id: str) -> LabelRequest:
    return LabelRequest(
        image_path=__import__("pathlib").Path("/tmp/fake.png"),
        image_id="img-1",
        policy_markdown="# policy\nrule text",
        policy_graph_version="MNIST_Digits.v0.1",
        prompt_version="v0.1",
        model_id=model_id,
        area="MNIST_Digits",
    )


class TestAnthropicCaching:
    def test_text_block_first_with_ephemeral_breakpoint(self) -> None:
        client = AnthropicClient(config=AnthropicClientConfig(model_name="claude-haiku-4-5"))
        messages = client._build_messages(
            prepared=_prepared(), policy_markdown="# policy doc"
        )
        content = messages[0]["content"]
        # Shared prefix (instructions + policy) leads; per-image bytes last.
        assert content[0]["type"] == "text"
        assert "[POLICY DOCUMENT]" in content[0]["text"]
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert content[1]["type"] == "image"

    def test_cache_usage_extraction(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=500,
                output_tokens=200,
                cache_read_input_tokens=6100,
                cache_creation_input_tokens=0,
            )
        )
        read, write = AnthropicClient._extract_cache_tokens(response)
        assert (read, write) == (6100, 0)

    def test_cache_usage_extraction_dict_shape(self) -> None:
        response = {"usage": {"cache_read_input_tokens": 10, "cache_creation_input_tokens": 3}}
        assert AnthropicClient._extract_cache_tokens(response) == (10, 3)

    def test_cache_usage_absent(self) -> None:
        assert AnthropicClient._extract_cache_tokens(SimpleNamespace(usage=None)) == (None, None)


class TestGeminiCaching:
    def test_text_part_precedes_image(self) -> None:
        client = GeminiClient(config=GeminiClientConfig(model_name="gemini-3.1-flash-lite"))
        contents = client._build_contents(
            prepared=_prepared(), policy_markdown="# policy doc"
        )
        parts = contents[0]["parts"]
        assert "text" in parts[0]
        assert "[POLICY DOCUMENT]" in parts[0]["text"]
        assert "inline_data" in parts[1]

    def test_cached_token_extraction(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=7000,
                candidates_token_count=300,
                cached_content_token_count=6100,
            )
        )
        assert GeminiClient._extract_cached_tokens(response) == 6100

    def test_cached_token_absent(self) -> None:
        response = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=7000))
        assert GeminiClient._extract_cached_tokens(response) is None


class TestOpenAICaching:
    def test_prompt_cache_key_is_stable_per_policy_version(self) -> None:
        request = _request("openai/gpt-5.4-mini-low")
        key = OpenAIClient._prompt_cache_key(request)
        assert key == "rush:MNIST_Digits:MNIST_Digits.v0.1"
        # Same policy version -> same key (routing affinity); the key must
        # not depend on the image.
        assert key == OpenAIClient._prompt_cache_key(_request("openai/gpt-5.4-mini-low"))

    def test_prompt_cache_key_withheld_for_local_models(self) -> None:
        # LM Studio reuses OpenAIClient; strict OpenAI-compat servers can
        # reject unknown body fields, so no key is sent for local/*.
        assert OpenAIClient._prompt_cache_key(_request("local/qwen2.5-vl-7b")) is None

    def test_api_params_carry_cache_key_via_extra_body(self) -> None:
        client = OpenAIClient(config=OpenAIClientConfig(model_name="gpt-5.4-mini"))
        params = client._build_api_params(
            messages=[{"role": "user", "content": "x"}], prompt_cache_key="rush:a:v0.1"
        )
        assert params["extra_body"] == {"prompt_cache_key": "rush:a:v0.1"}
        # And omitted when not supplied (keeps dry-run/test transports clean).
        params2 = client._build_api_params(messages=[{"role": "user", "content": "x"}])
        assert "extra_body" not in params2

    def test_cached_token_extraction(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=7000,
                completion_tokens=300,
                prompt_tokens_details=SimpleNamespace(cached_tokens=6016),
            )
        )
        assert OpenAIClient._extract_cached_tokens(response) == 6016

    def test_cached_token_absent(self) -> None:
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7000))
        assert OpenAIClient._extract_cached_tokens(response) is None


class TestCachedPricing:
    """compute_call_cost must follow each provider's usage semantics."""

    def _rate(self, model_id: str) -> tuple[float, float]:
        pricing = PRICING[model_id]
        return pricing["input_per_mtok"], pricing["output_per_mtok"]

    def _anthropic_model(self) -> str:
        return next(m for m in PRICING if m.startswith("anthropic/"))

    def _openai_model(self) -> str:
        return next(m for m in PRICING if m.startswith("openai/"))

    def _google_model(self) -> str:
        return next(m for m in PRICING if m.startswith("google/"))

    def test_no_cache_fields_matches_legacy_cost(self) -> None:
        model = self._openai_model()
        assert compute_call_cost(model, 1000, 100) == compute_call_cost(
            model, 1000, 100, cached_input_tokens=None, cache_creation_input_tokens=None
        )

    def test_anthropic_reads_and_writes_are_additive(self) -> None:
        model = self._anthropic_model()
        in_rate, out_rate = self._rate(model)
        pricing = PRICING[model]
        cost = compute_call_cost(
            model, 500, 200, cached_input_tokens=6000, cache_creation_input_tokens=100
        )
        expected = (
            in_rate * 500 / 1e6
            + in_rate * 6000 * 0.10 / 1e6
            + in_rate * 100 * 1.25 / 1e6
            + out_rate * 200 / 1e6
            + pricing["image_per_image"]
        )
        assert cost is not None and abs(cost - expected) < 1e-12
        # Caching a 6.5k-token prefix must be cheaper than paying it raw.
        raw = compute_call_cost(model, 6600, 200)
        assert cost < raw

    def test_openai_cached_tokens_are_discounted_subset(self) -> None:
        model = self._openai_model()
        in_rate, out_rate = self._rate(model)
        pricing = PRICING[model]
        cost = compute_call_cost(model, 7000, 300, cached_input_tokens=6000)
        expected = (
            in_rate * (7000 - 6000) / 1e6
            + in_rate * 6000 * 0.50 / 1e6
            + out_rate * 300 / 1e6
            + pricing["image_per_image"]
        )
        assert cost is not None and abs(cost - expected) < 1e-12
        assert cost < compute_call_cost(model, 7000, 300)

    def test_google_cached_tokens_are_discounted_subset(self) -> None:
        model = self._google_model()
        in_rate, out_rate = self._rate(model)
        pricing = PRICING[model]
        cost = compute_call_cost(model, 7000, 300, cached_input_tokens=6000)
        expected = (
            in_rate * (7000 - 6000) / 1e6
            + in_rate * 6000 * 0.25 / 1e6
            + out_rate * 300 / 1e6
            + pricing["image_per_image"]
        )
        assert cost is not None and abs(cost - expected) < 1e-12

    def test_cached_subset_clamped_to_input(self) -> None:
        model = self._openai_model()
        # A buggy provider reporting cached > prompt must not go negative.
        cost = compute_call_cost(model, 100, 0, cached_input_tokens=500)
        in_rate, _ = self._rate(model)
        pricing = PRICING[model]
        expected = in_rate * 100 * 0.50 / 1e6 + pricing["image_per_image"]
        assert cost is not None and abs(cost - expected) < 1e-12
