"""End-to-end wiring tests: per-project ontology drives each label call.

Covers the X2 integration step that connects the ontology CONTRACT
(:mod:`pipeline.providers.ontology`) to the label PATH:

  * ``LabelRequest.area`` selects the ontology in every provider client.
  * GenAI area reproduces today's exact prompt/schema (regression).
  * MNIST area sends the MNIST system prompt + digit-enum schema.
  * ``coerce_label_fields`` validates ``is_boundary_between`` against the
    area's L1 classes (required-when-boundary for MNIST).
  * The runner derives the area from the policy version and threads it into
    every :class:`LabelRequest`.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from pipeline.providers.anthropic_client import (
    AnthropicClient,
    AnthropicClientConfig,
)
from pipeline.providers.base import LabelRequest, coerce_label_fields
from pipeline.providers.gemini_client import GeminiClient, GeminiClientConfig
from pipeline.providers.ontology import (
    GENAI_ONTOLOGY,
    MNIST_ONTOLOGY,
    get_ontology,
)
from pipeline.providers.openai_client import OpenAIClient, OpenAIClientConfig
from pipeline.providers._prompts import (
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_INSTRUCTIONS,
    LABELING_RESPONSE_SCHEMA,
)

from tests.test_providers_smoke import (
    _RecordingAnthropicClient,
    _RecordingGeminiClient,
    _RecordingOpenAIClient,
    _fake_anthropic_response,
    _fake_gemini_response,
    _fake_openai_response,
)

MNIST_JSON_RESPONSE = (
    '{"label":"7","l2_label":"MD.digit.7",'
    '"justification":"Single top bar with a downstroke; no loop, reads as 7.",'
    '"policy_citations":["MD.digit.7"],"policy_quotes":["a 7 has a top bar"],'
    '"confidence":0.88,"difficulty":"medium","is_boundary":true,'
    '"is_boundary_between":["1","7"]}'
)


@pytest.fixture()
def image_path(tmp_path: Path) -> Path:
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    out = tmp_path / "digit.png"
    img.save(out, format="PNG")
    return out


def _mnist_request(image_path: Path) -> LabelRequest:
    return LabelRequest(
        image_path=image_path,
        image_id="mnist-001",
        policy_markdown="# MNIST digit policy\nMD.digit.7 governs sevens.",
        policy_graph_version="MNIST_Digits.v1",
        prompt_version="2026-07-01",
        model_id="openai/gpt-5.5",
        area="MNIST_Digits",
    )


def _genai_request(image_path: Path) -> LabelRequest:
    return LabelRequest(
        image_path=image_path,
        image_id="genai-001",
        policy_markdown="# GenAI policy\nViolation if AI-generated.",
        policy_graph_version="v0.1",
        prompt_version="2026-05-10",
        model_id="openai/gpt-5.5",
        # area defaults to Generative_AI
    )


# --------------------------------------------------------------------------- #
# Contract sanity: GenAI ontology == historical constants (regression)        #
# --------------------------------------------------------------------------- #


class TestGenAIRegression:
    def test_genai_ontology_prompts_are_byte_identical(self) -> None:
        assert GENAI_ONTOLOGY.system_prompt == LABELING_SYSTEM_PROMPT
        assert GENAI_ONTOLOGY.user_instructions == LABELING_USER_INSTRUCTIONS

    def test_genai_schema_matches_historical_core(self) -> None:
        # GenAI schema is the historical schema plus an OPTIONAL boundary pair
        # property (not in `required`). Everything else must be untouched.
        gen = GENAI_ONTOLOGY.response_schema
        for key, val in LABELING_RESPONSE_SCHEMA["properties"].items():
            assert gen["properties"][key] == val
        assert "is_boundary_between" not in LABELING_RESPONSE_SCHEMA.get(
            "properties", {}
        )
        assert "is_boundary_between" in gen["properties"]
        assert "is_boundary_between" not in gen.get("required", [])

    def test_get_ontology_defaults_to_genai(self) -> None:
        assert get_ontology() is GENAI_ONTOLOGY
        assert get_ontology(None) is GENAI_ONTOLOGY
        assert get_ontology("Generative_AI") is GENAI_ONTOLOGY


# --------------------------------------------------------------------------- #
# OpenAI: area selects the ontology in the client path                        #
# --------------------------------------------------------------------------- #


class TestOpenAIOntologySelection:
    def test_genai_request_sends_unchanged_genai_prompt(
        self, image_path: Path
    ) -> None:
        fake = _RecordingOpenAIClient(_fake_openai_response())
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.5"), client=fake
        )
        client.label(_genai_request(image_path))
        messages = fake.calls[0]["messages"]
        assert messages[0]["content"] == LABELING_SYSTEM_PROMPT
        assert LABELING_USER_INSTRUCTIONS in messages[1]["content"][0]["text"]

    def test_mnist_request_sends_mnist_prompt(self, image_path: Path) -> None:
        fake = _RecordingOpenAIClient(_fake_openai_response(MNIST_JSON_RESPONSE))
        client = OpenAIClient(
            config=OpenAIClientConfig(model_name="gpt-5.5"), client=fake
        )
        resp = client.label(_mnist_request(image_path))
        messages = fake.calls[0]["messages"]
        assert messages[0]["content"] == MNIST_ONTOLOGY.system_prompt
        assert MNIST_ONTOLOGY.user_instructions in messages[1]["content"][0]["text"]
        # The digit label flows through and the boundary pair is validated.
        assert resp.label == "7"
        assert resp.is_boundary is True
        assert resp.is_boundary_between == ["1", "7"]


# --------------------------------------------------------------------------- #
# Anthropic: system prompt swaps by area                                      #
# --------------------------------------------------------------------------- #


class TestAnthropicOntologySelection:
    def test_genai_system_is_historical(self, image_path: Path) -> None:
        fake = _RecordingAnthropicClient(_fake_anthropic_response())
        client = AnthropicClient(
            config=AnthropicClientConfig(model_name="claude"), client=fake
        )
        client.label(_genai_request(image_path))
        assert fake.calls[0]["system"] == LABELING_SYSTEM_PROMPT

    def test_mnist_system_is_mnist(self, image_path: Path) -> None:
        fake = _RecordingAnthropicClient(
            _fake_anthropic_response(MNIST_JSON_RESPONSE)
        )
        client = AnthropicClient(
            config=AnthropicClientConfig(model_name="claude"), client=fake
        )
        resp = client.label(_mnist_request(image_path))
        assert fake.calls[0]["system"] == MNIST_ONTOLOGY.system_prompt
        assert resp.label == "7"
        assert resp.is_boundary_between == ["1", "7"]


# --------------------------------------------------------------------------- #
# Gemini: response_schema swaps by area (digit enum for MNIST)                #
# --------------------------------------------------------------------------- #


class TestGeminiOntologySelection:
    def test_genai_schema_is_historical(self, image_path: Path) -> None:
        fake = _RecordingGeminiClient(_fake_gemini_response())
        client = GeminiClient(
            config=GeminiClientConfig(model_name="gemini"), client=fake
        )
        client.label(_genai_request(image_path))
        schema = fake.calls[0]["config"]["response_schema"]
        assert schema == GENAI_ONTOLOGY.response_schema

    def test_mnist_schema_has_digit_enum(self, image_path: Path) -> None:
        fake = _RecordingGeminiClient(_fake_gemini_response(MNIST_JSON_RESPONSE))
        client = GeminiClient(
            config=GeminiClientConfig(model_name="gemini"), client=fake
        )
        resp = client.label(_mnist_request(image_path))
        schema = fake.calls[0]["config"]["response_schema"]
        assert schema["properties"]["label"]["enum"] == [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "abstain",
        ]
        # System text carries the MNIST prompt (Gemini folds system into text).
        text = fake.calls[0]["contents"][0]["parts"][1]["text"]
        assert MNIST_ONTOLOGY.system_prompt in text
        assert resp.label == "7"


# --------------------------------------------------------------------------- #
# coerce_label_fields: is_boundary_between validated against L1 classes        #
# --------------------------------------------------------------------------- #


class TestBoundaryCoercion:
    def test_mnist_valid_pair(self) -> None:
        fields = coerce_label_fields(
            {"label": "7", "is_boundary": True, "is_boundary_between": ["1", "7"]},
            MNIST_ONTOLOGY,
        )
        assert fields["is_boundary_between"] == ["1", "7"]
        assert fields["boundary_between_invalid"] is False

    def test_mnist_invalid_pair_out_of_l1(self) -> None:
        fields = coerce_label_fields(
            {"label": "7", "is_boundary": True, "is_boundary_between": ["7", "z"]},
            MNIST_ONTOLOGY,
        )
        assert fields["boundary_between_invalid"] is True

    def test_mnist_invalid_pair_wrong_count(self) -> None:
        fields = coerce_label_fields(
            {"label": "7", "is_boundary": True, "is_boundary_between": ["7"]},
            MNIST_ONTOLOGY,
        )
        assert fields["boundary_between_invalid"] is True

    def test_genai_boundary_not_required(self) -> None:
        # GenAI has require_boundary_between=False: never flags invalid.
        fields = coerce_label_fields(
            {"label": "gen_ai", "is_boundary": True, "is_boundary_between": []},
            GENAI_ONTOLOGY,
        )
        assert fields["boundary_between_invalid"] is False

    def test_not_boundary_clears_pair(self) -> None:
        fields = coerce_label_fields(
            {"label": "7", "is_boundary": False, "is_boundary_between": ["1", "7"]},
            MNIST_ONTOLOGY,
        )
        assert fields["is_boundary_between"] == []
        assert fields["boundary_between_invalid"] is False


# --------------------------------------------------------------------------- #
# Runner: area derived from policy version + threaded into LabelRequest        #
# --------------------------------------------------------------------------- #


class TestRunnerThreadsArea:
    def test_area_from_policy_version(self) -> None:
        from pipeline.web.demo_area import area_from_policy_version

        assert area_from_policy_version("MNIST_Digits.v1") == "MNIST_Digits"
        assert area_from_policy_version("Generative_AI.v2") == "Generative_AI"
        assert area_from_policy_version("v0.1") == "Generative_AI"
        assert area_from_policy_version(None) == "Generative_AI"

    def test_build_request_carries_area(self) -> None:
        from pipeline.manifest import SampleRecord
        from pipeline.runner import ModelSpec, _build_request

        # _build_request only reads sample.absolute_path (a derived property)
        # and model_spec.model_id; it never opens the image.
        sample = SampleRecord(
            sample_id="s1",
            repo_rel_path="data/s.png",
            split="train",
            sme_label_raw="ai_generated",
            sme_label="gen_ai",
            dataset="ds",
            sha256="",
            sampling_version="v1",
        )
        spec = ModelSpec(model_id="openai/gpt-5.5")
        req = _build_request(
            sample,
            spec,
            policy_markdown="# p",
            policy_graph_version="MNIST_Digits.v1",
            prompt_version="v1",
            area="MNIST_Digits",
        )
        assert req.area == "MNIST_Digits"
