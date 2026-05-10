from __future__ import annotations

import pytest

from pipeline.providers._config import LABELING_TEMPERATURE, resolve_temperature
from pipeline.providers.anthropic_client import AnthropicClient, AnthropicClientConfig
from pipeline.providers.gemini_client import GeminiClient, GeminiClientConfig
from pipeline.providers.openai_client import OpenAIClient, OpenAIClientConfig
from pipeline.persistence import RUN_MANIFEST_SCHEMA, validate_record
from pipeline.runner import ModelSpec, _initial_manifest


def test_resolve_temperature_defaults_to_labeling_temperature() -> None:
    assert resolve_temperature("claude-opus-4-6") == pytest.approx(LABELING_TEMPERATURE)


@pytest.mark.parametrize("model_id", ["gpt-5.5-vision", "gpt-5.5"])
def test_resolve_temperature_omits_gpt_5_5_reasoning_models(model_id: str) -> None:
    assert resolve_temperature(model_id) is None


def test_resolve_temperature_returns_override() -> None:
    assert resolve_temperature("gemini-3.1-pro-preview", override=0.33) == pytest.approx(0.33)


def test_openai_api_params_never_send_temperature() -> None:
    client = OpenAIClient(
        config=OpenAIClientConfig(
            model_name="gpt-5.5",
            reasoning_effort="high",
            extra_params={"temperature": 0.9, "seed": 123},
        ),
        client=object(),
    )

    params = client._build_api_params(messages=[])

    assert params["model"] == "gpt-5.5"
    assert params["reasoning_effort"] == "high"
    assert params["seed"] == 123
    assert "temperature" not in params


def test_anthropic_api_params_send_resolved_temperature() -> None:
    client = AnthropicClient(
        config=AnthropicClientConfig(model_name="claude-opus-4-6"),
        client=object(),
    )

    params = client._build_api_params(messages=[])

    assert params["temperature"] == pytest.approx(LABELING_TEMPERATURE)


def test_anthropic_api_params_omit_temperature_when_resolution_is_none() -> None:
    client = AnthropicClient(
        config=AnthropicClientConfig(model_name="gpt-5.5"),
        client=object(),
    )

    params = client._build_api_params(messages=[])

    assert "temperature" not in params


def test_gemini_api_params_send_resolved_temperature() -> None:
    client = GeminiClient(
        config=GeminiClientConfig(model_name="gemini-3.1-pro-preview"),
        client=object(),
    )

    params = client._build_api_params(contents=[])

    assert params["config"]["temperature"] == pytest.approx(LABELING_TEMPERATURE)


def test_gemini_api_params_omit_temperature_when_resolution_is_none() -> None:
    client = GeminiClient(
        config=GeminiClientConfig(model_name="gpt-5.5-vision"),
        client=object(),
    )

    params = client._build_api_params(contents=[])

    assert "temperature" not in params["config"]


def test_run_manifest_records_resolved_temperature_per_model() -> None:
    manifest = _initial_manifest(
        run_id="20260510T210000-abcdef12",
        started_at="2026-05-10T21:00:00Z",
        sample_manifest_rel="data/images/genai-classification/manifests/combined_labels.jsonl",
        sample_ids=["dev_golden_0001"],
        models=[
            ModelSpec(model_id="openai/gpt-5.5"),
            ModelSpec(model_id="anthropic/claude-opus-4-6"),
        ],
        policy_graph_version="v0.1",
        prompt_version="v0.1",
        sampling_version="genai-gold-sampling-v1",
        split="dev_golden",
        limit=1,
        concurrency=1,
        expected_calls=2,
        dry_run=True,
    )

    by_model = {m["model_id"]: m for m in manifest["models"]}
    assert by_model["openai/gpt-5.5"]["resolved_temperature"] is None
    assert by_model["anthropic/claude-opus-4-6"]["resolved_temperature"] == pytest.approx(
        LABELING_TEMPERATURE
    )
    assert validate_record(manifest, RUN_MANIFEST_SCHEMA) == []
