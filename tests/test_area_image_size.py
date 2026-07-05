"""FIX 1 regression: area-aware image sizing.

MNIST source digits are 28x28; upscaling to 1024px blows compact local
reasoning models' context (gemma 400s / over-reasons + abstains). MNIST must
prepare small (<=112px); GenAI keeps the 1024px baseline. The runner threads
the ontology's max_image_size onto LabelRequest so prepare_image_for_labeling
downsamples per area.
"""
from __future__ import annotations

from pipeline.providers.ontology import get_ontology
from pipeline.providers.registry import MODEL_REGISTRY
from pipeline.runner import _build_request
from pipeline.manifest import SampleRecord


def test_mnist_ontology_caps_at_112():
    assert get_ontology("MNIST_Digits").max_image_size == (112, 112)


def test_genai_ontology_stays_1024():
    assert get_ontology("Generative_AI").max_image_size == (1024, 1024)


def _sample() -> SampleRecord:
    return SampleRecord(
        sample_id="s1",
        repo_rel_path="data/x.png",
        split="train",
        sme_label_raw="ai_generated",
        sme_label="gen_ai",
        dataset="mnist",
        sha256="0" * 64,
        sampling_version="v1",
    )


def _build(area: str):
    spec = MODEL_REGISTRY["openai/gpt-5.5"]
    return _build_request(
        _sample(),
        spec,
        policy_markdown="p",
        policy_graph_version=f"{area}.v1",
        prompt_version="v1",
        area=area,
    )


def test_runner_threads_mnist_small_image():
    req = _build("MNIST_Digits")
    assert req.max_image_size == (112, 112)
    assert max(req.max_image_size) <= 112


def test_runner_threads_genai_full_image():
    req = _build("Generative_AI")
    assert req.max_image_size == (1024, 1024)


def test_gemma_disables_reasoning():
    spec = MODEL_REGISTRY["local/gemma-4-26b-a4b-qat"]
    assert spec.params.get("reasoning_effort") == "none"
