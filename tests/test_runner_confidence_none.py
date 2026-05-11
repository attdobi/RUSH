"""Regression tests for the None-confidence path in pipeline.runner.

Real-world trigger (run job-20260511T005149-51434365): a provider that
correctly propagates ``confidence: None`` (Gemini after the null-vs-zero
fix) caused ``_build_llm_output`` / ``_build_label_vote`` to raise
``TypeError: float() argument must be a string or a real number, not
'NoneType'`` and abort the run mid-batch. The schemas explicitly allow
``confidence`` to be ``number | null``, so the runner must round-trip
``None`` instead of crashing.
"""
from __future__ import annotations

from pipeline.providers.base import LabelResponse
from pipeline.runner import (
    _build_label_vote,
    _build_llm_output,
    _coerce_optional_confidence,
)


def _make_response(confidence) -> LabelResponse:
    return LabelResponse(
        image_id="dev_golden_xxxx",
        model_id="google/gemini-3.1-pro-preview",
        label="not_gen_ai",
        l2_label="GA.negative.authentic_photo",
        justification="Sufficient justification for the test record.",
        confidence=confidence,
        difficulty="medium",
        is_boundary=False,
        raw_provider_payload={},
        error=None,
        latency_ms=42,
        attempts=1,
        prepared_image_sha256="a" * 64,
        prepared_image_width=1024,
        prepared_image_height=1024,
        prepared_image_mime_type="image/jpeg",
        prepared_image_byte_size=1234,
    )


def test_coerce_handles_none_and_garbage():
    assert _coerce_optional_confidence(None) is None
    assert _coerce_optional_confidence("nan-string") is None
    assert _coerce_optional_confidence(0.42) == 0.42
    assert _coerce_optional_confidence("0.7") == 0.7
    assert _coerce_optional_confidence(0) == 0.0


def test_build_llm_output_accepts_none_confidence():
    out = _build_llm_output(_make_response(None))
    assert out["confidence"] is None
    assert out["label"] == "not_gen_ai"


def test_build_llm_output_accepts_numeric_confidence():
    out = _build_llm_output(_make_response(0.85))
    assert out["confidence"] == 0.85


def test_build_label_vote_accepts_none_confidence():
    vote = _build_label_vote(
        _make_response(None),
        run_id="20260511T000000-deadbeef",
        policy_graph_version="Generative_AI.v0.1",
        prompt_version="v0.1",
    )
    assert vote["confidence"] is None
    assert vote["label"] == "not_gen_ai"
    assert vote["labeler_id"] == "google/gemini-3.1-pro-preview"
