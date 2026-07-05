"""FIX 2 regression: per-model speed telemetry must survive persist + rollup.

The runner measures response.latency_ms; it must (a) land in the persisted
llm_output envelope so X3 can build per-model speed, and (b) drive a non-null
avg in the per-model rollup. Guards against the earlier bug where latency_ms
was dropped before persist and the status rollup returned all-None.
"""
from __future__ import annotations

import json

from pipeline.providers.base import LabelResponse
from pipeline.runner import _build_llm_output, _build_label_vote
from pipeline.web.run_registry import _per_model_rollup


def _response(latency_ms: int) -> LabelResponse:
    return LabelResponse(
        image_id="img-1",
        model_id="local/gemma-4-26b-a4b-qat",
        label="7",
        l2_label="MD.digit.7",
        justification="A clearly written seven with a horizontal top stroke.",
        confidence=0.9,
        difficulty="low",
        is_boundary=False,
        raw_provider_payload={},
        error=None,
        latency_ms=latency_ms,
        attempts=1,
        output_tokens=152,
    )


def test_latency_ms_present_in_llm_output():
    out = _build_llm_output(_response(1900))
    assert out["latency_ms"] == 1900


def test_latency_ms_present_in_label_vote():
    vote = _build_label_vote(
        _response(1900),
        run_id="r1",
        policy_graph_version="MNIST_Digits.v1",
        prompt_version="v1",
    )
    assert vote["latency_ms"] == 1900


def test_per_model_rollup_non_null_from_votes(tmp_path):
    votes = tmp_path / "label_votes.jsonl"
    with votes.open("w", encoding="utf-8") as fh:
        for lat in (1900, 2100):
            fh.write(json.dumps({"model_id": "local/gemma-4-26b-a4b-qat", "latency_ms": lat}) + "\n")
    rollup = _per_model_rollup(votes, ["local/gemma-4-26b-a4b-qat"], elapsed_seconds=10.0, expected_calls=2)
    assert len(rollup) == 1
    row = rollup[0]
    assert row["avg_latency_ms"] == 2000.0
    assert row["calls_done"] == 2
    assert row["throughput_imgs_per_min"] is not None
