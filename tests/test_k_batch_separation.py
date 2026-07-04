"""k (sample count) and batch_size (dispatch grouping) are independent (X2)."""
from __future__ import annotations

import pytest

from pipeline.web._safety import APIError, validate_start_payload


def _base(**over):
    payload = {
        "models": ["openai/gpt-5.5"],
        "split": "dev_golden",
        "allow_spend": True,
        "limit": 50,
        "batch_size": 20,
    }
    payload.update(over)
    return payload


def test_limit_and_batch_size_are_independent():
    out = validate_start_payload(_base(limit=50, batch_size=20))
    assert out["limit"] == 50
    assert out["batch_size"] == 20


def test_batch_size_defaults_when_omitted():
    payload = _base()
    payload.pop("batch_size")
    out = validate_start_payload(payload)
    assert out["batch_size"] == 20
    assert out["limit"] == 50


def test_gpu_batch_20_is_settable_independent_of_k():
    # Attila's GPU pass: small k, large dispatch batch.
    out = validate_start_payload(_base(limit=10, batch_size=20))
    assert out["limit"] == 10
    assert out["batch_size"] == 20


def test_batch_size_must_be_positive():
    with pytest.raises(APIError):
        validate_start_payload(_base(batch_size=0))
