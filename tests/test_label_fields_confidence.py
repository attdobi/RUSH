import pytest

from pipeline.providers.base import coerce_label_fields


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, None),
        ({"confidence": "abc"}, None),
        ({"confidence": 0.7}, 0.7),
        ({"confidence": 1.5}, 1.0),
        ({"confidence": -0.2}, 0.0),
        ({"confidence": True}, None),
    ],
)
def test_confidence_missing_malformed_and_clamped(raw, expected):
    assert coerce_label_fields(raw)["confidence"] == expected
