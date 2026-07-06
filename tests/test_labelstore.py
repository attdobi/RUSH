"""Pure-logic tests for the label store (§4.1 tiers + golden resolution).

DB integration is exercised by scripts/labelstore_ingest.py against a live
Postgres; these tests cover the tier/materialization math with no DB needed.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import labelstore  # noqa: E402


def test_tiers_follow_section_4_1():
    assert labelstore.confidence_tier([], "x") == "seed_bpo"
    assert labelstore.confidence_tier(["7"], "7") == "sme_1"
    assert labelstore.confidence_tier(["7", "7"], "7") == "sme_2_confirmed"
    assert labelstore.confidence_tier(["7", "9"], "9") == "sme_2_contested"
    assert labelstore.confidence_tier(["7", "9", "7"], "7") == "sme_3"


def test_tier_weights_match_doc():
    assert labelstore.TIER_WEIGHTS == {
        "seed_bpo": 1.0,
        "sme_1": 1.0,
        "sme_2_confirmed": 2.0,
        "sme_2_contested": 0.5,
        "sme_3": 3.0,
    }


def test_resolve_golden_single_and_confirmed():
    assert labelstore.resolve_golden(["4"]) == "4"
    assert labelstore.resolve_golden(["4", "4"]) == "4"


def test_resolve_golden_contested_takes_latest():
    # §4.1: sme_2_contested resolves to the latest event.
    assert labelstore.resolve_golden(["4", "9"]) == "9"


def test_resolve_golden_majority_at_three():
    assert labelstore.resolve_golden(["4", "9", "4"]) == "4"
    assert labelstore.resolve_golden(["9", "4", "4"]) == "4"


def test_resolve_golden_empty_is_none():
    assert labelstore.resolve_golden([]) is None


def test_golden_row_materialization():
    row = labelstore.golden_row(["4", "9", "4"], seed_source="sme_single", last_epoch=2)
    assert row["current_label"] == "4"
    assert row["num_sme_labels"] == 3
    assert row["num_sme_agree_current"] == 2
    assert row["confidence_tier"] == "sme_3"
    assert row["at_cap"] is True
    assert row["last_epoch"] == 2


def test_golden_row_contested_not_at_cap():
    row = labelstore.golden_row(["4", "9"], seed_source="sme_single", last_epoch=1)
    assert row["current_label"] == "9"
    assert row["confidence_tier"] == "sme_2_contested"
    assert row["at_cap"] is False
    assert row["num_sme_agree_current"] == 1


def test_golden_row_requires_events():
    with pytest.raises(ValueError):
        labelstore.golden_row([], seed_source="sme_single", last_epoch=None)


def test_human_confidence_formula():
    # p = 1 - 1/(m + 0.2): a lone human label is weak evidence; agreement
    # compounds it, saturating toward 1.
    assert labelstore.human_confidence(1) == pytest.approx(1 - 1 / 1.2)   # 0.1667
    assert labelstore.human_confidence(2) == pytest.approx(1 - 1 / 2.2)   # 0.5455
    assert labelstore.human_confidence(3) == pytest.approx(1 - 1 / 3.2)   # 0.6875
    assert labelstore.human_confidence(10) == pytest.approx(1 - 1 / 10.2)
    # m=0 (no agreeing human evidence): the raw formula would go negative
    # (1 - 1/0.2 = -4); clamped to zero confidence.
    assert labelstore.human_confidence(0) == 0.0


def test_golden_row_carries_human_confidence():
    row = labelstore.golden_row(["4", "9", "4"], seed_source="sme_single", last_epoch=2)
    # m = num_sme_agree_current = 2 -> 1 - 1/2.2
    assert row["human_confidence"] == pytest.approx(1 - 1 / 2.2, abs=1e-6)
