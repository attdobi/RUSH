"""Tests for the cascade orchestrator's tier-2 accounting.

The invariant under test: an escalated image whose tier-2 call errored (no
consensus record in a completed-with-errors run) falls through to the SME
queue — it must never silently vanish from the ledger.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("run_cascade", ROOT / "scripts" / "run_cascade.py")
run_cascade = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_cascade)


def _rec(image_id):
    return {"image_id": image_id}


def test_all_judged_none_residual():
    judged, errored, residual = run_cascade.tier2_accounting(
        ["a", "b"], [], [_rec("a"), _rec("b")]
    )
    assert (judged, errored, residual) == (2, [], [])


def test_errored_escalation_falls_through_to_sme():
    # "b" was escalated but has no tier-2 record (its call errored).
    judged, errored, residual = run_cascade.tier2_accounting(
        ["a", "b"], [], [_rec("a")]
    )
    assert judged == 1
    assert errored == ["b"]
    assert residual == ["b"]


def test_residual_is_union_of_unresolved_and_errored():
    # "a" judged but still unresolved; "b" errored; "c" judged and resolved.
    judged, errored, residual = run_cascade.tier2_accounting(
        ["a", "b", "c"], ["a"], [_rec("a"), _rec("c")]
    )
    assert judged == 2
    assert errored == ["b"]
    assert residual == ["a", "b"]


def test_all_errored_reports_zero_judged():
    judged, errored, residual = run_cascade.tier2_accounting(["a", "b"], [], [])
    assert judged == 0
    assert errored == ["a", "b"]
    assert residual == ["a", "b"]


def test_stray_tier2_record_not_counted_as_judged():
    # A record for an id that was never escalated must not inflate `judged`.
    judged, errored, residual = run_cascade.tier2_accounting(
        ["a"], [], [_rec("a"), _rec("zzz")]
    )
    assert judged == 1
    assert errored == []
    assert residual == []
