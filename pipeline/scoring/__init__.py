"""RUSH scoring subpackage.

Public API:
    - decision_quality.compute_decision_quality
    - misalignment.compute_misalignment
    - borderline.compute_borderline
    - consensus.build_consensus_records / build_cohort_rollups
    - cost.aggregate_per_call_costs / attach_cost_to_labelers
    - flip_rate.build_flip_rate_records / cohort_rollups
    - exporters.write_web_exports
    - run_scoring / compute_flip_rate

All functions are stdlib-only and offline-safe. Schema validation is best-effort
via the optional :mod:`jsonschema` dependency; when unavailable, structural
checks still run but JSON Schema enforcement is skipped (a warning is logged).
"""
from . import borderline, consensus, cost, decision_quality, exporters, flip_rate, misalignment
from .run_scoring import compute_flip_rate, run_scoring

__all__ = [
    "borderline",
    "consensus",
    "cost",
    "decision_quality",
    "exporters",
    "flip_rate",
    "misalignment",
    "run_scoring",
    "compute_flip_rate",
]
