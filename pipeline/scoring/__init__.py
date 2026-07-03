"""RUSH scoring subpackage.

Public API:
    - decision_quality.compute_decision_quality
    - decision_quality_multiclass.compute_multiclass_metrics /
      compute_decision_quality_multiclass
    - tasks.ScoringTask / GENAI_BINARY / MNIST_MULTICLASS / get_task
    - misalignment.compute_misalignment
    - borderline.compute_borderline
    - consensus.build_consensus_records / build_cohort_rollups
    - cost.aggregate_per_call_costs / attach_cost_to_labelers
    - exporters.write_web_exports
    - run_scoring

All functions are stdlib-only and offline-safe. Schema validation is best-effort
via the optional :mod:`jsonschema` dependency; when unavailable, structural
checks still run but JSON Schema enforcement is skipped (a warning is logged).
"""
from . import (
    borderline,
    consensus,
    cost,
    decision_quality,
    decision_quality_multiclass,
    exporters,
    misalignment,
    tasks,
)
from .run_scoring import run_scoring, run_scoring_multiclass

__all__ = [
    "borderline",
    "consensus",
    "cost",
    "decision_quality",
    "decision_quality_multiclass",
    "exporters",
    "misalignment",
    "tasks",
    "run_scoring",
    "run_scoring_multiclass",
]
