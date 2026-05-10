"""RUSH scoring subpackage.

Public API:
    - decision_quality.compute_decision_quality
    - misalignment.compute_misalignment
    - borderline.compute_borderline
    - consensus.build_consensus_records / build_cohort_rollups
    - flip_rate.build_flip_rate_records / cohort_rollups
    - exporters.write_web_exports

All functions are stdlib-only and offline-safe. Schema validation is best-effort
via the optional :mod:`jsonschema` dependency; when unavailable, structural
checks still run but JSON Schema enforcement is skipped (a warning is logged).
"""
from . import borderline, consensus, decision_quality, exporters, flip_rate, misalignment

__all__ = [
    "borderline",
    "consensus",
    "decision_quality",
    "exporters",
    "flip_rate",
    "misalignment",
]
