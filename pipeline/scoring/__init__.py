"""RUSH scoring subpackage.

Public API:
    - decision_quality.compute_decision_quality
    - misalignment.compute_misalignment
    - borderline.compute_borderline
    - exporters.write_web_exports

All functions are stdlib-only and offline-safe. Schema validation is best-effort
via the optional :mod:`jsonschema` dependency; when unavailable, structural
checks still run but JSON Schema enforcement is skipped (a warning is logged).
"""
from . import borderline, decision_quality, exporters, misalignment

__all__ = [
    "borderline",
    "decision_quality",
    "exporters",
    "misalignment",
]
