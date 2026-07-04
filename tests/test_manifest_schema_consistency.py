"""Guardrail: run-manifest schema enums must cover what the registry emits.

This catches the class of bug where a new reasoning tier (e.g. ``low``/``medium``)
is added to the model registry but the run-manifest schema's ``reasoning_effort``
enum is not widened — which makes run_manifest writes fail schema validation at
run time (not at test time). See the 2026-07-04 gpt-5.4-mini-low incident.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.providers.registry import MODEL_REGISTRY

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = _REPO_ROOT / "schemas" / "run-manifest.schema.json"


def _schema_reasoning_effort_enum() -> set[str]:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    node = schema["properties"]["model_runtime_config"]["additionalProperties"]
    return set(node["properties"]["reasoning_effort"]["enum"])


def _registry_reasoning_efforts() -> set[str]:
    vals: set[str] = set()
    for spec in MODEL_REGISTRY.values():
        params = getattr(spec, "params", {}) or {}
        effort = params.get("reasoning_effort")
        if effort is not None:
            vals.add(effort)
    return vals


def test_schema_reasoning_effort_enum_covers_registry() -> None:
    enum = _schema_reasoning_effort_enum()
    emitted = _registry_reasoning_efforts()
    missing = emitted - enum
    assert not missing, (
        f"run-manifest schema reasoning_effort enum {sorted(enum)} is missing "
        f"registry values {sorted(missing)} — widen the schema enum."
    )
