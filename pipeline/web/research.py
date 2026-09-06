"""Read-only experiment evidence for the research workbench.

The graph timeline follows explicit generator_before/generator_after fields, not
version-name ordering. This adapter never changes a gate, label, or experiment.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

AREAS = {"Generative_AI": "GA.root", "MNIST_Digits": "MD.root"}
RUN_ID = re.compile(r"^_?exp-[0-9]{8}T[0-9]{6}-[a-f0-9]{6}$")
VERSION = re.compile(r"^v[0-9]+\.[0-9]+$")
MAX_BYTES = 8_000_000


def _safe_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Evidence file is missing or outside the configured root")
    if resolved.stat().st_size > MAX_BYTES:
        raise ValueError("Evidence file exceeds the read limit")
    raw = json.loads(resolved.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    if not isinstance(raw, dict):
        raise ValueError("Expected an experiment object")
    return raw


def _public(value: Any, depth: int = 0) -> Any:
    """Bound nested evidence and exclude non-finite numbers from JSON output."""
    if depth > 9:
        return None
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return value[:16_000]
    if type(value) in (int, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_public(v, depth + 1) for v in value[:500]]
    if isinstance(value, dict):
        return {str(k)[:200]: _public(v, depth + 1) for k, v in list(value.items())[:200]}
    return None


def _version(value: Any, area: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.removeprefix(area + ".")
    return value if VERSION.fullmatch(value) else None


def _complete(root: Path, area: str, version: str | None) -> bool:
    if version is None:
        return False
    path = (root / "policy-graph" / area / version).resolve()
    if not path.is_relative_to(root):
        return False
    for filename in (AREAS[area] + ".md", "edges.json"):
        item = (path / filename).resolve()
        if not item.is_relative_to(root) or not item.is_file():
            return False
    return True


def read_run(repo_root: str | Path, area: str, run_id: str) -> dict[str, Any]:
    if area not in AREAS or not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid area or experiment id")
    root = Path(repo_root).resolve()
    run = _safe_json(root, root / "data" / "experiments" / run_id / "experiment.json")
    if run.get("area") != area:
        raise ValueError("Experiment belongs to another policy area")
    if run.get("dry_run") is not False:
        raise ValueError("Dry or unverified runs are not measured research evidence")
    cycles = run.get("cycles")
    if not isinstance(cycles, list) or len(cycles) > 500:
        raise ValueError("Invalid or oversized cycle history")
    warnings: list[str] = []
    frames: list[dict[str, Any]] = []
    current = _version(run.get("base_version"), area)
    if _complete(root, area, current):
        frames.append({"version": current, "before_version": None, "k": 0,
                       "status": "baseline", "title": "k=0 · baseline"})
    else:
        warnings.append("Baseline snapshot is missing; no graph lineage can be replayed.")
    public_cycles = []
    lineage_open = bool(frames)
    seen_k: set[int] = set()
    for cycle in cycles:
        if not isinstance(cycle, dict) or type(cycle.get("k")) is not int or cycle["k"] < 0:
            warnings.append("Malformed cycle omitted; lineage stops at the last verified step.")
            lineage_open = False
            continue
        k = cycle["k"]
        if k in seen_k:
            warnings.append("Duplicate cycle index; lineage stops at the last verified step.")
            lineage_open = False
            continue
        seen_k.add(k)
        allowed = ("k", "kind", "status", "metrics", "gate", "generator_before", "generator_after",
                   "n_misaligned", "n_anchors", "proposal_id", "policy_blame", "errored_calls",
                   "gate_review", "started_at", "closed_at")
        public_cycles.append({key: _public(cycle[key]) for key in allowed if key in cycle})
        if k == 0:
            pins = [cycle.get("generator_before"), cycle.get("generator_after")]
            if any(pin is not None and _version(pin, area) != current for pin in pins):
                raise ValueError("Baseline generator identity does not match the policy snapshot")
            continue
        if not lineage_open:
            continue
        before = _version(cycle.get("generator_before"), area)
        after = _version(cycle.get("generator_after"), area)
        status = cycle.get("status")
        if before != current or (frames and k <= frames[-1]["k"]):
            warnings.append(f"k={k}: explicit parent/order mismatch; graph replay stops here.")
            lineage_open = False
            continue
        if status == "accepted":
            valid = after != current and _complete(root, area, after)
        else:
            valid = status in ("skipped", "rejected", "no_misalignments") and after == current
        if not valid:
            warnings.append(f"k={k}: missing snapshot or unresolved transition; graph replay stops here.")
            lineage_open = False
            continue
        frames.append({"version": after, "before_version": before, "k": k,
                       "status": status, "title": f"k={k} · {status}"})
        current = after
    config_keys = ("experiment_id", "area", "seed", "run_number", "status", "strategy", "gate_mode",
                   "gate_model", "drafter_model", "judge_models", "batch_n", "test_n", "k_max",
                   "max_changes", "epsilon", "test_mode", "cost_usd_total", "started_at", "finished_at")
    config = {key: _public(run[key]) for key in config_keys if key in run}
    # Report only what identifiers establish. Absence is not evidence of disjointness.
    splits = run.get("splits") if isinstance(run.get("splits"), dict) else {}
    test_ids = splits.get("test_ids")
    training = [c for c in cycles if isinstance(c, dict) and type(c.get("k")) is int and c["k"] > 0]
    ids_complete = (run.get("test_mode", "fixed") == "fixed" and isinstance(test_ids, list)
                    and bool(test_ids) and bool(training) and any(c.get("train_ids") for c in training) and all(isinstance(c.get("train_ids"), list) for c in training))
    overlap = None
    if ids_complete:
        if all(isinstance(i, str) for i in test_ids) and all(isinstance(i, str) for c in training for i in c["train_ids"]):
            overlap = len(set(test_ids).intersection(i for c in training for i in c["train_ids"]))
    return {"schema_version": 1, "origin": "recorded", "config": config, "cycles": public_cycles,
            "frames": frames, "warnings": warnings,
            "split_audit": {"train_gate_overlap": overlap, "gate_n": len(test_ids) if isinstance(test_ids, list) else None,
                            "holdout_n": _public(splits.get("holdout_n")),
                            "scope": "Recorded identifiers only; duplicate content and across-run reuse are not checked."},
            "readouts": {"holdout": _public(run.get("holdout")), "benchmark": _public(run.get("benchmark"))},
            "metric_status": "Stored measurements; legacy F1 requires re-scoring. No significance is inferred.",
            "source": f"data/experiments/{run_id}/experiment.json"}


def dispatch(repo_root: str | Path, url: str) -> tuple[int, dict[str, Any]]:
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    if parts.path != "/api/studio/research-run":
        return 404, {"error": "Unknown research endpoint"}
    if set(query) != {"area", "id"} or any(len(v) != 1 for v in query.values()):
        return 400, {"error": "One area and one experiment id are required"}
    try:
        return 200, read_run(repo_root, query["area"][0], query["id"][0])
    except (ValueError, OSError, TypeError, OverflowError):
        return 400, {"error": "Invalid, missing, or unreadable experiment evidence"}
