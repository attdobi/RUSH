"""Callable scoring orchestration for a completed labeling run.

This module contains the implementation behind ``scripts/score_labels.py`` so
web-triggered jobs can score in-process without shelling out.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import borderline as borderline_mod
from . import consensus as consensus_mod
from . import cost as cost_mod
from . import decision_quality as dq_mod
from . import decision_quality_multiclass as dq_mc_mod
from . import exporters as exporters_mod
from . import misalignment as mis_mod
from . import tasks as tasks_mod
from ._common import load_ground_truth, load_label_votes, try_validate


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def run_scoring(
    run_id: str,
    repo_root: Path,
    *,
    runs_root: Path | None = None,
    manifest: Path | None = None,
    policy_graph_version: str = "Generative_AI.v0.1",
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
    low_confidence_threshold: float = 0.6,
    validate_schemas: bool = False,
    task: tasks_mod.ScoringTask | None = None,
) -> dict[str, Any]:
    """Score one run and write canonical + web artifacts.

    ``task`` selects the label space. When omitted (or when a binary task is
    passed) the full cold-start GenAI pipeline runs unchanged (backward
    compatible). When a multiclass task is passed, the run dispatches to
    :func:`run_scoring_multiclass`, which writes the multiclass DQ artifact.

    The return value is JSON-serializable and suitable for API responses.
    """
    resolved_task = task or tasks_mod.DEFAULT_TASK
    if resolved_task.is_multiclass:
        return run_scoring_multiclass(
            run_id,
            repo_root,
            task=resolved_task,
            runs_root=runs_root,
            manifest=manifest,
            policy_graph_version=policy_graph_version,
            ground_truth_tier=ground_truth_tier,
            validate_schemas=validate_schemas,
        )

    root = repo_root.resolve()
    resolved_runs_root = (runs_root or root / "data" / "runs").resolve()
    run_dir = resolved_runs_root / run_id
    votes_path = run_dir / "label_votes.jsonl"
    if not votes_path.exists():
        raise FileNotFoundError(f"missing {votes_path}")

    manifest_path = manifest or (
        root / "data" / "images" / "genai-classification" / "manifests" / "combined_labels.jsonl"
    )
    schemas_dir = root / "schemas" if validate_schemas else None

    dq = dq_mod.compute_decision_quality(
        votes_path,
        manifest_path,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
        schemas_dir=schemas_dir,
    )
    mis = mis_mod.compute_misalignment(
        votes_path,
        manifest_path,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
    )
    bord = borderline_mod.compute_borderline(
        votes_path,
        manifest_path,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
        low_confidence_threshold=low_confidence_threshold,
    )

    votes_raw = load_label_votes(votes_path)
    cost_summary = cost_mod.aggregate_per_call_costs(votes_raw)
    dq = cost_mod.attach_cost_to_labelers(dq, cost_summary)
    if schemas_dir is not None:
        errs = try_validate(dq, schemas_dir / "decision-quality.schema.json", label="decision-quality")
        if errs:
            raise ValueError("decision-quality validation failed after cost attach: " + "; ".join(errs))

    consensus_records = consensus_mod.build_consensus_records(votes_raw, run_id=run_id)
    try:
        truth = load_ground_truth(manifest_path, truth_tiers=ground_truth_tier)
    except FileNotFoundError:
        truth = {}
    for record in consensus_records:
        gt = truth.get(str(record.get("image_id") or ""))
        if gt:
            record["repo_rel_path"] = gt.repo_rel_path
            record["sme_truth"] = gt.label
    consensus_rollup = consensus_mod.build_cohort_rollups(consensus_records, ground_truth=truth)
    consensus_summary = {
        "run_id": run_id,
        "policy_graph_version": policy_graph_version,
        "ground_truth_tier": [t for t in ground_truth_tier if t in {"gold", "platinum"}] or list(ground_truth_tier),
        "summary": consensus_rollup,
        "records": consensus_records,
    }

    scoring_dir = run_dir / "scoring"
    _atomic_write_json(scoring_dir / "decision_quality.json", dq)
    _atomic_write_json(scoring_dir / "cost.json", cost_summary)
    _atomic_write_json(scoring_dir / "misalignment.json", mis)
    _atomic_write_json(scoring_dir / "borderline.json", bord)
    _atomic_write_json(scoring_dir / "consensus.json", consensus_summary)
    _atomic_write_jsonl(run_dir / "consensus.jsonl", consensus_records)

    written = exporters_mod.write_web_exports(
        run_dir,
        decision_quality=dq,
        misalignment=mis,
        borderline=bord,
        consensus=consensus_summary,
        run_id=run_id,
    )
    return {
        "ok": True,
        "run_id": run_id,
        "scoring_done": True,
        "written": {name: str(path) for name, path in written.items()},
        "decision_quality": {"n_labelers": len(dq.get("labelers", []))},
        "misalignment_summary": mis.get("summary", {}),
        "borderline_summary": bord.get("summary", {}),
        "consensus_summary": consensus_rollup,
        "cost_summary": cost_summary,
    }


def run_scoring_multiclass(
    run_id: str,
    repo_root: Path,
    *,
    task: tasks_mod.ScoringTask,
    runs_root: Path | None = None,
    manifest: Path | None = None,
    policy_graph_version: str = "Generative_AI.v0.1",
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
    validate_schemas: bool = False,
) -> dict[str, Any]:
    """Score one run against a multiclass task and write canonical artifacts.

    Writes ``scoring/decision_quality_multiclass.json`` (new shape) and
    ``scoring/consensus.json`` (label-agnostic). The binary-specific
    misalignment / borderline analyses and web exporters remain binary-only for
    v1 and are intentionally NOT run here (deferred); this keeps the binary
    output shapes and schemas untouched. Cost aggregation is label-agnostic and
    is attached to the labeler rows.
    """
    root = repo_root.resolve()
    resolved_runs_root = (runs_root or root / "data" / "runs").resolve()
    run_dir = resolved_runs_root / run_id
    votes_path = run_dir / "label_votes.jsonl"
    if not votes_path.exists():
        raise FileNotFoundError(f"missing {votes_path}")

    manifest_path = manifest or (
        root / "data" / "images" / task.name / "manifests" / "combined_labels.jsonl"
    )
    schemas_dir = root / "schemas" if validate_schemas else None

    dq = dq_mc_mod.compute_decision_quality_multiclass(
        votes_path,
        manifest_path,
        task=task,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
        schemas_dir=schemas_dir,
    )

    votes_raw = load_label_votes(votes_path)
    cost_summary = cost_mod.aggregate_per_call_costs(votes_raw)
    dq = cost_mod.attach_cost_to_labelers(dq, cost_summary)

    consensus_records = consensus_mod.build_consensus_records(votes_raw, run_id=run_id)
    try:
        truth = load_ground_truth(
            manifest_path,
            truth_tiers=ground_truth_tier,
            label_coercer=dq_mc_mod.make_label_coercer(task.classes),
        )
    except FileNotFoundError:
        truth = {}
    for record in consensus_records:
        gt = truth.get(str(record.get("image_id") or ""))
        if gt:
            record["repo_rel_path"] = gt.repo_rel_path
            record["sme_truth"] = gt.label
    consensus_rollup = consensus_mod.build_cohort_rollups(consensus_records, ground_truth=truth)
    consensus_summary = {
        "run_id": run_id,
        "policy_graph_version": policy_graph_version,
        "task": task.name,
        "ground_truth_tier": [t for t in ground_truth_tier if t in {"gold", "platinum"}] or list(ground_truth_tier),
        "summary": consensus_rollup,
        "records": consensus_records,
    }

    scoring_dir = run_dir / "scoring"
    _atomic_write_json(scoring_dir / "decision_quality_multiclass.json", dq)
    _atomic_write_json(scoring_dir / "cost.json", cost_summary)
    _atomic_write_json(scoring_dir / "consensus.json", consensus_summary)
    _atomic_write_jsonl(run_dir / "consensus.jsonl", consensus_records)

    return {
        "ok": True,
        "run_id": run_id,
        "task": task.name,
        "multiclass": True,
        "scoring_done": True,
        "decision_quality": {"n_labelers": len(dq.get("labelers", []))},
        "consensus_summary": consensus_rollup,
        "cost_summary": cost_summary,
    }
