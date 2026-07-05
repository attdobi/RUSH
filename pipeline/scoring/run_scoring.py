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
from pipeline.web.demo_area import MNIST_POLICY_AREA, area_from_policy_version, normalize_policy_area


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


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _repo_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _task_for_manifest(manifest: dict[str, Any], fallback_version: str) -> tasks_mod.ScoringTask:
    raw_area = manifest.get("area")
    if isinstance(raw_area, str) and raw_area:
        area = normalize_policy_area(raw_area)
    else:
        area = area_from_policy_version(manifest.get("policy_graph_version") or fallback_version)
    return tasks_mod.MNIST_MULTICLASS if area == MNIST_POLICY_AREA else tasks_mod.DEFAULT_TASK


def _metrics_for_web(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add binary-shaped aliases for the MNIST web tables without losing detail."""
    if "macro_f1" not in metrics:
        return metrics
    out = dict(metrics)
    out.setdefault("precision", metrics.get("macro_precision"))
    out.setdefault("recall", metrics.get("macro_recall"))
    out.setdefault("f1", metrics.get("macro_f1"))
    return out


def _dq_for_web(dq: dict[str, Any]) -> dict[str, Any]:
    out = dict(dq)
    out["labelers"] = [
        {**row, "metrics": _metrics_for_web(dict(row.get("metrics", {})))}
        for row in dq.get("labelers", [])
        if isinstance(row, dict)
    ]
    return out


def _build_update_candidates(misalignment: dict[str, Any], *, cap: int = 50) -> list[dict[str, Any]]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    type_rank = {"consensus_wrong": 0, "model_vs_sme": 1, "model_vs_model": 2}
    rows: list[dict[str, Any]] = []
    for record in misalignment.get("records", []):
        if not isinstance(record, dict):
            continue
        if dq_mod.split_kind(record.get("split")) != "train":
            continue
        misalignment_type = record.get("misalignment_type")
        if misalignment_type == "all_agree":
            continue
        votes = record.get("votes", [])
        rows.append(
            {
                "image_id": record.get("image_id"),
                "sme_truth": record.get("sme_truth"),
                "misalignment_type": misalignment_type,
                "severity": record.get("severity"),
                "split": "train",
                "is_boundary": any(
                    bool(v.get("is_boundary", False)) for v in votes if isinstance(v, dict)
                ),
                "repo_rel_path": record.get("repo_rel_path", ""),
            }
        )
    rows.sort(
        key=lambda r: (
            severity_rank.get(str(r.get("severity") or ""), 9),
            type_rank.get(str(r.get("misalignment_type") or ""), 9),
            str(r.get("image_id") or ""),
        )
    )
    return rows[:cap]


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
    root = repo_root.resolve()
    resolved_runs_root = (runs_root or root / "data" / "runs").resolve()
    run_dir = resolved_runs_root / run_id
    run_manifest = _read_run_manifest(run_dir)
    resolved_task = task or _task_for_manifest(run_manifest, policy_graph_version)
    manifest_policy_version = run_manifest.get("policy_graph_version")
    if isinstance(manifest_policy_version, str) and manifest_policy_version:
        policy_graph_version = manifest_policy_version
    if manifest is None:
        manifest = _repo_path(root, run_manifest.get("sample_manifest_path"))
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
    dq["update_candidates"] = _build_update_candidates(mis)
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
            record["split"] = gt.split
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
        "update_candidates_count": len(dq.get("update_candidates", [])),
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
    dq_web = _dq_for_web(dq)

    mis = mis_mod.compute_misalignment(
        votes_path,
        manifest_path,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
        label_coercer=dq_mc_mod.make_label_coercer(task.classes),
    )
    bord = borderline_mod.compute_borderline(
        votes_path,
        manifest_path,
        policy_graph_version=policy_graph_version,
        ground_truth_tier=ground_truth_tier,
        label_coercer=dq_mc_mod.make_label_coercer(task.classes),
        classes=task.classes,
    )

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
            record["split"] = gt.split
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
    _atomic_write_json(scoring_dir / "decision_quality.json", dq_web)
    _atomic_write_json(scoring_dir / "cost.json", cost_summary)
    _atomic_write_json(scoring_dir / "misalignment.json", mis)
    _atomic_write_json(scoring_dir / "borderline.json", bord)
    _atomic_write_json(scoring_dir / "consensus.json", consensus_summary)
    _atomic_write_jsonl(run_dir / "consensus.jsonl", consensus_records)

    written = exporters_mod.write_web_exports(
        run_dir,
        decision_quality=dq_web,
        misalignment=mis,
        borderline=bord,
        consensus=consensus_summary,
        run_id=run_id,
    )

    return {
        "ok": True,
        "run_id": run_id,
        "task": task.name,
        "multiclass": True,
        "scoring_done": True,
        "decision_quality": {"n_labelers": len(dq.get("labelers", []))},
        "written": {name: str(path) for name, path in written.items()},
        "misalignment_summary": mis.get("summary", {}),
        "borderline_summary": bord.get("summary", {}),
        "consensus_summary": consensus_rollup,
        "cost_summary": cost_summary,
    }
