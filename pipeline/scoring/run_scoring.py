"""Callable scoring orchestration for a completed labeling run.

This module contains the implementation behind ``scripts/score_labels.py`` so
web-triggered jobs can score in-process without shelling out.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from . import borderline as borderline_mod
from . import consensus as consensus_mod
from . import cost as cost_mod
from . import decision_quality as dq_mod
from . import exporters as exporters_mod
from . import flip_rate as flip_rate_mod
from . import misalignment as mis_mod
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _scored_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(
        [
            path
            for path in runs_root.iterdir()
            if path.is_dir()
            and not path.name.startswith("_")
            and (path / "run_manifest.json").exists()
            and (path / "label_votes.jsonl").exists()
            and (path / "scoring" / "decision_quality.json").exists()
        ],
        key=lambda p: p.name,
    )


def _record_to_dict(record: Any) -> dict[str, Any]:
    row = asdict(record)
    row["single_run_only"] = int(row.get("n_runs") or 0) < 2
    run_ids = sorted(
        {
            str(row.get("first_seen_run_id") or ""),
            str(row.get("last_seen_run_id") or ""),
        }
        - {""}
    )
    # build_flip_rate_records only stores first/last; for current callers this
    # is enough for filtering repeated-run cohorts and preserves existing shape.
    row["run_ids"] = run_ids
    return row


def _flip_rate_summary(records: list[dict[str, Any]], *, computed_at: str) -> dict[str, Any]:
    total = len(records)
    stable = sum(1 for r in records if int(r.get("flip_count") or 0) == 0)
    flipped = sum(1 for r in records if int(r.get("flip_count") or 0) > 0)
    single = sum(1 for r in records if bool(r.get("single_run_only")))
    multi_rates = [float(r.get("flip_rate") or 0.0) for r in records if not r.get("single_run_only")]

    per_model: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("single_run_only"):
            continue
        model_id = str(row.get("model_id") or "unknown")
        slot = per_model.setdefault(model_id, {"_rates": [], "n_pairs": 0, "n_pairs_flipped": 0})
        slot["n_pairs"] += 1
        if int(row.get("flip_count") or 0) > 0:
            slot["n_pairs_flipped"] += 1
        slot["_rates"].append(float(row.get("flip_rate") or 0.0))

    per_model_out = {}
    for model_id in sorted(per_model):
        slot = per_model[model_id]
        rates = slot.pop("_rates")
        per_model_out[model_id] = {
            "n_pairs": slot["n_pairs"],
            "n_pairs_flipped": slot["n_pairs_flipped"],
            "mean_flip_rate": sum(rates) / len(rates) if rates else None,
        }

    top = sorted(
        records,
        key=lambda r: (
            -float(r.get("flip_rate") or 0),
            -int(r.get("flip_count") or 0),
            str(r.get("image_id") or ""),
            str(r.get("model_id") or ""),
        ),
    )[:20]
    top_flipped_images = [
        {
            "image_id": r.get("image_id"),
            "model_id": r.get("model_id"),
            "flip_rate": r.get("flip_rate"),
            "flip_count": r.get("flip_count"),
            "n_runs": r.get("n_runs"),
            "labels_observed": r.get("labels_observed", []),
        }
        for r in top
    ]

    return {
        "n_pairs_total": total,
        "n_pairs_stable": stable,
        "n_pairs_flipped": flipped,
        "n_pairs_single_run": single,
        "mean_flip_rate": (sum(multi_rates) / len(multi_rates)) if multi_rates else None,
        "per_model_flip_rate": per_model_out,
        "top_flipped_images": top_flipped_images,
        "computed_at": computed_at,
    }


def compute_flip_rate(
    *,
    repo_root: Path,
    runs_root: Path | None = None,
    min_runs: int = 2,
) -> dict[str, Any]:
    """Compute and persist global flip-rate artifacts when enough runs exist.

    Returns a small status dict. When fewer than ``min_runs`` scored runs are
    present, no files are written and ``skipped`` explains why.
    """
    root = repo_root.resolve()
    resolved_runs_root = (runs_root or root / "data" / "runs").resolve()
    run_dirs = _scored_run_dirs(resolved_runs_root)
    if len(run_dirs) < min_runs:
        return {
            "ok": True,
            "skipped": True,
            "reason": "flip-rate needs ≥2 scored runs",
            "n_scored_runs": len(run_dirs),
        }

    computed_at = _now_iso()
    run_ids_by_pair: dict[tuple[str, str], set[str]] = {}
    for run_dir in run_dirs:
        for vote in load_label_votes(run_dir / "label_votes.jsonl"):
            image_id = vote.get("image_id")
            model_id = vote.get("model_id") or vote.get("labeler_id")
            if not image_id or not model_id:
                continue
            run_ids_by_pair.setdefault((str(image_id), str(model_id)), set()).add(str(vote.get("run_id") or run_dir.name))

    records = []
    for record in flip_rate_mod.build_flip_rate_records(run_dirs):
        row = _record_to_dict(record)
        row["run_ids"] = sorted(run_ids_by_pair.get((record.image_id, record.model_id), set(row.get("run_ids", []))))
        records.append(row)
    stamp = computed_at.replace(":", "")
    out_dir = resolved_runs_root / "_flip_rate" / stamp
    summary = _flip_rate_summary(records, computed_at=computed_at)
    _atomic_write_json(out_dir / "flip_rate_summary.json", summary)
    _atomic_write_jsonl(out_dir / "flip_rate.jsonl", records)

    # Static web fallback used by the browser when the local API is unavailable.
    web_payload = {"summary": summary, "records": records}
    _atomic_write_json(root / "web" / "flip_rate.json", web_payload)
    return {
        "ok": True,
        "skipped": False,
        "output_dir": str(out_dir),
        "summary": summary,
    }


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
    compute_flip: bool = True,
) -> dict[str, Any]:
    """Score one run and write canonical + web artifacts.

    The return value is JSON-serializable and suitable for API responses.
    """
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
    flip_result = compute_flip_rate(repo_root=root, runs_root=resolved_runs_root) if compute_flip else {"skipped": True, "reason": "disabled"}

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
        "flip_rate": flip_result,
    }
