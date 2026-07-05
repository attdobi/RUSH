"""Web-ready JSON exporters for ``data/runs/<run_id>/web/*.json``.

The exporters take the canonical scoring artifacts written by
:mod:`pipeline.scoring.{decision_quality, misalignment, borderline}` and
produce slimmer, UI-friendly shapes the web app can consume directly.

Pass-through invariant
----------------------
Per the 2026-05-10 scope amendment, label records may include downsample-audit
metadata:

    prepared_image_sha256, prepared_image_width, prepared_image_height
    (+ optional prepared_image_mime, prepared_image_bytes)

These fields originate on the per-vote ``LabelVote`` row (X1 attaches them).
:func:`pipeline.scoring._common.extract_prep_metadata` is the single sieve
that pulls them forward; both ``misalignment.compute_*`` and
``borderline.compute_*`` already carry them on each vote block, so the
exporters simply preserve those fields verbatim. No image bytes ever leave
the JSON layer — only metadata about the prepared variant.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import _common

# Fields preserved per-vote in web exports (slimmer than the canonical
# scoring records but always retains prep-audit metadata when present).
_VOTE_WEB_FIELDS = (
    "labeler_id",
    "model_id",
    "label",
    "l2_label",
    "confidence",
    "is_boundary",
    "is_boundary_between",
    "difficulty",
    "justification",
    "policy_citations",
    "policy_quotes",
    "justification_too_long",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
)


def _slim_vote(vote: dict[str, Any]) -> dict[str, Any]:
    out = {k: vote.get(k) for k in _VOTE_WEB_FIELDS if k in vote}
    # prep-audit pass-through (vote dicts coming from misalignment/borderline
    # already include these; this guard makes the exporter independent of order)
    for f in _common.PREP_METADATA_FIELDS:
        if f in vote and vote[f] is not None:
            out[f] = vote[f]
    return out


def build_summary(
    decision_quality: dict[str, Any],
    misalignment: dict[str, Any],
    borderline: dict[str, Any],
    *,
    run_id: str,
    consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_id": run_id,
        "policy_graph_version": decision_quality.get("policy_graph_version"),
        "ground_truth_tier": decision_quality.get("ground_truth_tier", []),
        "labelers": [
            {
                "labeler_id": row["labeler_id"],
                "labeler_type": row["labeler_type"],
                "metrics": row["metrics"],
            }
            for row in decision_quality.get("labelers", [])
        ],
        "misalignment_summary": misalignment.get("summary", {}),
        "borderline_summary": borderline.get("summary", {}),
    }
    if consensus:
        out["consensus_summary"] = consensus.get("summary", {})
    return out


def build_consensus_web(consensus: dict[str, Any]) -> dict[str, Any]:
    """UI-friendly consensus payload: cohort rollups + per-image records.

    Voters are passed through verbatim (already slim — no justifications).
    Records are ordered by image_id (same as the canonical output).
    """
    records = list(consensus.get("records", []))
    records.sort(key=lambda r: r.get("image_id", ""))
    return {
        "run_id": consensus.get("run_id"),
        "policy_graph_version": consensus.get("policy_graph_version"),
        "ground_truth_tier": consensus.get("ground_truth_tier", []),
        "summary": consensus.get("summary", {}),
        "records": records,
    }


def build_misalignment_web(misalignment: dict[str, Any]) -> dict[str, Any]:
    """Slim, UI-friendly disagreement worklist (severity-sorted)."""
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    records = list(misalignment.get("records", []))
    # Push 'all_agree' to the bottom; sort the rest by severity then image_id.
    records.sort(
        key=lambda r: (
            r.get("misalignment_type") == "all_agree",
            severity_rank.get(r.get("severity", "low"), 9),
            r.get("image_id", ""),
        )
    )
    return {
        "policy_graph_version": misalignment.get("policy_graph_version"),
        "summary": misalignment.get("summary", {}),
        "records": [
            {
                "image_id": r["image_id"],
                "repo_rel_path": r.get("repo_rel_path", ""),
                "sme_truth": r["sme_truth"],
                "split": r.get("split", ""),
                "misalignment_type": r["misalignment_type"],
                "severity": r["severity"],
                "votes": [_slim_vote(v) for v in r.get("votes", [])],
            }
            for r in records
            if r.get("misalignment_type") != "all_agree"
        ],
    }


def build_borderline_web(borderline: dict[str, Any]) -> dict[str, Any]:
    """Web-friendly borderline list grouped by L0 (cold-start)."""
    groups_in = borderline.get("groups", {})
    groups_out: dict[str, list[dict[str, Any]]] = {}
    for l0, recs in groups_in.items():
        groups_out[l0] = [
            {
                "image_id": r["image_id"],
                "repo_rel_path": r.get("repo_rel_path", ""),
                "sme_truth": r.get("sme_truth"),
                "reasons": r.get("reasons", []),
                "votes": [_slim_vote(v) for v in r.get("votes", [])],
            }
            for r in recs
        ]
    return {
        "policy_graph_version": borderline.get("policy_graph_version"),
        "low_confidence_threshold": borderline.get("low_confidence_threshold"),
        "summary": borderline.get("summary", {}),
        "groups": groups_out,
    }


def write_web_exports(
    run_dir: Path,
    *,
    decision_quality: dict[str, Any],
    misalignment: dict[str, Any],
    borderline: dict[str, Any],
    consensus: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Write summary/borderline/misalignment/consensus JSON under ``run_dir/web/``.

    ``consensus`` is optional; when omitted, the legacy three-file output is
    preserved (additive, no breakage). Returns ``{name: path}`` for the files
    written.
    """
    web_dir = run_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    rid = run_id or run_dir.name

    paths: dict[str, Path] = {
        "summary": web_dir / "summary.json",
        "misalignment": web_dir / "misalignment.json",
        "borderline": web_dir / "borderline.json",
    }
    payload: dict[str, Any] = {
        "summary": build_summary(
            decision_quality,
            misalignment,
            borderline,
            run_id=rid,
            consensus=consensus,
        ),
        "misalignment": build_misalignment_web(misalignment),
        "borderline": build_borderline_web(borderline),
    }
    if consensus is not None:
        paths["consensus"] = web_dir / "consensus.json"
        payload["consensus"] = build_consensus_web(consensus)
    for name, path in paths.items():
        _atomic_write_json(path, payload[name])
    return paths


def _atomic_write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
