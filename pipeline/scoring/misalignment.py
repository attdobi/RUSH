"""Model-vs-SME and model-vs-model disagreement detection.

Output shape (X3-defined; persisted as ``scoring/misalignment.json``)::

    {
      "policy_graph_version": "Generative_AI.v0.1",
      "summary": {
        "total_images": int,
        "all_agree": int,
        "model_vs_sme": int,
        "model_vs_model": int,
        "consensus_wrong": int
      },
      "records": [
        {
          "image_id": str,
          "repo_rel_path": str,
          "sme_truth": "gen_ai" | "not_gen_ai",
          "misalignment_type": "all_agree" | "model_vs_sme" | "model_vs_model" | "consensus_wrong",
          "severity": "low" | "medium" | "high",
          "votes": [
            {
              "labeler_id": str,
              "model_id": str | None,
              "label": str,
              "l2_label": str,
              "confidence": float | null,
              "is_boundary": bool,
              "difficulty": str,
              "justification": str,
              "prepared_image_*": (passed through when present)
            },
            ...
          ]
        }
      ]
    }

Severity heuristic:
    * ``high``: model consensus disagrees with SME (``consensus_wrong``)
    * ``medium``: any model disagrees with SME OR mixed model-vs-model with one wrong
    * ``low``: only model-vs-model disagreement on a case where models that
      disagree are split across the SME truth
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import _common


def _classify(per_labeler: dict[str, str], sme: str) -> tuple[str, str]:
    decided = {l: lbl for l, lbl in per_labeler.items() if lbl != _common.ABSTAIN}
    if not decided:
        return "model_vs_sme", "medium"
    label_set = set(decided.values())
    matches_sme = [l for l, lbl in decided.items() if lbl == sme]
    misses_sme = [l for l, lbl in decided.items() if lbl != sme]
    if len(label_set) == 1:
        # full model consensus
        if sme in label_set:
            return "all_agree", "low"
        return "consensus_wrong", "high"
    # models disagree
    if not misses_sme:
        return "model_vs_model", "low"
    if not matches_sme:
        # all misses, but not unanimous label (rare for binary; possible if abstain mix)
        return "consensus_wrong", "high"
    return "model_vs_sme", "medium"


def _vote_block(vote: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "labeler_id": _common.labeler_id_for(vote),
        "model_id": vote.get("model_id"),
        "label": vote.get("label"),
        "l2_label": vote.get("l2_label", ""),
        "confidence": vote.get("confidence"),
        "is_boundary": bool(vote.get("is_boundary", False)),
        "is_boundary_between": list(vote.get("is_boundary_between") or []),
        "difficulty": vote.get("difficulty", ""),
        "justification": vote.get("justification", ""),
        "policy_citations": list(vote.get("policy_citations") or []),
        "policy_quotes": list(vote.get("policy_quotes") or []),
        "justification_too_long": bool(vote.get("justification_too_long", False)),
        "input_tokens": vote.get("input_tokens"),
        "output_tokens": vote.get("output_tokens"),
        "cost_usd": vote.get("cost_usd"),
    }
    block.update(_common.extract_prep_metadata(vote))
    return block


def compute_misalignment(
    label_votes_path: Path,
    manifest_path: Path,
    *,
    policy_graph_version: str,
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
    label_coercer: Any | None = None,
) -> dict[str, Any]:
    truth = _common.load_ground_truth(
        manifest_path,
        truth_tiers=ground_truth_tier,
        label_coercer=label_coercer,
    )
    votes = _common.load_label_votes(label_votes_path)

    by_image: dict[str, list[dict[str, Any]]] = {}
    for v in votes:
        image_id = v.get("image_id")
        if not image_id or image_id not in truth:
            continue
        by_image.setdefault(image_id, []).append(v)

    records: list[dict[str, Any]] = []
    summary = Counter()
    for image_id in sorted(by_image.keys()):
        gt = truth[image_id]
        per_labeler = {
            _common.labeler_id_for(v): v.get("label", _common.ABSTAIN)
            for v in by_image[image_id]
        }
        misalignment_type, severity = _classify(per_labeler, gt.label)
        summary[misalignment_type] += 1
        # deterministic vote ordering by labeler_id
        sorted_votes = sorted(by_image[image_id], key=_common.labeler_id_for)
        records.append(
            {
                "image_id": image_id,
                "repo_rel_path": gt.repo_rel_path,
                "sme_truth": gt.label,
                "split": gt.split,
                "misalignment_type": misalignment_type,
                "severity": severity,
                "votes": [_vote_block(v) for v in sorted_votes],
            }
        )

    return {
        "policy_graph_version": policy_graph_version,
        "summary": {
            "total_images": len(records),
            "all_agree": summary["all_agree"],
            "model_vs_sme": summary["model_vs_sme"],
            "model_vs_model": summary["model_vs_model"],
            "consensus_wrong": summary["consensus_wrong"],
        },
        "records": records,
    }
