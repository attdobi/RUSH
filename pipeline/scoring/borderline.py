"""Borderline / boundary-difficulty case clustering.

Cold-start v1: cluster by L0 SME truth bucket only. Once the policy graph
populates more L2 nodes for the GenAI label set, the same shape extends to
``groups[<l0>][<l2>]`` keyed lists.

A case is borderline when ANY of:
    * any vote has ``is_boundary == True``
    * any vote has ``difficulty == "high"``
    * any decided vote has confidence below ``low_confidence_threshold`` (default 0.6)
    * model-vs-model disagreement (>= 2 distinct decided labels)
    * any vote is ``abstain``

Output shape (persisted as ``scoring/borderline.json``)::

    {
      "policy_graph_version": "Generative_AI.v0.1",
      "low_confidence_threshold": 0.6,
      "summary": {"total_images": int, "borderline_images": int,
                  "by_l0": {"gen_ai": int, "not_gen_ai": int}},
      "groups": {
        "gen_ai":     [borderline_record, ...],
        "not_gen_ai": [borderline_record, ...]
      }
    }

A ``borderline_record`` carries pass-through ``prepared_image_*`` metadata for
each labeler vote where present, so reviewers can audit cost/quality without
the exporter having to reach into label_votes.jsonl twice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import _common


def _vote_block(vote: dict[str, Any]) -> dict[str, Any]:
    block = {
        "labeler_id": _common.labeler_id_for(vote),
        "model_id": vote.get("model_id"),
        "label": vote.get("label"),
        "l2_label": vote.get("l2_label", ""),
        "confidence": vote.get("confidence"),
        "is_boundary": bool(vote.get("is_boundary", False)),
        "difficulty": vote.get("difficulty", ""),
        "justification": vote.get("justification", ""),
    }
    block.update(_common.extract_prep_metadata(vote))
    return block


def _is_borderline(votes: list[dict[str, Any]], low_conf: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    decided_labels = set()
    for v in votes:
        label = v.get("label")
        if label == _common.ABSTAIN:
            reasons.append("abstain_vote")
            continue
        decided_labels.add(label)
        if v.get("is_boundary"):
            reasons.append("is_boundary_flag")
        if v.get("difficulty") == "high":
            reasons.append("difficulty_high")
        conf = _common.optional_confidence(v.get("confidence"))
        if conf is not None and conf < low_conf:
            reasons.append("low_confidence")
    if len(decided_labels) > 1:
        reasons.append("model_disagreement")
    # dedupe + stable order
    seen = []
    for r in reasons:
        if r not in seen:
            seen.append(r)
    return (bool(seen), seen)


def compute_borderline(
    label_votes_path: Path,
    manifest_path: Path,
    *,
    policy_graph_version: str,
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
    low_confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    truth = _common.load_ground_truth(manifest_path, truth_tiers=ground_truth_tier)
    votes = _common.load_label_votes(label_votes_path)

    by_image: dict[str, list[dict[str, Any]]] = {}
    for v in votes:
        image_id = v.get("image_id")
        if not image_id or image_id not in truth:
            continue
        by_image.setdefault(image_id, []).append(v)

    groups: dict[str, list[dict[str, Any]]] = {l: [] for l in _common.COLD_START_LABELS}
    by_l0_count: dict[str, int] = {l: 0 for l in _common.COLD_START_LABELS}
    borderline_n = 0

    for image_id in sorted(by_image.keys()):
        gt = truth[image_id]
        is_b, reasons = _is_borderline(by_image[image_id], low_confidence_threshold)
        if not is_b:
            continue
        borderline_n += 1
        by_l0_count[gt.label] = by_l0_count.get(gt.label, 0) + 1
        sorted_votes = sorted(by_image[image_id], key=_common.labeler_id_for)
        groups.setdefault(gt.label, []).append(
            {
                "image_id": image_id,
                "repo_rel_path": gt.repo_rel_path,
                "sme_truth": gt.label,
                "reasons": reasons,
                "votes": [_vote_block(v) for v in sorted_votes],
            }
        )

    return {
        "policy_graph_version": policy_graph_version,
        "low_confidence_threshold": low_confidence_threshold,
        "summary": {
            "total_images": len(by_image),
            "borderline_images": borderline_n,
            "by_l0": by_l0_count,
        },
        "groups": groups,
    }
