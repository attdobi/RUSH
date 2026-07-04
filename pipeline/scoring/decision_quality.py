"""Decision-quality metrics for RUSH bulk-labeling runs.

Computes the per-labeler row required by ``schemas/decision-quality.schema.json``
plus a synthesized ``majority_vote`` ensemble row.

Conventions (cold-start / GenAI v0.1):
    * Positive class = ``gen_ai``.
    * ``abstain`` predictions are excluded from numerator/denominator math
      and counted separately in ``n_abstained`` (carried in the labeler
      record's optional ``warning`` blob, not in the schema-required block).
    * ``n`` is the number of decided predictions for the labeler against
      a ground-truth image. Images without an SME truth row are skipped.
    * Metrics that would divide by zero are emitted as ``null`` (the schema
      allows null).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from . import _common

TRAIN_SPLIT_ALIASES = {"dev_golden", "train", "training", "development"}
TEST_SPLIT_ALIASES = {
    "holdout",
    "val",
    "validation",
    "test",
    "testing",
    "locked_holdout",
    "locked_holdout_decision_quality",
}


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _round(value: float | None, places: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, places)


def compute_metrics(
    predictions: list[str], truths: list[str], *, positive: str = _common.POSITIVE_CLASS
) -> dict[str, Any]:
    """Pure binary-classification metrics.

    Both inputs MUST be the same length and aligned positionally. Abstain
    rows must be filtered out before calling this; ``n`` is ``len(predictions)``.
    """
    if len(predictions) != len(truths):
        raise ValueError("predictions/truths length mismatch")
    n = len(predictions)
    tp = fp = tn = fn = 0
    for pred, truth in zip(predictions, truths):
        if pred == positive and truth == positive:
            tp += 1
        elif pred == positive and truth != positive:
            fp += 1
        elif pred != positive and truth != positive:
            tn += 1
        elif pred != positive and truth == positive:
            fn += 1
    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)  # TPR
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    if precision is None or recall is None or (precision == 0 and recall == 0):
        f1 = None
    else:
        f1 = _safe_div(2 * precision * recall, precision + recall)
    positive_proportion = _safe_div(tp + fp, n)
    if recall is None or fpr is None:
        informedness = None
    else:
        informedness = recall - fpr  # Youden's J
    return {
        "accuracy": _round(accuracy),
        "f1": _round(f1),
        "precision": _round(precision),
        "recall": _round(recall),
        "fpr": _round(fpr),
        "fnr": _round(fnr),
        "positive_proportion": _round(positive_proportion),
        "n": n,
        "informedness": _round(informedness),
    }


def _majority_vote(per_labeler_pred: dict[str, str]) -> str | None:
    """Majority across decided labelers for one image. Ties / all-abstain → None."""
    votes = [v for v in per_labeler_pred.values() if v != _common.ABSTAIN]
    if not votes:
        return None
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None  # tie → ensemble abstains
    return counts[0][0]


def split_kind(split: str | None) -> str | None:
    normalized = str(split or "").strip().lower()
    if normalized in TRAIN_SPLIT_ALIASES:
        return "train"
    if normalized in TEST_SPLIT_ALIASES:
        return "test"
    return None


def _compute_labelers_block(
    votes: list[dict[str, Any]],
    truth: dict[str, _common.GroundTruth],
    *,
    image_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    # per-labeler aligned predictions vs truth
    by_labeler: dict[str, list[tuple[str, str]]] = defaultdict(list)
    abstain_counts: dict[str, int] = defaultdict(int)
    # for ensemble: image_id -> {labeler -> label}
    per_image: dict[str, dict[str, str]] = defaultdict(dict)

    for v in votes:
        image_id = v.get("image_id")
        gt = truth.get(image_id) if image_id else None
        if not gt:
            continue
        if image_ids is not None and image_id not in image_ids:
            continue
        labeler = _common.labeler_id_for(v)
        label = v.get("label", _common.ABSTAIN)
        per_image[image_id][labeler] = label
        if label == _common.ABSTAIN:
            abstain_counts[labeler] += 1
            continue
        by_labeler[labeler].append((label, gt.label))

    labelers_block: list[dict[str, Any]] = []
    for labeler in sorted(by_labeler.keys() | abstain_counts.keys()):
        pairs = by_labeler.get(labeler, [])
        preds = [p for p, _ in pairs]
        truths = [t for _, t in pairs]
        labelers_block.append(
            {
                "labeler_id": labeler,
                "labeler_type": "llm",
                "metrics": compute_metrics(preds, truths),
            }
        )

    # majority-vote ensemble
    ens_pairs: list[tuple[str, str]] = []
    for image_id in sorted(per_image.keys()):
        gt = truth.get(image_id)
        if not gt:
            continue
        winner = _majority_vote(per_image[image_id])
        if winner is None:
            continue
        ens_pairs.append((winner, gt.label))
    if ens_pairs:
        labelers_block.append(
            {
                "labeler_id": "majority_vote",
                "labeler_type": "ensemble",
                "metrics": compute_metrics(
                    [p for p, _ in ens_pairs], [t for _, t in ens_pairs]
                ),
            }
        )

    return labelers_block, len(per_image), dict(sorted(abstain_counts.items()))


def compute_decision_quality(
    label_votes_path: Path,
    manifest_path: Path,
    *,
    policy_graph_version: str,
    ground_truth_tier: tuple[str, ...] = ("gold",),
    schemas_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute the DecisionQualitySnapshot dict matching the schema."""
    truth = _common.load_ground_truth(
        manifest_path, truth_tiers=ground_truth_tier or ("gold",)
    )
    votes = _common.load_label_votes(label_votes_path)

    labelers_block, _n_images_all, abstain_counts = _compute_labelers_block(votes, truth)
    split_image_ids: dict[str, set[str]] = {"train": set(), "test": set()}
    for image_id, gt in truth.items():
        kind = split_kind(gt.split)
        if kind in split_image_ids:
            split_image_ids[kind].add(image_id)
    by_split: dict[str, dict[str, Any]] = {}
    for kind in ("train", "test"):
        split_labelers, n_images, _split_abstain_counts = _compute_labelers_block(
            votes, truth, image_ids=split_image_ids[kind]
        )
        by_split[kind] = {"labelers": split_labelers, "n_images": n_images}

    snapshot: dict[str, Any] = {
        "policy_graph_version": policy_graph_version,
        "ground_truth_tier": [t for t in ground_truth_tier if t in {"gold", "platinum"}]
        or ["gold"],
        "labelers": labelers_block,
        "by_split": by_split,
        "reported_split": "test",
        "reported": by_split["test"],
    }
    # Carry abstain rates as a non-schema sidecar so the schema stays exact.
    if abstain_counts:
        snapshot["warning"] = json.dumps(
            {"abstain_counts": abstain_counts},
            sort_keys=True,
        )

    if schemas_dir is not None:
        errs = _common.try_validate(
            snapshot, schemas_dir / "decision-quality.schema.json", label="decision-quality"
        )
        if errs:
            raise ValueError("decision-quality validation failed: " + "; ".join(errs))
    return snapshot
