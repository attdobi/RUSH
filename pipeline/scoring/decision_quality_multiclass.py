"""Multiclass decision-quality metrics for RUSH scoring runs.

This module generalizes the binary metrics in
:mod:`pipeline.scoring.decision_quality` to an arbitrary closed label set
(e.g. the MNIST 0-9 demo) **without** touching the binary output shape. The
binary path (``compute_metrics`` / ``compute_decision_quality``) is unchanged;
multiclass metrics live here and are persisted to a NEW artifact
(``scoring/decision_quality_multiclass.json``) with its own shape.

Metric output shape (per labeler)
---------------------------------
:func:`compute_multiclass_metrics` returns::

    {
      "accuracy": float | None,          # (correct decided) / (n decided)
      "n": int,                          # decided predictions (abstains excluded)
      "n_abstained": int,                # abstain predictions (excluded from math)
      "macro_precision": float | None,   # mean of defined per-class precision
      "macro_recall": float | None,
      "macro_f1": float | None,
      "macro_fpr": float | None,
      "macro_fnr": float | None,
      "micro_precision": float | None,   # pooled one-vs-rest counts
      "micro_recall": float | None,
      "micro_f1": float | None,
      "micro_fpr": float | None,
      "micro_fnr": float | None,
      "per_class": {
        <class>: {
          "precision": float | None,     # tp / (tp + fp),  one-vs-rest
          "recall":    float | None,     # tp / (tp + fn)
          "f1":        float | None,     # harmonic mean of P/R
          "fpr":       float | None,     # fp / (fp + tn)
          "fnr":       float | None,     # fn / (fn + tp) = 1 - recall
          "support":   int               # # of decided rows whose truth == class
        }, ...
      },
      "confusion_matrix": { <truth>: { <pred>: count, ... }, ... }
    }

Conventions (mirrors the binary module)
----------------------------------------
* ``abstain`` predictions are excluded from every numerator/denominator and
  counted in ``n_abstained``; ``n`` counts only decided predictions.
* Divide-by-zero yields ``None`` (via :func:`decision_quality._safe_div`);
  every ratio is rounded to 6 decimals (via :func:`decision_quality._round`).
* **Macro averages** are the arithmetic mean over classes whose per-class value
  is *defined* (not ``None``); if no class is defined the macro value is
  ``None``. This keeps a never-predicted class (undefined precision) from
  poisoning the average.
* **Unknown labels**: a predicted label that is neither in ``classes`` nor the
  abstain sentinel is treated as *wrong* — it can never match a truth, so it
  contributes to that truth-class's false negatives and lowers accuracy. It is
  still recorded in the confusion matrix under its literal key so anomalies are
  auditable. (An unknown label in ``truths`` is handled symmetrically.)

The confusion matrix is pre-seeded with a ``classes x classes`` grid of zeros
for a predictable shape; unknown labels are added lazily as extra rows/columns.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from . import _common
from .decision_quality import _majority_vote, _round, _safe_div
from .tasks import ScoringTask


def _macro(values: list[float | None]) -> float | None:
    """Arithmetic mean over defined (non-None) values; None if none defined."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return _safe_div(sum(defined), len(defined))


def compute_multiclass_metrics(
    predictions: list[str],
    truths: list[str],
    *,
    classes: tuple[str, ...],
    abstain: str = _common.ABSTAIN,
) -> dict[str, Any]:
    """Pure multiclass-classification metrics (one-vs-rest per class + macro).

    ``predictions`` and ``truths`` MUST be the same length and aligned
    positionally. Unlike the binary helper, abstains do NOT need to be filtered
    out beforehand: any prediction equal to ``abstain`` is dropped (with its
    paired truth) and counted in ``n_abstained``.

    Args:
        predictions: Predicted labels (may include ``abstain``).
        truths: Ground-truth labels, aligned to ``predictions``.
        classes: The closed label vocabulary to score over.
        abstain: The non-decisive sentinel label.

    Returns:
        The multiclass metric dict documented in the module docstring.

    Raises:
        ValueError: if the inputs differ in length.
    """
    if len(predictions) != len(truths):
        raise ValueError("predictions/truths length mismatch")

    # Split off abstains; keep only decided (pred, truth) pairs.
    decided: list[tuple[str, str]] = []
    n_abstained = 0
    for pred, truth in zip(predictions, truths):
        if pred == abstain:
            n_abstained += 1
            continue
        decided.append((pred, truth))
    n = len(decided)

    # Confusion matrix seeded with a classes x classes grid of zeros for a
    # predictable shape; unknown truth/pred labels are added lazily.
    confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in classes} for t in classes
    }
    correct = 0
    for pred, truth in decided:
        if truth == pred:
            correct += 1
        row = confusion.setdefault(truth, {})
        row[pred] = row.get(pred, 0) + 1

    accuracy = _safe_div(correct, n)

    # One-vs-rest counts per class, plus pooled counts for micro averages.
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float | None] = []
    recalls: list[float | None] = []
    f1s: list[float | None] = []
    fprs: list[float | None] = []
    fnrs: list[float | None] = []
    pooled_tp = pooled_fp = pooled_tn = pooled_fn = 0
    for cls in classes:
        tp = fp = tn = fn = 0
        for pred, truth in decided:
            if pred == cls and truth == cls:
                tp += 1
            elif pred == cls and truth != cls:
                fp += 1
            elif pred != cls and truth == cls:
                fn += 1
            else:  # pred != cls and truth != cls
                tn += 1
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        fpr = _safe_div(fp, fp + tn)
        # fnr = fn / (fn + tp) = 1 - recall; computed from counts so it is
        # None exactly when recall is None (zero support).
        fnr = _safe_div(fn, fn + tp)
        if precision is None or recall is None or (precision == 0 and recall == 0):
            f1 = None
        else:
            f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[cls] = {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "fpr": _round(fpr),
            "fnr": _round(fnr),
            "support": tp + fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr)
        fnrs.append(fnr)
        pooled_tp += tp
        pooled_fp += fp
        pooled_tn += tn
        pooled_fn += fn

    # Micro = pooled one-vs-rest counts across classes. For single-label
    # multiclass, micro precision == micro recall == accuracy over decided rows
    # (every error is one FP for the predicted class and one FN for the true
    # class); micro FPR is the genuinely new pooled signal.
    micro_recall = _safe_div(pooled_tp, pooled_tp + pooled_fn)
    micro_precision = _safe_div(pooled_tp, pooled_tp + pooled_fp)
    micro_fpr = _safe_div(pooled_fp, pooled_fp + pooled_tn)
    micro_fnr = _safe_div(pooled_fn, pooled_fn + pooled_tp)
    if micro_precision is None or micro_recall is None or (
        micro_precision == 0 and micro_recall == 0
    ):
        micro_f1 = None
    else:
        micro_f1 = _safe_div(
            2 * micro_precision * micro_recall, micro_precision + micro_recall
        )

    return {
        "accuracy": _round(accuracy),
        "n": n,
        "n_abstained": n_abstained,
        "macro_precision": _round(_macro(precisions)),
        "macro_recall": _round(_macro(recalls)),
        "macro_f1": _round(_macro(f1s)),
        "macro_fpr": _round(_macro(fprs)),
        "macro_fnr": _round(_macro(fnrs)),
        "micro_precision": _round(micro_precision),
        "micro_recall": _round(micro_recall),
        "micro_f1": _round(micro_f1),
        "micro_fpr": _round(micro_fpr),
        "micro_fnr": _round(micro_fnr),
        "per_class": per_class,
        "confusion_matrix": {
            t: dict(sorted(confusion[t].items()))
            for t in sorted(confusion.keys())
        },
    }


def make_label_coercer(
    classes: tuple[str, ...],
) -> Callable[[str, int | None], str]:
    """Build a ground-truth label coercer for a multiclass task.

    The returned callable accepts ``(raw_label, label_int)`` from a manifest row
    and returns the canonical class string. It accepts the raw string (when
    already in ``classes``), the sampler's source-of-truth vocabulary mapped
    through :data:`pipeline.manifest.SME_LABEL_MAP` (GenAI manifests store
    ``ai_generated``/``not_ai_generated``, scored as ``gen_ai``/``not_gen_ai``),
    or the stringified ``label_int``.

    Raises:
        ValueError: if no form resolves to a member of ``classes``.
    """
    from pipeline.manifest import SME_LABEL_MAP

    allowed = set(classes)

    def _coerce(raw: str, label_int: int | None) -> str:
        raw_str = str(raw)
        if raw_str in allowed:
            return raw_str
        mapped = SME_LABEL_MAP.get(raw_str)
        if mapped in allowed:
            return mapped
        if label_int is not None and str(label_int) in allowed:
            return str(label_int)
        raise ValueError(
            f"Unrecognized multiclass label '{raw}' (label_int={label_int}); "
            f"expected one of {sorted(allowed)}"
        )

    return _coerce


def compute_decision_quality_multiclass(
    label_votes_path: Path,
    manifest_path: Path,
    *,
    task: ScoringTask,
    policy_graph_version: str,
    ground_truth_tier: tuple[str, ...] = ("gold",),
    schemas_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute a multiclass DecisionQualitySnapshot.

    Mirrors :func:`decision_quality.compute_decision_quality` but emits the
    multiclass metric shape (see module docstring) and a synthesized
    ``majority_vote`` ensemble row. This is a NEW artifact shape and is NOT
    validated against the binary ``decision-quality.schema.json``. When
    ``schemas_dir`` is given and ``decision-quality-multiclass.schema.json``
    exists, best-effort validation runs against it.
    """
    if task.is_binary:
        raise ValueError(
            f"task '{task.name}' is binary; use decision_quality.compute_decision_quality"
        )

    truth = _common.load_ground_truth(
        manifest_path,
        truth_tiers=ground_truth_tier or ("gold",),
        label_coercer=make_label_coercer(task.classes),
    )
    votes = _common.load_label_votes(label_votes_path)

    by_labeler: dict[str, list[tuple[str, str]]] = defaultdict(list)
    abstain_counts: dict[str, int] = defaultdict(int)
    per_image: dict[str, dict[str, str]] = defaultdict(dict)
    class_set = set(task.classes)

    for v in votes:
        image_id = v.get("image_id")
        gt = truth.get(image_id) if image_id else None
        if not gt:
            continue
        labeler = _common.labeler_id_for(v)
        label = v.get("label", task.abstain)
        if label not in class_set and label != task.abstain:
            import re

            match = re.fullmatch(r"MD\.digit\.(\d+)", str(v.get("l2_label", "")))
            if match and match.group(1) in class_set:
                label = match.group(1)
        per_image[image_id][labeler] = label
        if label == task.abstain:
            abstain_counts[labeler] += 1
            continue
        by_labeler[labeler].append((label, gt.label))

    labelers_block: list[dict[str, Any]] = []
    for labeler in sorted(by_labeler.keys() | abstain_counts.keys()):
        pairs = by_labeler.get(labeler, [])
        preds = [p for p, _ in pairs]
        truths = [t for _, t in pairs]
        # Re-inject this labeler's abstains so n_abstained is reported.
        preds_with_abstain = preds + [task.abstain] * abstain_counts.get(labeler, 0)
        truths_with_abstain = truths + [""] * abstain_counts.get(labeler, 0)
        labelers_block.append(
            {
                "labeler_id": labeler,
                "labeler_type": "llm",
                "metrics": compute_multiclass_metrics(
                    preds_with_abstain,
                    truths_with_abstain,
                    classes=task.classes,
                    abstain=task.abstain,
                ),
            }
        )

    # majority-vote ensemble (generic; ties → abstain/excluded)
    ens_preds: list[str] = []
    ens_truths: list[str] = []
    for image_id in sorted(per_image.keys()):
        gt = truth.get(image_id)
        if not gt:
            continue
        winner = _majority_vote(per_image[image_id])
        if winner is None:
            continue
        ens_preds.append(winner)
        ens_truths.append(gt.label)
    if ens_preds:
        labelers_block.append(
            {
                "labeler_id": "majority_vote",
                "labeler_type": "ensemble",
                "metrics": compute_multiclass_metrics(
                    ens_preds, ens_truths, classes=task.classes, abstain=task.abstain
                ),
            }
        )

    snapshot: dict[str, Any] = {
        "policy_graph_version": policy_graph_version,
        "task": task.name,
        "classes": list(task.classes),
        "ground_truth_tier": [t for t in ground_truth_tier if t in {"gold", "platinum"}]
        or ["gold"],
        "labelers": labelers_block,
    }

    if schemas_dir is not None:
        schema_path = schemas_dir / "decision-quality-multiclass.schema.json"
        if schema_path.exists():
            errs = _common.try_validate(
                snapshot, schema_path, label="decision-quality-multiclass"
            )
            if errs:
                raise ValueError(
                    "decision-quality-multiclass validation failed: " + "; ".join(errs)
                )
    return snapshot
