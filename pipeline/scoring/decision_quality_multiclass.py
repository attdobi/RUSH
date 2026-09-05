"""Multiclass scoring artifacts for RUSH, preserving the existing output schema.

The pure metric implementation is in multiclass_counts. Its count-based F1
fix retains zero-F1 classes in the macro average. Classes entirely absent
from predictions AND truth have undefined F1; averages omit only undefined
values. Abstains are excluded from decided metrics and reported separately.
Scores are therefore conditional on coverage, not full-population accuracy.

This changes the metric definition relative to historical artifacts. Re-score
saved predictions before comparing runs or resuming optimization from an old
metric baseline. No historical labels, metrics or decisions are rewritten.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from . import _common
from .decision_quality import _majority_vote
from .multiclass_counts import compute_multiclass_metrics as _count_metrics
from .tasks import ScoringTask


def compute_multiclass_metrics(predictions: list[str], truths: list[str], *, classes: tuple[str, ...], abstain: str = _common.ABSTAIN) -> dict[str, Any]:
    """Public API retained; count-based F1 = 2TP / (2TP + FP + FN)."""
    return _count_metrics(predictions, truths, classes=classes, abstain=abstain)


def make_label_coercer(classes: tuple[str, ...]) -> Callable[[str, int | None], str]:
    """Resolve a closed-vocabulary label from raw labels or a manifest integer."""
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
        raise ValueError(f"Unrecognized multiclass label '{raw}' (label_int={label_int}); expected one of {sorted(allowed)}")

    return _coerce


def compute_decision_quality_multiclass(label_votes_path: Path, manifest_path: Path, *, task: ScoringTask,
                                       policy_graph_version: str, ground_truth_tier: tuple[str, ...] = ("gold",),
                                       schemas_dir: Path | None = None) -> dict[str, Any]:
    """Compute a multiclass snapshot with a coverage-aware majority-vote row."""
    if task.is_binary:
        raise ValueError(f"task '{task.name}' is binary; use decision_quality.compute_decision_quality")
    truth = _common.load_ground_truth(manifest_path, truth_tiers=ground_truth_tier or ("gold",), label_coercer=make_label_coercer(task.classes))
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
        count = abstain_counts.get(labeler, 0)
        labelers_block.append({'labeler_id': labeler, 'labeler_type': 'llm', 'metrics': compute_multiclass_metrics(
            [p for p, _ in pairs] + [task.abstain] * count, [t for _, t in pairs] + [''] * count,
            classes=task.classes, abstain=task.abstain)})
    ens_preds: list[str] = []
    ens_truths: list[str] = []
    for image_id in sorted(per_image):
        gt = truth.get(image_id)
        if not gt:
            continue
        winner = _majority_vote(per_image[image_id])
        # A tie is an abstention, not an invisible disappearance from coverage.
        ens_preds.append(winner if winner is not None else task.abstain)
        ens_truths.append(gt.label)
    if ens_preds:
        labelers_block.append({'labeler_id': 'majority_vote', 'labeler_type': 'ensemble',
                               'metrics': compute_multiclass_metrics(ens_preds, ens_truths, classes=task.classes, abstain=task.abstain)})
    snapshot: dict[str, Any] = {'policy_graph_version': policy_graph_version, 'task': task.name, 'classes': list(task.classes),
                                'ground_truth_tier': [t for t in ground_truth_tier if t in {'gold', 'platinum'}] or ['gold'],
                                'labelers': labelers_block,
                                'warning': 'counts-f1-v2: zero-F1 classes are included; re-score historical predictions before comparing metrics. Accuracy excludes abstentions.'}
    if schemas_dir is not None:
        schema_path = schemas_dir / 'decision-quality-multiclass.schema.json'
        if schema_path.exists():
            errs = _common.try_validate(snapshot, schema_path, label='decision-quality-multiclass')
            if errs:
                raise ValueError('decision-quality-multiclass validation failed: ' + '; '.join(errs))
    return snapshot
