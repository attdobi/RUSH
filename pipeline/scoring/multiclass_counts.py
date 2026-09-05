"""Dependency-free count metrics. F1 is 0 when errors exist, not undefined.

The output shape and abstain convention match decision_quality_multiclass.
A class absent from both predictions and truth has undefined F1; macro F1
averages the defined classes. Unknown labels remain auditable, never correct.
Historical artifacts are not rewritten by importing or calling this module.
"""
from __future__ import annotations

from typing import Any

METRIC_DEFINITION = "counts-f1-v2"


def _div(a: int | float, b: int | float) -> float | None:
    return a / b if b else None


def _round(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _macro(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return _div(sum(defined), len(defined))


def compute_multiclass_metrics(predictions: list[str], truths: list[str], *, classes: tuple[str, ...], abstain: str = "abstain") -> dict[str, Any]:
    if len(predictions) != len(truths):
        raise ValueError("predictions/truths length mismatch")
    if not classes or len(set(classes)) != len(classes) or abstain in classes:
        raise ValueError("classes must be nonempty, unique and exclude the abstain sentinel")
    decided = [(p, t) for p, t in zip(predictions, truths) if p != abstain]
    n = len(decided)
    confusion = {t: {p: 0 for p in classes} for t in classes}
    correct = 0
    for pred, truth in decided:
        correct += pred == truth and truth in classes
        row = confusion.setdefault(truth, {})
        row[pred] = row.get(pred, 0) + 1
    per_class = {}
    names = ('precision', 'recall', 'f1', 'fpr', 'fnr')
    values: dict[str, list[float | None]] = {key: [] for key in names}
    ptp = pfp = ptn = pfn = 0
    for cls in classes:
        tp = sum(p == cls and t == cls for p, t in decided)
        fp = sum(p == cls and t != cls for p, t in decided)
        fn = sum(p != cls and t == cls for p, t in decided)
        tn = n - tp - fp - fn
        ratios = {'precision': _div(tp, tp + fp), 'recall': _div(tp, tp + fn),
                  'f1': _div(2 * tp, 2 * tp + fp + fn),
                  'fpr': _div(fp, fp + tn), 'fnr': _div(fn, fn + tp)}
        per_class[cls] = {**{k: _round(v) for k, v in ratios.items()}, 'support': tp + fn}
        for key, value in ratios.items():
            values[key].append(value)
        ptp += tp
        pfp += fp
        ptn += tn
        pfn += fn
    return {'accuracy': _round(_div(correct, n)), 'n': n, 'n_abstained': len(predictions) - n,
            **{f'macro_{key}': _round(_macro(values[key])) for key in names},
            'micro_precision': _round(_div(ptp, ptp + pfp)), 'micro_recall': _round(_div(ptp, ptp + pfn)),
            'micro_f1': _round(_div(2 * ptp, 2 * ptp + pfp + pfn)),
            'micro_fpr': _round(_div(pfp, pfp + ptn)), 'micro_fnr': _round(_div(pfn, pfn + ptp)),
            'per_class': per_class,
            'confusion_matrix': {t: dict(sorted(confusion[t].items())) for t in sorted(confusion)}}
