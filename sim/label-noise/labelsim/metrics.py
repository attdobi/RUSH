"""numpy-only metrics (macro-F1 matches the crank's gate convention)."""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over all n_classes; a class absent from both truth and
    prediction contributes 0 (zero_division=0 convention)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom > 0 else 0.0
    return float(f1s.mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney): P(score_pos > score_neg), ties 0.5.
    Returns nan when either class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gini(counts: np.ndarray) -> float:
    """Concentration of anchor picks over items (0 = even, ->1 = one item
    hoards the anchors — the hyper-focus signature)."""
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def mean_ci(values, z: float = 1.96):
    """(mean, half-width) normal-approx CI across seeds; nan-safe."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(len(v)))
