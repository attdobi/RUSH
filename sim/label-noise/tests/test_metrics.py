"""metrics: hand-computed macro-F1, rank AUROC edge cases, gini, mean_ci."""

import math

import numpy as np
import pytest

from labelsim.metrics import accuracy, auroc, gini, macro_f1, mean_ci


def test_macro_f1_binary_hand_computed():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # class 0: tp=1 fp=0 fn=1 -> 2/3 ; class 1: tp=2 fp=1 fn=0 -> 4/5
    assert macro_f1(y_true, y_pred, 2) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert macro_f1(y_true, y_true, 2) == pytest.approx(1.0)


def test_macro_f1_three_class_with_absent_class():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    # class 0: tp=1 fp=0 fn=1 -> 2/3 ; class 1: tp=1 fp=1 fn=0 -> 2/3
    # class 2 absent from truth and prediction -> contributes 0
    assert macro_f1(y_true, y_pred, 3) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_auroc_perfect_ties_and_degenerate():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]),
                 np.array([True, True, False, False])) == pytest.approx(1.0)
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]),
                 np.array([True, True, False, False])) == pytest.approx(0.0)
    assert auroc(np.full(6, 0.5),
                 np.array([1, 0, 1, 0, 1, 0], dtype=bool)) == pytest.approx(0.5)
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([True, True])))
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([False, False])))


def test_gini_equal_vs_concentrated():
    assert gini(np.full(4, 5.0)) == pytest.approx(0.0)
    assert gini(np.zeros(10)) == 0.0
    x = np.zeros(100)
    x[0] = 100.0
    assert gini(x) == pytest.approx(0.99)  # (n-1)/n, -> 1 as n grows
    assert gini(np.array([1.0, 3.0])) > gini(np.array([2.0, 2.0]))


def test_mean_ci_single_value_and_nan_safety():
    m, hw = mean_ci([3.0])
    assert (m, hw) == (3.0, 0.0)
    m, hw = mean_ci([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert hw == pytest.approx(1.96 * 1.0 / math.sqrt(3))
    m, hw = mean_ci([float("nan"), 4.0])
    assert (m, hw) == (4.0, 0.0)          # nans dropped
    m, hw = mean_ci([float("nan")])
    assert math.isnan(m) and math.isnan(hw)


def test_accuracy():
    assert accuracy(np.array([0, 1, 1]), np.array([0, 1, 0])) == pytest.approx(2 / 3)
