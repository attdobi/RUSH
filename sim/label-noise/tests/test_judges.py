"""judges.Panel: majority vote, tie-break, exclude semantics, CRN discipline."""

import numpy as np
import pytest

from labelsim import JudgeSpec, Panel, default_panel
from labelsim.policy import LogisticPolicy

# predicts 1 iff x > 0; sigma=0 judges read it exactly
POLICY = LogisticPolicy(np.array([1.0, 0.0]), 0.0)
X2 = np.array([[-3.0, 0.0], [3.0, 0.0]])


def test_majority_with_a_constant_judge():
    panel = Panel([JudgeSpec("a", 0.0), JudgeSpec("b", 0.0),
                   JudgeSpec("c", 0.3, constant=1)])
    labels, system, consensus = panel.vote(POLICY, X2, np.random.default_rng(0))
    assert labels.shape == (3, 2)
    np.testing.assert_array_equal(labels[0], [0, 1])
    np.testing.assert_array_equal(labels[2], [1, 1])   # ignores the policy
    np.testing.assert_array_equal(system, [0, 1])       # 2-1 majority wins
    assert consensus[0] == pytest.approx(2 / 3)
    assert consensus[1] == pytest.approx(1.0)


def test_tie_breaks_to_smallest_class_id():
    # multiclass tie 1-vs-2 -> 1
    panel = Panel([JudgeSpec("hi", 0.1, constant=2),
                   JudgeSpec("lo", 0.1, constant=1)])
    _, system, consensus = panel.vote(None, X2, np.random.default_rng(0))
    np.testing.assert_array_equal(system, [1, 1])
    np.testing.assert_allclose(consensus, [0.5, 0.5])
    # binary tie 0-vs-1 -> 0
    panel01 = Panel([JudgeSpec("one", 0.1, constant=1),
                     JudgeSpec("zero", 0.1, constant=0)])
    _, system01, _ = panel01.vote(None, X2, np.random.default_rng(0))
    np.testing.assert_array_equal(system01, [0, 0])


def test_exclude_removes_vote_but_labels_keep_k_rows():
    panel = Panel([JudgeSpec("a", 0.0), JudgeSpec("b", 0.0),
                   JudgeSpec("c", 0.3, constant=1)])
    labels, system, consensus = panel.vote(POLICY, X2, np.random.default_rng(0),
                                           exclude=frozenset({"c"}))
    assert labels.shape == (3, 2)                      # still K rows
    np.testing.assert_array_equal(labels[2], [1, 1])   # still labels
    np.testing.assert_array_equal(system, [0, 1])      # but is not counted
    np.testing.assert_allclose(consensus, [1.0, 1.0])  # 2 of 2 counted


def test_one_rng_draw_per_vote_keeps_same_seeded_panels_coupled():
    X = np.random.default_rng(3).normal(0.0, 1.5, size=(50, 2))
    pa, pb = default_panel(), default_panel()
    ra, rb = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(3):  # stays coupled across consecutive votes
        la, sa, ca = pa.vote(POLICY, X, ra)
        lb, sb, cb = pb.vote(POLICY, X, rb)
        np.testing.assert_array_equal(la, lb)
        np.testing.assert_array_equal(sa, sb)
        np.testing.assert_allclose(ca, cb)


def test_noise_tensor_drawn_for_all_k_even_with_constant_judge():
    # CRN across panel compositions: the compliant judges see identical draws
    # whether or not judge-4 was swapped for a constant one.
    X = np.random.default_rng(4).normal(0.0, 1.5, size=(40, 2))
    la, _, _ = default_panel().vote(POLICY, X, np.random.default_rng(7))
    lc, _, _ = default_panel(noncompliant=True, constant_class=0).vote(
        POLICY, X, np.random.default_rng(7))
    np.testing.assert_array_equal(la[:4], lc[:4])
    assert (lc[4] == 0).all()
