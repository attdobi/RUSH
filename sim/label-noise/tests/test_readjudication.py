"""readjudication: queue strategies honor budget/score>0, SME review model."""

import numpy as np

from labelsim import adjudicate, select_queue


def test_stack_rank_returns_top_scored():
    scores = np.array([0.0, 3.0, 1.0, 7.0, 0.5, 2.0])
    eligible = np.arange(6)
    picked = select_queue(scores, eligible, budget=3, strategy="stack_rank",
                          rng=np.random.default_rng(0))
    np.testing.assert_array_equal(picked, [3, 1, 5])  # descending score


def test_pps_and_random_respect_budget_and_only_pick_positive_scores():
    scores = np.zeros(20)
    live = [2, 5, 7, 11, 13]
    scores[live] = [1.0, 2.0, 3.0, 4.0, 5.0]
    eligible = np.arange(20)
    for strategy in ("pps", "random"):
        picked = select_queue(scores, eligible, budget=3, strategy=strategy,
                              rng=np.random.default_rng(1))
        assert len(picked) == 3
        assert len(set(picked.tolist())) == 3          # no repeats
        assert (scores[picked] > 0).all()              # only score>0 items
        # budget above pool size: everything live, nothing else
        picked_all = select_queue(scores, eligible, budget=50, strategy=strategy,
                                  rng=np.random.default_rng(2))
        assert sorted(picked_all.tolist()) == live


def test_off_zero_budget_or_dead_pool_return_empty():
    scores = np.zeros(5)
    eligible = np.arange(5)
    assert len(select_queue(scores, eligible, 3, "stack_rank",
                            np.random.default_rng(0))) == 0   # no mass anywhere
    scores[0] = 1.0
    assert len(select_queue(scores, eligible, 0, "stack_rank",
                            np.random.default_rng(0))) == 0   # no budget
    assert len(select_queue(scores, eligible, 3, "off",
                            np.random.default_rng(0))) == 0   # strategy off


def test_adjudicate_perfect_sme_restores_truth_and_counts():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_human = np.array([1, 1, 1, 0, 1, 0])   # wrong at 0, 2, 3
    idx = np.array([0, 2, 4])                # two wrong, one already right
    reviewed, overturned = adjudicate(idx, y_human, y_true, q_sme=1.0,
                                      rng=np.random.default_rng(0))
    assert (reviewed, overturned) == (3, 2)
    np.testing.assert_array_equal(y_human[idx], y_true[idx])  # truth restored
    assert y_human[3] == 0                    # unreviewed item stays wrong


def test_adjudicate_qsme_zero_and_empty_queue():
    y_true = np.array([0, 1])
    y_human = np.array([1, 0])
    reviewed, overturned = adjudicate(np.array([0, 1]), y_human, y_true,
                                      q_sme=0.0, rng=np.random.default_rng(0))
    assert (reviewed, overturned) == (2, 0)
    np.testing.assert_array_equal(y_human, [1, 0])  # nothing changed
    assert adjudicate(np.array([], dtype=int), y_human, y_true, 1.0,
                      np.random.default_rng(0)) == (0, 0)
