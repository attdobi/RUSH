"""confidence: Bayesian posterior direction/magnitude, adjudication freeze,
anchor_weight modes and pinned overrides."""

import numpy as np
import pytest

from labelsim import ConfidenceConfig, HumanConfidence, anchor_weight


def test_disagreement_lowers_w_agreement_raises_w():
    conf = HumanConfidence(3, ConfidenceConfig())
    w0 = conf.w.copy()
    assert np.allclose(w0, 0.9)  # prior c0
    conf.observe(np.array([0]), np.array([True]), np.array([1.0]))
    conf.observe(np.array([1]), np.array([False]), np.array([1.0]))
    assert conf.w[0] < w0[0]
    assert conf.w[1] > w0[1]
    assert conf.w[2] == pytest.approx(w0[2])  # unobserved item untouched


def test_stronger_consensus_moves_posterior_more():
    dis = HumanConfidence(2, ConfidenceConfig())
    dis.observe(np.array([0, 1]), np.array([True, True]), np.array([1.0, 0.6]))
    assert dis.w[0] < dis.w[1] < 0.9  # unanimous disagreement digs deeper
    agr = HumanConfidence(2, ConfidenceConfig())
    agr.observe(np.array([0, 1]), np.array([False, False]), np.array([1.0, 0.6]))
    assert agr.w[0] > agr.w[1] > 0.9  # unanimous agreement lifts higher


def test_mark_adjudicated_sets_high_conf_and_freezes_item():
    cfg = ConfidenceConfig()
    conf = HumanConfidence(2, cfg)
    conf.observe(np.array([0, 1]), np.array([True, True]), np.array([0.8, 0.8]))
    assert conf.queue_score[0] > 0
    conf.mark_adjudicated(np.array([0]))
    assert conf.w[0] == pytest.approx(cfg.adjudicated_conf)  # ~0.99
    assert conf.resolved[0] and not conf.resolved[1]
    assert conf.queue_score[0] == 0.0
    log_odds_before = conf.log_odds.copy()
    q1_before = conf.queue_score[1]
    conf.observe(np.array([0, 1]), np.array([True, True]), np.array([1.0, 1.0]))
    assert conf.log_odds[0] == log_odds_before[0]   # resolved: no more updates
    assert conf.queue_score[0] == 0.0               # resolved: leaves the queue
    assert conf.log_odds[1] < log_odds_before[1]    # live item still updates
    assert conf.queue_score[1] > q1_before


def test_suspicion_is_one_minus_w():
    conf = HumanConfidence(2, ConfidenceConfig())
    conf.observe(np.array([0]), np.array([True]), np.array([1.0]))
    np.testing.assert_allclose(conf.suspicion(), 1.0 - conf.w)


def test_anchor_weight_all_modes():
    w = np.array([0.9, 0.4, 0.6])
    np.testing.assert_allclose(anchor_weight("off", w), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(anchor_weight("deweight", w), w)
    np.testing.assert_allclose(anchor_weight("deweight_hard", w), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(anchor_weight("upweight", w, amp=2.0),
                               1.0 + 2.0 * (1.0 - w))
    with pytest.raises(ValueError):
        anchor_weight("nope", w)


def test_anchor_weight_pinned_override():
    w = np.array([0.9, 0.4, 0.6])
    idx = np.array([10, 11, 12])
    out = anchor_weight("deweight", w, pinned={11: 5.0}, idx=idx)
    np.testing.assert_allclose(out, [0.9, 5.0, 0.6])
    # pinned without idx is a no-op
    np.testing.assert_allclose(anchor_weight("deweight", w, pinned={11: 5.0}), w)
