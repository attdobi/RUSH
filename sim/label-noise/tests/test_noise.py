"""noise.apply_noise: counts, geometry, directionality, and no-op modes."""

import numpy as np
import pytest

from labelsim import NoiseConfig, apply_noise, make_genai
from labelsim.policy import fit_oracle

DS = make_genai(n=600, seed=13)
ORACLE = fit_oracle(DS)


def test_exact_flip_count_uniform_and_boundary():
    for model in ("uniform", "boundary"):
        _, flipped = apply_noise(DS, NoiseConfig(model=model, rate=0.15),
                                 seed=13, oracle=ORACLE)
        assert int(flipped.sum()) == round(0.15 * DS.n) == 90


def test_flip_count_respects_eligible_subset():
    eligible = np.arange(0, DS.n, 3)  # 200 items
    _, flipped = apply_noise(DS, NoiseConfig(model="uniform", rate=0.2),
                             seed=7, eligible=eligible, oracle=ORACLE)
    assert int(flipped.sum()) == round(0.2 * len(eligible)) == 40
    assert set(np.flatnonzero(flipped)) <= set(eligible.tolist())


def test_boundary_flips_sit_at_smaller_oracle_margin_than_uniform():
    _, fb = apply_noise(DS, NoiseConfig(model="boundary", rate=0.15),
                        seed=13, oracle=ORACLE)
    _, fu = apply_noise(DS, NoiseConfig(model="uniform", rate=0.15),
                        seed=13, oracle=ORACLE)
    margin_boundary = ORACLE.class_margin(DS.X[fb], DS.y[fb]).mean()
    margin_uniform = ORACLE.class_margin(DS.X[fu], DS.y[fu]).mean()
    assert margin_boundary < margin_uniform


def test_flip_from_restricts_flips_to_that_true_class():
    y_h, flipped = apply_noise(DS, NoiseConfig(model="boundary", rate=0.2,
                                               flip_from=1),
                               seed=7, oracle=ORACLE)
    assert flipped.any()
    assert (DS.y[flipped] == 1).all()
    # rate means fraction of ALL eligible labels (comparable to two-sided
    # cells at the same rate), capped at the flip_from pool size
    n_cls1 = int((DS.y == 1).sum())
    assert int(flipped.sum()) == min(round(0.2 * DS.n), n_cls1)
    assert (y_h[flipped] == 0).all()  # binary: away from class 1


def test_model_none_and_rate_zero_produce_no_flips():
    for cfg in (NoiseConfig(model="none", rate=0.5),
                NoiseConfig(model="boundary", rate=0.0),
                NoiseConfig(model="uniform", rate=0.0)):
        y_h, flipped = apply_noise(DS, cfg, seed=13, oracle=ORACLE)
        assert not flipped.any()
        assert (y_h == DS.y).all()


def test_flip_targets_never_equal_truth_boundary_model():
    y_h, flipped = apply_noise(DS, NoiseConfig(model="boundary", rate=0.3),
                               seed=99, oracle=ORACLE)
    assert flipped.any()
    assert (y_h[flipped] != DS.y[flipped]).all()
    assert (y_h[~flipped] == DS.y[~flipped]).all()


def test_flip_targets_never_equal_truth_uniform_model():
    y_h, flipped = apply_noise(DS, NoiseConfig(model="uniform", rate=0.3),
                               seed=99, oracle=ORACLE)
    assert flipped.any()
    assert (y_h[flipped] != DS.y[flipped]).all()


def test_uniform_flipped_mask_matches_actual_label_changes():
    y_h, flipped = apply_noise(DS, NoiseConfig(model="uniform", rate=0.15),
                               seed=13, oracle=ORACLE)
    assert int((y_h != DS.y).sum()) == int(flipped.sum())
