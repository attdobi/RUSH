"""engine: determinism, CRN coupling, clean-twin divergence, gate semantics,
test resampling, truth-blindness of the mitigation path, point influence."""

import numpy as np
import pytest

import labelsim.engine as engine
from labelsim import (CrankConfig, NoiseConfig, ReadjConfig, point_influence,
                      run_crank, run_pair)
from labelsim.datasets import Dataset
from labelsim.engine import build_world, clean_labels, divergence_series
from labelsim.policy import fit_oracle


def _cfg(**kw):
    base = dict(dataset="genai", seed=11, cycles=8, train_batch=40, test_n=80,
                noise=NoiseConfig(model="boundary", rate=0.25, target="both"))
    base.update(kw)
    return CrankConfig(**base)


def test_run_crank_is_deterministic():
    r1 = run_crank(_cfg())
    r2 = run_crank(_cfg())
    assert r1["series"]["policy_f1_true"] == r2["series"]["policy_f1_true"]
    assert r1["series"]["sys_f1_human_test"] == r2["series"]["sys_f1_human_test"]
    assert r1["series"]["accepted"] == r2["series"]["accepted"]
    np.testing.assert_array_equal(r1["policy"].param_vector(),
                                  r2["policy"].param_vector())


def test_crn_identical_universes_never_diverge():
    _, _, div = run_pair(_cfg(), a={}, b={})
    assert all(d == 0.0 for d in div["decision_disagreement"])
    assert all(d == 0.0 for d in div["param_dist"])
    assert all(g == 0.0 for g in div["f1_gap"])


def test_noisy_vs_clean_twin_diverges_under_one_way_noise():
    cfg = _cfg(cycles=10, noise=NoiseConfig(model="boundary", rate=0.3,
                                            target="both", flip_from=1))
    ds, labels = build_world(cfg)
    res_noisy = run_crank(cfg, ds=ds, labels=labels)
    res_clean = run_crank(cfg, ds=ds, labels=clean_labels(ds))
    div = divergence_series(res_noisy, res_clean)
    assert div["decision_disagreement"][0] == 0.0  # shared label-free v0
    assert max(div["decision_disagreement"]) > 0.0  # ...then the noise bites


def test_gate_off_accepts_exactly_the_anchored_cycles():
    res = run_crank(_cfg(gate="off"))
    accepted = res["series"]["accepted"]
    n_anchors = res["series"]["n_anchors"]
    assert accepted[0] is None  # k=0 baseline row
    for acc, n_anch in zip(accepted[1:], n_anchors[1:]):
        assert acc == (n_anch > 0)


def test_resample_test_changes_trace_but_stays_deterministic():
    fixed = run_crank(_cfg())
    res1 = run_crank(_cfg(resample_test=True))
    res2 = run_crank(_cfg(resample_test=True))
    assert res1["series"]["sys_f1_human_test"] == res2["series"]["sys_f1_human_test"]
    assert res1["series"]["sys_f1_human_test"] != fixed["series"]["sys_f1_human_test"]


def _garbage_twin(ds, seed=5):
    rng = np.random.default_rng(seed)
    return Dataset(name=ds.name, X=ds.X.copy(),
                   y=ds.y[rng.permutation(ds.n)].copy(),
                   class_names=ds.class_names, meta=dict(ds.meta))


def test_mitigation_path_never_reads_truth(monkeypatch):
    """With the truth-fit oracle pinned (it seeds v0 and places noise — both
    documented experiment-design uses), replacing ds.y with garbage must leave
    every human-label-driven series untouched; only oracle-measured metrics
    may move. Gate note: `accepted` reads only human labels; `gate_cell` is
    NOT ds.y-invariant by design (its second letter encodes oracle agreement).
    """
    cfg = _cfg(weighting="deweight", readj=ReadjConfig(strategy="off"))
    ds, labels = build_world(cfg)
    y_h, flipped = labels
    ds_garbage = _garbage_twin(ds)
    assert not np.array_equal(ds.y, ds_garbage.y)
    true_oracle = fit_oracle(ds)
    monkeypatch.setattr(engine, "fit_oracle", lambda _ds: true_oracle)
    res_a = run_crank(cfg, ds=ds, labels=(y_h.copy(), flipped.copy()))
    res_b = run_crank(cfg, ds=ds_garbage, labels=(y_h.copy(), flipped.copy()))
    # human-facing loop signal: identical
    assert res_a["series"]["sys_f1_human_test"] == res_b["series"]["sys_f1_human_test"]
    assert res_a["series"]["accepted"] == res_b["series"]["accepted"]
    assert res_a["series"]["n_anchors"] == res_b["series"]["n_anchors"]
    np.testing.assert_array_equal(res_a["policy"].param_vector(),
                                  res_b["policy"].param_vector())
    # oracle-measured metrics: different (they DO read ds.y)
    assert res_a["series"]["policy_f1_true"] != res_b["series"]["policy_f1_true"]


# NOTE: v0 is deliberately truth-anchored (a seeded distortion of the oracle
# boundary, disclosed in _make_v0's docstring) — it is label-NOISE-free and
# mode-independent, not label-free. The monkeypatched test above pins the
# oracle to prove the LOOP's own signal never reads ds.y.


def test_point_influence_records_and_pinned_weights_matter():
    cfg = _cfg(cycles=8, noise=NoiseConfig(model="boundary", rate=0.3,
                                           target="both"))
    records = point_influence(cfg, m_suspects=4, m_controls=2, horizon=8)
    assert len(records) == 6
    expected_keys = {"idx", "influence_dd", "influence_f1_gap", "suspicion",
                     "is_flipped", "times_anchored"}
    for r in records:
        assert set(r) == expected_keys
        assert isinstance(r["idx"], int)
        assert isinstance(r["is_flipped"], bool)
        assert r["influence_dd"] >= 0.0
        assert 0.0 <= r["suspicion"] <= 1.0
    suspects = records[:4]  # ordered suspects-then-controls
    assert max(r["influence_dd"] for r in suspects) > 0.0


def test_series_and_snapshot_lengths_are_cycles_plus_one():
    cfg = _cfg(cycles=5)
    res = run_crank(cfg)
    assert len(res["snapshots"]) == cfg.cycles + 1
    for key, vals in res["series"].items():
        assert len(vals) == cfg.cycles + 1, f"series {key!r} has wrong length"
