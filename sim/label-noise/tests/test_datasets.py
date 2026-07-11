"""datasets: genai geometry/reproducibility, MNIST realizable world, splits."""

import numpy as np
import pytest

from labelsim import make_genai, make_mnist, split_indices
from labelsim.datasets import MNIST_NPZ
from labelsim.metrics import macro_f1
from labelsim.policy import fit_oracle


def test_genai_shapes_and_label_values():
    ds = make_genai(n=601, seed=4)  # odd n exercises the halves split
    assert ds.X.shape == (601, 2)
    assert ds.y.shape == (601,)
    assert ds.X.dtype == np.float64
    assert set(np.unique(ds.y)) == {0, 1}
    np.testing.assert_array_equal(np.bincount(ds.y), [300, 301])
    assert ds.n == 601
    assert ds.n_classes == 2
    assert ds.class_names == ("not_gen_ai", "gen_ai")


def test_genai_reproducible_same_seed_identical():
    a = make_genai(seed=21)
    b = make_genai(seed=21)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    c = make_genai(seed=22)
    assert not np.array_equal(a.X, c.X)


@pytest.mark.skipif(not MNIST_NPZ.exists(), reason="repo MNIST npz missing")
def test_mnist_loads_and_realizable_truth_is_learnable():
    ds = make_mnist(n_per_class=30, seed=3, realizable=True)
    assert ds.X.shape == (300, 2)
    assert set(np.unique(ds.y)) <= set(range(10))
    assert ds.name == "mnist"
    # realizable=True: ground truth is the prototype boundary's own labeling,
    # so the oracle (nearest-prototype fit on truth) must score very high.
    oracle = fit_oracle(ds)
    f1 = macro_f1(ds.y, oracle.predict(ds.X), ds.n_classes)
    assert f1 >= 0.85


def test_split_indices_disjoint_and_covers_all_n():
    dev, test = split_indices(600, seed=13)
    assert len(test) == round(600 * 0.35) == 210
    assert len(dev) + len(test) == 600
    dev_s, test_s = set(dev.tolist()), set(test.tolist())
    assert dev_s.isdisjoint(test_s)
    assert dev_s | test_s == set(range(600))
    # seeded: same seed reproduces, different seed differs
    dev2, test2 = split_indices(600, seed=13)
    np.testing.assert_array_equal(dev, dev2)
    np.testing.assert_array_equal(test, test2)
    dev3, _ = split_indices(600, seed=14)
    assert not np.array_equal(dev, dev3)
