"""Datasets: 2D worlds where the policy is a visible decision boundary.

- genai: synthetic binary world shaped like the RUSH GenAI demo — an easy mass
  of real photos, an easy mass of obvious renders, and a contested ridge of
  photo-real generations where the classes genuinely interleave. Mislabels in
  the wild live on that ridge (adult vs racy), never in the easy mass (puppy).
- mnist: the repo's real MNIST archive, class-balanced subsample, PCA to 2D.
  Confusable pairs (4/9, 3/5, 7/1, 3/8) overlap naturally in the projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# sim/label-noise/labelsim/datasets.py -> parents[2] == the RUSH repo root
REPO_ROOT = Path(__file__).resolve().parents[2].parent
MNIST_NPZ = REPO_ROOT / "data" / "images" / "mnist-classification" / "mnist_full.npz"

MNIST_CONFUSABLE_PAIRS = ((4, 9), (3, 5), (7, 1), (3, 8))


@dataclass
class Dataset:
    name: str
    X: np.ndarray            # (n, 2) float
    y: np.ndarray            # (n,) int ground truth
    class_names: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1


def make_genai(n: int = 600, seed: int = 13, hard_frac: float = 0.35) -> Dataset:
    """Binary world: 0 = not_gen_ai, 1 = gen_ai.

    Each class is (1 - hard_frac) easy mass far from the boundary plus
    hard_frac contested mass on the ridge at x ~ 0 where classes interleave.
    """
    rng = np.random.default_rng([int(seed), 101])
    halves = [n // 2, n - n // 2]
    xs, ys = [], []
    for cls, n_cls, easy_mu, hard_mu in (
        (0, halves[0], (-2.2, 0.0), (-0.30, 0.0)),
        (1, halves[1], (+2.2, 0.3), (+0.30, 0.1)),
    ):
        n_hard = int(round(n_cls * hard_frac))
        n_easy = n_cls - n_hard
        easy = rng.normal(easy_mu, (0.9, 0.9), size=(n_easy, 2))
        # contested ridge: tight in x (interleaves across 0), tall in y
        hard = rng.normal(hard_mu, (0.55, 1.1), size=(n_hard, 2))
        xs.append(np.vstack([easy, hard]))
        ys.append(np.full(n_cls, cls))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(n)
    return Dataset(
        name="genai",
        X=X[perm].astype(float),
        y=y[perm].astype(int),
        class_names=("not_gen_ai", "gen_ai"),
        meta={"seed": seed, "hard_frac": hard_frac},
    )


def make_mnist(
    n_per_class: int = 40,
    seed: int = 13,
    npz_path: Path | str | None = None,
    realizable: bool = True,
) -> Dataset:
    """Real MNIST, class-balanced subsample, PCA(50) -> Fisher LDA to 2D.

    realizable=True (default) relabels each point with the class-prototype
    boundary's own prediction (teacher-student setup): the 2D geometry and
    confusion adjacency come from real MNIST, but ground truth is realizable
    in-world, so the truth ceiling is 1.0 and learning dynamics are not
    drowned by 2D-projection Bayes error. realizable=False keeps the original
    digit labels (ceiling ~0.57 for a prototype policy in 2D)."""
    path = Path(npz_path) if npz_path else MNIST_NPZ
    arch = np.load(path)
    labels = arch["labels"]
    rng = np.random.default_rng([int(seed), 202])
    picks = []
    for cls in range(10):
        idx = np.flatnonzero(labels == cls)
        picks.append(rng.choice(idx, size=n_per_class, replace=False))
    picks = np.concatenate(picks)
    imgs = arch["images"][picks].reshape(len(picks), -1).astype(float) / 255.0
    y = labels[picks].astype(int)
    # PCA to 50 dims, then Fisher LDA to 2: raw-pixel PCA-2D leaves ten
    # classes hopelessly overlapped, while the discriminant plane keeps the
    # real confusion structure (4/9, 3/5, 7/1) without drowning everything.
    # Truth is used to DESIGN the world here; the loop never sees it.
    centered = imgs - imgs.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    Z = centered @ vt[:50].T
    overall = Z.mean(axis=0)
    d = Z.shape[1]
    sw = np.zeros((d, d))
    sb = np.zeros((d, d))
    for cls in range(10):
        Zc = Z[y == cls]
        mu = Zc.mean(axis=0)
        dev = Zc - mu
        sw += dev.T @ dev
        gap = (mu - overall)[:, None]
        sb += len(Zc) * (gap @ gap.T)
    sw += 1e-3 * np.trace(sw) / d * np.eye(d)
    evals, evecs = np.linalg.eig(np.linalg.solve(sw, sb))
    order = np.argsort(-evals.real)
    comps = evecs[:, order[:2]].real.T
    signs = np.sign(comps[np.arange(2), np.abs(comps).argmax(axis=1)])
    comps = comps * signs[:, None]
    X = Z @ comps.T
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    if realizable:
        protos = np.vstack([X[y == c].mean(axis=0) for c in range(10)])
        d = np.linalg.norm(X[:, None, :] - protos[None, :, :], axis=2)
        y = d.argmin(axis=1)
    perm = rng.permutation(len(picks))
    return Dataset(
        name="mnist",
        X=X[perm],
        y=y[perm],
        class_names=tuple(str(d) for d in range(10)),
        meta={"seed": seed, "n_per_class": n_per_class, "realizable": realizable},
    )


def make_dataset(name: str, seed: int = 13, **kw) -> Dataset:
    if name == "genai":
        return make_genai(seed=seed, **kw)
    if name == "mnist":
        return make_mnist(seed=seed, **kw)
    raise ValueError(f"unknown dataset {name!r}")


def split_indices(n: int, seed: int, test_pool_frac: float = 0.35):
    """Seeded (dev_idx, test_pool_idx). Test pool is held out of training and
    re-adjudication-on-train; the fixed or per-cycle test set draws from it."""
    rng = np.random.default_rng([int(seed), 303])
    perm = rng.permutation(n)
    n_test = int(round(n * test_pool_frac))
    return perm[n_test:].copy(), perm[:n_test].copy()


def probe_grid(ds: Dataset, m: int = 60) -> np.ndarray:
    """Fixed (m*m, 2) grid over the padded data bbox — the 'document space'
    probe used to measure decision disagreement between two policies."""
    lo = ds.X.min(axis=0)
    hi = ds.X.max(axis=0)
    pad = 0.1 * (hi - lo)
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], m)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], m)
    xx, yy = np.meshgrid(gx, gy)
    return np.column_stack([xx.ravel(), yy.ravel()])
