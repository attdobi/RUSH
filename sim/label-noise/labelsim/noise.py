"""Human-label noise models.

The empirical picture from RUSH re-adjudication: of LLM-vs-SME disagreements
that went to review, ~33% were overturned in favor of the LLM (up to ~44%
for the sensitive-content area) — human labels are materially imperfect, and
the mistakes are boundary mistakes: adult vs racy gets confused, the puppy
never does. The noise models encode that geometry:

- uniform:  flip probability independent of position (the worst case the
            boundary model is compared against — interior mislabels are the
            catastrophic anchors).
- boundary: flip probability ~ exp(-margin / tau) under the ORACLE boundary,
            normalized to an overall rate; the flip target is the runner-up
            class (the confusable neighbor), never a random distant class.
- pair:     boundary noise restricted to confusable class pairs
            (mnist default: 4/9, 3/5, 7/1, 3/8).

The oracle is used only to PLACE noise (experiment design). No mitigation
ever sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import MNIST_CONFUSABLE_PAIRS
from .policy import fit_oracle


@dataclass
class NoiseConfig:
    model: str = "boundary"        # uniform | boundary | pair | none
    rate: float = 0.15             # overall fraction of labels flipped
    tau: float = 0.25              # boundary concentration (smaller = tighter)
    # 'both' is the realistic default: the same SMEs label train AND test, so
    # the gate's yardstick is corrupted too. 'train' isolates optimizer
    # damage (clean test -> the gate is secretly an oracle and rescues the
    # run); 'test' isolates gate damage (S4).
    target: str = "both"
    # Directional confusion: only items whose TRUE class == flip_from get
    # flipped. The count is still rate x len(all eligible) — comparable to
    # two-sided cells at the same rate — capped at the class pool. Two-sided
    # noise largely cancels at a boundary; the adult-vs-racy failure mode is
    # one-way and actually drags it. None = two-sided.
    flip_from: int | None = None
    pairs: tuple = MNIST_CONFUSABLE_PAIRS


def _flip_targets(ds, idx: np.ndarray, oracle, rng: np.random.Generator) -> np.ndarray:
    """Flip to the confusable neighbor: the oracle's runner-up class, unless
    that equals the true label (deep inside a rival region), then the oracle's
    own prediction, else a random other class."""
    ru = oracle.runner_up(ds.X[idx])
    pred = oracle.predict(ds.X[idx])
    target = np.where(ru != ds.y[idx], ru, pred)
    clash = target == ds.y[idx]
    if clash.any():
        target = target.copy()
        target[clash] = _random_other_class(ds.y[idx][clash], ds.n_classes, rng)
    return target


def _random_other_class(y: np.ndarray, n_classes: int, rng) -> np.ndarray:
    """Uniform draw over the n_classes-1 classes != y (no wraparound no-ops)."""
    draw = rng.integers(0, n_classes - 1, size=len(y))
    return draw + (draw >= y)


def apply_noise(ds, cfg: NoiseConfig, seed: int, eligible: np.ndarray | None = None,
                oracle=None):
    """Returns (y_human (n,), flipped (n,) bool). Only indices in `eligible`
    (default: all) may flip; the count is round(rate * len(eligible))."""
    y_human = ds.y.copy()
    flipped = np.zeros(ds.n, dtype=bool)
    if cfg.model == "none" or cfg.rate <= 0:
        return y_human, flipped
    rng = np.random.default_rng([int(seed), 404])
    eligible = np.arange(ds.n) if eligible is None else np.asarray(eligible)
    # rate always means "fraction of the eligible split flipped" so one-way
    # (flip_from) and two-sided cells are comparable at the same rate; the
    # one-way pool is smaller, so the count caps at the pool size.
    n_flip = int(round(cfg.rate * len(eligible)))
    if cfg.flip_from is not None:
        eligible = eligible[ds.y[eligible] == cfg.flip_from]
        n_flip = min(n_flip, len(eligible))
    if n_flip == 0 or len(eligible) == 0:
        return y_human, flipped
    oracle = oracle or fit_oracle(ds)

    if cfg.model == "uniform":
        pick = rng.choice(eligible, size=n_flip, replace=False)
        targets = _random_other_class(ds.y[pick], ds.n_classes, rng)
    elif cfg.model in ("boundary", "pair"):
        pool = eligible
        if cfg.model == "pair":
            ru = oracle.runner_up(ds.X[eligible])
            pairset = {frozenset(p) for p in cfg.pairs}
            mask = np.array([
                frozenset((int(a), int(b))) in pairset
                for a, b in zip(ds.y[eligible], ru)
            ])
            pool = eligible[mask]
            if len(pool) == 0:
                return y_human, flipped
            n_flip = min(n_flip, len(pool))
        margin = oracle.class_margin(ds.X[pool], ds.y[pool])
        weights = np.exp(-np.clip(margin, 0.0, None) / cfg.tau)
        p = weights / weights.sum()
        n_flip = min(n_flip, int((p > 0).sum()))
        pick = rng.choice(pool, size=n_flip, replace=False, p=p)
        targets = _flip_targets(ds, pick, oracle, rng)
    else:
        raise ValueError(f"unknown noise model {cfg.model!r}")

    y_human[pick] = targets
    flipped[pick] = True
    return y_human, flipped
