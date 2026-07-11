"""Budgeted SME re-adjudication.

Queue ordering is the design axis under test (it's the same question as the
anchor-sampling strategy — how do we spend scarce SME attention?):

- stack_rank: top-B by cumulative disagreement mass (production's default)
- pps:        sample B with probability proportional to that mass
              (probability-proportional-to-size — the winner's-curse hedge)
- random:     uniform over anything with nonzero mass
- off:        no re-adjudication

The SME at full attention is imperfect too: with probability q_sme the review
returns ground truth, otherwise it keeps the current label. Overturn events
(label actually changed) are tracked so the sim's emergent overturn rate can
be sanity-checked against the empirical 33-44%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReadjConfig:
    strategy: str = "off"        # off | stack_rank | pps | random
    budget: int = 5              # train-pool reviews per cycle
    q_sme: float = 0.95          # P(full-attention review returns truth)
    include_test: bool = False   # also review the test partition
    test_budget: int = 5


def select_queue(scores: np.ndarray, eligible: np.ndarray, budget: int,
                 strategy: str, rng: np.random.Generator) -> np.ndarray:
    """Pick up to `budget` dataset indices from `eligible` by queue strategy.
    Items need nonzero accumulated disagreement mass to be reviewable."""
    scores_e = scores[eligible]
    live = scores_e > 0
    if strategy == "off" or budget <= 0 or not live.any():
        return np.empty(0, dtype=int)
    pool = eligible[live]
    pool_scores = scores_e[live]
    k = min(budget, len(pool))
    if strategy == "stack_rank":
        order = np.argsort(-pool_scores, kind="stable")
        return pool[order[:k]]
    if strategy == "pps":
        p = pool_scores / pool_scores.sum()
        return rng.choice(pool, size=k, replace=False, p=p)
    if strategy == "random":
        return rng.choice(pool, size=k, replace=False)
    raise ValueError(f"unknown re-adjudication strategy {strategy!r}")


def adjudicate(idx: np.ndarray, y_human: np.ndarray, y_true: np.ndarray,
               q_sme: float, rng: np.random.Generator):
    """Review items in place. Returns (n_reviewed, n_overturned)."""
    idx = np.asarray(idx)
    if len(idx) == 0:
        return 0, 0
    hits = rng.random(len(idx)) < q_sme
    new_labels = np.where(hits, y_true[idx], y_human[idx])
    overturned = int((new_labels != y_human[idx]).sum())
    y_human[idx] = new_labels
    return len(idx), overturned
