"""The weighting methodology under test: Bayesian confidence per human label.

Key design commitment carried over from RUSH: a single human label (N=1) gets
a NON-PERFECT prior, c0 = 0.9 by default. Every time the panel's system vote
is observed against that label, the posterior updates:

    odds(human right) *= [ P(evidence | right) / P(evidence | wrong) ]^s

where s is the panel's consensus share (a strong unanimous disagreement is
stronger evidence than a 3-2 split), and the likelihoods come from two ASSUMED
constants — p_catch = P(panel disagrees | human wrong) and
p_false = P(panel disagrees | human right). These are operator priors about
panel quality, not measurements against truth: no oracle leaks in.

The posterior w = P(human label correct | history) is then used three ways:
- deweight   universe: anchor weight = w        (learn to ignore suspects)
- upweight   universe: anchor weight = 1 + (1-w) (hyper-focus, human assumed right)
- suspicion  = 1 - w ranks the re-adjudication queue and scores mislabel
  detection (AUROC vs the true flipped mask — measurement only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOGIT_MAX = 12.0  # keep posteriors away from exactly 0/1


@dataclass
class ConfidenceConfig:
    c0: float = 0.9        # prior P(single human label is correct), N=1
    p_catch: float = 0.75  # assumed P(system vote disagrees | human wrong)
    p_false: float = 0.25  # assumed P(system vote disagrees | human right)
    adjudicated_conf: float = 0.99  # confidence after SME full-attention review


class HumanConfidence:
    def __init__(self, n: int, cfg: ConfidenceConfig):
        self.cfg = cfg
        self.log_odds = np.full(n, _logit(cfg.c0))
        self.n_seen = np.zeros(n, dtype=int)
        self.queue_score = np.zeros(n)   # cumulative disagreement mass Σ s·1[dis]
        self.resolved = np.zeros(n, dtype=bool)  # adjudicated items leave the queue
        # highest suspicion ever reached — the detector's score of record.
        # NOT reset by adjudication, so detection AUROC is computed on a
        # stable population (re-adjudication arms would otherwise censor
        # exactly the items their detector caught).
        self.peak_suspicion = np.full(n, 1.0 - cfg.c0)

    @property
    def w(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def observe(self, idx: np.ndarray, disagree: np.ndarray, consensus: np.ndarray):
        """Sequential Bayes update for the items just labeled by the panel."""
        idx = np.asarray(idx)
        dis = np.asarray(disagree, dtype=bool)
        s = np.asarray(consensus, dtype=float)
        llr_dis = np.log(self.cfg.p_false / self.cfg.p_catch)
        llr_agr = np.log((1 - self.cfg.p_false) / (1 - self.cfg.p_catch))
        llr = np.where(dis, llr_dis, llr_agr) * s
        live = ~self.resolved[idx]  # adjudicated labels are settled
        upd = np.zeros(len(idx))
        upd[live] = llr[live]
        np.add.at(self.log_odds, idx, upd)
        np.clip(self.log_odds, -_LOGIT_MAX, _LOGIT_MAX, out=self.log_odds)
        np.add.at(self.n_seen, idx, 1)
        q = np.where(live & dis, s, 0.0)
        np.add.at(self.queue_score, idx, q)
        np.maximum.at(self.peak_suspicion, idx, 1.0 - self.w[idx])

    def mark_adjudicated(self, idx: np.ndarray):
        self.log_odds[idx] = _logit(self.cfg.adjudicated_conf)
        self.resolved[idx] = True
        self.queue_score[idx] = 0.0

    def suspicion(self) -> np.ndarray:
        return 1.0 - self.w


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def anchor_weight(mode: str, w: np.ndarray, amp: float = 1.0,
                  pinned: dict | None = None,
                  idx: np.ndarray | None = None) -> np.ndarray:
    """Per-anchor weight from the confidence posterior.

    off           -> 1 (trust every human label fully)
    deweight      -> w (soft: learn to ignore in proportion to suspicion)
    deweight_hard -> 1 if w >= 0.5 else 0 (drop suspects outright)
    upweight      -> 1 + amp*(1-w) (parallel universe: hyper-focus)

    `pinned` maps dataset index -> forced weight (per-point influence probes).
    """
    w = np.asarray(w, dtype=float)
    if mode == "off":
        out = np.ones_like(w)
    elif mode == "deweight":
        out = w.copy()
    elif mode == "deweight_hard":
        out = (w >= 0.5).astype(float)
    elif mode == "upweight":
        out = 1.0 + amp * (1.0 - w)
    else:
        raise ValueError(f"unknown weighting mode {mode!r}")
    if pinned and idx is not None:
        for j, ds_i in enumerate(np.asarray(idx)):
            if int(ds_i) in pinned:
                out[j] = pinned[int(ds_i)]
    return out
