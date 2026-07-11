"""The judge panel: noisy readers of the current policy.

A compliant judge labels x by applying the policy boundary at a perturbed
location x + bias + eps, eps ~ N(0, sigma^2 I): sigma is (inverse) capacity,
bias is systematic style. A non-compliant judge (the qwen failure mode)
ignores the policy and outputs a constant class — it has zero policy-text
gradient by construction.

CRN discipline: vote() takes a Generator and draws exactly one (K, n, 2)
noise tensor per call, so paired universes that pass the same per-cycle
stream stay coupled draw-for-draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class JudgeSpec:
    name: str
    sigma: float
    bias: tuple = (0.0, 0.0)
    constant: int | None = None  # non-compliant: always outputs this class


DEFAULT_SIGMAS = (0.15, 0.25, 0.35, 0.50, 0.70)


def default_panel(noncompliant: bool = False, constant_class: int = 0) -> "Panel":
    specs = [
        JudgeSpec(name=f"judge-{i}", sigma=s, bias=(0.0, 0.0))
        for i, s in enumerate(DEFAULT_SIGMAS)
    ]
    if noncompliant:
        specs[-1] = JudgeSpec(name="judge-4-noncompliant", sigma=0.3,
                              constant=constant_class)
    return Panel(specs)


@dataclass
class Panel:
    specs: list = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.specs)

    def vote(self, policy, X: np.ndarray, rng: np.random.Generator,
             exclude: frozenset = frozenset()):
        """Returns (labels (K, n), system (n,), consensus (n,)).

        System = majority over judges not in `exclude` (compliance deweight);
        ties break to the smallest class id (deterministic). Consensus is the
        top-vote share among counted judges.
        """
        n = X.shape[0]
        noise = rng.normal(0.0, 1.0, size=(self.k, n, 2))
        labels = np.empty((self.k, n), dtype=int)
        for j, spec in enumerate(self.specs):
            if spec.constant is not None:
                labels[j] = spec.constant
            else:
                perturbed = X + np.asarray(spec.bias) + spec.sigma * noise[j]
                labels[j] = policy.predict(perturbed)
        counted = [j for j in range(self.k) if self.specs[j].name not in exclude]
        n_classes = max(int(labels.max()) + 1, 2)
        counts = np.zeros((n_classes, n), dtype=int)
        for j in counted:
            np.add.at(counts, (labels[j], np.arange(n)), 1)
        system = counts.argmax(axis=0)  # argmax ties -> smallest class id
        consensus = counts.max(axis=0) / max(len(counted), 1)
        return labels, system, consensus
