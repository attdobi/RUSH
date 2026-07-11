"""Policies: the 'document' is a parametric decision boundary.

- LogisticPolicy (binary / genai): theta = (w, b); the boundary is a line.
- PrototypePolicy (multiclass / mnist): theta = one prototype per class
  (LVQ-style); the boundary is the nearest-prototype Voronoi diagram.

The textual-gradient step is a clipped, anchor-weighted parameter update —
step-norm clipping plays the role of the crank's 1-5-discrete-edits rule.
Document distance = L2 in normalized parameter space; the companion measure
decision_disagreement() is the sim analogue of a policy expert reading the
diff ("do these two documents decide differently, and how often?").
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


class LogisticPolicy:
    def __init__(self, w: np.ndarray, b: float):
        self.w = np.asarray(w, dtype=float).copy()
        self.b = float(b)

    # -- reading ------------------------------------------------------------
    def scores(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.scores(X) > 0).astype(int)

    def class_margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed distance to the boundary, positive when x sits on the side
        of its label y. Near zero = contested; negative = inside the wrong
        region (genuinely confusable under this policy)."""
        z = self.scores(X) / (np.linalg.norm(self.w) + _EPS)
        return np.where(np.asarray(y) == 1, z, -z)

    def runner_up(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.predict(X)

    # -- writing ------------------------------------------------------------
    def update(self, X, y, sample_weight, lr: float, clip: float) -> None:
        """One weighted logistic step toward labels y, step-norm clipped."""
        w_s = np.asarray(sample_weight, dtype=float)
        total = w_s.sum()
        if total <= 0:
            return
        w_s = w_s / total
        p = 1.0 / (1.0 + np.exp(-np.clip(self.scores(X), -30, 30)))
        err = p - np.asarray(y, dtype=float)
        step = -lr * np.concatenate([(w_s * err) @ X, [(w_s * err).sum()]])
        norm = np.linalg.norm(step)
        if norm > clip:
            step *= clip / norm
        self.w += step[:2]
        self.b += step[2]

    # -- bookkeeping ---------------------------------------------------------
    def clone(self) -> "LogisticPolicy":
        return LogisticPolicy(self.w, self.b)

    def param_vector(self) -> np.ndarray:
        """Scale-normalized (the decision rule is invariant to positive
        rescaling of (w, b)), so distances compare boundaries, not gains."""
        v = np.concatenate([self.w, [self.b]])
        return v / (np.linalg.norm(self.w) + _EPS)

    @classmethod
    def fit(cls, X, y, steps: int = 300, lr: float = 0.5, l2: float = 1e-3,
            rng: np.random.Generator | None = None) -> "LogisticPolicy":
        rng = rng or np.random.default_rng(0)
        w = rng.normal(0, 0.01, size=2)
        b = 0.0
        yf = np.asarray(y, dtype=float)
        n = len(yf)
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
            err = p - yf
            w -= lr * (err @ X / n + l2 * w)
            b -= lr * err.mean()
        return cls(w, b)


class PrototypePolicy:
    def __init__(self, protos: np.ndarray):
        self.protos = np.asarray(protos, dtype=float).copy()  # (C, 2)

    @property
    def n_classes(self) -> int:
        return self.protos.shape[0]

    def _dists(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[:, None, :] - self.protos[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._dists(X).argmin(axis=1)

    def class_margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Relative margin of label y vs best other class: positive when x is
        closer to its own prototype, ~0 on a boundary, negative when a rival
        prototype is closer."""
        d = self._dists(X)
        idx = np.arange(len(X))
        d_own = d[idx, np.asarray(y)]
        d_masked = d.copy()
        d_masked[idx, np.asarray(y)] = np.inf
        d_rival = d_masked.min(axis=1)
        return (d_rival - d_own) / (d_rival + d_own + _EPS)

    def runner_up(self, X: np.ndarray) -> np.ndarray:
        """Second-choice class — the confusable neighbor (4 for a 9, ...)."""
        d = self._dists(X)
        order = d.argsort(axis=1)
        return order[:, 1]

    def update(self, X, y, sample_weight, lr: float, clip: float,
               push_ratio: float = 0.5) -> None:
        """LVQ step toward human labels: pull the labeled class's prototype
        to each anchor, push the wrongly-predicted prototype away. The total
        movement (Frobenius norm) is clipped — the edit-clipping analogue."""
        w_s = np.asarray(sample_weight, dtype=float)
        total = w_s.sum()
        if total <= 0:
            return
        w_s = w_s / total
        pred = self.predict(np.asarray(X))
        delta = np.zeros_like(self.protos)
        for xi, yi, pi, wi in zip(np.asarray(X), np.asarray(y), pred, w_s):
            delta[yi] += lr * wi * (xi - self.protos[yi])
            if pi != yi:
                delta[pi] -= lr * wi * push_ratio * (xi - self.protos[pi])
        norm = np.linalg.norm(delta)
        if norm > clip:
            delta *= clip / norm
        self.protos += delta

    def clone(self) -> "PrototypePolicy":
        return PrototypePolicy(self.protos)

    def param_vector(self) -> np.ndarray:
        return self.protos.ravel().copy()

    @classmethod
    def fit(cls, X, y, n_classes: int,
            rng: np.random.Generator | None = None) -> "PrototypePolicy":
        rng = rng or np.random.default_rng(0)
        X = np.asarray(X)
        y = np.asarray(y)
        centroid = X.mean(axis=0) if len(X) else np.zeros(2)
        protos = np.zeros((n_classes, 2))
        for c in range(n_classes):
            mask = y == c
            if mask.any():
                protos[c] = X[mask].mean(axis=0)
            else:  # unseen class in the bootstrap: start near the centroid
                protos[c] = centroid + rng.normal(0, 0.25, size=2)
        return cls(protos)


def decision_disagreement(pol_a, pol_b, probe_X: np.ndarray) -> float:
    """Fraction of probe points the two policies decide differently — the
    'expert reads the diff' divergence measure."""
    return float((pol_a.predict(probe_X) != pol_b.predict(probe_X)).mean())


def param_distance(pol_a, pol_b) -> float:
    va, vb = pol_a.param_vector(), pol_b.param_vector()
    return float(np.linalg.norm(va - vb) / np.sqrt(len(va)))


def fit_oracle(ds) -> "LogisticPolicy | PrototypePolicy":
    """The true boundary, fit on ALL ground-truth labels. Used only for
    (a) placing boundary-concentrated noise and (b) measurement (anchor
    margin diagnostics). Never visible to any mitigation."""
    if ds.n_classes == 2:
        return LogisticPolicy.fit(ds.X, ds.y)
    return PrototypePolicy.fit(ds.X, ds.y, ds.n_classes)
