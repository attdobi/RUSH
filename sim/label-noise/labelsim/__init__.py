"""labelsim — simulation of the RUSH crank under human-label noise.

The real crank optimizes a policy document against SME labels: a judge panel
reads the policy and labels images, panel-vs-SME misalignment picks anchors,
a drafter edits the policy, a gate accepts/rejects on a test partition.
Human labels are the reward signal (RLHF both ways) — so mislabeled items
become anchor points and can steer the policy toward the wrong boundary.

This package abstracts that loop into a 2D geometry where everything is
measurable against ground truth:

- the policy document  -> a parametric decision boundary (logistic for the
  GenAI binary task, LVQ prototypes for MNIST 10-class)
- the judge panel      -> noisy readers of the current boundary (per-judge
  perception noise + bias; optional non-compliant constant judge)
- textual gradient     -> a clipped, anchor-weighted parameter step
  ("1-5 discrete edits" ~ step-norm clipping)
- document distance    -> distance in normalized parameter space plus
  decision-disagreement rate on a probe grid (the sim analogue of doc-vector
  distance / an expert reading the diff)
- human labels         -> ground truth corrupted by a noise model; mislabels
  concentrate near the true boundary (adult-vs-racy, never the puppy)
- re-adjudication      -> budgeted SME queue (stack-rank vs PPS vs random)
- the weighting method -> Bayesian confidence on each human label
  (non-perfect prior at N=1), used to de-weight — or, in the parallel
  universe, up-weight — suspect anchors

Everything is numpy-only and seeded; paired universes share common random
numbers so divergence is attributable to the intervention, not sampling noise.
"""

from .datasets import Dataset, make_dataset, make_genai, make_mnist, probe_grid, split_indices
from .policy import LogisticPolicy, PrototypePolicy, decision_disagreement, fit_oracle
from .judges import JudgeSpec, Panel, default_panel
from .noise import apply_noise, NoiseConfig
from .confidence import ConfidenceConfig, HumanConfidence, anchor_weight
from .readjudication import ReadjConfig, select_queue, adjudicate
from .engine import (CrankConfig, PanelConfig, build_world, clean_labels,
                     divergence_series, point_influence, run_crank, run_pair)
from . import metrics

__version__ = "0.1.0"
