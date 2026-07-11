# label-noise sim — what do wrong human labels do to the crank?

RLHF goes both ways. The crank optimizes a policy document against SME labels,
so a mislabeled image becomes an anchor point and can steer the policy toward
the wrong boundary — the catastrophic failure mode we normally handle through
re-adjudication. Empirically the labels ARE imperfect: of LLM-vs-SME
disagreements sent to review, ~33% were overturned in favor of the LLM (~44%
for the sensitive-content area). This package simulates the whole loop in a
2D geometry where ground truth is knowable, so we can measure two things the
real crank can't tell us:

1. **Divergence** — how far does label noise bend the learned policy away
   from the one a clean-labeled twin universe would have learned? Measured in
   "document space": distance between policy parameter vectors, plus the
   fraction of a probe grid the two policies decide differently (the sim
   analogue of a policy expert reading the diff).
2. **Convergence of the weighting methodology** — every human label carries a
   non-perfect Bayesian confidence (prior `c0 = 0.9` at N=1 — a key design
   commitment), updated each time the panel's system vote agrees/disagrees
   with it. Does de-weighting suspect anchors (or up-weighting them, in the
   parallel universe that assumes the human is right) actually converge to the
   true boundary? And how does that interact with budgeted SME
   re-adjudication (stack-rank vs PPS vs random queues)?

Nothing here touches the production pipeline; it's a standalone numpy package.

## The mapping

| Real crank | Simulation |
|---|---|
| Policy document (KG render) | Parametric decision boundary: logistic (GenAI binary), LVQ class prototypes (MNIST 10-class) |
| Judge panel reading the policy | 5 judges labeling `policy.predict(x + bias + N(0, σ_j²))`; σ = capacity; optional non-compliant constant judge |
| Drafter's 1–5 discrete edits | Anchor-weighted gradient step with step-norm clipping |
| Anchors (misalignment gradient) | Batch items where system vote ≠ human label, scored consensus × confidence-weight, picked by stack-rank / PPS / random |
| Gate (candidate vs incumbent macro-F1 on the test partition, human labels) | Same, on panel system votes; oracle-scored twin gate recorded for false-accept / false-reject accounting |
| SME re-adjudication queue | Budgeted per cycle, ordered by cumulative disagreement mass; SME returns truth with prob `q_sme` (0.95) |
| Document distance / expert reads the diff | Normalized parameter distance + decision-disagreement on a probe grid |
| Doc-space intuition | Everything lives in 2D: mislabels sit ON the true boundary (adult vs racy — never the puppy), placed by `exp(-margin/τ)` under the oracle, flipping to the runner-up class |

Datasets: `genai` — synthetic binary world with an easy mass per class and a
contested ridge (Bayes error lives there, like photo-real generations);
`mnist` — the repo's real MNIST archive, PCA(50)→Fisher-LDA to 2D, class
prototypes as realizable teacher labels (real confusion geometry: 4/9, 3/5,
7/1), truth ceiling 1.0 by construction.

## The plan (and what the first 12-seed sweep already shows)

**Protocol A (implemented here):** start from high-confidence labels (= ground
truth), inject noise at a known rate and geometry, run the crank, compare
against the clean twin universe over the SAME world with common random
numbers — every stochastic subsystem (batch draws, judge noise, queue
sampling, SME coin-flips) draws from streams keyed `(seed, subsystem, cycle)`,
so twin trajectories differ only through the labels, and divergence is
attributable. Noise geometries: `uniform` (interior mislabels — the
catastrophic anchors), `boundary` (flip probability `∝ exp(-margin/τ)`,
target = the confusable runner-up class), one-way (`flip_from`: gen_ai
systematically called not_gen_ai — directional adult/racy-style confusion;
two-sided flips at a boundary largely cancel, one-way flips drag it).

**Protocol B (for the real crank, when ready):** use the pre-corrected vs
post-corrected label sets from the re-adjudication log
(`data/adjudication_reviews.jsonl`, local-only) as the noisy/clean pair —
same measurement machinery, empirical noise. Calibration check: the EMERGENT
overturn rate among reviewed items in the mnist suites lands at 0.32–0.36,
inside the observed 33–44% band, with no tuning. (The genai S2 cell is a
deliberate stress test — 30% of all labels flipped one-way means 60% of the
gen_ai class is wrong, and reviews there overturn at 0.74, as they should.)
Noise rates always mean "fraction of ALL eligible labels", so one-way and
two-sided cells are comparable at the same rate.

Five pre-registered suites (12 seeds per cell, mean ± 95% CI; a full sweep of
both datasets runs in ~2 minutes):

### S1 — dose-response (divergence)
Noise bends the trajectory monotonically, and geometry matters more than rate:

- genai @ 30%: uniform two-way 0.785 vs boundary two-way 0.873 (clean 0.895).
  Interior mislabels hurt ~4× more than boundary mislabels at equal rate.
- Direction is what really drags a boundary: boundary ONE-WAY lands at 0.798
  @ 20% and 0.705 @ 30%, vs two-way's 0.894 / 0.873 at the same overall
  rates. Symmetric boundary flips largely cancel; directional confusion
  (adult→racy style) does not.
- mnist @ 30% uniform: 0.546 vs clean 0.755, decision-disagreement divergence
  0.30 — multiclass prototypes are far more fragile than a 3-parameter line.

### S2 — does the weighting methodology converge? (the headline)
Two regimes: genai is a heavy directional stress test (30% of all labels
one-way = 60% of the gen_ai class wrong; reviews overturn at 0.74); mnist is
the calibrated regime (30% two-sided boundary noise; overturn 0.32–0.36,
inside the empirical band):

| arm | genai F1 | mnist F1 |
|---|---|---|
| clean reference | 0.895 | 0.755 |
| no mitigation | 0.705 | 0.709 |
| deweight (soft) | 0.725 | **0.644** |
| deweight (hard) | 0.700 | 0.665 |
| upweight | 0.701 | 0.693 |
| readj stack-rank | 0.854 | 0.719 |
| readj PPS | 0.860 | 0.725 |
| readj random | 0.859 | 0.714 |
| **deweight + readj** | **0.859** | **0.746** |

Two lessons. (1) **Weighting alone does not converge** — it can't repair the
gate's corrupted yardstick. Under heavy directional corruption it nudges
(+0.02 on genai); in the calibrated multiclass regime soft deweighting is
WORSE than nothing: the posterior can't yet distinguish "hard but correct
boundary label" from "wrong label", so it throws away exactly the boundary
evidence the prototypes need. (2) **Deweight + re-adjudication is the winning
combination** (mnist 0.746 vs 0.755 clean; genai recovers 0.86 of a 0.19-point
hole): the confidence posterior's real job is routing scarce SME attention
(and the review then restores confirmed labels to full weight), not silently
ignoring points. Detection AUROC of the peak-suspicion score, measured on a
stable population (re-adjudication must not censor its own catches), is
0.61–0.77 — informative, not oracle.

### S3 — parallel universes ("did this label ever matter?")
Twin runs that only disagree about suspect labels (deweight vs upweight):
genai diverges 0.024–0.030 in decision space — for a gated binary world these
points mostly "never mattered much" — while mnist diverges 0.11–0.14: in
multiclass the assumption you make about suspects changes the document you
end up with. Per-point influence probes (one label pinned to weight 0 vs 2,
short twin horizon; random flipped vs random clean items, so the comparison
isn't confounded by the detector's own selection) separate them by ~7× on
genai (0.0080 vs 0.0012). On mnist the per-point gap narrows (0.047 vs
0.040): individual boundary flips matter little one at a time even though
collectively they diverge the twins — influence is a boundary-vs-interior
signal more than a flipped-vs-clean signal there. Still the sim analogue of
"small distance in document space ⇒ the point never mattered."

### S4 — the test set is not optional (his call, confirmed)
Noise ONLY in the test partition: the gate false-accepts 16–20% of steps
(genai @ 10–20% test noise) and false-rejects up to 26%. Re-adjudicating the
test set (budget 5/cycle, stack-ranked) collapses that to 1–3% FA / 3–9% FR.
The gate's yardstick deserves SME attention as much as the anchors do —
re-adjudicate the validation/test sets, not just training anchors.

### S5 — sampling strategy under contamination
At 20% uniform noise, anchors run heavily enriched in mislabels
(contamination 0.34–0.53 vs the 20% base rate — persistent misalignment
selects for them), and anchor picks CONCENTRATE: Gini over ever-eligible
items jumps from ~0.33 (clean) to 0.63–0.67 (noisy), with stack-rank the most
concentrated (0.67). The hyper-focus mechanism is real and visible. Final F1
still lands within each strategy's CI (the gate limits the damage any one
selection rule can do); PPS trends mildly better on mnist (0.601 vs 0.580
stack-rank). The interesting strategy question moved to the SME queue (S2:
random ≈ PPS ≈ stack-rank on equal budget — because at these rates almost any
review hits a mislabel; at low rates ranking quality should matter more.
Sweep `rate × strategy` before concluding.)

## How to run

```bash
# full sweep, both datasets, 12 seeds/cell (~2 min) -> results/*.json
./.venv/bin/python sim/label-noise/run_sim.py --suite all --dataset both

# one suite
./.venv/bin/python sim/label-noise/run_sim.py --suite S2 --dataset mnist --seeds 20

# tests
./.venv/bin/pytest sim/label-noise/tests -q
```

Notebook: `notebooks/label_noise_sim.ipynb` — the narrative version with
plots (committed with outputs, renders on GitHub).

Interactive: `web/index.html` — a d3 port of the GenAI binary arm with the
twin universe animated live (noise/weighting/re-adjudication sliders):

```bash
python3 -m http.server 8794 -d sim/label-noise/web   # then open :8794
```

## Design commitments (read before arguing with a result)

- **N=1 human labels get prior 0.9, never 1.0** — the posterior update uses
  assumed panel likelihoods (`p_catch=0.75`, `p_false=0.25`) scaled by
  consensus share. No ground truth ever enters the loop's own signal; truth is
  used only to PLACE noise (experiment design), inside the SME review model,
  and for measurement.
- **Noise defaults to `target="both"`** — the same SMEs label train and test,
  so the yardstick is corrupted too. With `target="train"` the clean test set
  makes the gate a secret oracle and it rescues almost any run (worth knowing:
  that's the fixed-yardstick argument for a large, well-audited test set).
- **v0 is a seeded 30–60° distortion of the oracle boundary** — truth-anchored
  (so absolute recovery levels are partly baked in by the start), but
  label-NOISE-free and mode-independent: twin universes and mitigation arms
  share their starting document exactly, so every RELATIVE comparison is
  attributable to the loop. Read the suites as relative claims.
- **mnist is realizable** (teacher labels from the prototype boundary over
  real MNIST LDA geometry) so multiclass dynamics aren't drowned in
  2D-projection Bayes error; `realizable=False` keeps original digit labels.
- The gate compares SYSTEM votes (panel majority), not raw policy predictions
  — same as production decision quality.

## Limits

A 2D parameter vector is not a policy document: no prompt-length effects, no
per-clause blame, no drafter language failures, judges are noisy readers of
the boundary rather than LLMs. The sim answers geometry questions (where
mislabels sit, what they drag, who should review what) — not language
questions. Treat effect DIRECTIONS and interactions as transferable, absolute
numbers as world-specific.

## Next steps for the sampling strategy

- Sweep `noise rate × SME queue strategy` (S2 grid extension) — the
  stack-rank vs PPS question likely inverts at low rates where ranking
  quality matters; that feeds directly into experiment E4/E8 of
  `docs/EXPERIMENT-PLAN-benchmark-grid.docx`.
- Influence-ranked SME queue: S3's per-point influence probe as the queue
  score (expensive in prod = twin short runs; cheap here) vs disagreement
  mass — is "would this label change the document?" a better use of SME
  attention than "does the panel keep disagreeing?".
- Protocol B on real data: pre- vs post-adjudication labels from
  `data/adjudication_reviews.jsonl` through the same divergence machinery.
- Panel-vs-single-judge optimization under noise (the cross-judge
  interference axis), and the non-compliant-judge arm
  (`PanelConfig(noncompliant=True)`) interacting with label noise.
