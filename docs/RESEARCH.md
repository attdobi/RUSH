# RUSH — research agenda: is textual policy-gradient descent a sound optimizer?

RUSH treats the **policy prompt as the parameter** and runs *textual gradient
descent* on it: each cycle samples misaligned examples, an LLM "drafter" writes
a small edit to the policy graph, and a trust-region gate accepts the edit only
if measured decision quality improves. The experiment crank
(`scripts/run_experiment.py`) is the **experimental harness** for asking whether
that optimizer is *sound* — the questions below are what we mean to characterize
and, eventually, publish.

The methodological spine is a single control: **every gradient / stack-ranked
strategy is compared against `random_misalignment` (S1) on the same seed and
config.** Random selection is the null hypothesis; the gradient formalism has to
beat it to earn its keep.

---

## Notation

- **Policy π** — the compiled policy-graph markdown bundle (the "parameter").
  Versions `v<run>.<k>`; accepted edits branch from the fixed baseline `v0.1`.
- **Judges** — the cheap LLM panel; the source of every decision-quality (DQ)
  metric. **SME truth `y`** — the golden (human) label.
- **Per-judge gradient** — for judge `j` with self-reported confidence `c` in its
  own label `ŷ`: `p = c if ŷ=y else 1−c`; magnitude `|g| = 1−p`; loss `−ln p`.
- **Panel signals** — SME agreement `a = (#judges matching y)/N`; LLM consensus
  `κ = (#judges on the modal label)/N` (SME-blind); misalignment `m = 1−a`.
- **Importance** — the four-tier stack rank `I_base = (m + κ(2m−1) + 1)/3`, with
  anchor value `= I_base·(1+mean|g|)·(1+½·boundary_rate)`. See the About tab.

---

## Q1 — Overfitting / generalization: "the blue ring is hot" vs "fire is hot"

When the drafter turns a handful of misaligned anchors into a policy edit, does
it learn a **general** rule ("fire is hot" — transfers to every heat source) or a
**hyper-specific** one ("the blue ring [of a stove] is hot" — fails on the red
ring, and in the worst case memorizes the exact training image)? This is the
policy-development analog of model overfitting: the *capacity* is the policy
document's expressiveness, the *training data* is the misaligned anchors, and the
failure mode is memorizing anchors instead of learning transferable rules.

**Regularizers already in the loop:**

- **Edit-size clip (≤5 node changes/version)** — capacity control per step.
- **The gate = trust region** — rejects edits that don't hold on the within-run
  test partition (a form of held-out early stopping).
- **Drafter constraints** — the prompt forbids encoding per-image answers or
  ground-truth labels; class-specific rules must target the owning node or a new
  boundary node, not the root.
- **Aligned anchors** — a sample of correctly-classified images is fed alongside
  the errors so the edit is pressured not to regress what already works.

**How we measure it:** the **train → test → holdout → benchmark generalization
gap.** An edit that lifts the per-run gate set but *not* the locked 500-image
holdout or the fixed 1,000-image cross-run benchmark is overfitting to the
sampled misalignments. The benchmark split exists precisely to make this
measurable across runs on identical images. Node-level probe: after an edit to
`MD.digit.9`, does accuracy rise on **all** 9s or only the anchor 9s?

---

## Q2 — Convergence, in two distinct senses

### (a) Learning-rate / eval convergence

Does DQ(v_n) plateau over accepted steps? By gate construction the **gate-set**
curve is monotone non-decreasing (`DQ(v0) ≤ DQ(v1) ≤ …`), but it is
optimistically biased — one noisy eval per candidate, ε=0, the incumbent's score
inherited rather than re-measured (the *winner's curse*). The **honest**
convergence signal is the locked-holdout / fixed-benchmark trajectory, which the
loop never adapts to. Open questions: does the holdout curve plateau (diminishing
returns / effective learning rate → 0)? Does the SME re-adjudication queue shrink
to a maintenance trickle — the *human-effort* convergence that the whole system
exists to produce?

### (b) Chaos / Lyapunov sense

Run the **same config with different random seeds** — the seed controls which
train images and anchors are drawn. Do the resulting **final policy documents**
converge to similar policies, or diverge? A **positive Lyapunov exponent** means
high sensitivity to initial conditions: seed → vastly different policy documents
even when DQ is comparable.

**How we measure it:** embed the final policy markdown bundle (and/or the KG
structure/edges) with a text-embedding model and compute pairwise distances
across N seeds. The reproducible-seed infrastructure plus the local
text-embedding models make this a concrete experiment. Interpretation:

| Seeds → policies | DQ across seeds | Reading |
|---|---|---|
| similar (low spread) | similar | well-posed / stable optimizer |
| different (high spread) | similar | many equivalent local optima — the policy is *not identifiable* from the data |
| different | different | unstable / fragile to the sampling |

The third regime is the danger sign; the second is philosophically interesting
(the "true" policy is under-determined and several valid rule-sets fit).

---

## Q3 — The central ablation: random vs stack-ranked (gradient) selection

- **Control:** `random_misalignment` (S1) — unbiased sampling of the error surface.
- **Treatment:** `top_importance` (four-tier: misalignment × consensus ×
  confidence × boundary) or `top_gradient` (`|g| = 1−p`).

Hold the seed and everything else fixed; swap only `--strategy`. Compare (a)
convergence speed (DQ gained per cycle), (b) final holdout/benchmark DQ, (c)
human touches to steady state, (d) the overfitting gap from Q1. **Hypothesis:**
gradient-ranked selection spends the drafter's ≤5-change budget on the most
informative errors — confident, unanimous, wrong panels are systematic policy
gaps rather than sampling noise — and therefore converges faster and/or higher
than random. If it does *not* beat random, the gradient formalism is not earning
its complexity, and that is itself a publishable result.

---

## Q4 — Prompt-tuning architectures

The current drafter is one architecture: an LLM rewrites markdown node files from
the anchors and their images, gated. The crank is a **fixed harness**; the
optimizer is swappable. Alternatives to A/B on the same seeds/splits:
reflective textual-gradient / GEPA-style prompt evolution, direct node-statistic
updates (edit frontmatter stats without free-form rewrite), retrieval-augmented
editing. The comparison is optimizer-vs-optimizer with everything else held.

---

## The hyperparameter response surface

The paper characterizes how the above respond to the knobs the crank exposes:

| Knob | Role |
|---|---|
| `epsilon` | gate margin — trust-region tightness; winner's-curse control |
| `max_changes` | edit capacity per step = regularization strength |
| `max_anchors` / `max_aligned_anchors` | batch size + the regularization ratio |
| `batch_n` | mini-batch size — sampling noise vs signal |
| `test_n` | gate-set size — winner's-curse magnitude |
| `gate_mode` | metric rule vs + agent veto |
| `drafter_model`, `strategy` | the optimizer itself |

---

## Known bias to fix before publishing (gate rigor)

The winner's curse (Q2a) is documented but unmitigated in the current gate:
one noisy eval, ε=0, inherited baseline. The mitigations to add and A/B — **ε>0,
paired incumbent re-eval, N-consecutive-wins** — directly harden the central
convergence claim, and the honest lift should always be reported from the
**holdout/benchmark**, never the gate set alone.
