# Optimizing the gradient descent — candidate selection & optimizer experiments

*Status: substrate + harness shipped, 2026-07-07. The label store views,
confidence fields, AND the experiment crank (`scripts/run_experiment.py` +
`rush.experiment*` tables + the §6 web panel) exist; the S2–S5 strategy
comparison is the next build on top. Sources: "Notation and the per-sample
gradient" (policy-optimization note), re-adjudication loop MVP §4, Attila's
2026-07-06 direction.*

## Philosophy: the policy is LLM-agnostic

Clear guidelines should not be tuned to one model's quirks. A well-grown
policy graph is one under which **every** judge improves — each LLM in the
panel, and ultimately humans too: BPO and SME decision quality should also
benefit from clear, comprehensive instructions. This is falsifiable and we
treat it as the core objective, not a slogan:

> **Objective.** Grow the policy document within a run (several mini-batches)
> such that decision quality improves for ALL selected LLMs on the eval set —
> not on average, and not for the best model. The panel is fixed (the user
> selects 2, 3, or 5 LLMs); the policy version is the only thing that moves.

Two readouts operationalize it:

- **DQ trajectory** — per-judge decision quality on the held-out eval set at
  each accepted policy version `v_0 … v_n`. Target: non-decreasing for every
  judge (gated today by SME review; the automated held-out gate is the next
  wiring step).
- **Agnosticism spread** — the per-judge DQ spread (max − min) at each
  version. A policy that helps one model while hurting another widens the
  spread; a genuinely clearer policy narrows it. Overfitting to one judge is
  a regression *even when average DQ rises*.

The escalation cascade remains the production economics story; it is
deliberately **out of scope** for these experiments. Experiments fix the
panel and study how the *policy* learns.

## Notation (per-sample gradient)

For judge *j* on evaluated sample *i* (binary for now; multiclass noted below):

| Symbol | Definition | Meaning |
|---|---|---|
| `y_i` | ground-truth label | golden set (`rush.golden_label.current_label`) |
| `ŷ_ij` | judge's predicted label | `rush.llm_label.decision` |
| `c_ij ∈ [0,1]` | judge's confidence in its OWN prediction | `rush.llm_label.confidence` |
| `p_ij` | probability assigned to the true class | `c` if correct, `1 − c` if wrong |
| `\|g_ij\| = 1 − p_ij` | gradient magnitude (hardness) | ~0 confident-correct (ignore), ~1 confident-wrong (focus: informative error), 0.5 uncertain |
| `h_ij = c_ij(1 − c_ij)` | hessian / curvature (uncertainty) | max at c=0.5, invariant to correctness — boundary detector |
| `l_ij = −log(p_ij)` | per-sample cross-entropy loss | the quantity the textual gradient should reduce |

Behavior check (from the note, reproduced by the live store):

| Regime | c | p | \|g\| | h | Interpretation | Measured (k=200 store) |
|---|---|---|---|---|---|---|
| Confident correct | 0.85 | 0.85 | 0.15 | 0.13 | low gradient — ignore | 1,844 verdicts, h̄=0.05 |
| Confident wrong | 0.9 | 0.1 | 0.9 | 0.09 | **focus: informative error** | 118 verdicts |
| Uncertain | 0.5 | 0.5 | 0.5 | 0.25 | high curvature — boundary | 85 verdicts, h̄=0.23 |

**Panel aggregation.** The learning signal is the *average across the
selected LLMs*: `ḡ_i = avg_j |g_ij|`, `c̄_i = avg_j c_ij`, plus split/boundary
/difficulty rollups. Implemented as `rush.panel_signal` (per item × policy
version): `n_judges, avg_confidence, avg_p_true, avg_grad_magnitude,
max_grad_magnitude, avg_hessian, avg_loss, is_split, any_boundary, n_wrong,
n_difficulty_high/medium`. Measured separation on the k=200 run: split panels
ḡ≈0.36 vs consensus panels ḡ≈0.03–0.09 — the signal is live and informative.

*Multiclass caveat:* `p = 1 − c` on a wrong prediction assumes the remaining
mass sits on the true class — exact for binary, an upper bound for 10-class
MNIST. Fine for ranking candidates; do not interpret `l_i` as a calibrated
loss in multiclass.

## Human-label confidence

Humans are judges too, and one human label is weak evidence ("the golden set
is not so golden"). With `m` = number of human labels **agreeing with the
current resolved label** (per category, or the high-level binary label):

> **p_human = 1 − 1/(m + 0.2)**

| m | p_human |
|---|---|
| 0 | 0.0 (clamped; raw formula −4) |
| 1 | 0.167 |
| 2 | 0.545 |
| 3 | 0.688 |
| 5 | 0.808 |

Stored on `rush.golden_label.human_confidence`, recomputed at every golden
materialization (`pipeline/labelstore.human_confidence`), and surfaced in
both gradient views. Uses: (a) weight the loss on items whose "truth" is
itself shaky — a confident-wrong verdict against an m=1 label is a
*re-adjudication candidate* before it is a *policy-edit candidate*; (b) the
doc's §4.1 tier weights govern sampling; `p_human` refines the loss weighting
within a tier. All current seeds are m=1 (p=0.167) — the number will move as
the re-adjudication loop adds real SME events.

## The four candidate-selection strategies

Each mini-batch, the optimizer selects K items whose disagreements drive the
next proposed policy edit. The strategies to A/B, all computable from
`rush.panel_signal` today:

| # | Strategy | Selection rule | Hypothesis |
|---|---|---|---|
| S1 | **Random misalignment anchors** (SVM-flavored baseline) | uniform sample from `n_wrong > 0` | disagreements are support vectors; unbiased coverage of the error surface |
| S2 | **Consensus-lack** | rank by `is_split`, then `avg_grad_magnitude` (equivalently low `c̄`) | where judges diverge, the *policy* is ambiguous — the most policy-fixable errors |
| S3 | **Boundary flags** | `any_boundary`, tie-break by `avg_hessian` | judges self-report gray zones; edits here sharpen definitions |
| S4 | **Difficulty rating** | `n_difficulty_high` desc, then medium | the judges' own hardness rating finds under-specified guidance |

Plus the composite the per-sample math suggests (test after the four pure
strategies establish baselines):

- **S5 gradient-weighted**: rank by `ḡ_i × p_human_i` — confident-wrong
  against trustworthy truth first; shaky-truth items route to re-adjudication
  instead.

## Experiment protocol

One experiment = one (strategy, optimizer) cell run under identical
conditions:

- **Fixed**: golden set (current 2,500-image MNIST gold; GenAI next), panel
  (2/3/5 LLMs from the §3 picker), batch budget (e.g. 5 mini-batches × k=20),
  seed policy `v0.1`, split discipline (train drives edits; locked holdout
  reports), acceptance gate (SME review today; ≤ ~5% token clip).
- **Varied**: candidate-selection strategy (S1–S5), optimizer (below).
- **Measured**, per accepted version and per judge, from the store:
  1. DQ trajectory on holdout (accuracy/F1 + FPR) — *the convergence curve*
  2. Agnosticism spread (max−min per-judge DQ)
  3. Edits-to-converge and accepted/rejected edit ratio
  4. `avg_loss` on holdout (the ḡ view of the same story)
  5. Cost per DQ point (labels spent ÷ DQ gained)
- **Honesty guards carried over**: prompt-lift vs label-lift decomposition
  when golden labels move mid-experiment; a random-agreement audit lane so
  selection-on-disagreement doesn't build its own reference standard
  (incorporation bias).

### Optimizers to compare

| Optimizer | What moves | RUSH mapping |
|---|---|---|
| **Textual gradient descent** (current) | one gated, clipped guideline edit per batch | analyst locates the gap from selected items; editor emits a single trackable diff |
| **PPO-style** (current gate, formalized) | same, with explicit trust region | accept only if held-out DQ non-decreasing AND edit ≤ ~5% tokens; reject = no step |
| **GEPA** (genetic-Pareto evolution) | population of policy variants | mutate guideline nodes per batch; select on the Pareto front of (per-judge DQ) — natural fit for the agnosticism objective since the front dominates on ALL judges |

The store makes every cell reproducible: `generator_version` records lineage
+ gate outcomes, `llm_label` dedup means re-scoring a version never re-pays
for covered (item, judge) pairs, and views recompute against the current
golden state after any overturn.

## The experiment crank (shipped 2026-07-07)

`scripts/run_experiment.py` is the harness the experiments run in — one
**experiment** = one numbered, seeded PPO run:

- **Seeded everything**: the master seed derives the fixed test partition
  (stratified out of dev_golden; the gate set), each cycle's train
  mini-batch (without replacement while the pool lasts), and the S1 anchor
  sample. Same seed → same data path; the LLMs are the only nondeterminism.
- **Per cycle**: label N train images with the fixed judge panel → S1 random
  misalignment anchors → drafter (gpt-5.5) proposes ONE edit **clipped to
  1..5 policy-node changes** (`max_changes`; hard cap 5 for human
  debuggability) → candidate bundle evaluated on the test partition → gate.
- **The PPO gate**: accept iff test **system macro-F1** (majority vote)
  strictly improves (`> before + epsilon`). The gate agent (gpt-5.5 default)
  reviews the metric table + unified diff + anchors and may **veto** a
  metric-passing edit (leakage, judge-specific hacks) but can never force a
  failing one. Accepted candidates become real `policy-graph` versions via
  the same `accept_proposal` path the §2 UI uses; skipped proposals archive.
- **Recorded per cycle, per judge AND the system, on train + test**:
  accuracy, F1, precision, recall, FPR, FNR (macro + micro + per-class) —
  `rush.experiment_metric` + portable `data/experiments/<id>/experiment.json`.
- **The human critic is post-hoc by design** (automation stays unblocked):
  every gate decision sits in the §6 ledger awaiting an SME
  correct/incorrect/unsure verdict → `rush.gate_review` — the recorded
  training data for future RLHF of the critic agent itself.
- **Split honesty**: the gate set is formally a validation set (the loop
  adapts to it); the 500-image locked holdout is scored only under the start
  and final versions (`--holdout-final`) — the untouched before/after
  readout for the paper.
- **Coverage-honest gate**: before/after F1 is computed over the
  *intersection* of test images where both policies produced a decided
  system verdict (`gate_comparison`), so errored provider calls or majority
  ties can never flip the gate on coverage alone; the compared-`n` is
  recorded on every gate decision.
- **Known bias, deliberately accepted for now (winner's curse)**: each
  candidate gets ONE noisy evaluation and the baseline value is inherited,
  not re-measured, so with `epsilon=0` the loop preferentially accepts
  upward noise; the accepted-version trajectory on the gate set is
  optimistically biased. This is exactly why the holdout readout exists —
  report policy lift from holdout start→final, never from the gate
  trajectory. Candidate mitigations to A/B later: `epsilon > 0`, re-scoring
  the incumbent alongside each candidate (paired eval), or requiring two
  consecutive wins.

## Build order (remaining)

1. S2–S5 in the crank: `--strategy` currently accepts only
   `random_misalignment` (S1) — add consensus-lack / boundary / difficulty /
   gradient-weighted selection over `rush.panel_signal` and the per-run
   misalignment records.
2. The (strategy × optimizer) comparison driver: same seed, same panel, same
   budget, one experiment per cell; compare DQ trajectories + agnosticism
   spread across cells (the `rush.experiment*` tables make this one SQL
   query per readout).
3. **Within-batch metric-driven descent** (Attila 2026-07-06): today one
   batch → one clipped edit. When batches are large and misalignments exceed
   what one edit can address, stack-rank them (by `ḡ_i × p_human_i`, S5) and
   iterate edits *within* the batch, gating each. Test whether the extra
   gate evaluations pay for themselves vs. simply running more cycles —
   this is a tokenomics question as much as a quality one.
4. §6 chart: add the agnosticism spread band (max−min per-judge DQ) and
   cross-experiment overlay for the comparison view.
