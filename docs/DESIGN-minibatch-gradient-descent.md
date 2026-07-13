# RUSH — Textual Policy-Gradient Descent: One Mini-Batch, End to End

*How the crank turns each cycle k: the per-sample gradient, the seeded sampling
that draws each mini-batch, and the PPO-style gate that accepts or rejects the
step. Every formula below is exactly what the code computes
(`pipeline/experiment/__init__.py`); the appendix maps each one to its function.*

Notation: subscript ₖ = cycle index, ᵢ = image, ⱼ = judge. ⊕ = "apply the edit
to the policy". All formulas use plain Unicode so they render as-is in Google
Docs, GitHub, or plain text.

---

## 0. What is being optimized

The **policy** (we also call it the *generator*) **Gₖ** is the versioned
prompt + policy-graph the judge panel runs at cycle k. It is the parameter
vector — except the parameters are **text** (markdown node files), not floats.

A **panel of N judges** (2–5 cheap models) labels each image i. Judge j returns:

- a label **ŷᵢⱼ**,
- a self-reported confidence **cᵢⱼ ∈ [0, 1]**,
- a difficulty ∈ {low, medium, high},
- an `is_boundary` flag (+ the confusion pair),
- a justification citing policy nodes.

The **SME golden label** for image i is **yᵢ**.

**Objective.** Maximize the panel's decision quality — the *system* (majority-
vote) macro-F1 on a fixed held-out test partition — by editing the policy,
subject to a trust region of at most 5 node-file changes per step:

```
maximize   DQ(G) = F1_test( system_vote(G) )
over        G reachable from Gₖ by one edit e
subject to  |e| ≤ 5 node-file changes            (the trust region)
```

There is no analytic ∇G. The **gradient is estimated from the panel's errors**
and **applied by a drafter model that rewrites the policy text**. That is the
whole idea of *textual* policy-gradient descent: the error signal is numeric,
the update is a language-model edit, and a gate keeps every step honest.

---

## 1. The per-sample gradient (one judge, one image)

For judge j on image i with self-reported confidence c in its own predicted
label ŷ, define the probability it placed on the **true** class:

```
p  =  c          if ŷ = y        (prediction correct)
p  =  1 − c      if ŷ ≠ y        (binary approximation)
```

From p we read three signals — the direct analogues of a gradient, a curvature,
and a loss:

```
|g|   =  1 − p                gradient magnitude
h     =  c · (1 − c)          curvature / uncertainty
ℓ     =  − ln p               per-sample cross-entropy loss
```

Reading them:

- **|g| = 1 − p** is the **most informative-error** signal. A confident-wrong
  judge (c ≈ 1, ŷ ≠ y ⟹ p ≈ 0) gives **|g| ≈ 1**: the policy is confidently
  steering it wrong — fix this first. A confident-correct judge gives |g| ≈ 0:
  nothing to learn. This is exactly what a gradient does — big where the model
  is confidently wrong, ~0 where it is confidently right.
- **h = c(1 − c)** peaks at c = 0.5 and is correctness-blind — pure model
  uncertainty (curvature of the same cross-entropy).
- **ℓ = −ln p** is the loss whose gradient we are following, in [0, ∞).

Abstains and missing confidences are **excluded** — an undecided judge carries
no gradient. (Computed by `vote_gradient`; p is clamped to [1e-6, 1] inside the
log so ℓ stays finite.)

---

## 2. Panel aggregation (all judges, one image)

A multi-LLM panel produces **two** agreement signals a single vote cannot. Over
the **N_dec** decisive (non-abstaining) judges on image i:

```
SME agreement   aᵢ  =  (# judges with ŷ = y) / N_dec      LLM ↔ HUMAN
misalignment    mᵢ  =  1 − aᵢ
LLM consensus   kᵢ  =  (# judges on the modal label) / N_dec   LLM ↔ LLM (SME-blind)
```

`aᵢ` measures whether the panel agrees with the human; `kᵢ` measures whether the
judges agree with **each other**, independent of the truth. Keeping them
separate is the point — a unanimous panel (kᵢ = 1) that is unanimously **wrong**
(aᵢ = 0) is the worst case, and only two signals can name it.

We also aggregate the per-vote gradient and the boundary flag over the panel:

```
mean|g|ᵢ       =  average of |g| over the decisive judges
boundary_rate  =  (# judges flagging is_boundary) / N          (all judges)
```

### 2.1 The four-tier importance score

Consensus flips meaning depending on alignment, which yields a natural ordering
of how much an image is worth learning from:

| Tier | Condition | Meaning |
|------|-----------|---------|
| **T1** | misaligned + high consensus | unanimous **and wrong** — systematic error (worst) |
| **T2** | misaligned + low consensus | the panel split and missed |
| **T3** | aligned + low consensus | right, but the panel argued (still instructive) |
| **T4** | aligned + high consensus | unanimous **and right** — the ideal state |

"High consensus" is **strict**: kᵢ > 0.5. An even split (1-1, 2-2) is a tie, not
consensus, so it lands in the low-consensus tier.

A single continuous score reproduces that T1 → T4 ordering:

```
I_base  =  ( m + k·(2m − 1) + 1 ) / 3            ∈ [0, 1]
```

(The raw form m + k(2m − 1) lives in [−1, 2]; the +1 and /3 just renormalize to
[0, 1].) Two derived scores amplify I_base by how *confident* and how *boundary-
heavy* the panel was:

```
anchor_value          =  I_base · (1 + 1.0·mean|g|) · (1 + 0.5·boundary_rate)
readjudication_value  =  anchor_value · (1 − p_human)
```

- **anchor_value** is the **policy-learning** priority — which misalignments the
  drafter should study this cycle (§3.3).
- **readjudication_value** is the **human-review** priority (the Adjudicate
  queue). It additionally fades by human-label confidence p_human, because a
  golden label an SME has already re-confirmed barely needs another look:

```
p_human  =  1 − 1 / (m_conf + 4)         m_conf = # SME confirmations
         →  0.000 (m=0),  0.800 (m=1),  0.833 (m=2),  0.857 (m=3)
```

(Computed by `importance_scores` + `human_confidence`; weights `GRAD_WEIGHT = 1.0`,
`BOUNDARY_WEIGHT = 0.5`, `CONSENSUS_HIGH = 0.5`.)

---

## 3. One mini-batch (cycle k)

Each cycle is one **gradient step**. It has four moves: **sample → draft → score
→ gate**. This section covers sample and draft; §4 covers the gate.

### 3.1 Sampling — everything is seeded and reproducible

Every random draw derives its own RNG from `(master_seed, role, k)` over
sample-id-sorted pools, so any single piece is reproducible from the run record
alone — there is no shared RNG state to keep in sync.

**Test partition (fixed for the whole run).** Carved once at k = 0 out of the
`dev_golden` pool, **stratified by golden label** so macro-F1 is stable, then
**reused every cycle** — it is the gate's fixed yardstick.

```
per class ℓ:  quota_ℓ = min( ⌊T / C⌋ + (1 if ℓ among first (T mod C) classes),  |pool_ℓ| )
              test_ids ← seeded sample from class-ℓ pool,  RNG = Random("{seed}:test:{ℓ}")
              (the min() clamps a small class to its own size; a lexical
               back-fill then tops up any resulting shortfall on lopsided pools)
train_pool  =  dev_golden \ test_ids
```

where T = test size, C = number of classes. (`partition_test_train`.)

**Train mini-batch Bₖ (fresh each cycle).** N images drawn **without replacement
across cycles** while the pool lasts; once fewer than N unused ids remain, the
batch tops up from already-used ids (a reused image is simply re-judged under a
newer policy — a distinct verdict):

```
fresh   =  train_pool \ used_ids
Bₖ      =  seeded sample of N from fresh                RNG = Random("{seed}:train:{k}")
           (if |fresh| < N:  all of fresh + a seeded top-up from the reused pool)
```

(`sample_train_batch`.)

**Anchors (the images the gradient step actually studies).** From the images in
Bₖ that **misaligned** (panel majority ≠ SME), select up to `a_neg` by the
chosen strategy; alongside them, up to `a_pos` correctly-classified ("aligned")
images. Both counts are hyperparameters (defaults 10 / 10; `a_pos = 0` disables
the aligned side).

```
misaligned pool  =  { i ∈ Bₖ : misalignment_type(i) ≠ all_agree }
aligned pool     =  { i ∈ Bₖ : misalignment_type(i) = all_agree  }

negatives (≤ a_neg), by --strategy:
   random_misalignment (S1)  uniform seeded sample     RNG = Random("{seed}:anchors:{k}")
   top_gradient              sort by  mean|g|  descending      (confident-wrong first)
   top_importance            sort by  anchor_value descending  (T1 unanimous-wrong first)

positives (≤ a_pos):
   uniform seeded sample of the aligned pool           RNG = Random("{seed}:aligned:{k}")
```

The **negatives are the gradient** — the errors to fix. The **positives are a
regularizer** — they show the drafter what the policy already gets right so its
edit does not over-correct past the margin. In SVM terms: the misaligned anchors
are the misclassified support vectors; the aligned anchors are the correctly-
classified points near the margin. (`select_anchors`, `select_aligned_anchors`.)

`random_misalignment` is the **null hypothesis** every ranked strategy must beat:
if seeing the "most important" errors first does not improve the learning curve
over a random sample, the ranking is not adding signal.

### 3.2 The gradient step — draft

The **drafter** (one model, cheap or frontier — it drafts, it never scores)
receives:

- the current policy graph **Gₖ** (the *exact* prompt the judges run), and
- per anchor: the **image pixels**, the **SME golden label**, and each judge's
  **label + confidence + justification**.

It returns **one edit eₖ** touching at most 5 node files. That edit **is the
textual gradient step**, and the ≤5-file clip **is the trust region / learning
rate**: it bounds how far the policy can move in one cycle and keeps every step
human-reviewable. A modification, addition, or removal of a node file each counts
as one change; the clip is a hard backstop when the drafter over-reaches
(`clip_changes`, hard cap 5). The candidate policy is **Gₖ ⊕ eₖ**.

### 3.3 Scoring

The panel **relabels the fixed test partition** under the candidate
**Gₖ ⊕ eₖ**, producing the candidate's system macro-F1. The incumbent's score
is inherited from its last evaluation on the same partition.

---

## 4. The gate — accept or reject the step

A step is committed only if it **provably improves decision quality**. The gate
is two layers: a hard deterministic rule, and an optional subtractive critic.

### 4.1 The deterministic rule (the trust-region wall)

```
accept(eₖ)  ⇔  F1_test(Gₖ ⊕ eₖ)  >  F1_test(Gₖ) + ε
```

where F1_test is the **system-of-judges** (majority-vote) macro-F1 on the run's
fixed test partition, and ε ≥ 0 is the acceptance margin (default 0). The
**expensive gate model never computes decision quality** — F1 always comes from
the cheap panel.

**Coverage-safe comparison.** Errored calls and majority-vote ties both remove
images from one side's decided verdicts. Comparing full-run F1 would then compare
two *different* subsets of the "fixed" partition and could flip the gate on
coverage alone. So both policies are scored over the **intersection of images
where both sides produced a decided system verdict**, and coverage is reported:

```
common      =  { i : both Gₖ and Gₖ⊕eₖ produced a decided system vote on i }
F1_test(·)  =  macro-F1 over `common` only
```

(`gate_comparison`, `metric_passes`.)

### 4.2 The gate agent (optional, subtractive)

An optional gate agent reviews the metric table + the unified diff + the anchor
evidence. It is a **one-way valve**: it can **veto** a metric-passing edit it
judges *unsound*, but it can **never force or accept** a metric-failing one. It
skips an edit that:

1. **overfits to named examples instead of stating a general guideline** — the
   over-specificity veto that fights memorization;
2. leaks the SME golden answer;
3. targets one judge model's quirks;
4. tells judges to abstain or defer instead of committing to a label;
5. piles class- or pair-specific rules into the root file instead of the owning
   class/boundary node;
6. is incoherent with the policy's structure.

Criterion (1) is the crank's most direct **overfitting guard** — it blocks a
hyper-specific rule (e.g. "the blue [ring] is hot") that would boost the metric
on the training anchors but fail to generalize. It sits alongside three other
guards: the ≤5-change trust region (§3.2), the drafter's "never encode per-image
answers" constraint, and the aligned anchors (§3.1). (`GATE_SYSTEM_PROMPT`.)

### 4.3 The decision truth table

```
gate mode / state                       outcome    decided_by
────────────────────────────────────────────────────────────────
gate OFF                                accept     gate_off        (metric recorded, never enforced)
rule fails (any agent verdict)          skip       metric_rule     (override_guard if agent said accept)
rule passes, no agent (metric_only)     accept     metric_rule
rule passes, agent accepts              accept     gate_agent
rule passes, agent vetoes               skip       gate_agent_veto
```

The rule is the hard wall; the agent can only subtract. (`resolve_gate_decision`.)

### 4.4 The update

```
accept:   G_{k+1}  =  Gₖ ⊕ eₖ   → committed as a new policy version  v<run>.<k>
skip:     G_{k+1}  =  Gₖ         → the incumbent stays; the candidate is archived
```

k = 0 is **fixed at v0.1** for every run, so all runs descend from the same
baseline and accepted versions branch from it (lineage recorded per run).

---

## 5. Why this is gradient descent

| SGD term | RUSH analogue |
|----------|---------------|
| parameters θ | the policy text Gₖ (markdown node files) |
| mini-batch | the seeded N-image train batch Bₖ (§3.1) |
| loss ℓ | −ln p per sample; DQ = system macro-F1 in aggregate |
| gradient ∇ℓ | the panel's errors, ranked by \|g\| / four-tier importance (§2) |
| the update θ ← θ − η∇ℓ | the drafter's edit Gₖ ⊕ eₖ (§3.2) |
| learning rate / trust region η | the ≤5 node-file clip (§3.2) |
| step acceptance (line search / PPO clip) | the gate: accept only if F1_test strictly improves (§4) |
| a rejected step | skip: the incumbent stays (§4.4) |

**The learning curve advances only on accepted steps.** The default x-axis ticks
once per accepted policy version (v0.1 → v0.2 → …), *not* per cycle: if k → k+1
did not clear the gate, nothing was learned — it was sampling noise, and the
policy did not move. Skipped candidates render as ghost points off the step axis.

---

## 6. Known bias — the winner's curse

The gate accepts on a **single noisy evaluation** of the candidate, with the
incumbent's score inherited (not re-measured) and ε = 0. Selecting the max over
noisy candidate evaluations biases the **gate-set** trajectory optimistically.
It is a measurement bias in the acceptance test, **not** in the method: report
lift from the **locked holdout** and the **fixed cross-run benchmark**
(start → final), never from the gate set alone. Mitigations to A/B later:
ε > 0, paired incumbent re-evaluation, two-consecutive-wins acceptance.

---

## Appendix — formula → code map (`pipeline/experiment/__init__.py`)

| Formula | Function |
|---------|----------|
| p, \|g\| = 1−p, h = c(1−c), ℓ = −ln p | `vote_gradient` |
| aᵢ, kᵢ, mean\|g\|, boundary_rate, tie | `panel_signal` |
| I_base, anchor_value, readjudication_value, tiers T1–T4 | `importance_scores` |
| p_human = 1 − 1/(m+4) | `human_confidence` |
| fixed stratified test partition | `partition_test_train` |
| seeded no-replacement train batch Bₖ | `sample_train_batch` |
| anchor selection (random / top_gradient / top_importance) | `select_anchors` |
| aligned-anchor selection | `select_aligned_anchors` |
| ≤5 node-file clip (trust region) | `clip_changes` |
| coverage-safe F1 before/after | `gate_comparison` |
| F1_after > F1_before + ε | `metric_passes` |
| accept/skip truth table | `resolve_gate_decision` |
| gate-agent soundness criteria | `GATE_SYSTEM_PROMPT` |
