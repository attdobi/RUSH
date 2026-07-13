# RUSH — Technical Report

**Textual policy-gradient descent for enterprise LLM auto-judges: architecture, formalism, evidence, and the experimental program**

*Status: living document (2026-07-08). Formulas are plain Unicode and are implemented
verbatim in `pipeline/experiment/__init__.py`; Appendix A maps each one to its function.
Companion documents: [RESEARCH.md](RESEARCH.md) (the research agenda),
[DESIGN-minibatch-gradient-descent.md](DESIGN-minibatch-gradient-descent.md) (one
mini-batch, end to end), [DESIGN-policy-gradient-experiments.md](DESIGN-policy-gradient-experiments.md)
(optimizer comparison plan), and the in-app About tab (project-agnostic metric reference).*

---

## 1. What RUSH is

**RUSH** (Reinforcement learning Using SME feedback and High-reasoning models) is an
enterprise auto-judge system that scales a subject-matter expert's decision quality to
cheap labelers — low-cost LLMs today; BPO reviewers and production ML downstream. The
economic thesis, from the README:

> Spend reasoning only where cheap consensus fails. Spend humans only where reasoning fails.

Two coupled loops implement that thesis:

1. **The labeling cascade** — a panel of 2–5 cheap LLM judges labels everything against a
   versioned, SME-owned policy; disagreements escalate to a high-reasoning judge; the
   residue lands in a human adjudication queue. Measured on the in-repo MNIST cascade run
   (400 images): the 5-model tier-1 panel labeled everything for ~$2.23 per 1,000 completed
   labels at 97.7% ensemble accuracy on majority-decided images (96.8% counting the 4
   no-verdict images as errors; 80 of 2,000 tier-1 calls errored), escalating 35.5% of images.
2. **The policy optimizer (the "crank")** — this report's subject. The policy prompt is
   treated as the **parameter**; each optimization cycle labels a fresh mini-batch, mines
   the panel's misalignments against SME golden labels into a **textual gradient**, has a
   drafter LLM propose one bounded policy edit, and accepts the edit only if a
   deterministic metric gate says decision quality improved.

The optimized artifact is not a model checkpoint. It is a **knowledge graph of the policy
itself** — Obsidian-compatible Markdown nodes plus typed edges (`edges.json`) that compile
deterministically into the generator prompt the judges run. *When we say the prompt is the
policy, it is literal: same bytes.* Every accepted step is a human-reviewable diff.

### Goals

- **Scale SME judgment, don't replace it.** Humans own the golden labels, adjudicate the
  stack-ranked residue, and can overturn the golden set itself ("the golden set is not so
  golden"). AI proposes; the gate and the SME dispose.
- **LLM-agnostic policy.** The objective is to grow a policy that improves decision
  quality for *every* judge on the panel — not on average, and not just for the best
  model. A policy that only helps one model has memorized that model's quirks.
- **DS-grade rigor.** Every run is numbered, seeded, and reproducible from its record;
  every metric has an honest split behind it; known biases (the gate's winner's curse) are
  documented with planned corrections rather than hidden.
- **A paper-grade harness.** The crank exists to answer a research question: *is textual
  policy-gradient descent a sound optimizer?* (§9).

---

## 2. Architecture: three decoupled machine roles + a human root of trust

RUSH deliberately splits the optimization loop across agents that cannot grade their own
homework (§8.1 develops why this is the first overfitting defense):

| Role | Who | Does | Never does |
|---|---|---|---|
| **Judges** | 2–5 cheap models (e.g. gpt-5.4-mini-low, gemini-3.1-flash-lite, local Qwen/Gemma on RTX 3090s) | Label every image; produce **every** decision-quality metric (F1, precision/recall, FPR/FNR, the learning curve, the gate metric) | Draft edits; veto anything |
| **Drafter** | One model, frontier or cheap (`ALLOWED_POLICY_MODELS`: gpt-5.5 variants, gpt-5.4-mini-low, opus-4-7) | Read the anchor evidence (including the actual image pixels) and emit ONE clipped policy edit per cycle | Score anything |
| **Gate** | A deterministic rule, optionally + one agent | Accept iff the *panel's* test macro-F1 strictly improves; the optional gate agent may **veto** a metric-passing edit as unsound | The agent can never force an accept ("the rule is the hard wall, the agent is a one-way valve") |
| **SME (human)** | The expert | Owns golden labels; reviews gate decisions; works the re-adjudication queue (confirm / overturn / uncertain) | Gets replaced |

No expensive model ever computes decision quality. The gate metric is the cheap panel's
own majority-vote macro-F1; gpt-5.5 only drafts and (optionally) vetoes.

### 2.1 The policy graph

Each demo area is a directory of Markdown nodes + `edges.json` under `policy-graph/<area>/v<X.Y>/`:

- **MNIST_Digits v0.1** — 12 files: `MD.root.md` (decision rule: assign exactly one digit
  by stroke topology/geometry; *never abstain* — doubt is expressed as lower confidence and
  higher difficulty) + `MD.digit.0..9` (positive criteria, distinguishing features, hard
  negatives) + `edges.json`. Boundary nodes (e.g. `MD.boundary.9_vs_0.md`, with
  `confused_with` edges to both digits and a "decisive test") are created by the optimizer
  itself — three exist today, all minted by run 7.
- **Generative_AI v0.1** — 17 nodes: `GA.root` plus evidence lanes (visual artifacts,
  surface texture, scene geometry, provenance) and a *designed* abstain lane
  (`GA.boundary.low_quality_uncertain`) for images too degraded to judge — confirmed
  abstains are excluded from decision-quality denominators by design.

### 2.2 One cycle, end to end

Every run starts, by default, from the same fixed baseline generator (k=0 = policy v0.1;
`--policy-version` can override) so runs are comparable. Per cycle k (driver:
`scripts/run_experiment.py`):

1. **Label.** A seeded train mini-batch of N images is labeled by the full panel under the
   current policy G_k. Every judge returns: label ŷ, confidence c ∈ [0,1], difficulty
   (low/medium/high), is_boundary (+ the confusion pair), a justification, **policy
   citations** (node ids) and **verbatim policy quotes** (§8.2).
2. **Select anchors.** Misaligned images — any image where a decisive judge disagreed with
   the SME truth *or* with the rest of the panel (everything but `all_agree`, so a 4-of-5
   correct panel is still eligible) — are ranked by the configured strategy —
   `random_misalignment` (S1, the null), `top_gradient` (avg |g| descending), or
   `top_importance` (the four-tier score, §6) — and the top ≤ `max_anchors` become the
   **negatives**. Up to `max_aligned_anchors` correctly-labeled images are sampled
   (deliberately unranked) as the **positives**.
3. **Draft.** The drafter receives the full current policy bundle, both anchor sets as
   compact vote records (SME truth, each judge's label/confidence/justification), and — in
   live runs — the anchor images themselves as provider-shaped image parts. It must emit
   the *single most impactful* edit as full-file Markdown changes.
4. **Clip.** `clip_changes` enforces the trust region: 1–5 node files touched, hard cap 5
   (`MAX_CHANGES_HARD_CAP`). No-ops are dropped; an all-no-op proposal skips the cycle.
5. **Score.** The candidate policy (base + overlay, materialized under
   `candidates/k<k>/`, never into `policy-graph/`) relabels the run's **fixed test
   partition** with the same panel.
6. **Gate.** `gate_comparison` computes macro-F1 before/after **over the intersection of
   images where both policies produced a decided system verdict** (so errored calls or
   majority ties can never flip the gate on coverage alone), then:

   ```
   accept(e_k)  ⇔  F1_test(G_k ⊕ e_k)  >  F1_test(G_k) + ε        (strict; ε ≥ 0, default 0)
   ```

   In `agent` mode the gate agent reviews the diff, the metric comparison *including the
   coverage block*, and the anchor evidence, and may veto (§8.2 lists its criteria). In
   `off` mode every clipped edit is accepted — the unfiltered-drift control arm; the
   metric is still recorded, and an unmeasurable edit (no decided verdicts) is never
   applied even then.
7. **Accept or skip.** Accepted edits mint policy version `v<run>.<k>` (run 7 accepting at
   cycle 2 mints v7.2; the scheme was introduced after run 3, whose accepts carry the
   legacy names v0.2/v0.3) and become the new incumbent, including inheriting the
   candidate's test evaluation as the next cycle's baseline. Skipped proposals are
   archived, not deleted. Failures are contained per cycle — a drafter error or a failed
   eval skips the cycle and the crank continues.

**End of run:** optional honest readouts — the **locked holdout** (~500 images, scored only
under start + final policy) and the **fixed cross-run benchmark** (1,000 images, same
images for every run ever; §8.3) — then the re-adjudication pass flags every image whose
latest evaluation is still misaligned into the cross-run SME queue.

### 2.3 Reproducibility: the seeding scheme

Every random draw derives its own `random.Random` from `(master_seed, role, k)` over
sample-id-sorted pools — no shared RNG state:

- test partition: `"{seed}:test:{class}"` per class, stratified by golden label, carved
  once at k=0 from the dev_golden pool and reused all run;
- train mini-batches: `"{seed}:train:{k}"`, drawn without replacement across cycles while
  fresh images last;
- anchor sampling: `"{seed}:anchors:{k}"` (S1) and `"{seed}:aligned:{k}"`;
  ranking strategies are deterministic (ties broken by image id).

Any single piece of a run is reproducible from the experiment record alone. The master
seed is `--seed` or a recorded `secrets.randbits(32)`.

---

## 3. The demos and what a run produces

### 3.1 The web UI — four views around one loop

Served by `scripts/rush_web_server.py` (locally and at rush.attiladobi.com):

1. **Run the loop** — the crank is the page: judge picker (cost badges), all knobs (§5),
   live per-model labeling telemetry with cancel, the learning curve (accepted-steps
   x-axis by default: a new tick only when an edit was actually accepted — skipped
   candidates are sampling noise, shown as hollow ghosts), the per-judge Δ-vs-k=0 table,
   the **gate ledger** (every proposed edit with its unified diff, its anchor images, each
   judge's vote on them, and the verdict + rationale), the policy-evolution knowledge
   graph with a cycle stepper, per-node diffs vs k=0 with the anchor images that taught
   each change, and the final 10×10 confusion grid.
2. **Run summary** — every judgment, per image: each judge's full response (label,
   confidence, difficulty, is_boundary + pair, justification, policy citations, verbatim
   quotes, tokens, cost), ranked misaligned-first.
3. **Adjudicate** — the cross-run human queue (§6.3): still-misaligned images deduped by
   content hash, stack-ranked by importance, with Confirm / Overturn / Uncertain actions
   that update the golden set's own confidence.
4. **About** — the project-agnostic metric reference (every formula in §6, every knob in §5).

### 3.2 Artifacts (the outputs)

| Artifact | Where | Truth status |
|---|---|---|
| Experiment record: config, every cycle (metrics, gate block, edit summary, anchors), summary, re-adjudication block | `data/experiments/<exp-id>/experiment.json` | **File is the truth**; Postgres `rush.*` mirror is soft-fail (a DB outage never blocks a run; `store_synced` records it) |
| Drafter + gate transcripts | `data/experiments/<exp-id>/agents/` | full audit trail per cycle |
| Candidate policy snapshots (accepted or not) | `data/experiments/<exp-id>/candidates/k<k>/` | every evaluated policy is preserved |
| Accepted policy versions | `policy-graph/<area>/v<run>.<k>/` | the shipped artifact; committed |
| Per-pass labeling runs: manifest, `label_votes.jsonl`, `llm_outputs.jsonl`, costs, errors, scoring | `data/runs/<run-id>/` | complete per-call ledger incl. latency, tokens, $ |
| SME re-adjudication log (confirm/overturn/uncertain) | `data/adjudication_reviews.jsonl` | append-only; file is the truth (created on the first SME review) |
| Label store: items, append-only human label events, materialized golden labels, gradient views | Postgres schema `rush` (`pipeline/labelstore/schema.sql`) | cross-experiment analysis layer; `rush.sample_gradient` mirrors the per-judge gradient (§6.1) as SQL; `rush.panel_signal` holds split/boundary rollups (the §6.2–6.3 signals live in Python only) |

Every judge vote validates against `schemas/label-vote.schema.json`; a real example:
gemini-3.1-flash-lite on `train_00605` returns label "3", confidence 0.95, difficulty
low, citations `[MD.digit.3, MD.digit.8, MD.digit.5]`, three verbatim policy quotes, a
stroke-topology justification, 7,035 input + 295 output tokens, $0.0022, 3.4s.

---

## 4. Evidence to date (runs 1–8, MNIST area)

| Run | Config (judges / gate / k×N×T / strategy) | Result | Cost |
|---|---|---|---|
| 1 | dry-run smoke (fakes) | plumbing only | $0 |
| 2 | 3 judges / agent gate / 2×10×25 / S1 | test system F1 saturated at **1.0** → gate correctly shipped nothing | $1.04 |
| 3 | 5 judges / agent gate / 5×40×100 / S1 | **2 accepts**: v0.2 (gate set 0.978→0.989, general rasterized-handwriting guidance in root) and v0.3 (→1.0 on the gate set — winner's-curse caveat applies, §8.3) | $9.50 |
| 4 | 4 judges / agent gate / 5×50×100 | **gate-agent veto observed**: at k=2 the metric passed (0.980→0.990) but the candidate eval covered only 98/100 test items, so the agent vetoed — "the apparent gain may be affected by missing examples" | $1.43 |
| 5 | 4 judges / gate off / 5×10×100 / S1 | stopped before any edit was scored — hence 0 accepts despite gate-off | $0.36 |
| 6 | 3 judges / **gate ON** (metric_only) / 3×15×50 / top_importance, seed 880808 | 0 accepts; gate-set F1 flat 0.9596 all run — **the gate held** while the drafter proposed an edit every cycle | $0.99 |
| 7 | identical config, **gate OFF**, seed 880808 | 3 forced accepts v7.1→v7.3; created 3 real boundary nodes (9↔0, 1↔7, 8↔6); gate-set F1 ended **0.9798 → 0.9564 → 0.9596** | $0.95 |
| 8 | 3 judges / metric_only / 5×20×100 / S1 | **1 correct rejection**: the k=1 candidate scored 0.9397 vs incumbent 0.9483 — the gate refused a worse edit; stopped at k=2 | $1.24 |

(k×N×T = cycles × train-batch size × gate-set size, §5.)

Three findings worth stating plainly:

1. **The gate mechanism behaves as designed** (runs 6 vs 7, plus run 8's refusal and run
   4's veto): with the gate on, nothing shipped that failed the acceptance test; with the
   gate off, the graph grew (the boundary nodes are genuinely well-formed policy) and the
   gate-set score ended two points below its own baseline. Honestly stated: that two-point
   drift is the same size as the noise floor in finding 3, on an n=1 pair with T=50 and no
   holdout/benchmark legs — so runs 6/7 demonstrate the *mechanism*, not yet a quantified
   harm. E2 and E5 (§9) upgrade this to statistical evidence.
2. **MNIST is near-saturated** for capable panels (F1 0.95–1.0 at baseline), so gated
   *lift* is hard to demonstrate there; runs 2 and 6 show correct null behavior instead.
   Headroom experiments belong in the GenAI area or with a deliberately degraded panel
   (experiment E7, §9).
3. **Panel measurement noise is real**: runs 6 and 7 share a seed (identical test
   partition) yet their baselines differ by ~2 F1 points (0.9596 vs 0.9798) from labeling
   stochasticity alone. Any acceptance rule with ε=0 is exposed to exactly this noise —
   the quantitative case for experiment E2 (§9).

---

## 5. Hyperparameters

The full knob surface of one run (web-UI defaults; CLI can exceed web ranges):

| Knob | Default | Range | Role in the optimizer |
|---|---|---|---|
| Judges (panel) | 2–5 models | hard 2–5 (web) | The measurement instrument. Every DQ number comes from this panel |
| Cycles `k_max` | 5 | 1–50 | Optimization steps per run; accepted edits mint `v<run>.<k>` |
| Train batch `N` (`batch_n`) | 20 | 2–200 | Mini-batch size — sampling noise vs signal per step |
| Test size `T` (`test_n`) | 100 | 10–1,000 | Gate-set size — controls acceptance-test noise (winner's-curse magnitude) |
| Seed | random (recorded) | any int | Reproducibility handle; the independent variable of the Lyapunov ablation (E4, §9) |
| Drafter | gpt-5.5 | `ALLOWED_POLICY_MODELS` | The step operator. Drafts, never judges; a cheap drafter is a legitimate config |
| Anchors (strategy) | `random_misalignment` | + `top_gradient`, `top_importance` | Which errors the drafter studies — random is the null hypothesis every ranked strategy must beat |
| Misaligned anchors (`max_anchors`) | 10 | 1–20 | The **negatives** per step (images attached to the drafter prompt) |
| Aligned anchors (`max_aligned_anchors`) | 10 | 0–20 | The **positives** — what already works; anti-over-correction regularizer |
| Max changes | 5 | 1–5 (hard cap) | The trust region: node files touched per edit ≈ step size / capacity |
| Gate mode | metric rule (UI) / agent (CLI) | agent · metric_only · off | Acceptance test; `off` is the drift control arm |
| Gate agent model | gpt-5.5 | openai/anthropic registry | The optional one-way veto valve |
| ε (epsilon) | 0 | ≥ 0; CLI + web API (no field in the web form) | Required improvement margin — the first winner's-curse mitigation (E2, §9) |
| Baseline policy (`policy_version`) | v0.1 | any existing version (CLI/API; UI pins v0.1) | The k=0 generator every run branches from |
| Holdout readout | off | flag | Locked ~500-image split scored under start + final only |
| Benchmark readout | off | flag (UI: "Benchmark readout") | Fixed 1,000-image cross-run validation split; two extra panel passes |
| Concurrency | 4 | 1–4 | In-flight calls **per hosted judge**; provider lanes are sized `concurrency × models-in-lane` and round-robin interleaved so same-provider judges run side by side. Local models ignore it (one call in flight per GPU card) |
| Child batch size | 10 (crank) | CLI | Images per provider batch inside a labeling pass |
| live / allow_spend | dry-run default (CLI) | both required for spend | Dry runs use deterministic fakes, never touch Postgres, never queue human work |

Read as an optimizer response surface: ε = trust-region margin; `max_changes` = edit
capacity (regularization strength); `max_anchors`:`max_aligned_anchors` = gradient batch
composition; `batch_n` = mini-batch size; `test_n` = evaluation noise floor; strategy +
drafter = the optimizer itself (Q3/Q4 in §9).

---

## 6. The gradient formalism

The optimization signal is built in three layers, implemented in
`pipeline/experiment/__init__.py`; the per-judge layer is additionally mirrored as the
`rush.sample_gradient` SQL view (Appendix A).

### 6.1 Per-judge gradient

Judge j returns label ŷ and self-reported confidence c ∈ [0,1] on image i with SME truth y:

```
p    = c        if ŷ = y      (correct)
p    = 1 − c    if ŷ ≠ y      (wrong)          — binary approximation of P(true class)

|g|  = 1 − p                   gradient magnitude: how informative this judgment is
h    = c·(1 − c)               curvature/uncertainty — peaks at c = 0.5, correctness-blind
loss = −ln max(p, 10⁻⁶)        per-sample cross-entropy
```

Abstentions and missing confidences carry no gradient (excluded, not zero-filled). The
four corners: confident-right (|g| ≈ 0, nothing to learn); **confident-wrong (|g| ≈ 1, the
most informative error** — either the policy fails here or the golden label is wrong);
unsure-right and unsure-wrong (both |g| ≈ 0.5 — the magnitude cannot distinguish them,
which is itself informative: an unsure judge teaches the same either way, that the item is
ambiguous under the current policy). The signal is live in recorded data: split panels
average |g| ≈ 0.36 vs 0.03–0.09 for consensus panels (k=200 label-store run; see
[DESIGN-policy-gradient-experiments.md](DESIGN-policy-gradient-experiments.md)).

### 6.2 Panel signals — two alignments that must not be conflated

Over the N_dec decisive (non-abstaining) judges on an image:

```
a = (# judges with ŷ = y) / N_dec        SME agreement   (LLM ↔ human, graded: 3/4, 2/4, …)
m = 1 − a                                 misalignment
κ = (# judges on the modal label) / N_dec LLM consensus   (LLM ↔ LLM, computed SME-BLIND)
b = (# judges flagging is_boundary) / N   boundary rate   (all judges in the denominator)
```

The two agreement numbers come apart exactly where it matters: a panel can be unanimous
(κ = 1) and entirely wrong (a = 0). For accounting, the panel collapses to its majority
vote (ties → no system verdict); for ranking, the graded signals are used in full.

**The four-tier hierarchy.** Consensus flips its meaning with alignment — when the panel
is wrong, agreement makes it *worse* (a systematic error); when right, agreement is the
ideal state:

| Tier | Alignment (majority) | LLM consensus | Meaning |
|---|---|---|---|
| **T1** | misaligned | high (κ > 0.5, strict) | Unanimous & wrong — the worst; top human priority |
| **T2** | misaligned | low (incl. exact ties) | Split & wrong |
| **T3** | aligned | low | Right, but the panel argued — boundary-instructive |
| **T4** | aligned | high | Unanimous & right — the ideal state |

A continuous score reproduces the ordering and interpolates the graded signals:

```
I_base = ( m + κ·(2m − 1) + 1 ) / 3        ∈ [0, 1]
```

(misaligned m→1: rises with κ; aligned m→0: falls with κ.)

### 6.3 Two derived scores, two consumers

```
amp             = (1 + 1.0·mean|g|) · (1 + 0.5·b)          confidence & boundary amplifier
anchor value    = I_base · amp                              → ranks policy-learning anchors
re-adjudication = anchor · (1 − p_human)                    → ranks the human queue
p_human         = 1 − 1/(m_SME + 4)                         m=1 → 0.800, m=2 → 0.833, m=3 → 0.857
```

**Anchor value** drives the `top_importance` selection strategy — which misalignments the
drafter studies. **Re-adjudication priority** is the same score faded by golden-label
confidence: the seed golden label counts as one human assertion (m_SME = 1); an SME
*confirm* raises it (the item fades); an *overturn* rewrites the effective truth and
**re-scores the whole panel against the new label** — a confidently-unanimous "error"
overturned to the panel's label flips T1 → T4 and drops off the queue. That is the "golden
set is not so golden" moment, made mechanical. An item is *resolved* only at
m_SME ≥ 2 — a lone overturn stays open (unresolved) awaiting a second human, ranked by its
re-scored importance: an overturn the panel now agrees with sinks (nothing left to argue),
while an overturn the panel still disputes stays high. Policy anchor value is deliberately
**not** faded: the policy must learn even a certain label.

### 6.4 Why this is gradient descent (and where the analogy is honest)

| SGD concept | RUSH realization |
|---|---|
| Parameters θ | The policy text G_k (Markdown nodes + edges) |
| Mini-batch | The seeded N-image train batch B_k |
| Per-sample loss | −ln p; aggregate DQ = system macro-F1 |
| Gradient | The panel's errors, ranked by \|g\| / four-tier importance, with images |
| Update θ ← θ − η·∇L | The drafter's edit G_k ⊕ e_k |
| Learning rate / trust region η | The ≤5 node-file clip |
| Step acceptance (line search) | The gate: strict improvement on the fixed test partition |
| Rejected step | Skip — incumbent persists, evidence archived |

This is an RL-*shaped* control loop over a text artifact — an orientation device, not a
theorem. The parameters are not differentiable; the "gradient" is evidence selection plus
an LLM's natural-language attribution; the step is acceptance-sampled, not computed.
Whether this optimizer is *sound* is precisely the research program of §9.

---

## 7. How RUSH compares: PPO, GEPA, VISTA

### 7.1 Positioning at a glance

| | **PPO** (Schulman et al. 2017, arXiv:1707.06347) | **GEPA** (Agrawal et al. 2025, arXiv:2507.19457, ICLR 2026 oral) | **VISTA** (Long et al. 2025, arXiv:2510.15831, CVPR 2026) | **RUSH** |
|---|---|---|---|---|
| Parameter | policy weights θ | prompts of a compound LLM program | a text-to-video generation prompt (scene plan) | the policy graph (Markdown = the judge prompt) |
| Update operator | gradient ascent on clipped surrogate | LLM reflective mutation + system-aware merge | "Deep Thinking Prompting Agent" rewrite from critiques | LLM drafter edit from misaligned anchors (+ pixels) |
| Reward / signal | environment reward or learned reward model | scalar task metric μ + textual feedback μ_f (traces, errors) | MLLM judge tournament + triadic critiques (model-generated only) | cheap-panel macro-F1 against **human golden labels** |
| Step control | clip ratio to 1±ε (trust region), KL-to-reference in RLHF | minibatch improvement over parent gates pool *admission* | champion carry-over via pairwise tournament | ≤5-file clip + strict-improvement gate on *deployment* + optional veto |
| Search state | single policy | **population** (instance-wise Pareto frontier) | single champion + fresh samples per iteration | single incumbent chain, branched per run |
| Human role | preference data (RLHF) trains the reward model offline | none in the loop | eval-only (66.4% preference study) | **in the loop**: golden labels, gate review, queue adjudication, golden-set overturns |
| Anti-local-optimum device | stochastic gradient noise; entropy bonus | Pareto frontier explicitly ("greedy extension of the single best gets stuck") | fresh sampling each iteration; adversarial judges | honest splits + random-selection null; population arm proposed (E6) |

### 7.2 PPO — the analogy RUSH is named for

PPO's clipped surrogate `L^CLIP = E[min(r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t)]` exists to stop
a policy update from moving so far that the local gradient estimate no longer holds. RUSH
implements the same *intent* with textual mechanics: the ≤5-node-file clip bounds the step
size; the strict-improvement gate on a fixed test partition is the acceptance test (closer
to a line search than to PPO's ratio clip); and the gate agent's unsoundness veto plus the
frozen root node play the role RLHF's KL-to-reference penalty plays — keeping the policy
from drifting into degenerate regions that happen to score well (in RLHF: reward hacking;
in RUSH: incorporation bias and memorized anchors). Honest differences: RUSH has no value
function or advantage estimation (the |g| ranking is a heuristic gradient magnitude, not a
baselined advantage); steps are proposed by an LLM rather than computed; the problem is a
single-step bandit (edit → measure), not a sequential MDP. The winner's curse (§8.3) is
the price of replacing an expected-value gradient step with a noisy acceptance test.

### 7.3 GEPA — the nearest optimizer family, and the sharpest contrast in search state

GEPA also treats prompts as parameters, also mutates them by LLM reflection over execution
traces, and argues the same premise RUSH's justification/citation traces embody — in the
paper's words, that "the interpretable nature of language can often provide a much richer
learning medium for LLMs, compared with policy gradients derived from sparse, scalar
rewards." Reported results (v2 abstract; v1 said 10% average): beats GRPO (24,000
rollouts, LoRA fine-tuning) by ~6% average and up to 20%, with up to 35× fewer rollouts
(678 on IFBench), and beats MIPROv2 by >10% aggregate.

The instructive differences:

1. **Search state.** GEPA keeps an *instance-wise Pareto frontier* of candidate prompts —
   every candidate that is best on at least one training instance stays alive as a
   possible parent — because always extending the single best candidate "can cause the
   optimizer to get stuck in a local optimum within the prompt space." RUSH currently
   keeps a single gated incumbent chain. This is RUSH's most credible local-optimum
   exposure, and the direct motivation for the population arm in E6 (§9).
2. **Signal source.** GEPA's feedback is self-referential to the system's own metric and
   traces; RUSH's gradient is anchored to *human* golden labels, and the labels themselves
   have a correction channel (overturns). GEPA optimizes what the metric can see; RUSH can
   repair the ruler.
3. **What acceptance protects.** Both systems test candidates — GEPA discards a mutation
   unless it improves over its parent on a minibatch, *then* admits it to the pool; RUSH's
   gate controls **deployment** of the single incumbent against the full fixed gate set,
   because in the enterprise setting each accepted version is a *shipped policy*, not a
   candidate in a pool. GEPA can afford to keep many survivors; RUSH must stand behind one.

### 7.4 VISTA — same algorithmic family, opposite trust model

VISTA (Google; test-time self-improvement for video generation) runs the same loop shape
as RUSH: sample candidate prompts → judge → select a champion → convert critiques into the
next edit, iterating ~5 rounds (the paper's iteration analysis reports ~0.7M tokens per
iteration; win rate over direct prompting rises across iterations to ~46% with the
remainder mostly ties, up to 60% pairwise vs state-of-the-art baselines, and 66.4% human
preference vs the strongest baseline). Its machinery is elegant: bidirectional pairwise
tournaments to cancel positional bias, and triadic judge courts (a normal judge, an
adversarial judge, a meta judge) per dimension — an idea worth borrowing for RUSH's gate
review (E6 note).

The fundamental difference is the **reward's provenance**: VISTA's entire signal is
model-generated — MLLM judges scoring an MLLM-refined prompt for an MLLM-consumed
generator, with humans only in the offline eval. That is the configuration RUSH's
architecture is explicitly built to avoid (§8.1): with no external anchor, judge
self-confirmation is unmeasurable *from inside the loop*. RUSH pays for the difference in
SME minutes — and then spends them only where the four-tier ranking says they buy the most.

Adjacent one-liners for completeness: **TextGrad** (arXiv:2406.07496) backpropagates
natural-language critiques through a computation graph ("autodiff with text");
**OPRO** (arXiv:2309.03409) prompts an LLM with a history of (solution, score) pairs and
asks for better ones; **MIPRO** (arXiv:2406.11695; shipped as MIPROv2 in DSPy) searches
instruction × few-shot-demo combinations with a Bayesian surrogate. RUSH's distinguishing bundle relative to all of
these: human-anchored reward with a repairable golden set, per-step gated acceptance with
an auditable ledger, and a bounded, reviewable edit as the only allowed move.

---

## 8. Preventing overfitting

The failure mode has a house name: **"the blue ring is hot" vs "fire is hot."** A drafter
staring at misaligned stove images can learn the general rule (fire is hot — transfers to
every heat source) or a hyper-specific one (the blue ring is hot — fails on the red ring;
worst case, it memorizes the training image). Everything in this section exists to force
the general rule. Three pillars carry the argument, then the structural regularizers.

### 8.1 Pillar 1 — Decoupled agent use

No agent in RUSH grades its own work:

- **Judges never draft; the drafter never scores.** The model that writes the edit cannot
  declare the edit good. All decision-quality numbers come from the panel of judges the
  edit is supposed to help.
- **The gate agent is subtractive only.** It can veto a metric-passing edit; it can never
  force one through. A compromised or sycophantic gate agent degrades to "no extra
  protection," never to "unearned accepts" (`resolve_gate_decision`'s override guard skips
  any accept-without-metric-pass).
- **The SME is the root of trust, and even the SME is instrumented.** Golden labels carry
  their own confidence (p_human), rise only through independent confirmations, and can be
  overturned — with the panel re-scored against the new truth.
- **Audit the agreements, not just the disagreements.** A random stream of *aligned* items
  goes to SMEs too; auditing only errors builds a confidently-wrong ruler (incorporation
  bias).
- **Multi-provider, multi-family panels** (OpenAI + Google + local Qwen/Gemma) make
  "target one judge's quirks" both detectable (per-judge Δ table: the objective is
  improvement for *every* judge) and vetoable (it is an explicit gate-agent criterion).

The VISTA contrast (§7.4) is the cautionary tale this pillar answers: a loop whose
proposer, scorer, and acceptor share a model family — or worse, a single self-referential
signal — cannot see its own systematic bias. T1 items (unanimous *and* wrong) are the
measured evidence of unanimous-panel/golden-label conflict — systematic cross-family panel
bias or a golden-set error, and the loop cannot tell which by itself. Either way they are
exactly what the architecture routes to humans first.

### 8.2 Pillar 2 — Interpretable traces with citation explanations (groundedness)

Every judgment and every step must be *explainable from the policy text* — not from vibes:

- **Per-vote grounding.** Every judge vote carries `policy_citations` (node ids) and
  `policy_quotes` (verbatim excerpts, hard-capped at parse time) plus a justification.
  A label that cannot cite the policy is visible as such in the Run-summary view.
- **Per-step grounding.** The drafter's proposal is a full-file unified diff of at most 5
  Markdown nodes; the gate ledger shows the diff next to the anchor images and every
  judge's vote on them, and the gate's rationale. The accepted knowledge-graph node shows,
  per node, *which images taught it* ("Node changes vs k=0").
- **The gate agent's veto criteria are themselves anti-overfitting rules**, applied to the
  edit text: ground-truth leakage, **overfitting to named examples instead of stating a
  general guideline**, targeting one judge's quirks, instructing judges to abstain, and
  piling pair-specific rules into the root instead of the owning node.
- **Full transcripts** (`agents/k<k>-drafter.json`, gate packets incl. the coverage block)
  make every decision re-derivable after the fact; gate verdicts are queued for SME review
  (`rush.gate_review`) as future training data for the critic.

Groundedness is an overfitting defense, not just UX: a hyper-specific rule ("image
train_00605 is a 3") *cannot survive* a pipeline where the edit must read as a general
guideline, must be cited by judges on unseen images to matter, and is reviewed as a diff
by a human who never saw the training batch.

### 8.3 Pillar 3 — Validation sets and a large fixed test set against local-optimum traps

RUSH separates three evaluation surfaces, in increasing order of honesty. A terminology
note for ML-trained readers, since RUSH's names are domain-flavored: the per-run "test
partition" is what ML convention calls a *validation set* (the optimizer selects against
it — we call it the **gate set** below); the locked holdout and the fixed benchmark play
the conventional *test set* role (never selected against).

1. **The per-run test partition (gate set, T=100 default)** — seeded, stratified, fixed
   within a run, disjoint from every train batch. The gate optimizes against it, therefore
   it is *biased by selection*: with one noisy candidate eval per step, an inherited
   incumbent score, and ε=0, the loop preferentially accepts upward noise — the
   **winner's curse**. This is a measurement bias in the acceptance test, not in the
   method, and it is documented rather than hidden (run 3's v0.3 "1.0" carries exactly
   this caveat; runs 6/7's same-seed ±2pt baseline gap shows the noise floor).
2. **The locked holdout (~500 images)** — never touched during optimization; scored only
   under the start and final policy. The *honest within-run lift*.
3. **The fixed cross-run benchmark (1,000 images)** — minted exactly once
   (stratified 100/digit from canonical MNIST test rows, disjoint from dev_golden and the
   holdout, seed 20260706, idempotent script), locked like the holdout, and identical for
   every run and every strategy forever. This is the surface on which runs, strategies,
   optimizers, and hyperparameters are compared — the defense against *selecting* a local
   optimum because the run-local gate set happened to like it.

Local-optimum traps are attacked from four directions: (a) the reporting rule — lift is
claimed from holdout/benchmark, never from the gate set; (b) planned acceptance-rule
hardening — ε > 0, paired incumbent re-evaluation, N-consecutive-wins (E2); (c) the
methodological null — every ranked selection strategy must beat seeded *random* anchor
selection on the same seed, so a gradient that merely relabels noise as signal is caught;
(d) search-diversity work borrowed from GEPA — a Pareto-pool optimizer arm (E6) tests
whether RUSH's single incumbent chain actually costs final quality.

### 8.4 Structural regularizers (the rest of the inventory)

- **Trust region:** ≤5 node files per edit, hard cap, no-ops uncounted; "fewer is better"
  is in the drafter contract.
- **Root freeze + node targeting:** class guidance belongs in the class's own node; one
  confusion pair per boundary node; the root is effectively frozen (root-dumping is a veto
  criterion). Prevents the policy degenerating into one unstructured mega-prompt.
- **Aligned anchors (the positives):** the drafter sees what already works, deliberately
  unranked, so fixing the negatives does not regress the positives — over-correction is
  the textual analogue of a too-large step.
- **Always-commit judging:** judges never abstain on MNIST; doubt lives in confidence and
  difficulty, so the gradient signal (|g|, h) stays populated instead of hiding in
  abstentions. (The GenAI area has a *designed* abstain lane for genuinely unjudgeable
  inputs — excluded from DQ denominators, never a place for the optimizer to hide errors.)
- **Coverage-safe gating:** before/after compared on the intersection of decided images,
  with the coverage block shown to the gate agent — run 4's veto is this guard firing in
  the wild.
- **Cycle containment:** a failed draft/eval skips one cycle instead of corrupting a run —
  no partial steps.

---

## 9. Next steps: the experimental program

The question the harness exists to answer: **is textual policy-gradient descent a sound
optimizer?** It decomposes into four research questions referenced throughout this
report: **Q1** generalization vs overfitting (blue ring vs fire), **Q2a** convergence in
the learning-rate sense (does honest DQ plateau?), **Q2b** convergence in the chaos sense
(seed sensitivity of the learned policy text), **Q3** ranked vs random anchor selection,
**Q4** optimizer architecture. Methodological spine, everywhere: *every ranked strategy is
compared against `random_misalignment` (S1) on the same seed and config* — random is the
null hypothesis the gradient must beat to earn its complexity. All experiments are runnable today with
`--strategy`, `--seed`, `--gate-mode`, `--holdout-final`, `--validation-final`; missing
pieces are flagged per experiment.

**E1 — Selection-strategy ablation (Q3, the central result).**
Hypothesis: importance-ranked anchors converge faster and higher, with fewer human
touches, than random. Protocol: seed-matched triples {S1, top_gradient, top_importance},
k_max ≥ 8, identical everything else; extend to S2 (consensus-lack), S3 (boundary flags),
S4 (difficulty), S5 (gradient × p_human composite) — computable from `rush.panel_signal`,
needs S2–S5 implemented as strategies. Readouts: benchmark ΔF1 (primary), accepted-step
learning curves, anchor-set overlap between strategies, Q1 generalization gap per
strategy. Decision rule: if no ranked strategy separates from S1 across ≥3 seeds, the
gradient formalism is not earning its keep — itself a publishable result.

**E2 — Gate rigor / winner's-curse quantification.**
Hypothesis: the gate-set curve overstates true lift by a measurable, ε-reducible amount.
Protocol: (a) noise floor — re-evaluate a *fixed* policy on the same test partition R
times (runs 6/7 imply σ ≈ 1–2 F1 points; measure it properly); (b) sweep
ε ∈ {0, σ/2, σ, 2σ}; (c) paired incumbent re-evaluation (re-score the incumbent alongside
every candidate — needs a driver flag); (d) N-consecutive-wins acceptance. Readouts:
(gate-set lift − holdout lift) per arm — the curse's size; accept rate; cost per accepted
point of *honest* lift. This experiment turns the acceptance test's known bias into a
calibrated design choice.

**E3 — Convergence, learning-rate sense (Q2a).**
Hypothesis: honest DQ plateaus; the trust region sets the effective learning rate.
Protocol: long runs (k_max 20–50) with periodic benchmark probes (cheap variant: probe
every 5 cycles — needs a `--benchmark-every` flag); sweep max_changes ∈ {1, 3, 5} and
batch_n ∈ {10, 20, 50}. Readouts: cycles-to-plateau, plateau height, accept-rate decay,
SME-queue drain rate (does the human queue shrink to a trickle as the policy learns?).

**E4 — Convergence, chaos sense (Q2b): seed sensitivity / Lyapunov probe.**
Hypothesis (to refute or confirm): different seeds → *vastly* different final policy
documents even at comparable DQ — a positive-Lyapunov-like regime where run history, not
the objective, dictates the learned policy. Protocol: one config × ≥5 seeds; embed each
final policy bundle (per-node text embeddings + graph-structure features); track
step-by-step divergence of same-config trajectories, not just endpoints. Readouts:
pairwise embedding distance vs within-run version drift (the natural scale); DQ spread vs
text spread quadrant plot. High text-divergence at equal DQ implies many equivalent optima
(fine for quality, fatal for "the policy" being a unique artifact — an important claim to
calibrate for the paper).

**E5 — Generalization-gap protocol (Q1: blue ring vs fire).**
Hypothesis: gated, clipped, positive-anchored edits generalize; unfiltered edits overfit.
Protocol: for every accepted edit record the quartet (train-batch ΔF1, gate-set ΔF1,
holdout ΔF1, benchmark ΔF1); the *shape* of decay across the quartet is the overfitting
measure. Add a targeted probe: for boundary-node edits (e.g. 9↔0), evaluate on held-out
images of exactly that confusion pair vs the rest — a rule that helps only the anchor
images and not the pair is a memorizer. Compare gate-on vs gate-off arms (run 6/7 pattern,
now with holdout/benchmark legs enabled) and max_changes 1 vs 5.

**E6 — Optimizer architecture A/B (Q4).**
Hypothesis: the gated single-chain drafter is competitive with population methods at equal
rollout budget — or it isn't, and RUSH should adopt a pool. Arms on identical seeds,
splits, and evaluation budget: (a) current drafter; (b) GEPA-style — maintain an
instance-wise Pareto frontier of candidate policies, mutate sampled parents, gate only at
*deployment* time; (c) node-statistic updates (edit frontmatter stats, no free-form
rewriting); (d) retrieval-augmented editing (drafter retrieves similar adjudicated cases).
Borrow VISTA's triadic review (normal + adversarial + meta) as an optional gate-agent
upgrade and measure veto precision against SME gate reviews. Report GEPA-style efficiency:
rollouts-to-target-DQ, not just final DQ. Needs: candidate-pool bookkeeping in the driver;
everything else exists.

**E7 — Headroom: escape MNIST saturation.**
MNIST at F1 0.95–1.0 demonstrates null-correctness, not lift (§4). Protocol: (a) GenAI
area runs (binary task, designed abstain lane, real ambiguity); (b) deliberately degraded
MNIST panels (weakest local models only) to reopen the gap between panel and policy
ceiling. This is a prerequisite for E1/E3 effect sizes, not an afterthought.

**E8 — Human-touch efficiency (the enterprise claim).**
Hypothesis: importance-ranked adjudication reaches a target golden-set quality with
materially fewer SME touches than FIFO/random queues. Protocol: replay recorded queues
under ranking policies (importance vs random vs confidence-only); measure overturn
discovery rate per touch (T1-first should surface golden-set errors fastest), queue drain
to resolved-state, p_human distribution shift. Readout: SME minutes per corrected label —
the third currency of the cascade, measured.

**Reporting standards for all of the above:** lift claimed from holdout/benchmark only;
gate-set curves always annotated as selection-biased; per-judge tables alongside system
metrics (the LLM-agnosticism objective); FPR reported first-class (downstream prevalence
correction consumes the operating point as (recall, FPR)); illustrative economics kept
clearly labeled as illustrative; every run reproducible from its committed
`experiment.json`.

---

## Appendix A — Formula → code map

| Formula | Function / view |
|---|---|
| p, \|g\| = 1−p, h = c(1−c), loss = −ln p | `vote_gradient` · mirrored as SQL `rush.sample_gradient` |
| a, m, κ, b, difficulty, tie handling | `panel_signal` (Python only; SQL `rush.panel_signal` holds coarser split/boundary rollups) |
| I_base, tiers T1–T4 (κ > 0.5 strict), anchor, re-adjudication | `importance_scores` (`CONSENSUS_HIGH=0.5`, `GRAD_WEIGHT=1.0`, `BOUNDARY_WEIGHT=0.5`) |
| p_human = 1 − 1/(m_SME + 4) | `human_confidence` |
| Fixed stratified test partition | `partition_test_train` (`"{seed}:test:{class}"`) |
| Seeded no-replacement mini-batch B_k | `sample_train_batch` (`"{seed}:train:{k}"`) |
| Anchor strategies S1 / top_gradient / top_importance | `select_anchors`; positives via `select_aligned_anchors` |
| Drafter packet (policy + votes + images) | `build_drafter_messages`, `DRAFTER_SYSTEM_PROMPT` |
| Trust-region clip (1–5 files) | `clip_changes` (`MAX_CHANGES_HARD_CAP = 5`) |
| Coverage-safe F1 before/after | `gate_comparison` |
| accept ⇔ F1_after > F1_before + ε | `metric_passes` (`GATE_METRIC = test_system_macro_f1`) |
| Accept/veto truth table | `resolve_gate_decision`; agent criteria in `GATE_SYSTEM_PROMPT` |
| Queue build / fold / overturn re-score | `build_readjudication`, `aggregate_readjudication`, `_recompute_importance`, `record_adjudication` |

## Appendix B — References

- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* 2017. arXiv:1707.06347.
- Ouyang et al. *Training language models to follow instructions with human feedback* (InstructGPT; PPO-with-KL-to-reference in RLHF). 2022. arXiv:2203.02155.
- Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* 2025. arXiv:2507.19457. ICLR 2026 (oral).
- Long, Wan, Nakhost, Lee, Pfister, Arık. *VISTA: A Test-Time Self-Improving Video Generation Agent.* 2025. arXiv:2510.15831. CVPR 2026.
- Yuksekgonul et al. *TextGrad: Automatic "Differentiation" via Text.* 2024. arXiv:2406.07496.
- Yang et al. *Large Language Models as Optimizers* (OPRO). 2023. arXiv:2309.03409.
- Opsahl-Ong et al. *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs* (MIPRO; MIPROv2 is the DSPy implementation). 2024. arXiv:2406.11695.
