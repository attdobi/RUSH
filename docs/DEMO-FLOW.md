# RUSH demo flow — presenter script

The demo is three tabs around one loop: **Run the loop** (the crank — config, live progress, learning curve, gate ledger, policy graph, confusion grid), **Run summary** (every judgment, per image), and **Adjudicate** (the cross-run SME queue). This script walks them in ten minutes for a VP/VC audience, with a 3-minute short version and a data-science verification appendix.

**Setup (before anyone is in the room):** start the server (`.venv/bin/python scripts/rush_web_server.py --host 127.0.0.1 --port 8766 --repo-root "$PWD"`, or use `rush.attiladobi.com`), open the UI, pick the **MNIST_Digits** demo, and confirm a completed run with accepted cycles is selected so the learning curve and ledger have data. Keep the **Generative_AI** demo one click away for the "real policy" beat.

**The three money moments**, in order: the **learning curve over accepted policy steps**, the **gate ledger with anchor-image evidence**, and the **adjudication queue** (the system telling the expert where their minutes matter). Everything else is connective tissue.

**One framing sentence to open:** *"Your policy team writes the guideline. RUSH turns it into a production judge — cheap models handle the easy majority, genuine ambiguity escalates, and humans only ever see the cases where they change the answer."*

---

## The 10-minute walk

### 1. The config panel — three roles, priced honestly (~2 min)

**Click:** expand "How this works" at the top of the loop view, then point at the three config groups.

**Say:** Three roles, three price points. The **judges** are your cheap panel — they label every image and they score every metric; every decision-quality number on this page comes from them, never from an expensive model. The **drafter** is the optimizer: once per cycle it reads the most instructive misaligned images — the actual pixels are attached to its prompt — and writes ONE policy edit of at most five files. It drafts, it never judges, so a cheap drafter is a legitimate choice. The **gate** is a deterministic rule — accept the edit only if the panel's test macro-F1 strictly improves — with an *optional* expensive agent that can veto a suspicious win but can never force one, and every rationale it writes lands in the ledger.

**Notice:** the judge picker shows measured $/1k-labels and seconds/image from real recorded runs (local gemma and qwen are free); the anchor selector offers unbiased random (S1) or top-gradient — confident-wrong panels first, |g| = 1−p.

### 2. Start a run — live, cancellable, seeded (~1 min)

**Click:** Start run. The live card appears: per-model calls, seconds/call, tokens/sec, cost so far, a progress bar, and a **Cancel run** button.

**Say:** Every run is numbered and seeded — fully reproducible — and starts from the same fixed baseline policy, so run numbers are comparable experiments, not vibes. Watch the phase line: it streams which cycle is labeling, how many calls are done, and what it has cost so far.

### 3. The learning curve + judges table — money moment #1 (~2 min)

**Say:** One line per judge plus the white system line. The x-axis only advances on *accepted* policy steps — a skipped candidate is sampling noise, shown as a hollow ghost, not learning. Accepted versions are named **v‹run›.‹k›** — v5.3 means run 5 accepted an edit at cycle 3. The judges table shows every metric with its delta against k=0 — per judge and for the system.

**Honesty beat, if a DS asks:** the gate set is formally a validation set (the loop adapts to it); that's exactly why the locked 500-image holdout and the fixed 1,000-image cross-run benchmark exist — tick "Benchmark readout" and run-vs-run numbers land on identical images.

### 4. The gate ledger — money moment #2 (~2 min)

**Click:** expand a cycle's evidence row: the anchor images that drove the edit, each judge's vote against SME truth, and the proposed diff.

**Say:** Every accepted step is small enough for a human to read — never more than five node files — and every gate decision carries its metric evidence, the diff, and (when the agent gate is on) the agent's rationale. The SME reviews the gate's decisions *after* the loop — correct, incorrect, unsure — and every verdict is recorded as training data for the critic itself. The expert isn't removed; they're repositioned.

**Notice:** the policy graph below steps through k with per-node diffs vs k=0 and the anchor images that taught each node; the 10×10 confusion grid at the bottom shows where the remaining difficulty lives under the final policy.

### 5. Run summary — every judgment, inspectable (~1 min)

**Click:** the Run summary tab; pick the run and a cycle; expand a misaligned row.

**Say:** Nothing here is a black box. Every image, every judge, the full response — label, confidence, difficulty, boundary flag, the policy nodes it cited, verbatim quotes from the policy, tokens, cost. Ranked misaligned-first so the interesting rows are on top.

### 6. Adjudicate — money moment #3: the SME queue (~2 min)

**Click:** the Adjudicate tab. Sort by the default composite, then flip to gradient.

**Say:** Whatever the loop could NOT fix, it hands to the expert — deduped across runs, with the run numbers that flagged it. Two rankings, two questions: the consensus→confidence→difficulty composite surfaces *panel-confused* items — the policy is unclear. The gradient ranking (|g| = 1−p, confident-wrong first) surfaces panels that are unanimously, confidently wrong — the strongest hint that the *golden label itself* may be wrong. That's the overturn workflow: the system doesn't just improve the policy; it ends each run by telling the SME exactly where their scarce minutes matter most.

---

## The 3-minute VC version

Three stops:

1. **The loop config (30 s).** *"Cheap models judge and score everything — about $2 per thousand labels measured. The expensive model only ever drafts policy edits or vetoes a suspicious win. Nobody expensive is in the scoring path."*
2. **The learning curve + ledger (90 s).** *"One click runs k cycles of label → draft → gate. Every accepted edit had to beat the current policy on a fixed test partition, every edit is five files or fewer so a human can read it, and the expert audits the gate instead of sitting inside the loop. The design target is a curve that only goes up while cost per decision goes down. Illustrative at platform scale: $710K of human review per million images → under $71K, with orders-of-magnitude faster turnaround."*
3. **Adjudicate (60 s).** *"And the residue is not a backlog, it's a ranked queue: lack of consensus means the policy is unclear; confident-wrong means the golden label might be. Human attention flows only where it changes a rule."*

If they ask "what's the moat?": open a policy node in the graph. *"The moat is this document — the policy, its edge cases, and the reasons, versioned and tested against production. Same bytes the judge runs on. Swap any model out; the asset stays."*

---

## DS appendix — claim → where to verify it

| Claim | Number | Verify in |
|---|---|---|
| Cheap-panel ensemble decision quality | 97.7% accuracy on the 400-image cascade run (396/400 decisive; 387/396 correct), 97.7% macro recall, 0.25% FPR | Judges table on any scored run; `data/runs/<id>/scoring/` |
| Cheap-tier cost | $4.28 for 1,920/2,000 completed calls ≈ $2.23 per 1k completed labels | judge-picker cost badges; `run_manifest.json` per-run ledger |
| Escalation cascade (tier-1 → tier-2 → SME) | 142/400 = 35.5% escalated; 258 cheap-resolved, all correct in that run | `scripts/run_cascade.py` / `POST /api/runs/start-cascade` artifacts (CLI/API; the §-based cascade UI was retired with the Inspect tab) |
| Split discipline | train updates / test reports / holdout + benchmark locked | experiment.json `splits`; manifest `split` fields; `HOLDOUT_SPLITS` in `pipeline/manifest.py` |
| Gate semantics | accept iff test system macro-F1 strictly improves; ≤5 changes/edit; agent can veto, never force | gate ledger rows + `rush.gate_decision`; `resolve_gate_decision` truth table tests |
| Version naming | v‹run›.‹k› = run R accepted at cycle k, branching from the fixed v0.1 baseline | policy-graph dir + KG version picker + `rush.generator_version.parent_id` |
| Gate auditability (RLHF of the critic) | every decision carries metric evidence, diff, rationale, post-hoc SME verdict | ledger review buttons; `rush.gate_review`; `data/experiments/<id>/agents/` |
| Cross-run benchmark | fixed 1,000 images (100/digit, canonical MNIST test rows), scored start + final | "Benchmark readout" checkbox; `summary.benchmark_system` |
| SME queue ranking | consensus → confidence → difficulty composite, or gradient \|g\| = 1−p | Adjudicate tab sort modes; `rush.sample_gradient` view |
| Error honesty | errored calls fall through, runs finalize completed-with-errors | run badges; `completed_with_errors` in status payloads |

Ground rules when challenged: every measured number traces to a recorded run under `data/runs/` or a unit-tested code path (521 tests pass). Tier-2 accuracy on the escalated set is **deliberately not quoted** — mechanics are shipped, numbers land with the next scored run. All Pinterest-scale figures ($710K → <$71K, ~24h vs multi-week BPO cycles, 85.7% internal-pilot consensus accuracy) are exec-brief targets and internal pilot figures, labeled illustrative, and should be presented as such.
