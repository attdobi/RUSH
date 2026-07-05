# RUSH Demo Review — Logic & Flow (2026-07-04, overnight)

**Reviewer:** Pista (CEO), for Attila. Deep step-by-step walkthrough of the demo against the RUSH theory.
**Branch reviewed:** `feat/mnist-ux-kdd` @ `1f3888a5` (== origin/main at review time), plus in-flight fixes noted inline.
**Method:** traced the §1–§7 narrative in `web/index.html`, the API route table, and the backend loop code (`policy_iterator.py`, `policy_diff.py`, `scoring/*`, `web/aggregator.py`). Exercised the live server on :8766.

---

## 0. TL;DR

**The RL engine is real and mostly wired.** The learning signal (misalignments + boundary cases → policy diff) and the version-materialization (accept → KG v(k+1)) genuinely work. The demo's weaknesses are **not** missing machinery — they are **(a) UX that hid the loop's closing move**, and **(b) a scientific-integrity gap in the reward signal on the MNIST path.** Fix those two and the demo tells a clean, honest RL story.

**Grade of the loop as-shipped:** engine B+, storytelling C (pre-tonight's fixes), scientific rigor B− (leak-safe for GenAI, NOT for MNIST).

---

## 1. The RUSH theory (what the loop is supposed to be)

RUSH = **R**einforcement learning **U**sing **S**ME feedback and **H**igh-reasoning models. The demo is the loop:

```
cold-start policy graph v0.1  (the "generator prompt" as a KG)
        │
        ▼
 §1 Sample golden set  →  §3 Label a batch  (models cite policy nodes, output structured labels + confidence + is_boundary)
        │
        ▼
 §4 Score vs SME truth  →  misalignments + boundary cases  (the LEARNING SIGNAL)
        │
        ▼
 §2 Propose policy diff (high-reasoning model reads the misalignments)  →  SME accepts/rejects
        │
        ▼
 accept → materialize KG v(k+1)  →  relabel  →  §5 measure decision quality on HELD-OUT test, per version  (the REWARD SIGNAL)
        │
        └────────────────────────── repeat, policy improves version over version ──────────────────────────┘
```

The two signals that make it *reinforcement learning* rather than a demo:
1. **Learning signal** — proposals must be *driven by* where the panel disagreed with SME truth (misalignment) and where models hedged (boundary). 
2. **Reward signal** — improvement must be measured on a **held-out test split**, per policy version, or you're just overfitting the policy to your eval set.

---

## 2. Step-by-step walkthrough

### §1 Sample — `/api/thumbnail`, sampler
- **Logic:** builds the golden pool (N per class). Clean. Feeds the label pool.
- **Flow:** OK. `N per class` → pool; `k per split` in §3/§4 draws from it. The relationship is stated in copy (good).
- **Verdict:** ✅ solid.

### §2 Grow the generator prompt — policy graph + proposals (RL CORE)
- **Logic:** renders the current KG version as a graph; lists pending proposals; `Accept update` → `accept_proposal()` copies base version → new dir, applies diff files, stamps `accepted_into_version`. **This works** — GenAI has `v0.1 → v0.2 → v0.3` on disk from real iterations.
- **GAP (was): the loop's closing move was invisible.** `policyGraphNextNote` = **"defined-not-executed"**, "Next version" disabled, and the misalignment-driven proposal entry point had been scattered into a per-row "Propose diff from this row" button. **→ Being fixed in-flight (run `5ea610a4`, restore dedicated iteration section, make "Next version" real, area-aware for MNIST).**
- **Verdict:** engine ✅, surfacing ⚠️ (fix landing tonight).

### §3 Label — `/api/runs/start`, `/status`, `/log`
- **Logic:** pick models, k, split, **policy version** (so you can label with v(k+1)), concurrency. Cites the selected KG.
- **Flow:** ✅ policy-version selector closes the loop back to §2. Now cancellable (feature E shipped).
- **Scheduler note:** pre-fix, the runner serialized the two local GPU models and starved hosted openai (model-major batch order + workers blocked on a size-1 semaphore). **→ Fixed on `wt/x1-scheduler-lanes` (per-lane executors), pending merge.**
- **Verdict:** ✅ (with scheduler fix merged).

### §4 Score — per-image audit (consensus / misalignment / borderline) + residual misalignments
- **Logic:** the audit surface. Misalignment tab ranks model-vs-SME disagreement; borderline surfaces hedging. This is exactly the learning-signal source, and `policy_iterator._select_priority_rows` **correctly** consumes high/medium-severity misalignments (excluding all-agree) + borderline highlights into the diff prompt. ✅ theory-sound.
- **GAP: `scoreAlgoBadge` = "intended pipeline · defined-not-executed"** and the honest footnote: scoring exports **one combined run-level snapshot**; train-vs-test separation "until backend separation is wired." See §Reward-signal gap below.
- **Verdict:** learning signal ✅, split discipline ⚠️.

### §5 Quality — `/api/decision-quality` (the REWARD signal)
- **Logic:** compares accuracy / F1 / FPR-FNR / review burden / cost **by labeler, run, and policy version**. `aggregate_decision_quality` **does** group by `policy_graph_version` and can filter to one version — so version-over-version comparison is structurally present. ✅
- **GAP (critical): held-out discipline is only half-wired.** `decision_quality.py` implements it correctly — `split_kind()`, `by_split` for train+test, `reported_split: "test"`, `reported = by_split["test"]` (report on holdout — exactly right). **But the actual MNIST run's `decision_quality.json` has NO `by_split`** — the 10-class `decision_quality_multiclass.py` path doesn't populate it. So: **leak-safe reward signal for GenAI/binary, NOT for MNIST.** If you iterate MNIST policy and read "improvement" off the combined snapshot, you risk train/test leakage → false RL signal.
- **Verdict:** ⚠️→❗ the single most important scientific fix.

### §6 Insights — `/api/insights`
- **Logic:** majority-wrong first, then model disagreement / boundary concentration / recurring pair disagreement. Good triage lens; complements §4.
- **UX (your point 1):** top block repositioning — in-flight.
- **Verdict:** ✅ content; minor layout fix landing.

### §7 About — theory / methodology / future work
- **Logic:** states the cold-start-graph-not-a-prompt-blob thesis, cost/quality frame, SME-feedback loop. Accurate and well-aligned to the actual system.
- **Verdict:** ✅.

---

## 3. Theory-vs-execution audit (the "defined-not-executed" tags)

| Claim in UI | Backend reality | Status |
|---|---|---|
| §2 "Next version" (defined-not-executed) | `accept_proposal()` fully materializes v(k+1); GenAI v0.1→0.3 on disk | Engine ✅, UI fix in-flight |
| §4 train/test "intended pipeline · defined-not-executed" | `decision_quality.py` separates; `decision_quality_multiclass.py` (MNIST) does NOT | ❗ port needed |
| §5 "by policy version" | `aggregate_decision_quality` groups by version | ✅ |
| Misalignment/boundary drive proposals | `_select_priority_rows` + `build_user_prompt` feed them (incl. images) | ✅ |

**Takeaway:** the demo has been *honest* by labeling aspirational bits "defined-not-executed" — good integrity — but two of those tags can now be made **true** with modest work.

---

## 4. Recommendations (prioritized)

**P0 — Reward-signal integrity (act tonight, X1 backend/scoring).** Port the train/test discipline from `decision_quality.py` into `decision_quality_multiclass.py`: emit `by_split` {train,test}, `reported_split: "test"`, `reported = by_split["test"]`, per policy version. Then flip §4/§5 copy from "defined-not-executed" to live. *Without this, MNIST RL "improvement" is not trustworthy.*

**P1 — Close the loop visibly (in-flight, run `5ea610a4`).** Restore the dedicated "misalignments → propose diff → accept → KG v(k+1) → relabel" section; make "Next version" execute; area-aware so MNIST can finally go v0.1→v0.2. Remove the per-row propose button. Render the actual image (not the path). Reposition §6 top block.

**P2 — Scheduler parallelism (committed, merge tonight).** `wt/x1-scheduler-lanes`: per-lane executors so both GPU cards + hosted providers run truly parallel, no starvation.

**P3 — Make the RL payoff legible (recommend, Attila's call).** Add a tiny "version delta" strip in §5: v(k) vs v(k+1) **test** accuracy/F1 with the arrow. That single widget is the money shot of the whole demo — it *shows* the policy learning. Currently a viewer must eyeball the version dropdown. (Flagging rather than auto-building — it's a design/story call.)

**P4 — Loop honesty polish (recommend).** The hero's 5-step loop should visibly light up step-by-step as the user completes each (Sample→Seed→Label→Propose→Accept). Right now it's static chrome. Low effort, high narrative payoff.

---

## 5. What I'm acting on overnight vs flagging

- **Acting (via named engineers, `[X#]`-tagged commits):** P0 (X1), P1 (in-flight X2/X3), P2 (merge X1's branch).
- **Flagging for your morning call:** P3 (version-delta widget — story/design decision) and P4 (animated loop stepper — nice-to-have). Both are cheap; I didn't want to guess at placement/story on your behalf.

*— Pista*
