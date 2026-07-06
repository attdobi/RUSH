# RUSH demo flow — presenter script

The five sections of the web demo *are* the story. This script walks them top to bottom in ten minutes for a VP/VC audience, with a 3-minute short version and a data-science verification appendix at the end.

**Setup (before anyone is in the room):** start the server (`.venv/bin/python scripts/rush_web_server.py --host 127.0.0.1 --port 8766 --repo-root "$PWD"`, or use `rush.attiladobi.com`), open the UI, pick the **MNIST_Digits** demo, and confirm a scored cascade run is loaded so §4 and §5 have data. Keep the **Generative_AI** demo one click away for the "real policy" beat.

**The three money moments**, in order: the **cascade lanes** (§4), the **confusion matrix** (§4), and the **DQ-per-policy-version convergence chart** (§5). Everything else is connective tissue.

**One framing sentence to open:** *"Your policy team writes the guideline. RUSH turns it into a production judge — cheap models handle the easy majority, a high-reasoning model handles genuine ambiguity, and humans only ever see the cases where they change the answer."*

---

## The 10-minute walk (§1 → §5)

### §1 Sample — "every honest metric starts with an honest sample" (~1 min)

**Click:** Sample the golden set — N images per class into **dev golden** (train) and **locked holdout** (test).

**Say:** Two splits with one job each. Train drives policy updates; test is the reported number. The exporter *refuses* to blend them — if a metric tile says "reported," it is test-split only, by construction.

**Notice (for the DS in the room):** if test ever comes out above train, we treat it as a leakage hunt, not good news.

### §2 Grow — "the policy graph is the product" (~2 min)

**Click:** Open the policy graph. Show a node's Markdown. Show the version badge (v0.0 → v0.1 → …). If a proposal is pending, open it.

**Say:** This document is what the policy team owns and what the models judge with — **same bytes**. There is no translation layer to drift. When the loop finds misalignments, it drafts a *policy diff* — one trackable edit with the evidence attached — and queues it here for SME review. Accepting it materializes the next version.

**Notice:** a non-engineer can read every rule and every proposed change. The SME's job shifts from labeling thousands of images to approving a handful of well-argued edits. AI surfaces, SME approves.

### §3 Label — the cascade runs (~2 min)

**Click:** Show the model picker with live per-model cost and speed from real recorded runs (local gemma ~3.2 s/img and qwen2.5-vl ~4 s/img, free; hosted cheap-tier models at fractions of a cent per image). Hit **Run cascade**.

**Say:** Five cheap models label everything — tier 1, and it costs almost nothing: in the measured run, **2,000 calls attempted, 1,920 completed for $4.28 of hosted spend** (80 errored; the run finalized completed-with-errors — the honesty semantics working as designed), about **$2.23 per thousand completed labels**. Where they agree confidently, we're done. Where they hedge or split, the image escalates to a single high-reasoning judge — tier 2. Whatever tier 2 can't resolve falls through, honestly, to the human SME queue — tier 3. Even errored calls fall through; nothing vanishes.

**Notice:** the escalation trigger is panel-size-aware and tuned on measurement, not intuition — single-hedger flags were measured to be pure noise (98 of 98 had correct majorities), so they don't escalate.

### §4 Score — money moments #1 and #2 (~3 min)

**Click:** Open the scored run: the **escalation-cascade lanes** first, then the **confusion matrix** and per-digit F1/recall/FPR.

**Say (lanes):** 35.5% of images escalated. The 258 that didn't were **100.0% correct** — and say the caveat out loud: on a saturated toy task, n=258, treat it as a trigger-validation result, ~98.6% lower bound, not a guarantee. Every single error in this run lives in the escalated lane — the expensive model and the human only ever see cases where they can change the answer. This is the tokenomics thesis as a picture: cost concentrates exactly where difficulty concentrates.

**Say (confusion matrix):** and here is *where* the remaining difficulty lives — which classes get confused with which. Those confusions feed straight back into the policy graph as `confused_with` edges and become the evidence for the next proposed edit. Measurement and policy improvement are the same pipeline.

**Notice:** the train/test lanes — only train-split misalignments become policy-update candidates; test stays a pure ruler.

### §5 Quality — money moment #3: the convergence chart (~2 min)

**Click:** The decision-quality table (accuracy/F1/precision/recall/FPR per labeler *and* the majority-vote ensemble, with cost per 1k labels), then the **accuracy-by-policy-version trend chart**.

**Say:** Two things converge here. First, the ensemble row: cheap consensus at **97.7% accuracy and 0.25% false-positive rate** at ~$2.23 per 1,000 completed labels — that is the production metric. Second, the trend: decision quality per policy version. Today an SME approves every diff before it ships; the automated held-out DQ gate — accept an edit only if it improves held-out decision quality — is the next wiring step. The design target is a non-decreasing accepted-version curve, and this chart is where that discipline shows up. And the two convergences are coupled: as the policy improves, fewer items hedge, escalation falls, and the same decision quality costs less each iteration.

**Notice:** the reported tiles are test-split only, and the x-axis is policy versions, not the calendar — the chart plots the policy team's work.

**Close:** *"Measured here: ~$2.23 per thousand completed labels with zero errors leaking past the trigger in this run. The at-scale comparison — $710K of 3× human review per million images versus under $71K, with prompt caching pushing toward 1/50th — is the exec-brief target this mechanism exists to hit, illustrative rather than measured."*

---

## The 3-minute VC version

Skip §1 and §2. Three stops:

1. **§3, Run cascade (30 s).** *"Five cheap models label everything for about $2.23 per thousand completed labels. Only genuine disagreement escalates."* Point at the live cost badge.
2. **§4, cascade lanes (90 s).** *"35.5% escalated. The 64.5% the cheap tier kept had zero errors in this run — every error lives in the escalated lane. Expensive judgment is spent only where it changes the answer. That is the entire cost thesis, measured — with the toy-task caveat in the appendix."*
3. **§5, convergence chart (60 s).** *"And it improves as it runs: the policy is a human-readable document graph. Today an SME approves every accepted edit; the automated held-out decision-quality gate is the next wiring step — the design target is a curve that only goes up while cost per decision goes down. Illustrative at platform scale: $710K of human review per million images → under $71K, with orders-of-magnitude faster turnaround (~24h vs multi-week BPO cycles)."*

If they ask "what's the moat?": open a §2 policy node. *"The moat is this document — the policy, its edge cases, and the reasons, versioned and tested against production. Same bytes the judge runs on. Swap any model out; the asset stays."*

---

## DS appendix — claim → where to verify it in the UI

| Claim | Number | Verify in |
|---|---|---|
| Cheap-tier ensemble decision quality | 97.7% accuracy on the 400-image run (396 of 400 had a decisive majority; 387/396 correct), 97.7% macro recall | §5 quality table, ensemble row (test-split tiles in §4) |
| False-positive rate, macro and micro | 0.25% / 0.25% | §5 quality table + §4 per-digit FPR |
| Cheap-tier cost | $4.28 for 1,920 of 2,000 calls completed (80 errored; run finalized completed-with-errors) ≈ $2.23 per 1k completed labels | §3 live cost badge; `run_manifest.json` per-run ledger |
| Escalation rate | 142/400 = 35.5% | §4 escalation-cascade lanes |
| Cheap-resolved correctness | 258/258 correct in this run (trigger-validation result, ~98.6% lower bound — saturated toy task) | §4 cheap-resolved lane vs golden labels |
| Single-hedger flags are noise | 98/98 correct majorities | §4 consensus/audit view (boundary-voter counts) |
| Split discipline | train updates / test reports, never blended | §1 split badges; §4 reported tiles are test-only |
| Policy versioning + gated edits | v_n → v_{n+1} only via SME-accepted diff | §2 version badge + proposal review flow |
| Convergence over accepted versions | SME approves every diff today; the automated held-out DQ gate is the next wiring step (design target: non-decreasing DQ(v_n) on holdout) | §5 accuracy-by-policy-version chart + §2 review flow |
| Error honesty | errored tier-2 calls join the SME queue | §4 SME-queue lane; completed-with-errors run badge |
| Per-model speed/cost provenance | gemma ~3.2 s/img, qwen ~4 s/img, $0 | §3 model picker (from recorded runs) |

Ground rules when challenged: every measured number above traces to recorded run `20260706T042415-1b258772` in `data/runs/` (gitignored — run artifacts live on the demo machine; the repo carries the code paths that regenerate them) or a unit-tested code path (458 tests pass; 461 collected, 3 skipped). Tier-2 accuracy on the escalated set is **deliberately not quoted** — mechanics are shipped, numbers land with the next scored run. All Pinterest-scale figures ($710K → <$71K, orders-of-magnitude faster turnaround, 85.7% internal-pilot consensus accuracy) are exec-brief targets and internal pilot figures, labeled illustrative, and should be presented as such.
