# Session Handoff — Fable-5, 2026-07-05 (the enterprise-repositioning + polish pass)

> **Audience:** the next AI operators of RUSH — Claude-Code, Fable-5, and Attila's **networked OpenClaw multi-agent system** (agents with access to the Gemma memory on the LM Studio GPU host). This is embedded into `memory-embeddings/` so you can semantically query it (`scripts/query_memory.py "…"`). Read `HANDOFF.md` first for the base transfer; this file is the delta from the 2026-07-05 Fable-5 session.
>
> **Author:** Fable-5 (Claude Code), working directly with Attila.

## 1. What changed conceptually — RUSH is now an enterprise auto-judge

RUSH was reframed from "image-classification demo" to an **enterprise auto-judge platform** for Trust & Safety (violative / non-violative), content quality, search relevance (graded 1–5), and ads. A "project" is a policy for one category. The image demos (`Generative_AI`, `MNIST_Digits`) are toy proofs of the loop; the real targets are things like a 50-page Adult-Content policy or a cold-started PII policy.

**The load-bearing new idea — the escalation cascade (cheap → high-reasoning → SME).** Not "run 3 big models and vote." The inverse:
1. **Tier 1 — cheap consensus (measure):** low-cost models label the whole stream and resolve the aligned majority; this tier *is* the production prevalence metric (~tens of thousands/day).
2. **Tier 2 — high-reasoning panel (validate + critic):** only misaligned / lack-of-consensus / boundary / difficult items escalate to a 3× panel (Cohen's κ). The **1×-vs-3× gap = measured deployment bias**.
3. **Tier 3 — human SME (adjudicate):** only residual boundary cases reach a person via **GoldMiner**, ranked by priority score (`is_boundary`, difficulty, judge-disagreement, prior human touches, L2 coverage). Humans spend budget only on the boundary.

Wrapped in **two loops** (fast measurement / slow learning) and **three flywheels** (prompt-tuning ⊂ GDS-polish/audit ⊂ policy approval). The prompt *is* the policy; tuning it is RLHF over a text policy — "gradient descent in a document," where a critic locates the gap and an actor emits one gated/clipped edit.

**Four honesty guardrails the design must keep (do NOT ship the naive cascade without them):**
1. **Gate/clip every escalated edit** — accept only if held-out DQ improves and the edit is small (≤ ~5% tokens); prefer principle-level over item-level edits.
2. **Aligned audit stream** — escalating only disagreements builds a confidently-wrong ruler (incorporation bias); send ~5% of *agreements* to SMEs too.
3. **"The golden set is not so golden"** — experts side with the model ~⅓–½ of the time on re-adjudicated misalignments (Trust / CQ+Search / Ads); the top rung is *re-adjudication* (overturn/confirm) with a per-item cap, not "ask once." Falling overturn rate = a converging golden set.
4. **Separate prompt-lift from label-lift** — a DQ gain from cleaned labels is a golden-set event, not a modeling win. Treat a 100% score as a flag, not a trophy.

Governing constraint: **decision quality is bounded by golden-set coverage relative to production.** Full source canon: the 11 strategy docs (2 KDD notes on convergence + overfitting/PPO, the RLHF note, `measurement_policy_loop`, the re-adjudication MVP, demo notes, the PII cold-start deck, the exec summary, 2 whiteboards). Standard vocabulary: GDS (Golden Decision Set), GoldMiner, prompt-as-policy, decision quality (DQ), overturn/confirm, hard anchor, textual gradients, 1×-vs-3× bias, drift, "write prompts not policy docs", "AI surfaces, SME approves".

## 2. Operational: reaching the GPUs / Gemma memory from a networked agent

The GPUs live on **DESKTOP-RTX** (2× RTX 3090, 48 GB VRAM) and the mac mini, shared over LM Studio's **LM Link** mesh. Key gotcha discovered this session: LM Link makes models usable *inside the LM Studio app* but does **not** auto-expose the OpenAI HTTP server. To get an endpoint a separate process (RUSH, an OpenClaw agent) can call:

- **On the machine running RUSH**, start the LM Studio server: `~/.lmstudio/bin/lms server start --port 1234` (bridges to the loaded remote models). Verify: `curl http://127.0.0.1:1234/v1/models`.
- **RUSH now honors `RUSH_LOCAL_BASE_URL`** repo-wide (labeling pipeline `pipeline/providers/registry.py`, plus `scripts/query_memory.py` and `scripts/build_memory_embeddings.py` as of this session). A networked agent can point the whole repo at a remote GPU host with `RUSH_LOCAL_BASE_URL=http://<host>:1234/v1` instead of assuming loopback.
- Verified working this session: `text-embedding-embeddinggemma-300m-qat` (memory search) and `google/gemma-4-26b-a4b-qat` (vision — read MNIST digits 7/3/5 correctly in ~1–2s at 112px). Models available: gemma-4-26b-a4b-qat, qwen/qwen3.6-27b, embeddinggemma-300m-qat, nomic-embed-text.
- **Do NOT subnet-scan to find the host** — the safety classifier will (correctly) block a `/24` port sweep as recon. Use `lms server start` locally, or ask Attila for the host IP and set `RUSH_LOCAL_BASE_URL`.

## 3. Code shipped this session (branch `feat/fable-polish-high-priority`, not yet merged)

- **P0 unblock — web runs work off the Mac mini.** `pipeline/web/run_registry.py` hardcoded `/Users/sacsimoto/GitHub/RUSH/.venv/bin/python` as the runner interpreter → `FileNotFoundError` on any other checkout. Replaced with `_runner_python(repo_root)` (prefers repo `.venv`, falls back to `sys.executable`), wrapped `Popen` to clean up + raise a clean `APIError` on spawn failure, and added a catch-all in `handlers_runs.py:handle_api` so non-APIError exceptions return JSON 500 instead of dropping the socket. +2 tests → **421 passing**. Verified end-to-end: `POST /api/runs/start` now spawns with the repo venv.
- **`RUSH_LOCAL_BASE_URL` in the memory scripts** (see §2).
- **README:** added the escalation-cascade section + enterprise positioning; fixed the broken "Run the web interface" instructions (they pointed at a dead static-server path).

## 4. What's next — implement the polished demo (VP + AI-researcher audience)

Full plan lives in the session's polish-plan artifact. Priority order:
1. **Phase 1 — RL-loop integrity (do before showing any researcher):** the loop corrupts its own policy files (LLM-echoed `<!-- name.md -->` markers break frontmatter in GenAI v0.2/v0.3 → untyped nodes, dropped edges); holdout misalignments leak into policy proposals (violates the train-only discipline the papers demand); MNIST grow-batch filters to zero (binary-hardcoded); the drafting model dropdown offers a model the backend 400s; stale-base accept silently reverts newer versions.
2. **Phase 2 — make the cascade real:** cheap Tier-1 first rung + priority-score escalation + 1×-vs-3× bias surface; un-break the split-discipline UI (backend computes `reported/by_split/update_candidates` but the exporter never writes them to `web/summary.json`, so §4 is stuck on "defined-not-executed").
3. **Phase 3 — demo polish:** 404 doc links, empty GenAI gallery groups (label-vocab mismatch `gen_ai` vs `ai_generated`), stale "not yet executed" copy, vendor d3 (CDN-only kills §2 offline), lead with the PII cold-start example, surface the economics.

**Open questions for Attila:** is `rush.attiladobi.com` internet-reachable (no API auth + uncapped `limit` = spend risk)? demo date + lead vertical? build the full cascade vs. build the visible surface and narrate the rest?
