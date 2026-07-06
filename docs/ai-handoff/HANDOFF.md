# RUSH — AI-to-AI Knowledge Transfer / Handoff

> **Audience:** the next AI operators of this repo — **Claude-Code** and **Fable-5** (running locally with access to Gemma-embeddings on Attila's GPUs), plus any human reviewer.
> **Author:** Pista (CEO agent, Claude Opus 4.8) with the OpenClaw office team (Theo/CTO, engineers X1–X3).
> **Date:** 2026-07-05.
> **Purpose:** transfer *everything* needed to continue the RUSH project without re-deriving it — architecture, the current state, every fix and decision made in the 2026-07-04/05 work, open issues, and how to use the embedded memory index.

This document is the human-readable master. It is also chunked + embedded (Gemma-embedding, 768-dim) into `docs/ai-handoff/memory-embeddings/` so you can semantically search the project's memory locally. See **§9 Using the embedded memory**.

---

## 1. What RUSH is (one paragraph)

**RUSH = Reinforcement learning Using SME feedback and High-reasoning models.** It is a policy-graph system that scales a subject-matter expert's (SME) decision quality to cheaper labelers (BPO reviewers, low-reasoning LLMs, production ML). Policy is explicit, **versioned**, **graph-structured** (Obsidian-style Markdown nodes + `edges.json`), and tested against golden examples. The loop: label a batch with generator-prompt version `v_k` → mine SME/model **misalignments + boundary cases** → propose a **policy diff** → SME accepts → **materialize `v_{k+1}`** → relabel. Each iteration grows the knowledge graph (KG) and raises decision quality, while a **multi-LLM consensus** resolves the easy majority cheaply and escalates only boundary/disagreement cases to expensive high-reasoning models (measured by per-batch cost).

Two demos/areas exist:
- **`Generative_AI`** — the original pilot (gen_ai vs not_gen_ai image classification). Has policy versions **v0.1, v0.2, v0.3** on disk (real iterations happened).
- **`MNIST_Digits`** — a 10-class digit demo added in the 2026-07-04 "MNIST-UX-KDD" overhaul. Only **v0.1** on disk until an SME accepts a proposal (then v0.2 materializes).

---

## 2. Repo map (where things live)

```
RUSH/
├── README.md                     # project intro (updated in this handoff to point here)
├── web/                          # dependency-free demo UI (vanilla JS, no build step)
│   ├── index.html                # single page, §1 Sample → §2 Grow → §3 Label → §4 Score → §6 Quality → §7 About (+ collapsed Provenance at bottom)
│   ├── app.js                    # §4 Score audit: per-digit table, consensus tabs, stat cards
│   ├── run-trigger.js            # §3 Label: start/cancel runs, live progress card, polling
│   ├── policy-graph.js           # §2 policy graph render + version picker + Next-version stepper
│   ├── policy-diff.js            # §2 propose-diff / accept / reject proposal flow
│   ├── policy-grow.js            # §2 grow-batch controls
│   ├── decision-quality.js       # §6 decision-quality metrics
│   ├── insights.js               # (post-2026-07-05) folded into §4; standalone §5 removed
│   ├── justifications.js         # per-image justification drawer + inline evidence image
│   ├── genai-sampler.js          # fallback sampler when manifests missing
│   └── demos.js                  # demo registry (GenAI / MNIST), copy, area wiring
├── pipeline/
│   ├── policy_diff.py            # propose_diff, accept_proposal (materializes v_{k+1}), list/get proposals
│   ├── web/handlers_policy.py    # /api/policy/* HTTP handlers
│   ├── runner.py                 # bulk labeling runner (per-lane executors — see §5 scheduler fix)
│   ├── run_registry.py           # job lifecycle: start/cancel/finalize, job-state files, speed summary
│   └── providers/                # provider clients + pricing.py (cost per token; local models = $0)
├── policy-graph/
│   ├── Generative_AI/{v0.1,v0.2,v0.3}/   # Markdown nodes + edges.json per version
│   └── MNIST_Digits/v0.1/                # MNIST digit policy (v0.2+ materialize on accept)
├── scripts/
│   ├── run_bulk_labeling.py      # headless labeling entrypoint (--area, --models, --split, --limit)
│   ├── rush_web_server.py        # the web server (served by LaunchAgent, see §4)
│   ├── finalize_orphaned_runs.py # clears zombie runs (job-state + manifest)
│   ├── sample_mnist_gold_sets.py / sample_genai_gold_sets.py
│   ├── score_labels.py, aggregate_costs.py, build_policy_pdf.py
│   └── build_memory_embeddings.py / query_memory.py   # (this handoff) embed + search this memory
├── data/
│   ├── runs/                     # run manifests (run_manifest.json) + _jobs/<job_id>.json job-state
│   ├── images/                   # prepared images (path-validated image route)
│   └── seed/                     # mock golden-set records
├── schemas/                      # JSON schemas (graph, image/split, label/vote, model-output, review, patch, export, metric)
├── tests/                        # pytest suite (406 passing as of 2026-07-05)
└── docs/                         # design + runbooks + THIS handoff (docs/ai-handoff/)
```

**Key architecture reads (already in `docs/`):** `architecture.md`, `architecture-bulk-labeling.md`, `label-hierarchy.md`, `DESIGN-per-project-ontology.md`, `runbook-bulk-labeling.md`, `mnist-prompt-v0.md`, `mnist-benchmarks-v0.md`, `DEMO-REVIEW-2026-07-04.md`.

---

## 3. How to run it (operational quickstart)

- **Web server:** runs under a macOS LaunchAgent `com.attdobi.rush-web` using the repo venv `.venv/bin/python scripts/rush_web_server.py --host 127.0.0.1 --port 8766 --repo-root <repo>`. Serves the working-tree `web/` and the `/api/*` endpoints.
  - Restart: `launchctl kickstart -k gui/$(id -u)/com.attdobi.rush-web`
  - **ALWAYS verify after restart:** `launchctl list | grep rush` for PID, then `ps -o pid,lstart -p <pid>` for start-time. See §6 "stale server" lesson.
  - URL: `http://127.0.0.1:8766/web/`
- **Headless labeling:** `cd <repo> && .venv/bin/python scripts/run_bulk_labeling.py --area MNIST_Digits --models <csv> --split dev_golden --live --allow-spend --concurrency 4 --limit 20`
  - **MUST use `.venv/bin/python`** (has `openai` 1.109.1). Bare system `python3` → `ModuleNotFoundError: openai` → instant abstain → fake `provider_error` on all calls. (Cost ~20 min once.)
- **Tests:** `cd <repo> && .venv/bin/python -m pytest tests -q` → **406 passing**.
- **Local models:** LM Studio at `http://127.0.0.1:1234/v1` (Gemma, qwen, embeddings) — Attila's GPUs, treated as local by the Mac mini host.

---

## 4. Providers, models, and the panel

- **Hosted:** `openai/gpt-5.4-mini-low`, `anthropic/claude-haiku-4-5-low`, `google/gemini-3.1-flash-lite` (+ others). Priced via `pipeline/providers/pricing.py`.
- **Local (LM Studio :1234):** `local/gemma-4-26b-a4b-qat`, `local/qwen3.6-27b` (+ `qwen3.6-27b-low`, `qwen3.6-35b-a3b` MoE). Local = $0 cost.
- **Embeddings (LM Studio :1234):** `text-embedding-embeddinggemma-300m-qat` — **Gemma-embedding, 768-dim.** This is what OpenClaw's memory search AND this handoff's embedded memory use.
- The panel labels each image across N models; **consensus** resolves the aligned majority; **misalignment/boundary** escalates. Per-model + per-batch cost recorded in `run_manifest.json` and shown live in §3.

---

## 5. The 2026-07-04/05 work — what changed and why (the important part)

This is the substance of the recent effort. Each item = a real fix on branch `feat/mnist-ux-kdd`.

### 5.1 MNIST demo end-to-end + 10-class scoring
Added the `MNIST_Digits` area threaded through the web run path: `run-trigger.js → _safety.py → run_registry → run_bulk_labeling(--area) → runner LabelRequest.area → get_ontology`. Emits digit labels `MD.digit.N` (conf ~1.0), NOT gen_ai/not_gen_ai. 10-class scoring (consensus / run_scoring / misalignment / aggregator + schemas).

### 5.2 Zombie-run fix (job-state vs manifest)
**Root cause:** a run shows "running" via `is_job_running()` reading `data/runs/_jobs/<job_id>.json` (returncode/finished_at), NOT `run_manifest.json`. Retro-finalizing only manifests did NOT clear the UI. Fix: crash/cancel now auto-finalizes **both** manifest + job-state; added `_finalize_dead_job` liveness + `scripts/finalize_orphaned_runs.py`. **Lesson:** to clear a zombie you must finalize the job-state file, not just the manifest.

### 5.3 Area-aware image sizing (gemma MNIST bug)
**Root cause:** MNIST 28px images upscaled to 1024×1024 = **8541 input tokens** → blew gemma's LM Studio context (400 "Context size exceeded" or over-reasoned 1900–2400 tok → abstain). Fix: **area-aware `max_image_size`** (MNIST ~112px). PROVEN: at 112px gemma labels a digit in **1.9s / 152 tok**. Also faster/cheaper for all models.

### 5.4 qwen `reasoning_effort="none"` knob
Registered `local/qwen3.6-27b-low` etc.; LM Studio 0.3.x honors `reasoning_effort="none"` (0 reasoning tokens). On MNIST this drops qwen from ~65s/2890tok to **~9.5s / 240 tok** and still correct.

**DECISION (Attila, 2026-07-05) — DO NOT globally disable qwen reasoning.** `reasoning_effort="none"`/low is acceptable ONLY for **trivial perceptual areas like MNIST_Digits** (a digit is a digit; the policy graph is simple stroke/topology). For **Generative_AI and any policy-grounded area, KEEP REASONING ON.** Reason: qwen's reasoning is load-bearing — it cites policy nodes, grounds the justification in the KG, sets `is_boundary`, and produces the *reasoned disagreement* that the RL loop mines into policy diffs. Disabling it degrades policy-node citation, justification grounding, boundary detection, AND the misalignment signal that feeds the learning loop (the "H" — High-reasoning — in RUSH). **Fix qwen speed at the GPU/infra layer (KV f16→f8, avoid CPU offload — see §7.1), NOT by turning off reasoning.** Do not "optimize" reasoning off globally.

### 5.5 Per-lane executors (scheduler starvation fix)
**Root cause:** `runner.py` used model-major batch order + a semaphore-held worker, so gemma→qwen serialized on the 2 GPU cards and hosted openai got starved to 0 (observed live: gemini40/haiku40/gemma33/qwen0/openai0). Fix: **per-lane executors** — hosted lane = concurrency-wide; each local model = 1 dedicated worker on its card; no cross-lane starvation. True hosted∥gemma∥qwen parallelism. (`4a91305d [X1]`.)

### 5.6 Cancel-run button (Feature E)
`POST /api/runs/{id}/cancel|stop` → `registry.cancel_run`: SIGTERM→5s→SIGKILL, finalizes both job-state + manifest as `status='canceled'` (idempotent, 404 on unknown). Frontend live cancel button + confirm + "canceling…" + terminal state.

### 5.7 Live per-model speed summary
`latency_ms` was dropped before persist → per-model speed table returned None. Fixed persist; status endpoint now surfaces `model_speed_summary` (tok/s + img/min) live. Attila wants per-model speed + cost visible.

### 5.8 RL policy-iteration loop restore — **the make-or-break** (2026-07-05)
**Symptom:** Attila: "The prompt generator iteration is still gone!!!" — the core of the demo appeared deleted for MNIST.
**Real root cause (X2, 2026-07-05):** in `web/policy-graph.js`, `demoUsesLocalPolicyGraph()` returned true for MNIST, routing the MNIST demo through a **static, no-API path** (reading `policy-graph/MNIST_Digits/v0.1/edges.json` + per-node `.md` over plain `fetch()`) that **skipped ALL `/api/policy/*` calls**. So for MNIST the §2 version picker (`#policyGraphVersion`), the "Next version" stepper (`#policyGraphNextVersion`), the propose-diff/accept flow (`policy-diff.js`), and the proposal list never talked to the backend RL loop → it rendered as a **dead skeleton**. The backend RL loop (`pipeline/policy_diff.py` `accept_proposal` materializes `v_{k+1}` by copytree base→new + apply files_added/changed/removed + stamp `accepted_into_version`) was **fully intact** the whole time — this was a **frontend routing gap**, not a backend rebuild.
**Fix (`2cd85e31 [X2]`):** MNIST now uses the LIVE `/api/policy` path when the API is up (static = offline fallback only). Version picker populates from `/api/policy/versions?area=MNIST_Digits`, graph renders via `/api/policy/graph?area=MNIST_Digits`, propose/accept/reject hit the backend with `area=MNIST_Digits`, "Next version" actually executes, and accepting a MNIST proposal materializes `MNIST_Digits/v0.2`. Covered by an end-to-end test.

### 5.9 §5 Insights folded into §4 (2026-07-05)
The standalone §5 Insights section (Majority-wrong / Model-disagreement / Boundary-concentration) was **redundant** with §4's panel-decision block + residuals and wasted a screen. Removed the standalone section; folded its cuts into §4 as tabs. (`d4139d66 [X3]`.)

### 5.10 §4 control consolidation + provenance relocation (2026-07-05, latest)
- §4 had BOTH a horizontal tab strip AND a separate "FILTER" dropdown doing overlapping filtering. **Removed the redundant dropdown**; the tab strip is the single switcher (added **Unanimous** + **Split** tabs; **Boundary-flagged** folded into **Borderline**). Tabs drive `runState.consensusFilter` → filter table + stat cards + "Showing N of M". (`92950973 [X3]`.)
- The **Provenance** section ("How this policy was created") was moved out of the §2→§3 flow into a **collapsed `<details id="provenance">` at the bottom** of `<main>`, content intact.

### 5.11 Recurring infra lessons (READ THESE)
- **Cache-buster:** `web/index.html` loads JS with `?v=demo-mnist-kdd-rN`. **Any JS/CSS change MUST bump `rN`** across all `<script>` tags + styles.css, or browsers serve STALE cached JS after a server restart (this bit us hard — Attila "didn't see updates" because r6 never changed). Currently at **r9**.
- **Stale server:** after a restart, VERIFY the new PID + `ps -o lstart`. `launchctl kickstart -k` does NOT help if a non-agent python is squatting port 8766 — check for and `kill -9` squatters. Exercise the REAL web path (POST `/api/runs/start`), not just a CLI proxy.
- **Verify the actual user path:** a CLI `run_bulk_labeling.py` success ≠ the web path works; always test `POST /api/runs/start` + manifest area + labels.

---

### 5.12 Per-local-model reasoning toggle (2026-07-05, latest) — resolves §7.4
Added a **Reasoning On/Off toggle per local-model card** in the §3 model-selection UI (`local/qwen3.6-27b`, `local/gemma-4-26b-a4b-qat`; hosted models unaffected). The toggle flows: UI switch → `local_reasoning` map in the `POST /api/runs/start` payload → `_safety._validate_local_reasoning` → `run_registry` → `run_bulk_labeling --local-reasoning "id=on,id=off"` → `_local_reasoning_runtime_params` → the LM Studio `/v1/chat/completions` `reasoning_effort` field. Mapping: **OFF → `reasoning_effort="none"`, max_completion_tokens=4000**; **qwen ON → `"low"`/6000**; **gemma ON → `"medium"`/6000** (generalized the override path to ALL local models — gemma previously had none). **Defaults are area-aware:** MNIST_Digits → Off (both), Generative_AI → On (both); the user can override per model. Live-verified: OFF → `reasoning_tokens=0`, qwen ON → `reasoning_tokens=1263`. Tests 406 → **414**. (`[X2]` backend `3f9bdb1c`, `[X3]` frontend `92580c88`.) **This replaces the hardcoded `reasoning_effort="none"` with user control — the §7.4 concern is now resolved for the web path.**

### 5.13 Per-model timing freeze + persisted speed/cost telemetry (2026-07-05, latest)
**Bug:** per-model IMAGES/MIN kept decaying for a model that had already finished all its calls, because `_per_model_rollup` divided `calls_done` by the RUN-level elapsed clock (start→now) shared across models. A finished gemma (40/40) kept dropping while qwen still ran.
**Fix (`[X1]` `e144e54b`):** compute a **per-model window** from `llm_outputs.jsonl` — `model_start = min(recorded_at - latency_ms)`, `model_end = max(recorded_at)` — and use `active_elapsed_s = model_end - model_start` as the denominator. Because `model_end` stops advancing once a model drains, its `images_per_min` **freezes at completion** even while other models keep running (synthetic test: run elapsed +3600s, gemma stays 30.0→30.0, qwen 15.0). Unified the rollup so TOKENS/SEC, TOTAL INPUT/OUTPUT TOKENS, and TOTAL COST populate live + at finalize (local = $0 but tokens still shown). **Persisted for later analysis:** a `per_model_timing` block (`{per_model:[{model_id, calls, first_started_at, last_finished_at, active_elapsed_s, avg_s_per_call, images_per_min, tokens_per_sec, total_input_tokens, total_output_tokens, total_cost_usd}], total:{...}}`) is written into BOTH `run_manifest.json` (top-level) and `model_speed_summary.json` at finalize; `schemas/run-manifest.schema.json` updated. **Frontend (`[X3]` `e2cd8d3d`):** table reads the frozen `images_per_min`, fills the token/cost columns, and marks a model row "done" when `calls_done==calls_total`. Cache-buster r11, 414→**415** pytest.
**For model analysis:** to compare model speed/cost after runs, read `run_manifest.json.per_model_timing` (or `model_speed_summary.json`) across runs — that's the durable per-model record.

### 5.14 Partial per-image errors no longer fail the whole run (2026-07-05, latest)
**Bug:** a single recoverable per-image error (e.g. qwen `parse_failed` on 1/40 images) marked the ENTIRE run FAILED and skipped scoring, even though 39/40 succeeded and `run_manifest.json` said `completed`. Root cause (pre-existing since 2026-05-10, surfaced once qwen joined the panel): `run_bulk_labeling.py` returned exit **1** whenever `errored_calls>0`, and `run_registry` maps `returncode!=0` → job-state `aborted`/FAILED + skips auto-scoring → manifest(completed) vs job-state(aborted) disagreement.
**Fix (`[X1]` `4822aa35`):** partial success now **exits 0**; the manifest stays `status=completed`, sets `completed_with_errors=true`, and surfaces `errored_calls`; the run **still auto-scores** the successful labels. FATAL cases stay fatal (nonzero exit + `status=failed`): **all calls failed, OR all calls for any single model failed**, plus the existing setup/validation `return 2` paths. Also **`parse_failed` now gets exactly one runner-level retry** (recoverable truncation/format flake; recovery persists `attempts=2`). **Frontend (`[X3]` `ed0426ef`):** `completed` + `errored_calls>0` renders an amber **"Completed · N errored"** badge (not red FAILED), keeps **Score now** enabled, and shows the `errors.jsonl` per-image rows; `failed`/`aborted` still render red. Cache-buster r12, 415→**420** pytest.
**Guidance:** a run showing "Completed · N errored" is a SUCCESS with a few recoverable per-image errors — inspect `errors.jsonl` for the affected images; a genuinely broken run (bad provider, all-fail) still shows FAILED.

### 5.15 Portable data fixture — run both demos from a fresh clone (2026-07-05, latest)
**Why:** the full image corpora are local-only and huge (GenAI `source-datasets/` ≈ 12 GB, 70k MNIST PNGs ≈ 45 MB), so a clone on a second machine (Attila's Mac Pro) couldn't run the demos. Now a small **committed fixture (~56 MB total)** makes both demos work out-of-the-box.
**What's committed (`[X1]`/`[X2]`):**
- **MNIST:** the 2,500 demo gold PNGs under `source-datasets/mnist/<digit>/` (~1.6 MB) **+** the full 70k set packed as `data/images/mnist-classification/mnist_full.npz` (~11.5 MB; uint8 `images (70000,28,28)`, `labels`, `index`). Scripts: `scripts/pack_mnist_full.py` (rebuild npz from `~/Downloads/mnist_png`) and `scripts/unpack_mnist.py` (expand to PNGs; `--layout digit|flat`, default all 70k).
- **GenAI:** a balanced **72-image** sample (12 per dataset×class, both splits, original bytes → sha256 valid, ~43 MB) under `data/images/genai-classification/sample/`, with `manifests/combined_labels.portable.jsonl`. Builder: `scripts/build_portable_fixture.py --max-mb 50 --per-stratum 12`.
**Auto-select (`[X2]`):** `pipeline/io_paths.py:genai_manifest_default()` returns the portable 72-row manifest when `RUSH_PORTABLE=1/true/yes` **or** the full `source-datasets/` image tree is absent; otherwise the full 200-row manifest. `pipeline/web/_safety.py` static allowlist extended to serve `data/images/genai-classification/sample/`. `run_bulk_labeling.py --manifest` still overrides.
**Full parity / where to find datasets:** see README "Run the demos on another machine" + "Where to find the full datasets" and both data READMEs. MNIST public sources = Kaggle / GitHub `mnist_png`; GenAI full tree = `rsync` from the Mac mini (no canonical public URL recorded). GenAI dataset identities: `midjourney`=Midjourney vs real, `sdv1_4`=Stable Diffusion v1.4 vs real, `wfir`=StyleGAN faces ("Which Face Is Real"-style) vs real.

## 6. Current state (as of 2026-07-05 ~10:20 PDT)

- **Branch `feat/mnist-ux-kdd`** holds all of §5. HEAD around `819158fd` before the handoff commits. **406 pytest passing.** Cache-buster **r9**. Server restarted onto the merged build (verify PID live).
- **`main`** was fast-forwarded to `1f3888a5` earlier (cancel-run) and pushed; the RL-loop + insights-fold + §4-cleanup work is **ahead of main on `feat/mnist-ux-kdd`** and is being merged to main + pushed as part of THIS handoff.
- **MNIST v0.2 not yet on disk** — it materializes only when an SME accepts a proposal from a scored MNIST run. Proven by test; needs a live batch + accept to see it.
- Worktrees `RUSH-wt-*` have accumulated; safe to prune stale ones (`git worktree remove`).

---

## 7. OPEN ISSUES for the next AI to test/fix (Attila flagged these)

### 7.1 Why is qwen so slow vs gemma? (investigation handoff)
**Observed:** qwen3.6-27b at full reasoning is a **hard floor of ~60–66s/call**; gemma-4-26b ~43s; hosted models seconds. On a run of K images, qwen dominates: total ≈ K × ~63s while everything else finishes free underneath.
**Why (current understanding):**
1. **Reasoning-token volume.** At full reasoning depth qwen emits far more reasoning/output tokens (~2890 tok on a genai image) than gemma. Generation is the long pole, not prompt processing.
2. **Image tokenization.** Before the area-aware sizing fix, MNIST 28px→1024px = 8541 input tokens made qwen over-reason/abstain. With MNIST ~112px it is far cheaper. Verify qwen's `max_image_size` is area-aware too.
3. **Per-card serialization.** `LOCAL_MODEL_MAX_CONCURRENCY=1` serializes calls on qwen's card; distinct local models run parallel across cards (post per-lane fix), but qwen's own calls are sequential on its GPU.
**Mitigations already available:** `reasoning_effort="none"` (qwen → ~9.5s/240tok on MNIST, still correct); smaller area-aware images; or drop qwen from the panel. 
**Operator's leading hypothesis (Attila, 2026-07-05) — START HERE:** the 5x gap (vs an expected ~2x of Gemma-4) is most likely a **GPU→CPU offload**: qwen3.6-27b's weights and/or KV cache don't fully fit in VRAM, so part spills to host RAM/CPU and tanks throughput. If it were purely compute/reasoning it should be within ~2x of Gemma-4, not ~5x. **Planned fix (Attila will patch on his end):** drop the **KV cache from f16 → f8** in LM Studio, which halves KV-cache VRAM and should keep the model fully resident on-GPU. Expectation: qwen falls back toward the ~2x-of-Gemma regime. Whoever continues this: after Attila's KV-f8 change, re-measure qwen latency and confirm no CPU offload (nvidia-smi memory + util on the GPU host during a call).
**Other things to investigate:** (a) confirm GPU compute-bound (nvidia-smi util during a qwen call) vs LM Studio scheduling/queueing vs the offload above; (b) compare qwen3.6-27b dense vs qwen3.6-35b-a3b MoE latency; (c) test `reasoning_effort` low/medium sweet spot for accuracy-vs-latency; (d) check LM Studio context/kv-cache settings + whether the model reloads between calls. There is **no GPU util exposed via LM Studio** (`/api/v0/gpu|hardware|system` 200 but empty/erroring) — real util needs an `nvidia-smi` exporter on the GPU host (Feature F, still parked).

### 7.2 Page refresh loses live labeling progress (real UX bug)
**Symptom:** while a run is in-flight (Attila was waiting on qwen to finish), refreshing the web page **loses track of labeling progress** — the live progress card / current run is not restored.
**Root-cause hypothesis:** the live run progress is held in **front-end state** in `run-trigger.js` (the polling loop + the live card are created when the user clicks Start). On reload there is no re-attach: the page does not query `/api/runs` for a still-running job and rebuild the live card + resume polling. The `run_id` isn't persisted client-side.
**Suggested fix (tee'd up for testers):** on page load, `GET /api/runs`; if any run has `finished_at == null` (running), **re-render the live progress card and resume polling its status** (the status endpoint already returns progress + `model_speed_summary`). Optionally persist the active `run_id` in `localStorage` as a hint and reconcile against `/api/runs`. Make progress **server-authoritative** so refresh is safe. This pairs well with the per-model speed summary already in the status endpoint.

### 7.4 Local-model reasoning — RESOLVED via the §3 toggle (2026-07-05)
**Status: RESOLVED for the web path.** The former problem (below) is fixed by the per-local-model reasoning toggle in §5.12: the UI now controls `reasoning_effort` per local model, area-aware by default (MNIST off / GenAI on), so GenAI gets real reasoning while MNIST stays fast. Remaining note: the **registry still hardcodes `reasoning_effort="none"` as the CLI/default** for `local/qwen3.6-27b`, `local/qwen3.6-35b-a3b`, `local/gemma-4-26b-a4b-qat` — that default only applies to direct `run_bulk_labeling` CLI invocations WITHOUT `--local-reasoning`; the web UI always sends explicit toggle state. If you run headless CLI for GenAI, pass `--local-reasoning "local/qwen3.6-27b=on,local/gemma-4-26b-a4b-qat=on"`.

<details><summary>Original issue (pre-toggle, for context)</summary>

**Was: Local-model reasoning is globally OFF — conflicts with the §5.4 doctrine (ACTION ITEM)**
**Current code state (verify in `pipeline/providers/registry.py`):** BOTH local models default to `reasoning_effort="none"`:
- `local/gemma-4-26b-a4b-qat` (~L336–349) — hard `none`, added to stop gemma over-reasoning (1900–2400 reasoning tokens → abstain) on trivial MNIST digits. **No per-run override path for gemma.**
- `local/qwen3.6-27b` (~L299–315) — defaults `none`; `local/qwen3.6-27b-low` is a separate `low` variant. qwen CAN be per-run overridden to high/xhigh (`build_client` honors it for `qwen/` model names; `_safety.py` allows only high|xhigh from the web payload).

**The conflict:** these defaults are **global per-model, NOT area-scoped**, so gemma and base qwen run reasoning=OFF for **Generative_AI** too — exactly what §5.4 says NOT to do for policy-grounded areas (loses policy-node citation, justification grounding, is_boundary, and the misalignment signal that feeds the RL loop). It was set for MNIST but leaks to GenAI.

**ACTION ITEM (make reasoning area-aware):** for MNIST_Digits keep `none` (fast, digits are trivial); for Generative_AI / any policy-grounded area, run local models with reasoning ON. — **DONE via the §5.12 toggle.** Attila is separately fixing local-model *speed* at the GPU layer via KV f16→f8, which is orthogonal (speed vs reasoning-depth are different levers).

</details>

### 7.3 Other parked items
- **Feature F (live GPU info):** LM Studio exposes only model-state, not GPU util/VRAM. Option A = LM Studio model-state + our telemetry panel (no fake gauges). Option B = real `nvidia-smi` util via an exporter on the GPU host. Awaiting Attila's A/B pick.
- **`run_manifest` `manifest` field is None** (area + policy_version drive ontology, so non-blocking).
- **haiku-low degenerate on genai** (99/99 not_gen_ai in one dead run) — separate open issue to investigate.

---

## 8. How this team worked (AI collaboration context — for Fable/Claude-Code)

RUSH was built by an **OpenClaw multi-agent office**, orchestrated for Attila:
- **Pista (CEO, Opus 4.8)** — planning, talks to Attila, delegates, owns finalization (verify tests/tree, final commit on the feature branch, capture engineer learnings).
- **TheoMaximus (CTO, Opus 4.8)** — architects, segments work, delegates to engineers, verifies. (Quirk: his `mode=run` dispatch often ends at yield after spawning engineers; Pista backstops the finalize.)
- **Engineers X1–X5 (Codex gpt-5.5)** — X1 backend/data-pipeline, X2 full-stack integration, X3 frontend/viz, X4/X5 flex. Every engineer commit is prefixed `[X#]` and lands on its own `wt/*` worktree; work is merged (never committed directly to `main`) onto the feature branch, then to main only on Attila's green-light.
**Hard rules that shaped the git history you'll read:** named `agentId` only (no anonymous subagents), `[X#]` commit prefixes so specialization is legible in `git log`, feature-branch-only for multi-file work, and active supervision (engineers must actually run, not be cosplayed). If you (Fable/Claude-Code) continue in this repo, expect and preserve these conventions.

---

## 9. Using the embedded memory (semantic search of this project)

This handoff + key docs are chunked and embedded with **Gemma-embedding** so a local AI can semantically search the project's memory.

- **Location:** `docs/ai-handoff/memory-embeddings/`
  - `index.jsonl` — one row per chunk: `{id, source_path, chunk_index, text, embedding: float[768], metadata}`.
  - `manifest.json` — embedding model, dims, created timestamp, chunk params, source file list, counts, total size.
- **Model / endpoint:** `text-embedding-embeddinggemma-300m-qat` (768-dim) served by **LM Studio at `http://127.0.0.1:1234/v1`** on Attila's GPUs. Fable-5 runs locally with access to these.
- **Query:** `python scripts/query_memory.py "your question"` embeds the query via the same endpoint and returns the top-k nearest chunks (cosine). Use it before re-deriving context.
- **Regenerate:** `python scripts/build_memory_embeddings.py` re-chunks + re-embeds the curated source set and rewrites `index.jsonl` + `manifest.json`. Total size is capped to a few MB (well under the ~5–10 MB budget); embeddings are float32 text, not binary blobs.
- **Why Gemma-embedding:** it is the same embedding model OpenClaw's own memory search uses here, so the vector space is consistent across the human's assistant memory and this project memory.

---

## 10. TL;DR for the incoming AI

1. Read this file top-to-bottom, then `git log --oneline` on `feat/mnist-ux-kdd`/`main` (commits are `[X#]`-tagged by specialty).
2. Start the server (LaunchAgent) and hard-reload `:8766/web/` (cache-buster is r9). Run a MNIST batch, then exercise §2: propose → accept → confirm **v0.2 materializes** and the version picker + Next-version light up. That is the RL loop; guard it.
3. Two things to fix/test first: **qwen slowness** (§7.1) and **refresh-loses-progress** (§7.2).
4. Never commit multi-file changes to `main`; branch first. Keep pytest green (406). Bump the `web/index.html` cache-buster on any JS/CSS change.
5. Use `scripts/query_memory.py` to search this memory instead of guessing.
