# EXECUTION-PLAN: Bulk Labeling Pipeline v1

**Status:** in-flight
**Branch:** `feat/bulk-labeling-v1` (cut from `feat/genai-sampler-demo`)
**Owner:** Theo (CTO)
**Engineers:** X1, X2, X3, X4 (parallel)
**Sample:** existing `data/images/genai-classification/manifests/combined_labels.jsonl` (100 dev_golden + 100 holdout, balanced)

---

## 1. Goal

Wire RUSH's first real bulk-LLM labeling pass across three providers, score the results against SME ground truth, surface misalignments and borderlines, and propose policy patches — all without committing a single secret.

---

## 2. Module layout (file slice ownership)

> Each engineer owns the listed files exclusively. No two engineers touch the same file in this pass. Shared schemas/docs are owned by X2 (schemas) and X4 (docs touchpoints) — coordinate via the data contracts in §3.

```
RUSH/
├── pipeline/                              # NEW python package
│   ├── __init__.py                        # X1
│   ├── providers/                         # X1
│   │   ├── __init__.py
│   │   ├── base.py                        # LabelClient ABC + LabelRequest/LabelResponse dataclasses
│   │   ├── openai_client.py               # GPT models (Responses API, vision, structured JSON)
│   │   ├── anthropic_client.py            # Claude Opus (Messages API, image content blocks)
│   │   ├── gemini_client.py               # Gemini (multimodal, inline_data)
│   │   ├── registry.py                    # MODEL_REGISTRY: model_id -> (provider, params, phase)
│   │   ├── auth.py                        # dotenv loading; explicit env var lookups; never logs values
│   │   └── retries.py                     # exponential backoff w/ Retry-After honor; per-provider 429 handling
│   ├── runner.py                          # X2 — orchestrates a labeling RUN across N images × M models
│   ├── io_paths.py                        # X2 — input/output paths, gitignored output tree, run_id minting
│   ├── manifest.py                        # X2 — load combined_labels.jsonl; ground-truth join helpers
│   ├── persistence.py                     # X2 — atomic JSONL append, schema validation hooks
│   ├── scoring/                           # X3
│   │   ├── __init__.py
│   │   ├── decision_quality.py            # accuracy/F1/precision/recall per labeler vs SME truth
│   │   ├── misalignment.py                # model-vs-SME and model-vs-model disagreements
│   │   ├── borderline.py                  # cluster boundary/difficulty cases by L2 (cold-start: L0 only)
│   │   └── exporters.py                   # write web-ready JSON: data/runs/<run_id>/web/*.json
│   ├── policy_iterator.py                 # X3 — call GPT-5.5 reasoning=high with policy MDs + misclassifications -> PolicyPatch
│   └── pdf/                               # X4
│       ├── __init__.py
│       └── policy_pdf.py                  # policy-graph/Generative_AI/v0.1/*.md -> single policy.pdf
├── scripts/                               # X2 owns runner CLIs; X4 owns pdf CLI
│   ├── run_bulk_labeling.py               # X2 — CLI wrapper around pipeline.runner
│   ├── score_labels.py                    # X3 — CLI wrapper around pipeline.scoring
│   ├── propose_policy_patch.py            # X3 — CLI wrapper around pipeline.policy_iterator
│   └── build_policy_pdf.py                # X4 — CLI wrapper around pipeline.pdf
├── schemas/                               # X2 (additions only; do NOT mutate semantics of existing fields)
│   ├── llm-output.schema.json             # X2 — extend `label` enum to allow gen_ai/not_gen_ai (cold-start)
│   ├── label-vote.schema.json             # already supports gen_ai/not_gen_ai — leave alone
│   └── run-manifest.schema.json           # X2 NEW — per-run manifest (run_id, model_id, sample_ids, params)
├── web/                                   # X4 owns all web changes
│   ├── index.html                         # X4 — add PDF download CTA + borderline/misalignment sections
│   ├── app.js                             # X4 — wire new sections, fetch run JSON exports
│   ├── styles.css                         # X4 — minimal additions
│   └── runs/                              # X4 — placeholder loader; reads from data/runs/<run_id>/web/
├── docs/
│   ├── EXECUTION-PLAN-bulk-labeling-v1.md # this file
│   ├── architecture-bulk-labeling.md      # X4 — short addendum (data flow diagram + how-to-run)
│   └── runbook-bulk-labeling.md           # X2 — operator runbook (env vars, smoke test, full run)
├── tests/                                 # NEW
│   ├── test_providers_smoke.py            # X1 — mocked HTTP smoke (no network) per provider
│   ├── test_runner.py                     # X2 — runner determinism + persistence
│   ├── test_scoring.py                    # X3 — DQ math + misalignment correctness
│   └── test_pdf_builder.py                # X4 — policy MD -> PDF smoke
├── data/runs/                             # gitignored output tree
└── requirements.txt                       # X1 — pin openai, anthropic, google-genai, jsonschema, reportlab, pytest
```

### .gitignore additions (X2 owns)
```
data/runs/
*.pdf       # policy.pdf is a build artifact; only commit if explicitly placed under web/policy.pdf? See §6.
```
(Decision: build the PDF on-demand; do not commit PDFs. Web UI fetches `data/runs/<run_id>/policy.pdf` or a built `web/policy.pdf` produced locally.)

---

## 3. Data contracts (cross-engineer)

### 3.1 LabelRequest (in-process, X1↔X2)
```python
@dataclass
class LabelRequest:
    image_path: Path                # absolute, on-disk
    image_id: str                   # sample_id from manifest
    policy_markdown: str            # concatenated MD bundle of policy-graph/Generative_AI/v0.1/*.md
    policy_graph_version: str       # "v0.1"
    prompt_version: str             # bumped per prompt change
    model_id: str                   # e.g. "openai/gpt-5.5", "anthropic/claude-opus-4-6"
```

### 3.2 LabelResponse (in-process, X1↔X2)
```python
@dataclass
class LabelResponse:
    image_id: str
    model_id: str
    label: str                      # cold-start: "gen_ai" | "not_gen_ai" | "abstain"
    l2_label: str                   # policy graph node id; "" if model abstains/no fit
    justification: str              # >=10 chars (matches schema)
    confidence: float               # 0..1
    difficulty: str                 # "high" | "medium" | "low"
    is_boundary: bool
    raw_provider_payload: dict      # opaque, persisted for audit; never logged
    error: str | None               # populated if call failed permanently
    latency_ms: int
    attempts: int
```

### 3.3 Persisted shapes (X2 writes; X3 reads; X4 reads exporter outputs)

```
data/runs/<run_id>/
├── run_manifest.json              # see schemas/run-manifest.schema.json
├── label_votes.jsonl              # one LabelVote per (image_id, model_id), validated against label-vote.schema.json
├── llm_outputs.jsonl              # raw LLMOutput records validated against llm-output.schema.json
├── errors.jsonl                   # per-attempt failures (no secrets)
├── scoring/
│   ├── decision_quality.json      # validated against decision-quality.schema.json
│   ├── misalignment.json          # NEW shape (X3 defines)
│   └── borderline.json            # NEW shape (X3 defines)
├── policy_patches.jsonl           # validated against policy-patch.schema.json
├── policy.pdf                     # X4 builds; web fetches this
└── web/
    ├── summary.json               # rollup for the web UI hero
    ├── borderline.json            # web-friendly list grouped by L0/L2
    └── misalignment.json          # web-friendly disagreement worklist
```

### 3.4 run_id format
`run_id = YYYYMMDDTHHMMSS-<short-uuid>` (UTC). Sample: `20260510T172300-a1b2c3d4`.

---

## 4. Schema changes (X2)

1. **`schemas/llm-output.schema.json`** — extend `label` enum to `["violative","non_violative","abstain","gen_ai","not_gen_ai"]`. Add `$comment` noting cold-start GenAI labels coexist with warm-start labels. Bump `title` to `RUSH LLMOutput v1.1`.
2. **`schemas/run-manifest.schema.json`** (new) — required: `run_id, started_at, finished_at|null, sample_manifest_path, sample_ids, models[], policy_graph_version, prompt_version, sampling_version`.
3. Existing `label-vote.schema.json`, `policy-patch.schema.json`, `decision-quality.schema.json` are sufficient — do not edit semantics.

---

## 5. Hard constraints (every engineer)

1. **Branch:** all work on `feat/bulk-labeling-v1`. Run `git branch --show-current` before every commit. Never commit to `main`.
2. **Secrets:** load from `.env` via `pipeline.providers.auth`. Never log keys, never include them in error strings, never echo via `print(os.environ)` or similar. Never pass via CLI args. CI-friendly: support env var override `RUSH_DOTENV_PATH`.
3. **No image bytes in JSON manifests** (we keep this discipline). Provider payloads may include base64 in-memory only; persisted `raw_provider_payload` MUST have any base64 image content stripped (replace with `"<image-bytes-omitted>"`).
4. **Validate every persisted record** against its schema before write. Invalid records go to `errors.jsonl` with reason.
5. **Rate limits:** honor `Retry-After` headers, exponential backoff (base 1s, jitter, cap 60s, max 6 attempts). Concurrency cap: 4 in-flight per provider by default.
6. **Determinism where possible:** runner sorts sample_ids before dispatch; per-image (model, prompt) pairs are stable.
7. **No tight loops:** no `while True: poll()` — always `time.sleep` with backoff.
8. **Tests must run offline.** Provider clients accept an injected transport for tests.

---

## 6. Web UI additions (X4)

- **Hero CTA:** "Download bound policy (PDF)" → links to `/web/policy.pdf` or `data/runs/<run_id>/policy.pdf`.
- **Section: Borderline Inspector** — grouped by L0 bucket at cold start; expand to L2 once policy graph has more nodes for this label set.
- **Section: Misalignment Worklist** — table with columns: image thumbnail (path-based, no embedded bytes), SME truth label, per-model labels (3 cols), agreement chip, disagreement reason (when X3 emits one), proposed policy_patch_id (link).
- **No fake data.** If no run exists yet, show empty-state copy: "Run `python scripts/run_bulk_labeling.py --models <ids>` to populate this view."
- **Run picker:** dropdown listing runs found at `data/runs/*/web/summary.json`. Defaults to most-recent.

---

## 7. Test plan

| Stage | Command | Owner | Acceptance |
|-------|---------|-------|------------|
| Unit (offline) | `pytest tests/ -v` | all | green |
| Schema validation | `python scripts/validate_foundation.py` | X2 | green |
| Sampler still deterministic | `python scripts/sample_genai_gold_sets.py` (in tmpdir) | X2 | byte-equal across 2 runs |
| **Smoke 1×1×1** | `python scripts/run_bulk_labeling.py --models openai/gpt-5.5 --limit 1 --split dev_golden` | X2 | 1 LabelVote written, validated |
| **Mini batch 5×3** | `--models openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4-6 --limit 5` | X2 | 15 LabelVotes, scoring works |
| **Full N=100 dev_golden ×3** | (Phase 2; Attila approval needed for spend) | Pista | gated |
| **Holdout** | NEVER auto-run; locked behind `--allow-holdout` flag w/ confirmation | X2 | guarded |

> Engineers MUST NOT execute Smoke 1×1×1 or beyond — Pista controls when to spend on real API calls. Just wire it and stop.

---

## 8. Branching / commit plan

- Each engineer commits small, focused commits to `feat/bulk-labeling-v1` directly (this is a small team; no nested branches).
- Commit message convention: `[X1] providers: ...` / `[X2] runner: ...` / `[X3] scoring: ...` / `[X4] web: ...`.
- Theo runs `pytest` + `validate_foundation.py` after each engineer reports back.
- Theo opens no PR yet; final report goes to Pista with branch + file list, Attila decides merge.

---

## 9. Engineer briefs (sent verbatim)

See subagent spawns below. Each engineer receives:
- their slice
- the data contracts above
- explicit "do NOT touch these files" guardrail
- offline acceptance criteria

---

## 10. Out of scope for this pass

- Real network calls to LLM providers (wired but not executed; Pista runs the smoke).
- SME re-review queue UI (stub only — show "Pending SME review" badge).
- Adaptive boundary-discovery batching.
- Multi-policy-version A/B (single `v0.1` only).
- L1 enforcement labels (cold-start = L0 only per `docs/scope-additions-2026-05-09.md`).
