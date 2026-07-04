# RUSH

**Reinforcement learning Using SME feedback and High-reasoning models**

RUSH is a policy-graph system for scaling subject matter expert (SME) decision quality to non-expert labelers: BPO reviewers, lower-reasoning LLMs, and production ML systems. The core idea is simple and annoyingly powerful: if policy is explicit, versioned, graph-structured, and tested against golden examples, label quality improves for everyone.

The first pilot is a **cold-start Generative AI image-classification policy**. We begin with images and labels, grow the policy as an Obsidian-style Markdown graph, capture SME/LLM misalignments, and propose policy diffs that an SME can approve or reject.

## What this repository contains now

- `web/` — dependency-free VC-demo interface: a simplified narrative flow for GenAI sampling, SME policy graph building, future bulk LLM labeling, and decision-quality/misalignment review. `web/genai-sampler.js` remains the static client-side fallback sampler for demos when local manifests are unavailable.
- `policy-graph/Generative_AI/v0.1/` — initial Obsidian-compatible Markdown policy graph for GenAI detection.
- `schemas/` — JSON schemas for policy nodes/edges, image records, split assignments, label votes, structured LLM outputs, arbiter decisions, SME reviews, label-tier records, decision-quality records, policy patches, export records, and metric snapshots.
- `data/seed/` — mock-only placeholder golden-set records, labels, not-enough-data metrics, and policy-suggestion examples.
- `docs/visuals/` — SVG architecture visuals used by README and the web UI.
- `scripts/validate_foundation.py` — lightweight repository validation with no external dependencies.

## RUSH in one picture

![RUSH system loop](docs/visuals/rush-system.svg)

## Demos and the §1→§4 flow

The web interface ships two self-contained demos, selectable from the demo picker:

- **Generative_AI** — the flagship cold-start pilot: binary `gen_ai` / `not_gen_ai` detection with expandable L2 subcategories and boundary/hard-negative nodes.
- **MNIST_Digits** — a multiclass reference demo (digits 0–9) that shows the same policy-graph machinery generalizing beyond binary decisions.

Both demos walk the same four-section loop:

1. **§1 Data preview** — sample and preview the image pool (N per class) for the active project's splits (dev golden / holdout).
2. **§2 Seed the Generator Prompt `v_k`** — grow/seed the policy graph version that renders into the labeling prompt. Each version `v_k` is an explicit, reviewable artifact.
3. **§3 Run panel** — pick panel models and a settable **k per split**, then launch a labeling pass. `split=all` runs up to *k* training + *k* test images; the run button reflects the current k (e.g. `Run panel · k=20`).
4. **§4 Decision-quality audit** — score the run and review train/test decision quality and SME/LLM misalignments, which feed policy diffs.

## Ontology

RUSH keeps the **ontology per project**, so the same engine serves different decision shapes:

- **Generative_AI** — a binary L1 decision (`gen_ai` / `not_gen_ai`) with expandable **L2 subcategories** (e.g. `GA.visual_artifacts.anatomy.hands`, `GA.surface_texture.plastic_skin`) plus boundary/hard-negative nodes.
- **MNIST_Digits** — a **multiclass** ontology (one node per digit) demonstrating that the framework is not limited to yes/no labels.

Cross-cutting ontology primitives make ambiguity first-class: `is_boundary` marks a node as a boundary/gray-zone concept, and `is_boundary_between` links the two (or more) classes a case sits between. This structure generalizes naturally beyond classification to **rating and relevance scales** (e.g. relevance graded 1–5, quality tiers), where boundaries live between adjacent grades.

## Knowledge graph / policy graph

The generator prompt is not an opaque string — it is a **knowledge graph rendered as an Obsidian-style Markdown node graph**, versioned as `v_k`:

- **Nodes:** a single `MD.root` (project root, e.g. `GA.root`) plus one node per class/subcategory. Each node is a Markdown file with machine-readable frontmatter (definitions, criteria, hard negatives, coverage targets, examples, source anchors) and human-readable policy text.
- **Edges:** typed relationships between nodes. `confused_with` edges capture empirically observed confusions between classes and directly inform audit priority and prompt context packs.
- **Roadmap:** a dedicated **boundary node-type** to represent gray-zone concepts natively (paired with the `is_boundary` / `is_boundary_between` ontology primitives), so boundaries become first-class graph citizens rather than annotations.

Rendering the policy as a graph keeps coverage and ambiguity visible to SMEs (Obsidian backlinks + graph view) while staying trivially compilable into prompts.

## Considerations

- **Model panel + cost tiers.** §3 exposes a cost-tiered model picker: **HIGH** (e.g. Opus 4.6/4.7, GPT-5.5 high/low, Gemini 3.1 Pro), **MEDIUM** (Sonnet 4.6, Sonnet 5, GPT-5.4-mini high/xhigh, Gemini flash), and **LOW / FREE** (GPT-5.4-mini low, Haiku 4.5, Gemini flash-lite, local models). High-tier models are unchecked by default so demo spend stays intentional. Note: Opus 4.7+ uses a newer tokenizer that emits ~30% more tokens, so its effective cost is ~1.3x its list rate. Claude Haiku 4.5 is the cheap/fast vision model recommended for image labeling; Sonnet 5 lists at intro pricing (2.0/10.0) through 2026-08-31.
- **Local GPU support.** Local models via **LM Studio** run free: `gemma` (fast) and `qwen` (slower, higher quality) are wired as `local/*` at $0.00, useful for offline iteration and cost-free sweeps.
- **Cost tracking + batching.** Per-call USD cost is tracked from usage tokens (`pipeline/providers/pricing.py`, mirrored in `web/run-trigger.js` — kept in exact sync). Cost is amortized two ways: **multi-image request amortization** (see below) and provider **Batch APIs** (~50% discount for asynchronous throughput). See *Image batching* below.
- **Output-token budgets (reasoning-safe).** The justification prompt enforces a hard **≤ 300-word (~400 token)** cap. Providers that expose a clean *visible*-output cap separate from hidden reasoning (Anthropic `max_tokens`, Gemini `max_output_tokens`) are set to **~768** visible tokens, with thinking budgets kept separate and untouched. OpenAI reasoning models use a *combined* budget (`max_completion_tokens`), so they are **not** hard-capped at the visible size — they are bounded at reasoning headroom + ~768 visible (~2000 for low reasoning, ~4000 for high/xhigh). Local models keep 4000–6000 so `gemma` stays snappy and `qwen` does not truncate.
- **Train drives updates / test drives metrics.** The training split drives policy/prompt **updates**; the test split drives **decision-quality metrics** only and never leaks into prompt tuning or adaptive node discovery.

### Image batching (multi-image request amortization)

`OpenAIClient.batch_label` (`pipeline/providers/openai_client.py`) sends **N images in a single request** that share **one** policy + system + user-instructions block, and the model returns one JSON `items` array (one entry per image, echoing `image_id` in input order). The recommended batch size is **5 images per request**.

**Why it saves money:** the policy bundle is large (~3.7k tokens) and is otherwise re-sent for every single-image call. Batching sends it **once** and amortizes it across all N images; the images themselves stay ~700 input tokens each. That collapses input roughly **~3x** for a batch of 5, which works out to roughly **25–50% total per-image cost savings** — nearer the high end for cheap/short-output models (input dominates) and nearer the low end for reasoning-heavy models (output/reasoning dominates).

> **FOLLOW-UP (decision quality):** confirm decision quality does not degrade under image batching — run a batched-vs-single-image A/B before making batching the default for scored runs.

**Future option — provider async Batch APIs.** OpenAI, Anthropic, and Gemini each offer asynchronous **Batch APIs** (~50% off list, ~24h turnaround). These are complementary to request-level amortization and are **not implemented yet** — documented here as a future throughput/cost lever only.

## Dataset images are local-only

Dataset image bytes are **not committed to git**. The repository tracks lightweight **manifests** that reference local images under `data/images/**`; the actual `.png`/`.jpg`/`.jpeg` files stay on the machine and are excluded via `.gitignore`. Generate/refresh local manifests with the sampler CLIs (see *GenAI sampler paths* below); they read local image files and write ignored manifests without adding image bytes to git.

## Design principles

1. **SME-reviewed labels are truth; LLM consensus is signal.** 3/3 model agreement can still be a correlated failure, so it should feed audit priority rather than become silent ground truth.
2. **Keep labels separate.** Legacy labels, model votes, arbiter decisions, SME canonical labels, label tiers, and downstream exports are distinct records.
3. **Policy is the product and the measurement surface.** Nodes carry definitions, criteria, hard negatives, coverage targets, examples, and source anchors.
4. **Graph diffs are reviewable.** Cold-start and warm-start refinements become Markdown/JSON patches that SMEs can approve or reject.
5. **Metrics must be split-aware and denominator-explicit.** Use gold/platinum labels for final decision-quality metrics, show denominators and confidence intervals, protect holdouts, and keep adaptive discovery batches separate from prevalence estimates.

## First pilot: cold-start GenAI image classification

Initial binary label:

- `gen_ai` — image is likely fully or materially AI-generated/synthetic under current policy.
- `not_gen_ai` — image is likely authentic, conventionally edited, CGI/game/rendered, or insufficiently evidenced.

Initial positive subcategories are intentionally expandable:

- `GA.visual_artifacts.anatomy.hands` — impossible hands/fingers/limbs.
- `GA.visual_artifacts.text_symbols` — garbled text, pseudo-logos, broken symbols.
- `GA.surface_texture.plastic_skin` — overly smooth/waxy synthetic skin and pore absence.
- `GA.scene_geometry.inconsistent_perspective` — impossible reflections, shadows, object geometry.
- `GA.provenance.synthetic_disclosure` — metadata/watermark/context explicitly indicating generation.

Initial hard-negative/boundary nodes:

- `GA.boundary.photo_editing` — real photos with filters, healing brush, retouching, compression, or stylization.
- `GA.boundary.cgi_game_render` — non-photographic CGI/game/3D assets not treated as GenAI by default.
- `GA.boundary.low_quality_uncertain` — low-resolution/cropped/ambiguous cases requiring abstain or SME review.

## Run the web interface

```bash
cd /Users/sacsimoto/GitHub/RUSH
python3 -m http.server 8766 --bind 127.0.0.1
# open http://127.0.0.1:8766/web/
```

## Guardrails baked into the scaffold

- **Consensus audit:** 3/3 LLM agreement is not truth. It is a routing signal, and a stratified sample of consensus cases must still go to SME audit to catch correlated model failures.
- **Gold/platinum truth:** Final metrics can only use SME-reviewed gold/platinum labels. Provisional LLM, arbiter, or majority labels are triage records until promoted by review.
- **Split leakage protection:** Validation, locked holdout, boundary holdout, and sentinel examples must not leak into prompt/context packs, policy tuning, or adaptive node discovery.
- **Adaptive vs sentinel separation:** Adaptive boundary batches are for finding hard cases and improving coverage; production sentinel/random batches are for prevalence and monitoring. Do not mix them when reporting rates.
- **Mock-only metrics:** Seed metrics intentionally report `not_enough_data` with null accuracy/precision/recall until real media, canonical truth, and paired predictions exist.

## GenAI sampler paths

- Real local image manifests: run `python3 scripts/sample_genai_gold_sets.py` against ignored `data/images/genai-classification/source-datasets/` image folders. The CLI reads local image files and writes ignored manifest files; it does not add image bytes to git.
- Static web demo reset: the web interface defaults to 100 dev golden + 100 locked holdout records. It first tries ignored local manifests under `data/images/genai-classification/manifests/` so local images can render in the demo; if those manifests are unavailable, `web/genai-sampler.js` exposes `window.RushGenaiSampler.runDemoReset({ seed, nDev, nHoldout, mode })` as a synthetic browser fallback. The current web flow intentionally does not invoke LLMs; bulk model labeling is the next milestone.

## Validate the scaffold

```bash
python3 scripts/validate_foundation.py
node scripts/validate_web_sampler.js
```

The foundation validator checks that graph node IDs are unique, `GA.root` is the single root node, parent chains are not orphaned, edge endpoints and seed references exist, schemas parse, seed metrics do not masquerade as reportable when mock-only/not-enough-data, and required web/docs assets are present. The web sampler validator checks deterministic sampling, split disjointness, balanced totals, required assumptions, and SME override handling.

## Near-term roadmap

| Milestone | Exit criteria |
| --- | --- |
| M1 — Graph parse / cold start | Markdown policy nodes, edge manifest, source anchors, schema validation, SME-readable skeleton. |
| M2 — Golden-set registry | Image records, label versions, LLM vote slots, SME canonical labels, split assignments. |
| M3 — Labeling queue | Human + future LLM labels with justifications, confidence, evidence refs, and SME audit routing. |
| M4 — Metrics dashboard | Accuracy, precision, recall, FPR, positive proportion, informedness, graph coverage, gray-zone mass by node/version/split. |
| M5 — Policy diff workflow | Proposed node additions, clarifications, examples, exceptions, and rejected-change memory. |
| M6 — LLM ensemble integration | Multiple API-backed model votes with structured outputs, policy-node citations, consensus audits, prompt/context packs from graph nodes, and SME resurfacing for misalignments. |

## Why Obsidian-style Markdown?

The policy graph should be easy for humans to read and easy for machines to compile. Each node is a normal Markdown file with machine-readable frontmatter and human-readable policy text. Obsidian backlinks and graph view make policy coverage and ambiguity visible without trapping the project inside a custom editor.
