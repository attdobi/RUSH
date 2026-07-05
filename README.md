# RUSH

**Reinforcement learning Using SME feedback and High-reasoning models**

RUSH is a policy-graph system for scaling subject matter expert (SME) decision quality to non-expert labelers: BPO reviewers, lower-reasoning LLMs, and production ML systems. Policy is explicit, versioned, graph-structured, and tested against golden examples so label quality improves over time.

The first pilot is cold-start **Generative AI image classification**: start with images and labels, grow an Obsidian-style Markdown policy graph, capture SME/LLM misalignments, and propose policy diffs for SME approval.

![RUSH system loop](docs/visuals/rush-system.svg)

## Main outcomes

RUSH exists to deliver three headline outcomes:

- **RLHF with minimal human intervention.** Humans are spent only where they move the needle: the loop highlights and routes images on the **decision boundaries** for SME re-review, instead of asking people to re-label the easy majority.
- **A self-improving policy loop.** Each iteration re-runs the **generator** prompt, grows the **knowledge graph (KG)**, and **enhances decision quality** — misalignments and boundary cases become versioned policy updates that raise the next round's accuracy.
- **A multi-LLM price-optimization system.** The "easy"/aligned cases (the majority) are resolved cheaply by **consensus** across low-cost models. On **misalignment**, RUSH escalates to a **high-reasoning model** (or a **judge system** of models when needed). **Boundary / difficult / lack-of-consensus** cases are flagged for **human re-review and intervention**.

**Cost per batch is the measuring stick.** Every run records per-image and per-batch cost in `run_manifest.json` (usage tokens x `pipeline/providers/pricing.py`; local models = $0) and surfaces it live in §3. Because spend is tracked at each iteration, the price-optimization above is *measurable*: you can watch cost fall as consensus absorbs the easy cases and expensive high-reasoning calls stay reserved for the boundary.

## Repo contents

- `web/` - dependency-free demo UI for GenAI sampling, policy graph building, future bulk LLM labeling, and decision-quality review. `web/genai-sampler.js` is the fallback when manifests are missing.
- `policy-graph/Generative_AI/v0.1/` - initial Obsidian-compatible GenAI policy graph.
- `schemas/` - JSON schemas for graph, image/split, label/vote, model-output, review, patch, export, and metric records.
- `data/seed/` - mock golden-set records, labels, not-enough-data metrics, and policy-suggestion examples.
- `docs/visuals/` - SVG visuals for the README and web UI.
- `scripts/validate_foundation.py` - dependency-free validation.

## AI Handoff & Embedded Memory

The current MNIST-UX-KDD project state, recent fixes, open issues, and operator handoff live in `docs/ai-handoff/HANDOFF.md`; read that file for full detail before continuing the work. The local semantic memory is in `docs/ai-handoff/memory-embeddings/`: `index.jsonl` stores chunk text plus 768-dim embeddings, and `manifest.json` records the embedding model, source files, chunk parameters, counts, and byte size.

Embeddings use Gemma-embedding `text-embedding-embeddinggemma-300m-qat` at `http://127.0.0.1:1234/v1` via LM Studio. Query with `./.venv/bin/python scripts/query_memory.py "why is qwen slow"` and regenerate with `./.venv/bin/python scripts/build_memory_embeddings.py`.

## Demos and §1 -> §4 flow

Two demos ship in the web UI:

- **Generative_AI** - flagship AI-generated image pilot: binary `gen_ai` / `not_gen_ai`, expandable L2 subcategories, and boundary/hard-negative nodes.
- **MNIST_Digits** - multiclass digit demo (0-9) showing the same graph machinery beyond binary decisions.

Both use the same four-section loop:

1. **§1 Data preview** - sample and preview N images per class for dev golden / holdout splits.
2. **§2 Seed the Generator Prompt `v_k`** - grow or seed the policy graph version rendered into the labeling prompt.
3. **§3 Run panel** - choose panel models and **k per split**; `split=all` runs up to *k* train + *k* test images.
4. **§4 Decision-quality audit** - score train/test decision quality, review SME/LLM misalignments, and feed policy diffs.

## Ontology

Ontology is **per project**, so the same engine supports different decision shapes:

- **Generative_AI** - binary L1 decision (`gen_ai` / `not_gen_ai`) with expandable L2 nodes such as hands and plastic-skin artifacts, plus boundary/hard-negative nodes.
- **MNIST_Digits** - multiclass ontology, one node per digit.

Ambiguity is explicit: `is_boundary` marks gray-zone concepts, and `is_boundary_between` links the classes involved. The same structure can support rating and relevance scales with boundaries between adjacent grades.

## Knowledge graph / policy graph

The generator prompt is a **versioned knowledge graph** rendered as Obsidian-style Markdown:

- **Nodes** - one `MD.root` project root (for example `GA.root`) plus one node per class/subcategory. Files include frontmatter and policy text: definitions, criteria, hard negatives, coverage targets, examples, and source anchors.
- **Edges** - typed relationships. `confused_with` edges capture observed class confusions and inform audit priority and prompt context packs.
- **Roadmap** - a dedicated boundary node type will make gray-zone concepts graph citizens instead of annotations.

Markdown keeps policy readable for SMEs through backlinks and graph view, while remaining easy to compile into prompts.

## Models, cost, batching

- **Model tiers.** §3 exposes **HIGH** (Opus 4.6/4.7, GPT-5.5 high/low, Gemini 3.1 Pro), **MEDIUM** (Sonnet 4.6/5, GPT-5.4-mini high/xhigh, Gemini flash), and **LOW / FREE** (GPT-5.4-mini low, Haiku 4.5, Gemini flash-lite, local models). High-tier models are unchecked by default. Opus 4.7+ emits ~30% more tokens; Haiku 4.5 is the cheap/fast vision default.
- **Local GPU support.** LM Studio models run locally at $0.00: `local/gemma` is fast, `local/qwen` is slower and higher quality. Use them for offline iteration and cost-free sweeps.
- **Cost tracking.** Per-call USD cost comes from usage tokens (`pipeline/providers/pricing.py`, mirrored in `web/run-trigger.js`, kept in exact sync). Costs are reduced by request-level image batching now and provider async Batch APIs later.
- **Output budgets.** Justifications are capped at <= 300 words (~400 tokens). Anthropic/Gemini use ~768 visible-token caps with separate thinking budgets. OpenAI reasoning models use combined `max_completion_tokens`, so they keep reasoning headroom plus ~768 visible tokens (~2000 low reasoning, ~4000 high/xhigh). Local models use 4000-6000.
- **Split discipline.** Training split runs can update policy/prompts; test split results drive metrics only.

### Image batching

`OpenAIClient.batch_label` (`pipeline/providers/openai_client.py`) sends **N images in one API request** sharing one policy/system/instruction block and returns one JSON `items` array in `image_id` order. The cost-win default is **about 5 images per request**.

Batching sends the policy bundle (~3.7k tokens) once instead of once per image; each image remains ~700 input tokens. A batch of 5 cuts input roughly 3x, for **25-50% per-image cost savings** depending on output/reasoning cost. **Local models run single-image, not batched.** Before making batching default for scored runs, run a batched-vs-single-image A/B. Future provider async Batch APIs can add ~50% off list with ~24h turnaround; they are not implemented yet.

## Dataset images are local-only

Image bytes are **never committed to git**. The repo tracks manifests that reference local files under `data/images/**`; actual `.png`, `.jpg`, and `.jpeg` files stay on the machine and are excluded by `.gitignore`.

Generate manifests with `python3 scripts/sample_genai_gold_sets.py` against ignored `data/images/genai-classification/source-datasets/` folders. The CLI reads local images and writes ignored manifests without adding bytes to git.

## Labels and guardrails

Initial labels: `gen_ai` means likely fully/materially AI-generated or synthetic; `not_gen_ai` means likely authentic, conventionally edited, CGI/game/rendered, or insufficiently evidenced.

Expandable positives cover impossible hands/fingers/limbs, garbled text/logos/symbols, plastic skin, inconsistent perspective/reflections/shadows/geometry, and synthetic disclosure metadata/watermark/context. Boundary nodes cover photo editing, CGI/game/3D assets, and low-quality uncertain cases.

Guardrails: SME-reviewed labels are truth; LLM consensus is audit signal, not ground truth. Keep legacy labels, model votes, arbiter decisions, SME canonical labels, tiers, and exports separate. Use reviewable Markdown/JSON graph diffs. Metrics must be split-aware, denominator-explicit, confidence-interval-aware, and based on gold/platinum labels. Validation/holdout/boundary/sentinel examples must not leak into prompts, policy tuning, or adaptive discovery. Adaptive batches improve coverage; sentinel/random batches measure prevalence. Seed metrics report `not_enough_data` until real media, truth, and paired predictions exist.

## Run the web interface

```bash
cd /Users/sacsimoto/GitHub/RUSH
python3 -m http.server 8766 --bind 127.0.0.1
# open http://127.0.0.1:8766/web/
```

The web demo defaults to 100 dev golden + 100 locked holdout records. It first tries ignored local manifests under `data/images/genai-classification/manifests/`; otherwise `window.RushGenaiSampler.runDemoReset({ seed, nDev, nHoldout, mode })` provides a synthetic fallback. The current web flow does not invoke LLMs; bulk model labeling is next.

## Validate

```bash
python3 scripts/validate_foundation.py
node scripts/validate_web_sampler.js
```

Foundation validation checks graph IDs, single `GA.root`, parent chains, edges, seeds, schemas, mock metric safety, and required web/docs assets. Web sampler validation checks deterministic sampling, split disjointness, balanced totals, assumptions, and SME overrides.

## Near-term roadmap

- **M1 Graph parse / cold start** - Markdown policy nodes, edge manifest, source anchors, schema validation, SME-readable skeleton.
- **M2 Golden-set registry** - image records, label versions, LLM vote slots, SME canonical labels, split assignments.
- **M3 Labeling queue** - human + future LLM labels with justifications, confidence, evidence refs, and SME audit routing.
- **M4 Metrics dashboard** - accuracy, precision, recall, FPR, positive proportion, informedness, graph coverage, gray-zone mass by node/version/split.
- **M5 Policy diff workflow** - proposed node additions, clarifications, examples, exceptions, and rejected-change memory.
- **M6 LLM ensemble integration** - API-backed model votes with structured outputs, policy-node citations, consensus audits, graph-node context packs, and SME resurfacing for misalignments.
