# RUSH

**Reinforcement learning Using SME feedback and High-reasoning models**

RUSH is a policy-graph system for scaling subject matter expert (SME) decision quality to non-expert labelers: BPO reviewers, lower-reasoning LLMs, and production ML systems. Policy is explicit, versioned, graph-structured, and tested against golden examples so label quality improves over time.

The first pilot is cold-start **Generative AI image classification**: start with images and labels, grow an Obsidian-style Markdown policy graph, capture SME/LLM misalignments, and propose policy diffs for SME approval.

![RUSH system loop](docs/visuals/rush-system.svg)

## Main outcomes

RUSH exists to deliver three headline outcomes:

- **RLHF with minimal human intervention.** Humans are spent only where they move the needle: the loop highlights and routes images on the **decision boundaries** for SME re-review, instead of asking people to re-label the easy majority.
- **A self-improving policy loop.** Each iteration re-runs the **generator** prompt, grows the **knowledge graph (KG)**, and **enhances decision quality** — misalignments and boundary cases become versioned policy updates that raise the next round's accuracy.
- **A multi-LLM price-optimization system — the escalation cascade.** Start **cheap**: low-cost model consensus resolves the aligned majority (this tier is also the production metric). Escalate only **misaligned / lack-of-consensus** items to a **high-reasoning panel**, and only the residual **boundary / difficult** cases to **human SME** re-review. Expensive judgment is reserved for the boundary. See [The escalation cascade](#the-escalation-cascade--cheap--high-reasoning--sme).

**Cost per batch is the measuring stick.** Every run records per-image and per-batch cost in `run_manifest.json` (usage tokens x `pipeline/providers/pricing.py`; local models = $0) and surfaces it live in §3. Because spend is tracked at each iteration, the price-optimization above is *measurable*: you can watch cost fall as consensus absorbs the easy cases and expensive high-reasoning calls stay reserved for the boundary.

## The escalation cascade — cheap → high-reasoning → SME

RUSH is an **enterprise auto-judge**: a self-improving system that scales a subject-matter expert's decision quality to production, then keeps the resulting metric honest as policy and content drift. The same machinery serves **Trust & Safety** (violative / non-violative), **content quality**, **search relevance** (graded 1–5 scales), and **ads relevance** — a "project" is a policy for one category. The image demos (`Generative_AI`, `MNIST_Digits`) are the toy proofs of the loop; the target verticals are things like a 50-page Adult-Content policy or a cold-started PII policy.

The organizing idea is a **three-tier judge cascade** that spends the expensive resources only where cheap ones fail. Do not confuse it with "run three big models and vote" — the point is the opposite:

1. **Tier 1 — cheap consensus (measure).** Low-cost models label the whole stream and reach consensus on the aligned majority. This tier *is* the production metric — e.g. daily prevalence (percent of violative impressions) at scale (~tens of thousands of labels/day), reported with confidence intervals.
2. **Tier 2 — high-reasoning panel (validate + critique).** Only the **misaligned / lack-of-consensus / boundary / difficult** items escalate to a small panel of high-reasoning models (e.g. 3× consensus, Cohen's kappa across raters). The **1×-vs-3× gap is the measured deployment bias** each policy version must shrink; the panel also acts as the RL *critic* that locates where the guideline loses decision quality.
3. **Tier 3 — human SME (adjudicate).** Only the residual boundary cases the panel still can't resolve reach a human, through the **GoldMiner** labeling interface, ranked by a priority score (`is_boundary`, difficulty, judge-disagreement, prior human touches, L2 coverage). Human attention is spent *only* on the boundary — the scarcest resource optimized around.

**Two loops turn at different speeds.** A **fast measurement loop** (cheap Tier 1, runs continuously at production scale) produces the metric; a **slow learning loop** (Tiers 2–3 on a small sample) grows the golden set and auto-tunes the **prompt-as-policy**, gated by version control that only promotes a converged, cost-acceptable version. Signal compounds inside-out through three flywheels: **prompt-tuning** (fast) ⊂ **golden-set polish/audit** (medium) ⊂ **policy approval** (slow, human-governed).

**The prompt is the policy; tuning it is RLHF.** The generator prompt is the policy (π); an SME label is the teacher; **decision quality** (F1 / informedness / Cohen's kappa vs the golden set on a *held-out* slice) is the reward; each guideline edit is one gated **textual-gradient** step — "gradient descent in a document." A *critic* model diagnoses the policy gap in words; an *actor* model emits exactly one trackable edit; a gate accepts it only if held-out DQ improves. Unlike standard RLHF, the reward model is a **maintained, certified golden set (GDS)**, not a frozen network — an overturn in adjudication *is* an update to the reward model.

### Guardrails that keep the cascade honest

The naive cascade ("escalate disagreements, trust the human, keep the biggest wins") is statistically unsound. RUSH's design owes four non-obvious corrections to the internal technical notes, and the demo is meant to *show* them:

- **Gate and clip every escalated edit.** A high-reasoning model's proposed edit is a high-variance action. Accept only if held-out DQ improves *and* the edit changes a small fraction of the guideline (default ≤ ~5% of tokens); otherwise shrink-and-retry or reject. Prefer principle-level edits over item-level ones — "an edit that names an item is a memorized point in disguise."
- **Audit the agreements, not just the disagreements.** Escalating only disagreements builds a confidently-wrong ruler (*incorporation bias*: the judge helps construct its own reference standard). A small random **aligned audit stream** (~5% of agreements) goes to SMEs too, so the label-error rate on agreements is actually measured.
- **The golden set is not so golden.** When seed labels were re-adjudicated against zero-shot LLM labels, experts sided with the *model* roughly **⅓ to ½ of the time** (Trust / content-quality + search / ads). So the top rung is *re-adjudication* — overturn or confirm, with a per-item cap on human touches — not "ask the human once." A falling **overturn rate** is the signature of a golden set that is converging.
- **Separate prompt-lift from label-lift.** Decompose each decision-quality change into "the prompt got better" (ruler fixed) vs "the ruler moved" (labels cleaned). A cycle dominated by label-lift is a golden-set quality event, not a modeling win, and must not be reported as one. And treat a reported score of 100% as a flag to investigate, not a trophy.

**One governing constraint:** decision quality is bounded by golden-set coverage relative to production. The cascade's job is to grow that coverage cheaply and spend human judgment only where it changes the answer.

> **Status:** Tiers 2–3 (high-reasoning panel + SME misalignment routing) and the prompt-as-policy tuning loop exist in the demo today; the **cheap Tier-1 first rung**, production prevalence measurement, GoldMiner re-adjudication with overturn/confirm, the aligned audit stream, and the prompt-lift/label-lift decomposition are the near-term build-out. Headline economics quoted in exec material (>100× throughput and >10× cost reduction vs 3× human BPO, ~$710K → <$71K per million labels) are Pinterest-scale targets from the exec brief, **not yet measured in this repo**.

## Repo contents

- `web/` - dependency-free demo UI for GenAI sampling, policy graph building, future bulk LLM labeling, and decision-quality review. `web/genai-sampler.js` is the fallback when manifests are missing.
- `policy-graph/Generative_AI/v0.1/` - initial Obsidian-compatible GenAI policy graph.
- `schemas/` - JSON schemas for graph, image/split, label/vote, model-output, review, patch, export, and metric records.
- `data/seed/` - mock golden-set records, labels, not-enough-data metrics, and policy-suggestion examples.
- `docs/visuals/` - SVG visuals for the README and web UI.
- `scripts/validate_foundation.py` - dependency-free validation.

## AI Handoff & Embedded Memory

> **If you are an AI taking over this repo (e.g. Fable-5 or Claude-Code): ABSORB THIS KNOWLEDGE BASE FIRST, before you touch any code.** This is an AI-to-AI transfer of everything the previous team (Pista/Theo/X-engineers) knows about RUSH.
>
> 1. **Read `docs/ai-handoff/HANDOFF.md` end to end**, then **`docs/ai-handoff/SESSION-2026-07-05-fable5.md`** (the enterprise-repositioning + polish delta: the escalation cascade, the honesty guardrails, what shipped, and the polish plan). Together they are the full transfer: project thesis, architecture, repo map, run ops, fixes with root causes, current state, team conventions, and the OPEN ISSUES to work next.
> 2. **Load and query the embedded project memory** in `docs/ai-handoff/memory-embeddings/` via `scripts/query_memory.py` whenever you need prior context, instead of re-deriving it. Agents with access to the Gemma-embedding server on Attila's GPUs (see below) can query this index directly.
> 3. **Preserve the working conventions** (HANDOFF §8): named engineers, `[X#]` commit prefixes, feature-branch-only for multi-file work, and bump the `web/index.html` cache-buster on any JS/CSS change.

The local semantic memory is in `docs/ai-handoff/memory-embeddings/`: `index.jsonl` stores chunk text plus 768-dim embeddings, and `manifest.json` records the embedding model, source files, chunk parameters, counts, and byte size. Embeddings use Gemma-embedding `text-embedding-embeddinggemma-300m-qat` served by LM Studio.

- **Query the memory:** `./.venv/bin/python scripts/query_memory.py "why is qwen slow"`
- **Regenerate after editing docs:** `./.venv/bin/python scripts/build_memory_embeddings.py`

**Reaching the GPUs from a networked agent.** The models (Gemma-embedding, `gemma-4-26b-a4b-qat`, `qwen3.6-27b`) run on a GPU host (2× RTX 3090) shared over LM Studio's LM Link mesh. LM Link makes them usable *inside the LM Studio app* but does not auto-expose an HTTP server, so a separate process must reach a real endpoint:

- On the machine running RUSH, start the local server: `~/.lmstudio/bin/lms server start --port 1234` (it bridges to the loaded remote models), then verify `curl http://127.0.0.1:1234/v1/models`.
- All three entry points — the labeling pipeline (`pipeline/providers/registry.py`), `scripts/query_memory.py`, and `scripts/build_memory_embeddings.py` — honor **`RUSH_LOCAL_BASE_URL`** (default `http://127.0.0.1:1234/v1`). Point the whole repo at a remote GPU host with `export RUSH_LOCAL_BASE_URL=http://<host>:1234/v1`. Do **not** subnet-scan to find the host — start the local bridge or set the variable.

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

## Run the demos on another machine (e.g. Mac Pro)

Both demos ship with a **committed portable fixture** (~56 MB total), so a fresh `git clone` runs end-to-end **without** the local ~12 GB source tree.

What's committed:
- **MNIST** — the full 2,500-image demo gold set as PNGs under `data/images/mnist-classification/source-datasets/mnist/<digit>/` (~1.6 MB), **plus** the entire 70k MNIST set packed compactly in `data/images/mnist-classification/mnist_full.npz` (~11.5 MB).
- **GenAI** — a balanced 72-image sample (12 per dataset×class, both splits, original bytes, ~43 MB) under `data/images/genai-classification/sample/`, with `manifests/combined_labels.portable.jsonl`.

### Option A — clone and go (portable fixture)

```bash
git clone <repo> RUSH && cd RUSH
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# MNIST demo (offline dry-run, no API keys needed):
./.venv/bin/python scripts/run_bulk_labeling.py \
  --area MNIST_Digits --models openai/gpt-5.5 --split dev_golden --limit 5

# GenAI demo — force the committed portable fixture:
RUSH_PORTABLE=1 ./.venv/bin/python scripts/run_bulk_labeling.py \
  --area Generative_AI --models openai/gpt-5.5 --split dev_golden --limit 5

# Web UI:
./.venv/bin/python scripts/rush_web_server.py   # http://127.0.0.1:8766/web/
```

`RUSH_PORTABLE=1` forces the portable GenAI manifest; even without it, the pipeline auto-falls back to the portable fixture whenever the full `source-datasets/` image tree is absent (a sparse clone). Live model calls need a local **LM Studio** endpoint running plus provider keys in `.env`, and `--live --allow-spend`.

### Option B — full parity (all images)

Only needed to resample gold sets larger than the committed fixture.

Expand the full 70k MNIST set locally from the committed archive, then resample any size:
```bash
./.venv/bin/python scripts/unpack_mnist.py            # writes all 70k PNGs into source-datasets/mnist/<digit>/
./.venv/bin/python scripts/sample_mnist_gold_sets.py \
  --n-train 2000 --n-val 500 --seed 20260703 --source-root ~/Downloads/mnist_png --force
```

Pull the full ~12 GB GenAI source tree from the Mac mini and regenerate its manifests:
```bash
rsync -avh <mac-mini-host>:/Users/sacsimoto/GitHub/RUSH/data/images/genai-classification/source-datasets/ \
  data/images/genai-classification/source-datasets/
./.venv/bin/python scripts/sample_genai_gold_sets.py --n-dev 100 --n-holdout 100 --seed 20260510 --force
```

### Where to find the full datasets

- **MNIST (70k):** already shipped compact in-repo as `mnist_full.npz` — unpack with `scripts/unpack_mnist.py`, or regenerate from `~/Downloads/mnist_png/` with `scripts/pack_mnist_full.py`. Publicly, MNIST as image files + labels is widely available on **Kaggle** and many **GitHub** repos (e.g. `mnist_png`-style exports).
- **GenAI (~12 GB, ~20k raw images):** byte-exact copies live on the **Mac mini** at `data/images/genai-classification/source-datasets/{midjourney,sdv1_4,wfir}/{ai_generated,not_ai_generated}/` — `rsync` from there for full parity (command above). Dataset identities: `midjourney` = Midjourney-generated vs real, `sdv1_4` = Stable Diffusion v1.4 generated vs real, `wfir` = StyleGAN faces (“Which Face Is Real”-style) vs real. No single canonical public URL is recorded in-repo; the Mac mini tree is the source of truth.

## Labels and guardrails

Initial labels: `gen_ai` means likely fully/materially AI-generated or synthetic; `not_gen_ai` means likely authentic, conventionally edited, CGI/game/rendered, or insufficiently evidenced.

Expandable positives cover impossible hands/fingers/limbs, garbled text/logos/symbols, plastic skin, inconsistent perspective/reflections/shadows/geometry, and synthetic disclosure metadata/watermark/context. Boundary nodes cover photo editing, CGI/game/3D assets, and low-quality uncertain cases.

Guardrails: SME-reviewed labels are truth; LLM consensus is audit signal, not ground truth. Keep legacy labels, model votes, arbiter decisions, SME canonical labels, tiers, and exports separate. Use reviewable Markdown/JSON graph diffs. Metrics must be split-aware, denominator-explicit, confidence-interval-aware, and based on gold/platinum labels. Validation/holdout/boundary/sentinel examples must not leak into prompts, policy tuning, or adaptive discovery. Adaptive batches improve coverage; sentinel/random batches measure prevalence. Seed metrics report `not_enough_data` until real media, truth, and paired predictions exist.

## Run the web interface

The web UI needs the RUSH server (not a static file server): it serves `web/` **and** the `/api/*` endpoints that drive §2 policy growth, §3 labeling, and §4 scoring.

```bash
# from the repo root, using the repo venv (has openai/anthropic/etc.)
.venv/bin/python scripts/rush_web_server.py --host 127.0.0.1 --port 8766 --repo-root "$PWD"
# open http://127.0.0.1:8766/web/
```

In production this runs under the macOS LaunchAgent `com.attdobi.rush-web`; restart with `launchctl kickstart -k gui/$(id -u)/com.attdobi.rush-web` and verify the new PID (see `docs/ai-handoff/HANDOFF.md` §3). A bare `python3 -m http.server` will serve the page but every `/api/*` call 404s, so the labeling and policy loops are dead.

The web demo defaults to 100 dev golden + 100 locked holdout records. It first tries local manifests under `data/images/genai-classification/manifests/`; otherwise `window.RushGenaiSampler.runDemoReset({ seed, nDev, nHoldout, mode })` provides a synthetic fallback. Bulk multi-LLM labeling runs from §3 (`POST /api/runs/start`); §4 scores the results.

**Get local image data first.** Image bytes are never committed, but a small sample rides in the repo as a zip. Unpack it so §1 previews real images and small labeling batches work:

```bash
python scripts/load_sample_data.py --demo all
```

See [docs/data-loading.md](docs/data-loading.md) for the sample vs. full-dataset workflow and how to rebuild the samples on the Mac mini.

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
