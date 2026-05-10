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
