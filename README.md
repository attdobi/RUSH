# RUSH

**Reinforcement learning Using SME feedback and High-reasoning models**

RUSH is a policy-graph system for scaling subject matter expert (SME) decision quality to non-expert labelers: BPO reviewers, lower-reasoning LLMs, and production ML systems. The core idea is simple and annoyingly powerful: if policy is explicit, versioned, graph-structured, and tested against golden examples, label quality improves for everyone.

The first pilot is a **cold-start Generative AI image-classification policy**. We begin with images and labels, grow the policy as an Obsidian-style Markdown graph, capture SME/LLM misalignments, and propose policy diffs that an SME can approve or reject.

## What this repository contains now

- `web/` — dependency-free foundation web interface with tabs for About, Policy Graph, Golden Set, Labeling, Policy Diff, and Metrics.
- `policy-graph/Generative_AI/v0.1/` — initial Obsidian-compatible Markdown policy graph for GenAI detection.
- `schemas/` — JSON schemas for policy nodes, image records, label votes, SME reviews, policy patches, and metric snapshots.
- `data/seed/` — placeholder golden-set records, labels, metrics, and policy-suggestion examples.
- `docs/visuals/` — SVG architecture visuals used by README and the web UI.
- `scripts/validate_foundation.py` — lightweight repository validation with no external dependencies.

## RUSH in one picture

![RUSH system loop](docs/visuals/rush-system.svg)

## Design principles

1. **SME-reviewed labels are truth; LLM consensus is signal.** 3/3 model agreement can still be a correlated failure, so it should feed audit priority rather than become silent ground truth.
2. **Keep labels separate.** Legacy labels, model votes, arbiter decisions, SME canonical labels, label tiers, and downstream exports are distinct records.
3. **Policy is the product and the measurement surface.** Nodes carry definitions, criteria, hard negatives, coverage targets, examples, and source anchors.
4. **Graph diffs are reviewable.** Cold-start and warm-start refinements become Markdown/JSON patches that SMEs can approve or reject.
5. **Metrics must be split-aware.** Use gold/platinum labels for final decision-quality metrics, protect holdouts, and keep adaptive discovery batches separate from prevalence estimates.

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

## Validate the scaffold

```bash
python3 scripts/validate_foundation.py
```

The validator checks that graph node IDs are unique, edge endpoints exist, seed data references valid nodes, JSON files parse, and required web/docs assets are present.

## Near-term roadmap

| Milestone | Exit criteria |
| --- | --- |
| M1 — Graph parse / cold start | Markdown policy nodes, edge manifest, source anchors, schema validation, SME-readable skeleton. |
| M2 — Golden-set registry | Image records, label versions, LLM vote slots, SME canonical labels, split assignments. |
| M3 — Labeling queue | Human + future LLM labels with justifications, confidence, evidence refs, and SME audit routing. |
| M4 — Metrics dashboard | Accuracy, precision, recall, FPR, positive proportion, informedness, graph coverage, gray-zone mass by node/version/split. |
| M5 — Policy diff workflow | Proposed node additions, clarifications, examples, exceptions, and rejected-change memory. |
| M6 — LLM ensemble integration | 3 high-reasoning model votes plus arbiter; consensus audits; prompt/context packs from graph nodes. |

## Why Obsidian-style Markdown?

The policy graph should be easy for humans to read and easy for machines to compile. Each node is a normal Markdown file with machine-readable frontmatter and human-readable policy text. Obsidian backlinks and graph view make policy coverage and ambiguity visible without trapping the project inside a custom editor.
