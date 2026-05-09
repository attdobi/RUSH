# RUSH Foundation Hardening — Execution Plan

**Branch:** `feat/foundation-web-policy-graph`
**Status:** Pre-merge hardening pass
**Date:** 2026-05-09
**Produced by:** TheoMaximus (Tech Lead)

---

## Summary of findings

The foundation scaffold on `feat/foundation-web-policy-graph` is structurally sound:
9 policy nodes, 12 edges, 6 JSON schemas, 3 placeholder image/label records, a
working web UI, SVG diagrams, and a validator that passes. However, it needs
hardening before merging to `main` in five areas:

1. **Missing data contracts** — 5 schemas from v0.1/v0.2 specs have no JSON schema.
2. **Weak metric schema** — no denominators, CIs, macro-F1, calibration, or graph-location fields.
3. **Misleading UI** — perfect 1.0/0.0 placeholder metrics displayed without
   denominator context; data hardcoded in app.js instead of loaded from seed JSON.
4. **Thin validator** — no schema-shape checks, no graph-invariant enforcement,
   no split-leakage detection.
5. **Incomplete policy graph** — missing negative/exception nodes required by v0.1/v0.2
   specs (authentic-photo negative, compression-artifact exception, medical/prosthetic
   exception, repeated-detail artifacts, low-quality abstain enforcement).

Plus one bug: `GA.root.md` uses `node_type: category` — should be `node_type: root`.

---

## Workstreams

### Workstream 1 — X1: Schemas, Data Contracts, Validator Hardening

**Scope:** `schemas/`, `scripts/`, `data/seed/`, one `.md` frontmatter fix.

#### Task 1.1 — Add 5 missing JSON schemas

Create each file under `schemas/`. Follow the existing style (JSON Schema draft 2020-12,
`title`, `required` array, `properties` block).

| File | Key required fields | Source spec reference |
|---|---|---|
| `schemas/arbiter-decision.schema.json` | `arbiter_run_id`, `image_id`, `final_label` (enum: gen_ai / not_gen_ai / abstain), `final_placements` (array of {node_id, weight}), `difficulty_tier` (enum: easy / medium / hard / sme-misalignment), `consensus_status` (enum: unanimous / two_of_three / no_majority), `reconciliation_method`, `per_labeler_votes` (array of {labeler_id, label, placements, confidence}), `policy_graph_version` | v0.1 §5.2 ArbiterOutput, v0.2 §5.1 |
| `schemas/label-tier-record.schema.json` | `image_id`, `current_tier` (enum: provisional / silver / gold / platinum / deprecated / superseded), `previous_tier`, `promoted_by` (enum: sme_review / consensus_audit / repeat_review / policy_owner), `promotion_reason`, `promoted_at` (ISO 8601), `policy_graph_version` | v0.2 §2 label tiers |
| `schemas/split-assignment.schema.json` | `image_id`, `split` (enum matching image-record split enum), `assigned_at`, `dedupe_cluster_id`, `near_duplicate_cluster_id`, `assignment_version`, `leak_check_passed` (boolean) | v0.2 §8 split discipline |
| `schemas/policy-edge.schema.json` | `source_node_id`, `target_node_id`, `edge_type` (enum: subtype_of / exception_to / boundary_with / confused_with / clarifies / example_of / negative_example_of), `confidence` (number 0–1), `provenance`, `version` | v0.1 §4.2, v0.2 §3 |
| `schemas/export-record.schema.json` | `export_id`, `policy_graph_version`, `label_version`, `truth_label_tiers` (array of gold/platinum), `split`, `holdout_report_ref`, `item_count`, `denominators` (object), `created_at` | v0.2 §10 |

**Acceptance criteria:**
- [ ] All 5 files exist under `schemas/` and parse as valid JSON
- [ ] Each uses `$schema: "https://json-schema.org/draft/2020-12/schema"`
- [ ] Each has a `required` array with the fields listed above
- [ ] Enums match the values specified in v0.1/v0.2 specs
- [ ] `validate_foundation.py` loads them without error

#### Task 1.2 — Strengthen `schemas/metric-snapshot.schema.json`

Edit in place. Add these fields/substructures:

```
"denominators": {
  "type": "object",
  "required": ["n_total", "n_positive", "n_negative"],
  "properties": {
    "n_total": {"type": "integer", "minimum": 0},
    "n_positive": {"type": "integer", "minimum": 0},
    "n_negative": {"type": "integer", "minimum": 0},
    "n_abstain": {"type": "integer", "minimum": 0}
  }
}
```

Add to `metrics.properties`:
- `macro_f1` (number 0–1)
- `calibration_ece` (number ≥ 0, expected calibration error)
- `graph_location_accuracy` (number 0–1)

Add to `graph_health.properties`:
- `consensus_audit_error_rate` (number 0–1)
- `sme_override_rate` (number 0–1)
- `per_node_difficulty` (object, additionalProperties: number)
- `per_node_coverage` (object, additionalProperties: number)

Add `confidence_intervals` object:
```
"confidence_intervals": {
  "type": "object",
  "properties": {
    "method": {"enum": ["wilson", "clopper_pearson", "bootstrap", "none"]},
    "alpha": {"type": "number"},
    "intervals": {"type": "object"}
  }
}
```

Constrain `split` to an enum (same as image-record split enum).

Add `"denominators"` to the top-level `required` array.

**Acceptance criteria:**
- [ ] Schema file parses as valid JSON
- [ ] `denominators` is required at top level
- [ ] `metrics` includes `macro_f1`, `calibration_ece`, `graph_location_accuracy`
- [ ] `graph_health` includes `consensus_audit_error_rate`, `sme_override_rate`, `per_node_difficulty`, `per_node_coverage`
- [ ] `confidence_intervals` structure present with method enum
- [ ] `split` constrained to enum

#### Task 1.3 — Fix GA.root.md node_type

**File:** `policy-graph/Generative_AI/v0.1/GA.root.md`

Change `node_type: category` → `node_type: root`

**Acceptance criteria:**
- [ ] Frontmatter reads `node_type: root`
- [ ] Validator still passes

#### Task 1.4 — Strengthen `scripts/validate_foundation.py`

Expand the validator (still zero external dependencies — stdlib only) to check:

1. **Frontmatter shape validation:** For each policy node .md, verify:
   - All `required` keys from policy-node.schema.json are present
   - `node_type` value is in the schema's enum
   - `polarity` value is in the schema's enum
   - `status` value (if present) is in the schema's enum
   - `coverage_target` is a dict (not null/string)

2. **Graph invariants:**
   - Exactly one node with `node_type: root` per area
   - No orphan nodes (every non-root node is the source of at least one `subtype_of` edge)
   - Every leaf node (no incoming `subtype_of` edges) has non-empty body text below frontmatter containing a decision rule or boundary section

3. **Edge validation:**
   - Every `edge_type` in edges.json is in the allowed enum: `subtype_of`, `exception_to`, `boundary_with`, `confused_with`, `clarifies`, `example_of`, `negative_example_of`
   - Both `source_node_id` and `target_node_id` reference known node IDs (existing check — keep it)

4. **Split leakage check:**
   - Load image-records.json, group by `dedupe_cluster_id`
   - If any cluster has images in both a holdout split (`locked_holdout`, `boundary_holdout`) AND a non-holdout split (`development`, `validation`), report an error

5. **Seed data cross-references:**
   - Label records reference valid image IDs and node IDs (existing — keep)
   - Suggestion records reference valid node IDs (existing — keep)
   - Metrics JSON has `denominators` key (new)

6. **Summary output:** Print counts for nodes, edges, images, labels, and any new schemas found.

**Acceptance criteria:**
- [ ] `python3 scripts/validate_foundation.py` passes on the hardened repo
- [ ] Introducing a deliberate error (e.g., duplicate node_type: root, orphan node, bad edge_type, cross-split cluster) causes a clear error message
- [ ] No external dependencies (stdlib only)
- [ ] Exit code 0 on success, 1 on failure

#### Task 1.5 — Update `data/seed/metrics.json` to match hardened schema

Add the new required and recommended fields with explicit placeholder values:

```json
{
  "denominators": {
    "n_total": 2,
    "n_positive": 1,
    "n_negative": 1,
    "n_abstain": 1
  },
  "confidence_intervals": {
    "method": "none",
    "alpha": 0.05,
    "intervals": {}
  },
  "metrics": {
    "accuracy": null,
    "precision": null,
    "recall": null,
    "fpr": null,
    "positive_proportion": null,
    "informedness": null,
    "macro_f1": null,
    "calibration_ece": null,
    "graph_location_accuracy": null
  },
  "graph_health": {
    "consensus_audit_error_rate": null,
    "sme_override_rate": null,
    "per_node_difficulty": {},
    "per_node_coverage": {}
  }
}
```

Set all decision-quality metrics to `null` (not fake 1.0 values). Keep `graph_health.node_count`, `edge_count`, `coverage`, `gray_zone_mass`, `orphan_nodes` as they are (those are real structural counts).

Add/update `warning`: `"Placeholder — all decision-quality metrics are null until golden-set images are labeled and SME-reviewed. Graph-health values reflect v0.1 scaffold structure only."`

**Acceptance criteria:**
- [ ] `denominators` present with correct small-n counts
- [ ] All decision-quality metrics are `null`, not fake perfect values
- [ ] `confidence_intervals.method` is `"none"`
- [ ] `warning` field explains placeholder status
- [ ] Graph-health structural counts preserved
- [ ] Validator passes

---

### Workstream 2 — X2: Web UI Hardening + Metrics Masking

**Scope:** `web/index.html`, `web/app.js`. Do not modify `web/styles.css` unless needed for new elements.

#### Task 2.1 — Load seed data from JSON files instead of hardcoded arrays

Refactor `app.js` so that `images`, `labels`, `suggestions`, and `metrics` are loaded
from the seed JSON files at startup via `fetch()`:

- `../data/seed/image-records.json` → images
- `../data/seed/label-records.json` → labels
- `../data/seed/policy-suggestions.json` → suggestions
- `../data/seed/metrics.json` → metrics

The `nodes` and `edges` arrays for the **graph visualization** may remain as compiled
constants in `app.js` (the .md graph has no single compiled JSON yet). But they MUST
include the new nodes that X3 adds (see Task 2.5).

Wrap all render functions in an `init()` async function that fetches then renders.
Handle fetch errors gracefully (show "Failed to load data" in each panel).

**Acceptance criteria:**
- [ ] No `const images = [...]`, `const labels = [...]`, `const suggestions = [...]`,
      or `const metrics = {...}` hardcoded arrays remain in app.js
- [ ] Data loads from `../data/seed/*.json` via fetch
- [ ] All 6 tabs still render correctly when served via `python3 -m http.server`
- [ ] Fetch failure shows a user-friendly error, not a blank panel or console crash

#### Task 2.2 — Mask placeholder metrics with denominator context

When metrics values are `null` or denominator `n_total < 30`:

- Display the value as **"—"** (em-dash) instead of a number
- Add a `placeholder` CSS class that dims the card (opacity 0.5)
- Show `n = {n_total}` under each metric value
- Display CI range `[lower, upper]` when `confidence_intervals.intervals` has data

Add a **prominent banner** at the top of the metrics panel:

```html
<div class="metrics-banner warning">
  ⚠️ Seed metrics are placeholders (n&nbsp;=&nbsp;2). Real decision-quality metrics
  require golden-set images, model votes, and SME-reviewed labels. Graph-health
  values reflect scaffold structure only.
</div>
```

Style the banner with `background: rgba(255,209,102,.12); border: 1px solid var(--gold);
border-radius: 14px; padding: 16px; margin-bottom: 18px; color: var(--gold);`.

**Acceptance criteria:**
- [ ] No perfect 1.0 or 0.0 decision-quality values displayed
- [ ] Null metrics show em-dash, not "null" or "NaN"
- [ ] Denominator `n = X` shown under each metric card
- [ ] Warning banner visible in Metrics tab
- [ ] Graph-health values (node_count, edge_count, etc.) still display normally

#### Task 2.3 — Add split + tier indicators to Golden Set and Labeling panels

**Golden Set panel:**
- Show `split` as a badge on each golden card (e.g., `development`, `locked_holdout`)
- Style holdout splits distinctly (e.g., `border-color: var(--red)`)

**Labeling panel:**
- Add `label_tier` column to the table
- Show tier as badge (gold/provisional styling already exists)

**Acceptance criteria:**
- [ ] Each golden card shows its split assignment
- [ ] Holdout cards are visually distinct
- [ ] Label table has a Tier column with badge styling

#### Task 2.4 — Verify hero intro emphasizes the three pillars

The current hero text already mentions high-reasoning models, voting/consensus, and
Obsidian-style graph policies. Verify the About cards still read:

1. **"High-reasoning committee"** — models vote independently, cite evidence, expose disagreements
2. **"Obsidian policy graph"** — Markdown nodes with frontmatter, examples, edges, source anchors
3. **"Consensus ≠ truth"** — SME labels are gold; consensus is risk signal, not ground truth

If the current text matches these three pillars, leave it. If any pillar is weak or missing,
strengthen the card body text to match the language above.

**Acceptance criteria:**
- [ ] All three About cards present and emphasize distinct pillars
- [ ] Hero `<h1>` mentions high-reasoning consensus and graph policies
- [ ] Lede paragraph mentions all three: model voting, consensus audits, Obsidian-style graph

#### Task 2.5 — Add new graph nodes to visualization

After X3 creates the new policy-graph .md files, add corresponding entries to the
`nodes` and `edges` arrays in app.js for the graph canvas:

New nodes to add (get exact IDs from X3's .md files):
- `GA.negative.authentic_photo` — type: category, polarity: negative
- `GA.exception.compression_artifacts` — type: exception, polarity: hard-negative
- `GA.exception.medical_prosthetic` — type: exception, polarity: hard-negative
- `GA.visual_artifacts.repeated_details` — type: category, polarity: positive

New edges to add (from X3's edges.json updates):
- subtype_of edges to GA.root
- exception_to / confused_with edges to relevant positive nodes

Position new nodes to avoid overlap. Suggested layout:
- `GA.negative.authentic_photo` → x:10, y:74 (left side, negative row)
- `GA.exception.compression_artifacts` → x:30, y:90
- `GA.exception.medical_prosthetic` → x:50, y:90
- `GA.visual_artifacts.repeated_details` → x:72, y:50

**Acceptance criteria:**
- [ ] All 4 new nodes appear on the graph canvas
- [ ] Clicking each shows its title, ID, type, polarity, and rule
- [ ] Edges render correctly between new and existing nodes
- [ ] No visual overlap with existing nodes
- [ ] Graph still renders correctly on mobile (< 900px)

**⚠️ Dependency:** Task 2.5 depends on X3 completing Tasks 3.1–3.4. X2 should
complete Tasks 2.1–2.4 first, then do 2.5 after X3's .md files exist.

---

### Workstream 3 — X3: Policy Graph Expansion

**Scope:** `policy-graph/Generative_AI/v0.1/` (.md files and edges.json only).

#### Task 3.1 — Add authentic-photo real-negative node

**File:** `policy-graph/Generative_AI/v0.1/GA.negative.authentic_photo.md`

Frontmatter:
```yaml
---
id: GA.negative.authentic_photo
version: Generative_AI.v0.1
title: Authentic photograph (real negative)
area: Generative_AI
node_type: category
parent: GA.root
polarity: negative
status: draft
coverage_weight: 1.0
coverage_target:
  easy_negative: 50
  hard_negative: 30
  platinum_min: 10
source_anchors:
  - "v0.2 §2: real-negative root for authentic photographs"
canonical_examples: []
---
```

Body:
- **Decision rule:** Image is an authentic, unmodified or conventionally-edited photograph
  with no generative-model provenance. Classify as `not_gen_ai`.
- **Positive criteria for this node (all must hold):**
  1. No visible generative artifacts (hands, text, skin, geometry).
  2. No provenance evidence of generation.
  3. Image characteristics consistent with camera capture or conventional editing.
- **Hard negatives for the overall policy that land here:**
  - Professional retouching with heavy smoothing (not GenAI without stronger evidence).
  - HDR/computational photography output from phone cameras.
  - Panorama stitching artifacts.

**Acceptance criteria:**
- [ ] File exists with valid frontmatter (all required schema fields present)
- [ ] `node_type: category`, `polarity: negative`, `parent: GA.root`
- [ ] Body has Decision rule, Positive criteria, Hard negatives sections
- [ ] Validator passes after adding

#### Task 3.2 — Add compression/artifact exception node

**File:** `policy-graph/Generative_AI/v0.1/GA.exception.compression_artifacts.md`

Frontmatter:
```yaml
---
id: GA.exception.compression_artifacts
version: Generative_AI.v0.1
title: Compression and encoding artifacts
area: Generative_AI
node_type: exception
parent: GA.root
polarity: hard-negative
status: draft
coverage_weight: 1.2
coverage_target:
  easy_negative: 30
  hard_negative: 40
  platinum_min: 10
source_anchors:
  - "v0.1 §4.1: compression artifacts as hard negatives"
  - "v0.2 §13: compression artifacts in confusable-with list"
canonical_examples: []
---
```

Body:
- **Decision rule:** JPEG/WebP/HEIC compression artifacts, re-encoding degradation,
  and low-bitrate video frame extraction can mimic GenAI texture artifacts. These are
  `not_gen_ai` unless independent generative evidence exists.
- **Common false-positive triggers:**
  1. Block artifacts from heavy JPEG compression imitating texture smoothing.
  2. Chroma subsampling causing skin-tone banding.
  3. Re-encoded screenshots with staircase edges.
  4. Video keyframe extraction with motion-compensation ghosts.
- **Boundary warnings:**
  - Compression on top of a generated image does not make it `not_gen_ai`.
  - When compression is severe enough to destroy classification evidence → route to
    `GA.boundary.low_quality_uncertain` for abstain/SME.

**Acceptance criteria:**
- [ ] File exists with `node_type: exception`, `polarity: hard-negative`
- [ ] Body explains compression vs GenAI distinction
- [ ] Boundary warnings reference `GA.boundary.low_quality_uncertain`

#### Task 3.3 — Add medical/prosthetic exception node

**File:** `policy-graph/Generative_AI/v0.1/GA.exception.medical_prosthetic.md`

Frontmatter:
```yaml
---
id: GA.exception.medical_prosthetic
version: Generative_AI.v0.1
title: Medical conditions and prosthetics
area: Generative_AI
node_type: exception
parent: GA.visual_artifacts.anatomy.hands
polarity: hard-negative
status: draft
coverage_weight: 1.5
coverage_target:
  easy_negative: 20
  hard_negative: 30
  platinum_min: 10
source_anchors:
  - "v0.1 §4.1: polydactyly, syndactyly as hard negatives for hand artifacts"
  - "v0.2 §3: medical-conditions as exception_to anatomical nodes"
canonical_examples: []
---
```

Body:
- **Decision rule:** Real medical conditions, congenital anomalies, surgical outcomes,
  and prosthetics that produce hand/limb/face appearances similar to generative
  artifacts. These are `not_gen_ai`.
- **Specific exceptions:**
  1. Polydactyly (extra digits), syndactyly (fused digits).
  2. Limb differences, amputations, prosthetic limbs.
  3. Post-surgical facial reconstruction.
  4. Dermatological conditions affecting skin texture (mimics plastic-skin node).
- **Boundary with positive nodes:**
  - [[GA.visual_artifacts.anatomy.hands]] — real polydactyly vs generated extra fingers.
  - [[GA.surface_texture.plastic_skin]] — post-procedure skin vs synthetic texture.
  - When in doubt, route to SME with note: "possible medical condition."

**Acceptance criteria:**
- [ ] File exists with `node_type: exception`, `parent: GA.visual_artifacts.anatomy.hands`
- [ ] References polydactyly, syndactyly, prosthetics explicitly
- [ ] Boundary section links to hand and skin nodes via Obsidian wikilinks

#### Task 3.4 — Add repeated-detail artifact node

**File:** `policy-graph/Generative_AI/v0.1/GA.visual_artifacts.repeated_details.md`

Frontmatter:
```yaml
---
id: GA.visual_artifacts.repeated_details
version: Generative_AI.v0.1
title: Repeated or cloned detail artifacts
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 30
  hard_positive: 20
  easy_negative: 10
  hard_negative: 15
  platinum_min: 5
source_anchors:
  - "v0.1 data/seed/policy-suggestions.json patch.seed.002"
  - "v0.2 §13: repeated-detail artifact expected subcategory"
canonical_examples: []
---
```

Body:
- **Decision rule:** Diffusion models sometimes repeat fine details (teeth, jewelry,
  buttons, texture patches, fabric patterns) in unnatural ways. Visible repetition
  that violates physical plausibility is evidence of generation.
- **Positive criteria (any one suffices):**
  1. Identical or near-identical small objects repeated in a pattern that real scenes
     don't produce (e.g., three identical earrings, cloned teeth).
  2. Texture tiling visible in organic surfaces (skin, fabric, foliage).
  3. Symmetric detail duplication across an axis that real anatomy doesn't share.
- **Hard negatives:**
  - Actual patterned objects (tiled floors, wallpaper, jewelry designs).
  - Clone-stamp or content-aware-fill edits → route to [[GA.boundary.photo_editing]].
  - Compression-induced ringing near repeated edges → see [[GA.exception.compression_artifacts]].

**Acceptance criteria:**
- [ ] File exists with `node_type: category`, `polarity: positive`, `parent: GA.root`
- [ ] Decision rule covers teeth, jewelry, texture tiling, symmetric duplication
- [ ] Hard negatives reference photo_editing and compression_artifacts nodes

#### Task 3.5 — Strengthen low-quality uncertain node

**File:** `policy-graph/Generative_AI/v0.1/GA.boundary.low_quality_uncertain.md`

Edit the existing file's body to add/strengthen:

- **Default action:** `abstain` — route to SME review queue.
- Add explicit statement: "The default label for images that land here is `abstain`.
  LLM labelers and human reviewers MUST NOT guess; they must output `abstain` and
  the image is routed to the SME disagreement queue."
- Add **Routing criteria** section:
  1. Resolution below 200×200 effective pixels.
  2. More than 60% of the image occluded, cropped, or watermarked.
  3. Combined labeler confidence below 0.5 across all three models.
  4. Image is a screenshot-of-a-screenshot or heavily re-encoded.
- Add **SME review expectations:** SME may reclassify as gen_ai/not_gen_ai with
  rationale, or confirm abstain (excluded from metrics denominators).

**Acceptance criteria:**
- [ ] Body contains explicit "default label is abstain" statement
- [ ] Routing criteria section with resolution, occlusion, confidence thresholds
- [ ] SME review expectations section
- [ ] Frontmatter unchanged (already correct for boundary node)

#### Task 3.6 — Update edges.json with new edges

Add these edges to `policy-graph/Generative_AI/v0.1/edges.json`:

```json
[
  {"source_node_id": "GA.negative.authentic_photo", "target_node_id": "GA.root", "edge_type": "subtype_of", "confidence": 1.0, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.exception.compression_artifacts", "target_node_id": "GA.root", "edge_type": "subtype_of", "confidence": 1.0, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.exception.compression_artifacts", "target_node_id": "GA.surface_texture.plastic_skin", "edge_type": "confused_with", "confidence": 0.8, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.exception.compression_artifacts", "target_node_id": "GA.boundary.low_quality_uncertain", "edge_type": "boundary_with", "confidence": 0.7, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.exception.medical_prosthetic", "target_node_id": "GA.visual_artifacts.anatomy.hands", "edge_type": "exception_to", "confidence": 1.0, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.exception.medical_prosthetic", "target_node_id": "GA.surface_texture.plastic_skin", "edge_type": "confused_with", "confidence": 0.6, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.visual_artifacts.repeated_details", "target_node_id": "GA.root", "edge_type": "subtype_of", "confidence": 1.0, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"},
  {"source_node_id": "GA.visual_artifacts.repeated_details", "target_node_id": "GA.boundary.photo_editing", "edge_type": "confused_with", "confidence": 0.7, "provenance": "hardening_v1", "version": "Generative_AI.v0.1"}
]
```

**Acceptance criteria:**
- [ ] 8 new edges appended to existing 12 (total: 20)
- [ ] All source/target node IDs reference nodes that exist
- [ ] All edge_types are in the allowed enum
- [ ] Validator passes with updated edges.json

---

## Dependency graph

```
X1 (schemas/validator/data)  ─────────────────────┐
                                                   │
X3 (policy-graph .md + edges) ───┐                 ├──► Theo: final integration review
                                 │                 │         + run validator
X2 Tasks 2.1–2.4 (UI core) ─────┤                 │
                                 │                 │
X2 Task 2.5 (graph viz) ────────┘ (after X3) ─────┘
```

X1, X2 (tasks 2.1–2.4), and X3 run **in parallel**.
X2 task 2.5 runs **after X3 completes** (needs new node IDs and edge data).
Final integration review after all three complete.

---

## File ownership (no merge conflicts)

| Engineer | Files owned (exclusive write) |
|---|---|
| **X1** | `schemas/*.schema.json` (all 11), `scripts/validate_foundation.py`, `data/seed/metrics.json`, `policy-graph/.../GA.root.md` (frontmatter fix only) |
| **X2** | `web/index.html`, `web/app.js`, `web/styles.css` |
| **X3** | `policy-graph/.../GA.negative.*.md`, `policy-graph/.../GA.exception.*.md`, `policy-graph/.../GA.visual_artifacts.repeated_details.md`, `policy-graph/.../GA.boundary.low_quality_uncertain.md` (body only), `policy-graph/.../edges.json` |

---

## Acceptance: branch is merge-ready when

1. `python3 scripts/validate_foundation.py` passes with 13 nodes, 20 edges
2. No perfect (1.0/0.0) decision-quality metrics displayed in web UI
3. All 11 JSON schemas parse and cover v0.1/v0.2 data contracts
4. Web UI loads data from `data/seed/*.json`, not hardcoded arrays
5. Graph visualization shows all 13 nodes with edges
6. Hero intro emphasizes high-reasoning models, voting/consensus, Obsidian graph
7. GA.root.md has `node_type: root`
8. Validator catches: orphan nodes, duplicate roots, bad edge types, split leakage
9. `data/seed/metrics.json` has null decision-quality metrics with denominator context

---

---

## Scope Addition: Warm-Start Label Hierarchy + Decision Quality Tab

Attila directive (2026-05-09 09:23 PDT). These are **pre-merge deliverables**.

### Context

RUSH warm-starts from an existing golden set with a fixed label hierarchy:

- **L0** — `violative` / `non_violative` (binary policy decision)
- **L1** — `ignore` / `hide` / `deactivate` (enforcement action, downstream of L0)
- **L2** — Subcategories from the policy graph (e.g., `GA.visual_artifacts.anatomy.hands`)

Labels are **fixed** for warm-start. L2 only changes when an SME approves policy changes.
Labels are tied to the policy document (.pdf or .md) fed into the LLM prompt with the image.

### LLM Structured Output Spec

Every LLM labeler must return:

```json
{
  "label": "violative",
  "l2_label": "GA.visual_artifacts.anatomy.hands",
  "justification": "Six fingers visible on left hand, violating anatomical plausibility per policy §4.1",
  "confidence": 0.82,
  "difficulty": "high",
  "is_boundary": true
}
```

- `label` — L0 binary (violative/non_violative/abstain)
- `l2_label` — L2 policy-graph node ID
- `justification` — grounded in policy text, not vibes
- `confidence` — 0–1
- `difficulty` — high/medium/low (labeler's self-assessment)
- `is_boundary` — true if the image is a hard positive or hard negative

`is_boundary` and `difficulty` are the most important signals for policy ambiguity reduction.

### Consensus-Based Ambiguity Model

- Full consensus (3/3) → easy case, clear policy.
- Lack of consensus → one or more of:
  - Policy needs clarification (missing rule)
  - Non-experts need guidance (training gap)
  - Expert was wrong (SME error)
- Majority-vote disagreements flag missing or ambiguous policy regions.

### SME Re-Review Sampling

At each iteration, flag a sample of golden-set images most beneficial for SME re-review.
Selection criteria: high `is_boundary` rate, high `difficulty`, consensus splits, recent
policy changes affecting that node. If SME overturns a label → recompute ALL decision-
quality metrics for that policy version.

### Decision Quality Tab (NEW — Web UI)

A new 7th tab: **"Decision Quality"**.

**Rows** (one per labeler):
| Labeler | Type |
|---|---|
| GPT-5.4 | LLM |
| GPT-5.5 | LLM |
| GPT-5.5-high | LLM (high-reasoning) |
| Gemini-3.1-pro | LLM |
| Majority Vote | Ensemble |
| Non-expert | Human |

**Columns:** accuracy, F1, precision, recall, FPR, FNR, positive_proportion, N, informedness.

**Ground truth:** SME latest label (gold/platinum tier).

**Evolution chart:** Line chart at the bottom showing key metrics (F1, informedness, FPR)
across policy graph versions. X-axis = policy version, Y-axis = metric value. One line per
labeler.

**Data source:** `data/seed/decision-quality.json` (new file, placeholder structure matching
the table above with null metric values until real labels exist).

### Next Phase (post-merge, out of scope)

Image gathering + LLM labeling via API calls. Foundation must support this by having
correct schemas, prompt templates, and metric infrastructure.

---

### Amended Workstream Assignments

#### X1 Additional Tasks (steered to running session)

##### Task 1.6 — Add label-hierarchy schema
**File:** `schemas/label-hierarchy.schema.json`

Captures the L0/L1/L2 hierarchy:
```json
{
  "l0_labels": ["violative", "non_violative", "abstain"],
  "l1_labels": ["ignore", "hide", "deactivate"],
  "l2_source": "policy_graph_node_ids",
  "hierarchy_version": "string",
  "policy_document_ref": "string",
  "frozen": true
}
```
Required: l0_labels, l1_labels, l2_source, hierarchy_version.

##### Task 1.7 — Add LLM structured-output schema
**File:** `schemas/llm-output.schema.json`

```json
{
  "label": {"enum": ["violative", "non_violative", "abstain"]},
  "l2_label": {"type": "string"},
  "justification": {"type": "string", "minLength": 10},
  "confidence": {"type": "number", "minimum": 0, "maximum": 1},
  "difficulty": {"enum": ["high", "medium", "low"]},
  "is_boundary": {"type": "boolean"}
}
```
All 6 fields required.

##### Task 1.8 — Add decision-quality snapshot schema
**File:** `schemas/decision-quality.schema.json`

Per-labeler metrics table:
```json
{
  "policy_graph_version": "string",
  "ground_truth_tier": ["gold", "platinum"],
  "labelers": [
    {
      "labeler_id": "string",
      "labeler_type": "llm|ensemble|human",
      "metrics": {
        "accuracy": "number|null",
        "f1": "number|null",
        "precision": "number|null",
        "recall": "number|null",
        "fpr": "number|null",
        "fnr": "number|null",
        "positive_proportion": "number|null",
        "n": "integer",
        "informedness": "number|null"
      }
    }
  ],
  "evolution": [{"version": "string", "labeler_id": "string", "metrics": {}}]
}
```
Required: policy_graph_version, ground_truth_tier, labelers.

##### Task 1.9 — Add SME re-review sampling schema
**File:** `schemas/sme-rereview-sample.schema.json`

```json
{
  "sample_id": "string",
  "policy_graph_version": "string",
  "selection_criteria": ["high_boundary_rate", "high_difficulty", "consensus_split", "recent_policy_change"],
  "image_ids": ["string"],
  "expected_yield": "number",
  "status": "pending|in_progress|completed",
  "overturn_count": "integer",
  "metrics_recomputed": "boolean"
}
```

##### Task 1.10 — Create seed decision-quality data
**File:** `data/seed/decision-quality.json`

Placeholder with correct structure, null metrics, 6 labeler rows (GPT-5.4, GPT-5.5,
GPT-5.5-high, Gemini-3.1-pro, majority_vote, non_expert). Empty evolution array.

##### Task 1.11 — Update label-vote schema for L0/L2/boundary/difficulty
Edit `schemas/label-vote.schema.json`:
- Add `l0_label` enum: violative/non_violative/abstain
- Add `l2_label` string (policy graph node ID)
- Add `is_boundary` boolean
- Add `difficulty` enum: high/medium/low
- Keep existing `label` field for backward compat

#### X2 Additional Tasks

##### Task 2.6 — Build Decision Quality tab
**Files:** `web/index.html`, `web/app.js`, `web/styles.css` (if needed)

- Add 7th tab button: `<button class="tab" data-tab="dq" role="tab">Decision Quality</button>`
- Add `<section id="dq">` panel with:
  - Header: "Decision Quality" with eyebrow "Per-labeler accuracy vs SME ground truth"
  - Table with 6 labeler rows × 9 metric columns (see spec above)
  - Load data from `../data/seed/decision-quality.json`
  - Show null/placeholder metrics as "—" with dimmed styling
  - Add denominator (N) column
  - Evolution chart placeholder: `<div id="dqChart">` with message
    "Evolution chart renders after ≥2 policy versions have metrics."
  - Same warning banner style as metrics tab for placeholder data

#### X3 Additional Tasks

##### Task 3.7 — Create LLM prompt template document
**File:** `docs/llm-prompt-template.md`

Document the structured output contract for LLM labelers:
- Input: image + policy document (.pdf/.md) + graph context pack
- Output: the 6-field JSON spec (label, l2_label, justification, confidence, difficulty, is_boundary)
- Explain each field with examples
- Explain consensus model (full agreement = easy, split = ambiguity signal)
- Include example prompt skeleton
- Reference `schemas/llm-output.schema.json` for formal contract

##### Task 3.8 — Document warm-start label hierarchy
**File:** `docs/label-hierarchy.md`

- Explain L0/L1/L2 hierarchy with examples
- State that labels are FIXED for warm-start
- L2 changes only via SME-approved policy changes
- Reference `schemas/label-hierarchy.schema.json`
- Diagram: L0 → L1 → L2 flow

---

## Post-merge roadmap items (out of scope for this pass)

- Generate `graph-compiled.json` from .md files so UI doesn't need hardcoded graph data
- Add jsonschema validation (requires pip dependency) vs current stdlib-only checks
- Wire real golden-set images for Round 0 labeling
- Implement LLM ensemble labeling pipeline
- Add consensus audit sampling logic
- Build SME review queue UI
- Implement auto-research loop with split protection
