# Per-Project Labeling Ontology (design)

Author: Theo (CTO/architect). Status: for Attila review. Branch: `feat/mnist-ux-kdd`.

## Problem

The labeling prompt (`pipeline/providers/_prompts.py`) and response schema were
hardcoded to the **GenAI** ontology: `label ∈ {gen_ai, not_gen_ai, violative,
non_violative, abstain}`, `l2_label = policy node id`, `is_boundary: bool`.

MNIST reuses this GenAI-shaped prompt, which is wrong. Today the MNIST digit is
smuggled through `l2_label` (`MD.digit.N`) while `label` stays a GenAI enum, so
accuracy cannot be measured on the digit at L1. The **ontology differs per
project (demo area)** and must be declared explicitly, once, per area.

## Design: ontology = single source of truth per demo area

New module `pipeline/providers/ontology.py`. One `Ontology` per demo area
(`Generative_AI`, `MNIST_Digits`), keyed by the existing `demo_area` concept
(`pipeline/web/demo_area.py`). The ontology is the single source that drives the
SYSTEM prompt, the USER instructions, and the RESPONSE FORMAT/schema — so the
three providers (OpenAI/Anthropic/Gemini) stay strictly comparable *within* a
project.

### Data model

```python
@dataclass(frozen=True)
class Ontology:
    area: str                       # "Generative_AI" | "MNIST_Digits"
    l1_classes: tuple[str, ...]     # classes ACCURACY is measured on (L1)
    label_enum: tuple[str, ...]     # l1_classes + ("abstain",) [+ warm-start for GenAI]
    abstain_label: str              # "abstain"
    scoring_task: str               # pipeline.scoring.tasks name ("genai_binary"/"mnist_multiclass")
    l2_semantics: str               # what l2_label means for this area (policy node id)
    boundary_semantics: str         # what is_boundary means for this area
    require_boundary_between: bool   # MNIST: True; GenAI: False (supported, not required)
    response_keys: tuple[str, ...]
    response_schema: dict           # provider-facing JSON schema (constrained output)
    system_prompt: str              # SYSTEM prompt fragment (domain-specific)
    user_instructions: str          # USER instruction fragment (domain-specific)
```

### Selection

`get_ontology(area)` resolves an area string (validated by
`demo_area.normalize_policy_area`) to its `Ontology`. Callers that only have a
`demo`/`policy_graph_dir` normalize first, then look up.

- **Prompt + schema selection** — every provider client reads
  `request.area`, calls `get_ontology(area)`, and assembles messages from
  `ontology.system_prompt`, `ontology.user_instructions`, and
  `ontology.response_schema` (OpenAI json_object/json_schema, Anthropic system,
  Gemini `response_schema`). No provider hardcodes GenAI copy anymore.
- **Scoring selection** — `ontology.scoring_task` maps to the existing
  `pipeline.scoring.tasks` registry (`genai_binary` binary / `mnist_multiclass`
  multiclass). Accuracy is measured on **L1** (`label`).

### How it flows

```
demo/area ──normalize_policy_area──▶ area
   │
   ├─▶ get_ontology(area) ──▶ Ontology
   │        │
   │        ├─ system_prompt / user_instructions / response_schema ─▶ clients
   │        └─ scoring_task ─▶ tasks.get_task ─▶ scorer (accuracy on L1 label)
   │
LabelRequest.area (new field) carries area into client.label()
run_labeling(policy_area=...) derives area from policy_graph_dir when unset
```

Backward compatibility: `LabelRequest.area` defaults to `Generative_AI`;
`get_ontology()` default is GenAI; `coerce_label_fields(parsed, ontology=None)`
defaults to GenAI behavior. Existing GenAI callers/tests are unchanged.

## GenAI ontology (baseline — preserve behavior exactly)

- L1 = `{gen_ai, not_gen_ai}`; label_enum also keeps warm-start
  `{violative, non_violative}` + `abstain` (byte-identical to today's enum).
- l2_label = policy subcategory node id (e.g. `GA.visual_artifacts.anatomy.hands`).
- is_boundary = bool at the L2 level.
- `is_boundary_between` = OPTIONAL at L2 (parity), NOT required.
- System/user copy = the current GenAI strings, verbatim.
- Schema = the current `LABELING_RESPONSE_SCHEMA`, verbatim (plus optional
  `is_boundary_between`).

## MNIST ontology (new)

- L1 = multiclass digits `["0".."9"]`. **The DIGIT is `label`** (accuracy
  measured here), NOT smuggled through l2_label.
- l2_label = the policy node id applied (`MD.digit.N`).
- Response schema `label` enum = the 10 digits + `"abstain"`.
- **`is_boundary_between`** (new conditional field): when `is_boundary=true`,
  REQUIRE exactly TWO L1 class ids (e.g. `["1","7"]`); when `is_boundary=false`
  it is empty/absent. Validated in `coerce_label_fields`.
- Prompt copy talks about DIGIT stroke/topology and confusion pairs (1↔7,
  4↔9, 3↔5/8, 5↔6, etc.), NOT generative imagery/anatomy/hands.

### MNIST response schema (final field list)

| field | type | rule |
|---|---|---|
| `label` | enum | one of `0..9` or `abstain` (**L1, accuracy measured here**) |
| `l2_label` | string | policy node id `MD.digit.N`; `""` only if abstain |
| `justification` | string | ≥10 chars, ≤~1500 chars (soft cap) |
| `policy_citations` | string[] | node ids invoked; `[]` only if abstain |
| `policy_quotes` | string[] | ≤6 verbatim snippets from policy markdown |
| `confidence` | number | `[0,1]` |
| `difficulty` | enum | high/medium/low |
| `is_boundary` | bool | on a documented confusion boundary |
| `is_boundary_between` | string[] | **required exactly-2 L1 ids when `is_boundary=true`; empty/absent when false** |

## Scoring / decision quality

- Accuracy measured on **L1**: MNIST digit `label`; GenAI `gen_ai/not_gen_ai`.
- Multiclass scorer already reads `label` from votes. We add a **safe
  fallback/migration**: when a vote's `label` is not in the class set but its
  `l2_label` matches `MD.digit.<N>`, derive the digit (covers legacy runs that
  smuggled the digit into l2_label). New runs read L1 directly.
- Boundary status (`is_boundary` + the `is_boundary_between` pair) is SECONDARY
  metadata feeding boundary analysis + residual-misalignment views, never the
  accuracy numerator.

## Boundary edge→node promotion (scaffold now, build later)

Today confusion pairs are `confused_with` EDGES (see MNIST node frontmatter +
`edges.json`). Attila wants boundaries to become a first-class NODE TYPE as the
system iterates, analogous to GenAI's `is_boundary` but as PAIRS.

Now (this PR):
- (a) `boundary` is already in `schemas/policy-node.schema.json` `node_type`
  enum — keep it; also accept the MNIST `digit_class` node_type in a per-area
  validation note (existing MNIST nodes use `digit_class`).
- (b) Define the `MD.boundary.<a>x<b>` node shape: `node_type: boundary`,
  `parent: MD.root`, edges `boundary_of → MD.digit.<a>` and `→ MD.digit.<b>`,
  holding the boundary region's example images in `canonical_examples`.
- (c) Document + minimally scaffold the edge→node PROMOTION path:
  `confused_with(a,b)` ⇒ materialize `MD.boundary.<min>x<max>` with two
  `boundary_of` edges. A helper (`pipeline/policy_graph/boundary_promotion.py`,
  scaffold) enumerates candidate promotions from existing `confused_with` edges;
  full materialization + example attachment + UI is **iterate-time**, not now.

### Promotion design (for Attila review)

1. Scan edges for `confused_with` pairs, dedupe unordered `(a,b)`.
2. For each pair, propose node id `MD.boundary.{a}x{b}` (sorted).
3. On promotion: create node frontmatter (`node_type: boundary`), add
   `boundary_of` edges to both digit nodes, keep the original `confused_with`
   edges (edges stay as relationship metadata; the node adds a home for boundary
   examples + a target for `is_boundary_between`).
4. Attach boundary example images later (labeler `is_boundary_between=["a","b"]`
   rows become candidates for `MD.boundary.{a}x{b}.canonical_examples`).

Open question for Attila: should promoted boundary nodes REPLACE `confused_with`
edges or COEXIST? Recommendation: **coexist** — edges are cheap relationship
metadata; nodes carry examples/coverage.

## UX

Surface `is_boundary_between` where boundary cases appear (§4 audit / residual
misalignment lanes): show the pair as a chip (e.g. `1 ↔ 7`). Demo selector +
panel keep working for both projects. Don't overbuild.

## Test plan

- per-area ontology selection (`get_ontology` returns correct enum/schema/copy)
- MNIST digit label enum (`0..9` + abstain)
- `is_boundary_between` required-when-boundary validation (coerce path)
- scorer reads L1 digit `label` (+ legacy l2_label fallback)
- boundary `node_type` schema acceptance
- GenAI regression: existing ~275 tests stay green.
