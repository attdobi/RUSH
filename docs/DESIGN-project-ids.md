# Project IDs Inside Demo Areas

Status: PROPOSAL ONLY, for Attila review. Not implemented.

## Goal

RUSH currently treats each demo area (`Generative_AI`, `MNIST_Digits`, later
`ai_gen`, etc.) as a mostly single-track generator/policy loop. Attila wants
project IDs even inside the main demos so the system can restart from
Generator Prompt `v0` and show independent iteration histories:

```text
project A: v0 -> label -> metrics -> propose -> accept -> v1 -> ...
project B: v0 -> label -> metrics -> propose -> accept -> v1 -> ...
```

Each `(area, project_id)` should be an independent instance of the generator
iteration loop. It owns its generator prompt history, policy graph revisions,
label/run artifacts, metrics, and review trail. This keeps the demo area as
the ontology/task boundary while allowing multiple experiments, customer
setups, or story arcs to coexist inside the same area.

The design stays directory/JSON based. There is no SQL database.

## Current Grounding

The existing scaffold already has most of the pieces, but they are single-track:

- Demo areas are selected through `web/demos.js` and normalized in
  `pipeline/web/demo_area.py`.
- Policy graphs live at `policy-graph/<Area>/<version>/`, for example
  `policy-graph/Generative_AI/v0.3/` and `policy-graph/MNIST_Digits/v0.1/`.
- Runs live at `data/runs/<run_id>/` with `run_manifest.json`,
  `label_votes.jsonl`, `llm_outputs.jsonl`, `scoring/`, and `web/`.
- The §3 run payload currently passes `demo`, `area`, and `policy_version`.
- The §2/§3 version-stepper UI is driven by `policyGraphVersion`,
  `runTriggerPolicyVersion`, `window.RUSH_API.catalog.policyVersions`, and
  `currentPolicyVersion`.
- The earlier ontology proposal in
  `docs/DESIGN-per-project-ontology.md` makes the demo area the ontology source
  of truth. This proposal keeps that idea: ontology is per area, while
  project ID selects an independent run of that area's generator loop.

## ID Scheme

### Project ID Format

Use a human-friendly slug plus a short unique suffix:

```text
<slug>-<suffix>
mnist-baseline-a7f3
hands-hard-negatives-09bc
fashion-boundary-pass-61de
```

Rules:

- `slug`: lowercase ASCII, `a-z0-9-`, normalized from a user title.
- `suffix`: 4 hex chars at minimum; increase to 6 or 8 only on collision.
- Full `project_id`: max 64 chars.
- Generated IDs are immutable. The display title can change in the manifest.
- Collision handling: if `projects/<area>/<project_id>/` exists, remint the
  suffix. Do not overwrite an existing project by default.

### Reserved Default Project

Reserve one stable project for migration/back-compat:

```text
project_id = "default"
```

Alternative user-facing label: `Main`. Internally use `default` so historical
single-track URLs and run manifests can be mapped without a generated suffix.

### Composite Key

The logical key for every generator-iteration artifact is:

```text
area + project_id + generator_version
```

Example:

```json
{
  "area": "MNIST_Digits",
  "project_id": "mnist-baseline-a7f3",
  "generator_version": "v2",
  "key": "MNIST_Digits/mnist-baseline-a7f3/v2"
}
```

`generator_version` is the user-facing loop version: `v0`, `v1`, `v2`, ...
It can map to existing policy-graph directory names during migration
(`v0.1`, `v0.2`, etc.), but the project UI should present the clean generator
sequence.

## Proposed Directory Layout

Introduce a project tree that owns the iteration history:

```text
projects/
  <area>/
    <project_id>/
      manifest.json
      v0/
        manifest.json
        generator_prompt.md
        policy-graph/
          *.md
          edges.json
        labels/
          run_refs.json
        metrics/
          decision_quality.json
          summary.json
      v1/
        manifest.json
        generator_prompt.md
        policy-graph/
          *.md
          edges.json
        labels/
          run_refs.json
        metrics/
          decision_quality.json
          summary.json
```

Concrete example:

```text
projects/
  MNIST_Digits/
    mnist-baseline-a7f3/
      manifest.json
      v0/
        manifest.json
        generator_prompt.md
        policy-graph/
          MD.root.md
          MD.digit.0.md
          ...
          edges.json
        labels/
          run_refs.json
        metrics/
          summary.json
      v1/
        manifest.json
        generator_prompt.md
        policy-graph/
          MD.root.md
          MD.digit.0.md
          ...
          MD.boundary.4x9.md
          edges.json
        labels/
          run_refs.json
        metrics/
          decision_quality_multiclass.json
          summary.json
```

### Project Manifest

`projects/<area>/<project_id>/manifest.json` is the project-level index:

```json
{
  "schema_version": 1,
  "area": "MNIST_Digits",
  "project_id": "mnist-baseline-a7f3",
  "title": "MNIST baseline",
  "status": "active",
  "created_at": "2026-07-04T00:00:00Z",
  "created_from": {
    "kind": "seed",
    "source_policy_graph_version": "MNIST_Digits.v0.1"
  },
  "default_generator_version": "v0",
  "current_generator_version": "v2",
  "versions": ["v0", "v1", "v2"]
}
```

### Iteration Manifest

`projects/<area>/<project_id>/<generator_version>/manifest.json` records one
step in the generator loop:

```json
{
  "schema_version": 1,
  "area": "MNIST_Digits",
  "project_id": "mnist-baseline-a7f3",
  "generator_version": "v1",
  "previous_generator_version": "v0",
  "policy_graph_ref": {
    "kind": "project_local",
    "path": "projects/MNIST_Digits/mnist-baseline-a7f3/v1/policy-graph",
    "legacy_policy_graph_version": "MNIST_Digits.v0.2"
  },
  "generator_prompt_path": "generator_prompt.md",
  "run_ids": ["20260704T180000-abc12345"],
  "metrics": {
    "latest_summary_path": "metrics/summary.json",
    "latest_decision_quality_path": "metrics/decision_quality_multiclass.json"
  },
  "proposal": {
    "accepted_from_run_id": "20260704T170000-def67890",
    "proposal_id": "20260704T173000-7890abcd"
  }
}
```

## Relationship to Existing Layouts

### `data/runs/`

Keep `data/runs/<run_id>/` as the canonical high-volume run artifact location.
Do not duplicate `label_votes.jsonl` or provider outputs into `projects/`.

Add project coordinates to new run manifests:

```json
{
  "run_id": "20260704T180000-abc12345",
  "area": "MNIST_Digits",
  "project_id": "mnist-baseline-a7f3",
  "generator_version": "v1",
  "policy_graph_version": "MNIST_Digits/mnist-baseline-a7f3/v1",
  "prompt_version": "v1"
}
```

Then `projects/<area>/<project_id>/<version>/labels/run_refs.json` is a small
index:

```json
{
  "run_ids": ["20260704T180000-abc12345"],
  "latest_run_id": "20260704T180000-abc12345"
}
```

This preserves the existing ignored `data/runs/` behavior and lets §4 read the
same scoring/export files it already knows how to read.

### `policy-graph/<Area>/`

There are two reasonable storage modes:

1. Project-local policy graphs under `projects/<area>/<project_id>/<version>/policy-graph/`.
2. Back-compat mirrors under `policy-graph/<Area>/<project_id>/<version>/`.

Recommendation for review: make `projects/` the source of truth for project
iterations and keep `policy-graph/<Area>/<version>/` as the legacy/default
lookup path. A thin resolver can accept either:

```text
legacy:  policy-graph/Generative_AI/v0.3
project: projects/Generative_AI/hands-hard-negatives-09bc/v2/policy-graph
```

If Obsidian compatibility is critical for every project, expose or mirror the
project graph into a predictable `policy-graph/<Area>/<project_id>/<version>/`
path later. That can be a follow-up; the proposal does not require it for the
first pass.

### Ontology-by-Project Directory

`docs/DESIGN-per-project-ontology.md` says the ontology differs per project
demo area and must be declared explicitly once per area. This proposal refines
the naming:

- `area` remains the ontology key (`Generative_AI`, `MNIST_Digits`).
- `project_id` is the independent experiment or customer/story instance inside
  that ontology.
- `generator_version` is the iteration step inside the project.

So "per-project ontology" should be read as "per demo area ontology, with many
project instances under that area."

## Tie-In to the §3 Version-Stepper

The current UI has one version-stepper and one current policy version. Project
IDs add one selector ahead of that stepper:

```text
Demo area: MNIST_Digits
Project:   mnist-baseline-a7f3
Version:   v0 | v1 | v2
```

Flow:

1. User selects a demo area.
2. UI loads `/api/projects?area=MNIST_Digits`.
3. User selects `project_id`.
4. UI loads `/api/projects/MNIST_Digits/mnist-baseline-a7f3/versions`.
5. Existing `policyGraphVersion` and `runTriggerPolicyVersion` options become
   the generator versions for that project.
6. §3 run payload includes:

```json
{
  "demo": "mnist",
  "area": "MNIST_Digits",
  "project_id": "mnist-baseline-a7f3",
  "generator_version": "v1",
  "policy_version": "MNIST_Digits/mnist-baseline-a7f3/v1"
}
```

The backend resolver turns that composite policy version into a concrete graph
directory and passes the same policy markdown to provider clients. Scoring and
§4 filter runs by `(area, project_id, generator_version)` instead of only
`policy_graph_version`.

`policyGraphNextVersion` can keep its current semantics:

- If `v2` exists, "Next version" moves to `v2`.
- If a proposal is pending for `v2`, show "defined-not-executed" or "pending
  review" against the project timeline.
- If the user chooses "Restart from v0", create a new project (recommended) or
  explicitly reset the selected project, depending on Attila's decision.

## Migration Plan

Migration should be additive and non-breaking:

1. Create `projects/Generative_AI/default/manifest.json`.
2. Map existing `policy-graph/Generative_AI/v0.1`, `v0.2`, `v0.3` into
   generator versions. Proposed display mapping:
   - `policy-graph/Generative_AI/v0.1` -> `generator_version: "v0"`
   - `policy-graph/Generative_AI/v0.2` -> `generator_version: "v1"`
   - `policy-graph/Generative_AI/v0.3` -> `generator_version: "v2"`
3. Create `projects/MNIST_Digits/default/manifest.json`.
4. Map `policy-graph/MNIST_Digits/v0.1` -> `generator_version: "v0"`.
5. Leave existing `policy-graph/<Area>/<version>/` directories in place.
6. Leave existing `data/runs/<run_id>/` directories in place.
7. For historical runs that lack `project_id`, treat them as:

```json
{
  "project_id": "default",
  "generator_version": null,
  "migration_inferred": true
}
```

When a historical `run_manifest.json.policy_graph_version` clearly maps to a
known migrated graph version, the UI can display it under the default project.
Otherwise show it in the default project's "legacy runs" list.

## Minimal UI

Keep the first pass small:

- Project selector in the header or §2 controls, scoped to the active demo area.
- "New project from v0" action that mints a project ID and seeds generator
  version `v0` from the area's default prompt/task brief.
- "Restart from v0" action. Recommended behavior: create a new project, not
  destructive reset.
- Iteration timeline in §2/§3:

```text
v0  labels 20  acc 0.72  proposals 1 accepted
v1  labels 40  acc 0.81  proposals 1 accepted
v2  labels 20  acc 0.84  current
```

Each timeline item links to:

- the policy graph for that generator version;
- run IDs and labels produced with that version;
- metrics/scoring artifacts for that version;
- proposal(s) that created the next version.

No new dashboard is required. The existing §3 run panel and §4 audit can read
the selected project context.

## Open Questions for Attila

- Should the reserved project be called `default`, `main`, or something else in
  URLs and manifests?
- Should "Restart from v0" always fork a new project, or should there also be a
  destructive reset mode for the selected project?
- Should project IDs be user-supplied, generated from a title, or both?
- Should the suffix be 4 hex chars for readability, or 8 chars to match
  `run_id` uniqueness style?
- Should old projects be kept forever, archived under `_archive/`, or pruned by
  retention rules?
- Should cross-project comparison be part of the first UI pass, or deferred
  until project-local loops work?
- Should `ai_gen`, `mnist`, and future demos all share the same project ID
  rules, or can an area define stricter naming?
- Should `projects/` be gitignored by default, tracked by default, or split so
  source-like project manifests/graphs are tracked while labels/runs/metrics
  stay ignored?
- Should project-local policy graphs be mirrored into `policy-graph/<Area>/`
  for Obsidian and existing API compatibility, or should the resolver learn the
  `projects/` source path directly?
- How should historical GenAI versions `v0.1`, `v0.2`, `v0.3` be displayed in
  the new `v0`, `v1`, `v2` generator-version timeline?
- Should proposal IDs include project coordinates, or is manifest metadata
  enough?
- Are projects scoped only within an area, or should a future "workspace
  project" coordinate multiple areas under one shared human-readable project?

## Recommendation

Use `area/project_id/generator_version` as the project loop key, create a
reserved `default` project for migration, keep run artifacts in `data/runs/`,
and make `projects/<area>/<project_id>/<version>/policy-graph/` the future
source of truth for per-project generator versions. The first UI change should
be only a project selector, a "new project from v0" action, and a compact
iteration timeline that reuses the current version-stepper/run/audit flow.
