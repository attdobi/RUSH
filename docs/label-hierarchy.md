# Warm-Start Label Hierarchy

This document defines the fixed label hierarchy used for warm-start labeling and evaluation.

## 1. L0 — Binary policy decision

L0 is the primary classification:

- `violative` — the image violates the bound policy.
- `non_violative` — the image does not violate the bound policy.
- `abstain` — available evidence is insufficient for a grounded decision.

All downstream analysis starts with L0. Reviewers and LLM labelers must abstain rather than guess when evidence is insufficient.

## 2. L1 — Enforcement action

L1 is the downstream enforcement action:

- `ignore` — no enforcement action.
- `hide` — reduce visibility or suppress display.
- `deactivate` — remove, disable, or otherwise deactivate the violating item.

L1 applies only when `L0 = violative`. If L0 is `non_violative` or `abstain`, the default L1 action is `ignore` or no action, depending on product workflow.

## 3. L2 — Policy graph subcategory

L2 provides structural attribution using policy-graph node IDs, for example:

- `GA.visual_artifacts.anatomy.hands`
- `GA.surface_texture.plastic_skin`
- `GA.exception.compression_artifacts`
- `GA.boundary.low_quality_uncertain`
- `GA.negative.authentic_photo`

L2 explains where the image belongs in the policy graph and supports coverage tracking, ambiguity reduction, SME review, and policy diffs.

## 4. Warm-start constraint

Warm-start labels are FIXED. Labelers may select among existing labels and valid L2 nodes, but they may not invent new labels, rename nodes, or move L2 placements ad hoc.

L2 changes only when an SME approves a policy change through the policy-diff workflow. Approved policy diffs create a new graph or label version rather than silently mutating prior labels.

## 5. Policy document binding

Labels are tied to the exact policy document (`.pdf` or `.md`) supplied in the prompt alongside the image. The policy document defines the label version.

Changing the policy document changes the interpretation context and therefore creates a new label version. Historical labels should remain traceable to the policy document and policy graph version used when they were produced.

## 6. Hierarchy diagram

```text
Image + bound policy document
        |
        v
L0: Binary policy decision
    - violative
    - non_violative
    - abstain
        |
        | if L0 = violative
        v
L1: Enforcement action
    - ignore
    - hide
    - deactivate
        |
        v
L2: Policy graph subcategory
    - GA.visual_artifacts.anatomy.hands
    - GA.surface_texture.plastic_skin
    - GA.visual_artifacts.repeated_details
    - GA.exception.compression_artifacts
    - GA.boundary.low_quality_uncertain
```

Example flows:

- Six implausible fingers on a generated-looking hand → `L0 = violative` → `L1 = hide` → `L2 = GA.visual_artifacts.anatomy.hands`.
- Authentic professional photo with conventional retouching → `L0 = non_violative` → `L1 = ignore` → `L2 = GA.negative.authentic_photo` or `GA.boundary.photo_editing`.
- Screenshot-of-screenshot too degraded for a grounded decision → `L0 = abstain` → no enforcement action → `L2 = GA.boundary.low_quality_uncertain`.

## 7. Reference

See `schemas/label-hierarchy.schema.json` for the formal hierarchy contract.
