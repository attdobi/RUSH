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
# Repeated or cloned detail artifacts

## Decision rule
Classify under this positive node when semantically meaningful image details are implausibly repeated, cloned, tiled, or symmetrically duplicated in ways consistent with generative synthesis rather than real-world patterning or conventional editing.

## Positive criteria
Use this node for repeated or cloned details such as:

1. Repeated teeth, fingers, eyelashes, jewelry, buttons, logos, buckles, beads, or other small objects that should vary individually.
2. Texture tiling in skin, fabric, hair, foliage, crowds, gravel, shelves, or backgrounds where patches recur with unnatural spacing or orientation.
3. Symmetric duplication of scene details, accessories, facial features, reflections, or object fragments that violates the expected geometry of the scene.
4. Locally plausible details that become implausible because the same mark, shape, or micro-texture appears several times with inconsistent lighting or perspective.

## Hard negatives
- Actual patterns, uniforms, printed fabric, wallpaper, tiles, beadwork, teeth aligners, product grids, or repeated architecture.
- Clone-stamp, healing-brush, or copy/paste edits from conventional photo manipulation; route these to `[[GA.boundary.photo_editing]]` when editing is the better explanation.
- Compression ringing, mosquito noise, macroblocking, and repeated encoding ghosts; route these to `[[GA.exception.compression_artifacts]]` when artifact repetition follows compression boundaries.

## Boundary guidance
This node is strongest when repeated details co-occur with other positive cues such as impossible anatomy, inconsistent perspective, malformed text, or synthetic provenance. If repetition is plausible real-world design, use a negative or boundary node instead.
