---
id: GA.boundary.photo_editing
version: Generative_AI.v0.1
title: Conventional photo editing, filters, and compression
area: Generative_AI
node_type: boundary
parent: GA.root
polarity: hard-negative
status: draft
coverage_weight: 1.3
coverage_target:
  easy_positive: 0
  hard_positive: 0
  easy_negative: 25
  hard_negative: 40
  platinum_min: 8
source_anchors:
  - RUSH v0.1: photo edits and healing brush as confusable node
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.surface_texture.plastic_skin}
  - {type: boundary_with, to: GA.visual_artifacts.anatomy.hands}
canonical_examples: []
---
# Conventional photo editing, filters, and compression

## Negative rule
Do not classify as `gen_ai` solely because an image is edited, filtered, retouched, compressed, upscaled, or beautified.

## Examples to collect
- Beauty filters that mimic plastic skin.
- Healing-brush artifacts that distort hands or backgrounds.
- Compression causing text/edge artifacts.

## SME review triggers
Escalate when edits materially create synthetic objects or when provenance suggests generative fill rather than conventional retouching.
