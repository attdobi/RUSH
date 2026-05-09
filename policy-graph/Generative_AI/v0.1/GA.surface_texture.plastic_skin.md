---
id: GA.surface_texture.plastic_skin
version: Generative_AI.v0.1
title: Synthetic surface and plastic skin texture
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.1
coverage_target:
  easy_positive: 20
  hard_positive: 30
  easy_negative: 20
  hard_negative: 30
  platinum_min: 6
source_anchors:
  - RUSH v0.1 graph example: plastic skin/style texture
edges:
  - {type: subtype_of, to: GA.root}
  - {type: confused_with, to: GA.boundary.photo_editing}
canonical_examples: []
---
# Synthetic surface and plastic skin texture

## Positive criteria
Use this node when surfaces have a synthetic, over-smoothed, waxy, or poreless appearance that is coherent across the image and not explained by filters or lighting.

## Hard negatives
- Beauty filters, airbrushing, cosmetic retouching, and social-media smoothing.
- Studio lighting or makeup producing smooth skin.
- Low-resolution images where texture detail is unavailable.

## Boundary note
This node is weak alone. Prefer pairing with another positive node or provenance cue before assigning high confidence.
