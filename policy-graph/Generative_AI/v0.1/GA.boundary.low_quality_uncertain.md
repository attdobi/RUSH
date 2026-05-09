---
id: GA.boundary.low_quality_uncertain
version: Generative_AI.v0.1
title: Low-quality, cropped, or insufficient-evidence cases
area: Generative_AI
node_type: boundary
parent: GA.root
polarity: negative
status: draft
coverage_weight: 1.1
coverage_target:
  easy_positive: 0
  hard_positive: 0
  easy_negative: 25
  hard_negative: 40
  platinum_min: 8
source_anchors:
  - RUSH v0.1 difficulty tiers: hard/abstain cases to SME queue
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.visual_artifacts.anatomy.hands}
  - {type: boundary_with, to: GA.visual_artifacts.text_symbols}
canonical_examples: []
---
# Low-quality, cropped, or insufficient-evidence cases

## Decision rule
If the image is too small, blurry, cropped, compressed, or ambiguous to ground a GenAI decision, classify as `not_gen_ai` or `abstain` according to workflow settings and route high-impact cases to SME review.

## Why this node exists
Cold-start systems love overcalling blurry nonsense. This node exists to make uncertainty explicit instead of pretending every pixel is a confession.
