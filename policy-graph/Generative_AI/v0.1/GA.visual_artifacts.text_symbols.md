---
id: GA.visual_artifacts.text_symbols
version: Generative_AI.v0.1
title: Garbled text and symbol artifacts
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.2
coverage_target:
  easy_positive: 25
  hard_positive: 25
  easy_negative: 15
  hard_negative: 25
  platinum_min: 6
source_anchors:
  - RUSH user brief 2026-05-09: GenAI subtype set grows over time
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.low_quality_uncertain}
canonical_examples: []
---
# Garbled text and symbol artifacts

## Positive criteria
Use this node when text-like regions show common generation failures:

- Pseudo-letters that resemble text but do not form language.
- Brand marks or logos that are malformed in ways inconsistent with blur alone.
- Repeated nonsensical typography integrated into the generated scene.

## Hard negatives
- Low-resolution compression making real text unreadable.
- Motion blur, depth-of-field blur, or intentionally fictional signage.
- Non-Latin scripts unfamiliar to the labeler; do not classify as garbled without evidence.
