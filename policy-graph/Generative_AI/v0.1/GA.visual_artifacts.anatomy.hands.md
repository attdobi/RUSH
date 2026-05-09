---
id: GA.visual_artifacts.anatomy.hands
version: Generative_AI.v0.1
title: Anatomical hand and limb artifacts
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.4
coverage_target:
  easy_positive: 30
  hard_positive: 30
  easy_negative: 15
  hard_negative: 30
  platinum_min: 8
source_anchors:
  - RUSH v0.1 worked GenAI example: hand/finger artifacts
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.photo_editing}
  - {type: boundary_with, to: GA.boundary.low_quality_uncertain}
canonical_examples: []
---
# Anatomical hand and limb artifacts

## Positive criteria
Use this node when visible anatomy includes strong generative artifacts such as:

1. Extra, missing, fused, webbed, or duplicated digits without plausible real-world explanation.
2. Hands or limbs that violate skeletal articulation or perspective.
3. Blended fingers, palms without coherent thumbs, or inconsistent limb continuity.

## Hard negatives
- Motion blur or occlusion that only appears like extra fingers.
- Real medical conditions, injuries, prosthetics, or gloves.
- Tiny crops where the hand is not clearly visible.

## SME review triggers
Escalate when the hand is cropped, highly stylized, medically atypical, or the artifact could plausibly come from editing/compression.
