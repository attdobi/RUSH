---
id: GA.boundary.cgi_game_render
version: Generative_AI.v0.1
title: CGI, game, 3D render, and stylized illustration boundary
area: Generative_AI
node_type: boundary
parent: GA.root
polarity: hard-negative
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 0
  hard_positive: 0
  easy_negative: 25
  hard_negative: 30
  platinum_min: 6
source_anchors:
  - RUSH v0.2 pilot recommendations: Generative_AI hard-negative boundary regions
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.scene_geometry.inconsistent_perspective}
canonical_examples: []
---
# CGI, game, 3D render, and stylized illustration boundary

## Negative rule
Do not classify as `gen_ai` solely because the image is non-photographic, rendered, stylized, or from a game/CGI pipeline.

## Positive crossover
Classify as `gen_ai` only when policy evidence indicates generative model output or material generative manipulation, not merely digital art.
