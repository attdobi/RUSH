---
id: GA.scene_geometry.inconsistent_perspective
version: Generative_AI.v0.1
title: Inconsistent scene geometry, reflections, and shadows
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.2
coverage_target:
  easy_positive: 20
  hard_positive: 30
  easy_negative: 20
  hard_negative: 25
  platinum_min: 6
source_anchors:
  - RUSH user brief 2026-05-09: boundary conditions and policy evolution
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.cgi_game_render}
canonical_examples: []
---
# Inconsistent scene geometry, reflections, and shadows

## Positive criteria
Use this node when multiple spatial cues are mutually incompatible:

- Reflections show objects or people that do not match the scene.
- Shadows imply conflicting light sources without plausible explanation.
- Objects merge, intersect, or change perspective in physically impossible ways.

## Hard negatives
- Fisheye lenses, mirrors, panoramas, artistic staging, or compositing.
- Game/CGI assets where non-realism is expected and not policy-violative by itself.
