<!-- GA.boundary.cgi_game_render.md -->
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
  - RUSH v0.2 pilot recommendations - Generative_AI hard-negative boundary regions
  - RUSH holdout review 2026-05-11 - stylized generated illustration confused with benign illustration
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.scene_geometry.inconsistent_perspective}
  - {type: boundary_with, to: GA.visual_artifacts.nature_wildlife}
canonical_examples: []
---
# CGI, game, 3D render, and stylized illustration boundary

## Negative rule
Do not classify as `gen_ai` solely because the image is non-photographic, rendered, stylized, painterly, illustrated, or from a game/CGI pipeline.

## Positive crossover
Do not use this boundary as evidence against `gen_ai`. A stylized image can still be generative. Classify as `gen_ai` when policy evidence indicates generative model output or material generative manipulation, including:

1. Explicit synthetic provenance or an AI-generation watermark tied to the image.
2. Painterly or rendered details that are locally plausible but globally incoherent, melted, overblended, or inconsistent across object boundaries.
3. Implausible animal, fish, human, hand, paw, eye, splash, fabric, or object structure within the stylized scene.
4. Repeated or cloned brush-like details that do not match deliberate patterning.
5. Scene geometry, reflection, lighting, or shadow contradictions beyond ordinary artistic stylization.

## Hard negatives
Use this boundary for conventional digital art, game screenshots, 3D renders, paintings, comics, concept art, and illustrations when the only suspicious fact is that they are not photographs.

## Routing guidance
For stylized wildlife, fish, water, landscapes, foliage, or terrain with synthesis-like local detail, consider `[[GA.visual_artifacts.nature_wildlife]]`. For ordinary CGI/game non-realism with no generative evidence, keep this hard-negative boundary.
