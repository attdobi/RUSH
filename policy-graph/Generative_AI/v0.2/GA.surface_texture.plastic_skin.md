---
id: GA.surface_texture.plastic_skin
version: Generative_AI.v0.1
title: Synthetic surface, fur, and plastic-skin texture
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
  - RUSH v0.1 graph example - plastic skin/style texture
  - RUSH holdout review 2026-05-11 - animal fur, foliage, water, and terrain texture misses
edges:
  - {type: subtype_of, to: GA.root}
  - {type: confused_with, to: GA.boundary.photo_editing}
  - {type: related_to, to: GA.visual_artifacts.nature_wildlife}
canonical_examples: []
---
# Synthetic surface, fur, and plastic-skin texture

## Positive criteria
Use this node when surfaces have a synthetic, over-smoothed, waxy, poreless, plastic, or procedurally uniform appearance that is coherent across the image and not explained by filters, lighting, camera processing, compression, or deliberate CGI style.

This node can apply to skin and to non-human or non-skin surfaces, including fur, feathers, leaves, rock, water, terrain, fabric, and backgrounds.

Common positive cues include:

1. Human skin that is poreless, waxy, or uniformly sharpened while retaining implausibly perfect wrinkle or blemish patterns.
2. Animal fur or mane texture that forms overly regular strands, fabric-like waves, melted clumps, or uniformly glossy synthetic sheen.
3. Leaves, grass, moss, rocks, waterfalls, foam, or terrain with airbrushed smoothness or repeated micro-texture inconsistent with natural variation.
4. Surface detail that is sharp and highly rendered locally but loses physical structure at boundaries, occlusions, or object transitions.
5. Multiple materials in the same scene sharing the same artificial smoothness, sheen, or procedural texture.

## Hard negatives
- Beauty filters, airbrushing, cosmetic retouching, and social-media smoothing.
- Studio lighting, makeup, wet fur, grooming, or species-specific coat sheen.
- Long-exposure water, shallow depth of field, motion blur, HDR, denoising, and phone-camera computational photography.
- Low-resolution images where texture detail is unavailable.

## Boundary note
This node is weak alone. Prefer pairing with another positive node or provenance cue before assigning high confidence. For wildlife, landscapes, water, foliage, terrain, and animal subjects, consider `[[GA.visual_artifacts.nature_wildlife]]` when the texture issue is part of broader natural-scene synthesis.
