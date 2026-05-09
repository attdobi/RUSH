---
id: GA.negative.authentic_photo
version: Generative_AI.v0.1
title: Authentic photograph (real negative)
area: Generative_AI
node_type: category
parent: GA.root
polarity: negative
status: draft
coverage_weight: 1.0
coverage_target:
  easy_negative: 50
  hard_negative: 30
  platinum_min: 10
source_anchors:
  - "v0.2 §2: real-negative root for authentic photographs"
canonical_examples: []
---
# Authentic photograph (real negative)

## Decision rule
Classify as `not_gen_ai` when the image is an authentic, unmodified or conventionally edited photograph with no evidence of generative provenance.

## Positive criteria
Use this real-negative node when the image shows:

1. No visible generative artifacts in anatomy, text, surface texture, geometry, or object continuity.
2. No provenance evidence indicating synthetic generation, generative fill, or AI-assisted composition.
3. Characteristics consistent with camera capture, including plausible optics, lighting, sensor noise, depth of field, and scene continuity.

## Hard negatives
Collect difficult authentic-photo negatives that can look suspicious without being generative:

- Professional retouching, studio lighting, makeup, or cosmetic cleanup.
- HDR, computational photography, night mode, portrait mode, depth compositing, or phone-camera processing.
- Panorama stitching, long exposure, motion blur, lens distortion, or other camera-pipeline artifacts.

## Boundary guidance
If conventional editing is the dominant ambiguity, route to `[[GA.boundary.photo_editing]]`. If the image is too degraded to verify authenticity, route to `[[GA.boundary.low_quality_uncertain]]` rather than guessing.
