---
id: GA.exception.compression_artifacts
version: Generative_AI.v0.1
title: Compression and encoding artifacts
area: Generative_AI
node_type: exception
parent: GA.root
polarity: hard-negative
status: draft
coverage_weight: 1.2
coverage_target:
  easy_negative: 30
  hard_negative: 40
  platinum_min: 10
source_anchors:
  - "v0.1 §4.1: compression artifacts as hard negatives"
  - "v0.2 §13: compression artifacts in confusable-with list"
canonical_examples: []
---
# Compression and encoding artifacts

## Decision rule
Do not classify an image as `gen_ai` solely because JPEG, WebP, video, screenshot, or platform re-encoding artifacts mimic synthetic texture or malformed detail.

## Positive criteria
Use this hard-negative exception when the suspicious evidence is better explained by lossy encoding or repeated transcoding than by generative provenance. Common false-positive triggers include:

1. Block artifacts around edges, faces, hands, text, logos, or high-contrast boundaries.
2. Chroma subsampling that smears color channels, creates color halos, or makes skin and fabric look plastic.
3. Re-encoded screenshots, social-media uploads, thumbnails, or messaging-app copies that destroy fine detail.
4. Video keyframe ghosts, motion-compensation trails, interlacing, rolling-shutter artifacts, or frame-blending residue.

## Boundary warnings
- Compression applied to a generated image does **not** make the image `not_gen_ai`; it only weakens visual evidence if provenance or underlying artifacts remain.
- Severe compression can erase decisive cues. When the image is too degraded to support a grounded decision, route to `[[GA.boundary.low_quality_uncertain]]` and use `abstain`.
- Compression ringing can resemble `[[GA.surface_texture.plastic_skin]]`, repeated teeth, warped text, or geometry seams. Require criteria beyond encoding damage before assigning a positive GenAI node.

## Hard negatives
Prioritize collection of low-bitrate social media photos, re-encoded screenshots, compressed video stills, and thumbnails that non-expert labelers are likely to overcall as generative.
