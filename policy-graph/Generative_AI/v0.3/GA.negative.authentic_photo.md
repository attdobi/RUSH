---
id: GA.negative.authentic_photo
version: Generative_AI.v0.2
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
  - v0.2 section 2 - real-negative root for authentic photographs
  - RUSH holdout review 2026-05-11 - authentic-photo overuse on photorealistic GenAI
  - RUSH holdout review 2026-05-12 - portrait, egret reflection, and reef-fish positives undercalled as authentic
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.visual_artifacts.nature_wildlife}
  - {type: boundary_with, to: GA.scene_geometry.inconsistent_perspective}
canonical_examples: []
---
# Authentic photograph (real negative)

## Decision rule
Classify as `not_gen_ai` when the image is an authentic, unmodified or conventionally edited photograph with no evidence of generative provenance **and** no specific positive visual artifacts after review of the relevant positive nodes.

Do not assign this node merely because the image has plausible optics, lighting, bokeh, depth of field, high detail, or a stock-photo composition. Modern generative images can imitate camera capture well.

## Positive criteria
Use this real-negative node when the image shows:

1. No visible generative artifacts in anatomy, eyes, text, surface texture, natural-scene detail, geometry, reflections, or object continuity.
2. No provenance evidence indicating synthetic generation, generative fill, or AI-assisted composition.
3. Characteristics consistent with camera capture, including plausible optics, lighting, sensor noise, depth of field, and scene continuity.
4. A better real-world explanation for any suspicious feature, such as motion blur, lens distortion, long exposure, makeup, retouching, species anatomy, real signage, decorative embossing, or compression.

## Required exclusions
Do not use this node if any of the following are present without a stronger real-world explanation:

- Clear reflection or shadow mismatch, especially water or mirror reflections that show objects, clouds, body parts, or light sources absent from the visible scene.
- Wading birds, animals, people, boats, or objects in still water where the reflection directly below the subject depicts the wrong body part, a different pose or proportion, impossible leg or contact geometry, or ripple patterns inconsistent with the visible subject.
- Animal or human digit, paw, limb, eye, tooth, or facial-structure artifacts.
- High-quality portraits where in-focus skin across the face is uniformly poreless, waxy, plastic, or clay-like while nearby hair, eyes, fabric, or accessories retain enough detail that real skin texture should be visible.
- Water, fur, foliage, terrain, or surface textures that are locally coherent but globally synthetic, repeated, melted, waxy, or physically implausible.
- Underwater, aquarium, reef, fish, or coral scenes where scales, fins, coral, anemone tentacles, polyps, or material boundaries are over-uniform, cloned, melted, or physically confused.
- Text, logos, labels, or signage that are intended to be legible but are malformed in ways not explained by blur, focus, language, or compression.

## Hard negatives
Collect difficult authentic-photo negatives that can look suspicious without being generative:

- Professional retouching, studio lighting, makeup, or cosmetic cleanup.
- HDR, computational photography, night mode, portrait mode, depth compositing, or phone-camera processing.
- Panorama stitching, long exposure, motion blur, lens distortion, reflections from off-frame objects, or other camera-pipeline artifacts.
- Wildlife and nature photos with telephoto compression, shallow depth of field, golden-hour light, long-exposure waterfalls, unusual but real animal poses, or decorative medals and logos.

## Boundary guidance
If conventional editing is the dominant ambiguity, route to `[[GA.boundary.photo_editing]]`. If the image is too degraded to verify authenticity, route to `[[GA.boundary.low_quality_uncertain]]` rather than guessing. If the scene is photorealistic wildlife, landscape, water, foliage, terrain, underwater life, or reef detail with suspicious local detail, consider `[[GA.visual_artifacts.nature_wildlife]]` before assigning this node. If a reflective surface is decision-relevant, check `[[GA.scene_geometry.inconsistent_perspective]]` before relying on plausible camera-like optics.
