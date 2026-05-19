<!-- GA.visual_artifacts.eyes.md -->
---
id: GA.visual_artifacts.eyes
version: Generative_AI.v0.1
title: Eye and gaze artifacts
area: Generative_AI
node_type: category
parent: GA.visual_artifacts
polarity: positive
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 20
  hard_positive: 25
  easy_negative: 20
  hard_negative: 25
  platinum_min: 6
source_anchors:
  - RUSH UX cleanup 2026-05-10 - L2 stub for policy graph drill-down
  - RUSH holdout review 2026-05-11 - eye artifacts and catchlight false positives
edges:
  - {type: subtype_of, to: GA.visual_artifacts}
  - {type: boundary_with, to: GA.boundary.photo_editing}
  - {type: boundary_with, to: GA.negative.authentic_photo}
canonical_examples: []
---
# Eye and gaze artifacts

## Decision rule
Use this positive node when eyes contain structural or optical anomalies that are better explained by generative synthesis than by lighting, reflection, lenses, makeup, image quality, or normal anatomy.

## Positive criteria
Strong positive cues include:

1. Pupils that are jagged, melted, non-elliptical, duplicated, missing, or not centered in a plausible iris.
2. Iris texture that is smudged, fragmented, inconsistent between eyes, or lacks coherent radial anatomy while the surrounding face is sharp.
3. Catchlights or reflections that are impossible for the lighting setup, inconsistent between eyes, or show unrelated scene fragments without plausible reflective source.
4. Eyelids, eyelashes, tear ducts, sclera, or eye corners that merge, duplicate, disappear, or violate facial anatomy.
5. Gaze direction, pupil shape, or eye placement that conflicts with head pose beyond ordinary strabismus, expression, or lens distortion.

## Hard negatives
Do not classify as `gen_ai` solely because of:

- Star-shaped, ring-shaped, window-like, or studio-softbox catchlights.
- Reflections of sky, windows, trees, phones, ring lights, or nearby objects in a real eye.
- Contact lenses, novelty lenses, makeup, cosmetic editing, flash effects, red-eye correction, medical conditions, or infant eye appearance.
- High-resolution iris detail that looks intricate, speckled, or galaxy-like but preserves plausible pupil, iris, eyelid, and reflection structure.
- Blur, compression, small crops, or unfamiliar animal-eye anatomy.

## Boundary guidance
Eye evidence is strongest when paired with other positive cues such as synthetic skin texture, facial asymmetry, malformed hands, or provenance. If the only issue is a decorative or unusual catchlight and the eye anatomy is otherwise plausible, route to `[[GA.negative.authentic_photo]]`, `[[GA.boundary.photo_editing]]`, or `[[GA.boundary.low_quality_uncertain]]` as appropriate.
