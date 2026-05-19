<!-- GA.scene_geometry.inconsistent_perspective.md -->
---
id: GA.scene_geometry.inconsistent_perspective
version: Generative_AI.v0.2
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
  - RUSH user brief 2026-05-09 - boundary conditions and policy evolution
  - RUSH holdout review 2026-05-11 - missed water-reflection mismatch
  - RUSH holdout review 2026-05-12 - wading-bird reflection undercalled as authentic
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.cgi_game_render}
  - {type: boundary_with, to: GA.negative.authentic_photo}
canonical_examples: []
---
# Inconsistent scene geometry, reflections, and shadows

## Positive criteria
Use this node when spatial cues are mutually incompatible in ways better explained by generative synthesis than by camera optics, cropping, or staging.

Strong positive cues include:

1. Reflections show objects, people, clouds, trees, lights, colors, body parts, or silhouettes that do not match the visible scene when the corresponding source area is visible.
2. Water reflections contain bright clouds, sky patches, tree lines, buildings, animals, or silhouettes absent from the sky or shoreline above the reflecting surface.
3. Wading birds, animals, people, boats, or objects whose reflection directly below them shows the wrong body part, a different pose or proportion, missing or thickened legs, an extra head or neck, or water contact inconsistent with the visible subject.
4. Shadows imply conflicting light sources without plausible explanation.
5. Objects merge, intersect, float, or change perspective in physically impossible ways.
6. Repeating terrain ridges, paths, fences, walls, windows, or horizon lines do not converge or occlude consistently.
7. Mirrors or glossy surfaces show missing, duplicated, or impossible reflected subjects.

## Review requirement
Before assigning `[[GA.negative.authentic_photo]]` to lakes, ponds, mirrors, windows, polished metal, or other reflective scenes, explicitly check whether the reflection has a plausible source in the visible or reasonably off-frame environment. For wading birds, shorebirds, animals, or people in water, compare the reflected legs, body, neck, head, and contact ripples against the visible subject rather than accepting a plausible overall wildlife composition.

## Hard negatives
- Fisheye lenses, mirrors, panoramas, long exposures, artistic staging, reflections from off-frame objects, and real compositing.
- Cropped scenes where a reflected object could plausibly be outside the frame.
- Game/CGI assets where non-realism is expected and not policy-violative by itself.
- Real water distortion, ripples, waves, mud, reeds, or glare that blur a reflection without changing it into a different object or body part.

## Boundary guidance
A single clear reflection contradiction can support `gen_ai` when it cannot be explained by crop, viewing angle, off-frame source, water distortion, or conventional compositing. Weak or ambiguous reflection oddities should be paired with another positive cue, such as synthetic surface texture or nature/wildlife artifacts.
