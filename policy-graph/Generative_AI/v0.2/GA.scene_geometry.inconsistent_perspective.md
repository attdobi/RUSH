<!-- GA.scene_geometry.inconsistent_perspective.md -->
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
  - RUSH user brief 2026-05-09 - boundary conditions and policy evolution
  - RUSH holdout review 2026-05-11 - missed water-reflection mismatch
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

1. Reflections show objects, people, clouds, trees, lights, or colors that do not match the visible scene when the corresponding source area is visible.
2. Water reflections contain bright clouds, sky patches, tree lines, buildings, or silhouettes absent from the sky or shoreline above the reflecting surface.
3. Shadows imply conflicting light sources without plausible explanation.
4. Objects merge, intersect, float, or change perspective in physically impossible ways.
5. Repeating terrain ridges, paths, fences, walls, windows, or horizon lines do not converge or occlude consistently.
6. Mirrors or glossy surfaces show missing, duplicated, or impossible reflected subjects.

## Review requirement
Before assigning `[[GA.negative.authentic_photo]]` to lakes, ponds, mirrors, windows, polished metal, or other reflective scenes, explicitly check whether the reflection has a plausible source in the visible or reasonably off-frame environment.

## Hard negatives
- Fisheye lenses, mirrors, panoramas, long exposures, artistic staging, reflections from off-frame objects, and real compositing.
- Cropped scenes where a reflected object could plausibly be outside the frame.
- Game/CGI assets where non-realism is expected and not policy-violative by itself.

## Boundary guidance
A single clear reflection contradiction can support `gen_ai` when it cannot be explained by crop, viewing angle, off-frame source, water distortion, or conventional compositing. Weak or ambiguous reflection oddities should be paired with another positive cue, such as synthetic surface texture or nature/wildlife artifacts.
