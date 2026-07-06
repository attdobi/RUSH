---
id: GA.visual_artifacts.nature_wildlife
version: Generative_AI.v0.2
title: Photorealistic nature, wildlife, and landscape synthesis artifacts
area: Generative_AI
node_type: category
parent: GA.visual_artifacts
polarity: positive
status: draft
coverage_weight: 1.3
coverage_target:
  easy_positive: 25
  hard_positive: 35
  easy_negative: 20
  hard_negative: 35
  platinum_min: 8
source_anchors:
  - RUSH holdout review 2026-05-11 - photorealistic animals, water, foliage, and landscapes missed as authentic
  - RUSH holdout review 2026-05-12 - reef fish, coral, and aquarium-like scenes undercalled as authentic or uncertain
edges:
  - {type: subtype_of, to: GA.visual_artifacts}
  - {type: boundary_with, to: GA.negative.authentic_photo}
  - {type: boundary_with, to: GA.boundary.cgi_game_render}
  - {type: related_to, to: GA.surface_texture.plastic_skin}
  - {type: related_to, to: GA.scene_geometry.inconsistent_perspective}
  - {type: related_to, to: GA.visual_artifacts.anatomy.hands}
canonical_examples: []
---
# Photorealistic nature, wildlife, and landscape synthesis artifacts

## Decision rule
Use this positive node when a nature, wildlife, animal, water, foliage, underwater, reef, aquarium, or landscape image appears camera-like overall but contains local or cross-scene artifacts better explained by generative synthesis than by photography, editing, CGI, compression, or real natural variation.

This node exists because photorealistic generated nature images often have plausible bokeh, lighting, and composition while failing in water behavior, animal anatomy, fur, scales, vegetation, coral, terrain, reflections, or material boundaries.

## Positive criteria
Classify here when one strong cue or multiple weaker cues are present:

1. Animal paws, claws, toes, legs, ears, teeth, muzzles, eyes, fins, gills, tails, or body contours are fused, duplicated, missing, or structurally incoherent.
2. Fur, mane, feathers, fish scales, reef-fish skin, whiskers, or similar animal texture forms over-uniform strands, candy-smooth fields, fabric-like waves, melted clumps, impossible halos, hyper-regular scale rows, or repeated micro-texture that does not follow anatomy.
3. Water, foam, spray, droplets, bubbles, or waterfalls appear like solid strands, glassy ribbons, repeated beads, suspended blobs, or physically implausible splash patterns relative to the animal or terrain.
4. Leaves, grass, moss, flowers, rocks, bark, coral, anemones, polyps, tentacles, or reef detail become generic, tiled, over-smoothed, melted, or repeated while the scene remains otherwise sharp.
5. Coral or anemone structures show many identical white-tipped polyps, cloned tentacle clusters, repeated reef fragments, or texture patterns that continue through occlusions and depth changes instead of following real organism structure.
6. Terrain, hills, ridges, shorelines, rocks, or paths have implausibly regular, wave-like, sculpted, or procedural geometry not explained by geology, agriculture, landscaping, or lens effects.
7. Reflections in water show clouds, trees, sky, animals, or light patterns that do not match the visible scene; route to `[[GA.scene_geometry.inconsistent_perspective]]` if reflection mismatch is the dominant cue.
8. Boundaries between animal, fish, water, grass, coral, rock, and background are locally blended or semantically confused despite high apparent image quality.
9. A cinematic wildlife, aquarium, reef, or stock-nature composition combines unusually perfect subject pose, lighting, bokeh, saturation, and two or more local material, anatomy, or pattern inconsistencies.

## Hard negatives
Do not use this node solely because an authentic nature or wildlife photo is beautiful, dramatic, rare, saturated, or polished. Hard negatives include:

- Telephoto wildlife photography, golden-hour lighting, shallow depth of field, high-speed water spray, long-exposure waterfalls, and HDR landscapes.
- Aquarium photography, macro reef photography, underwater color correction, artificial tank lighting, glass distortion, backscatter, bubbles, and ordinary reef or anemone complexity.
- Wet fur, groomed manes, seasonal coats, species-specific markings, real scars, unusual animal poses, or partial occlusion by grass, water, rocks, coral, or shadow.
- Repeating real-world patterns such as ripples, waves, leaf clusters, crops, terraces, rocks, fish scales, coral branches, anemone tentacles, fur direction, or geological strata.
- Conventional CGI, game renders, paintings, or stylized illustrations where non-realism is expected and no generative evidence is present.
- Compression, motion blur, low resolution, or screenshot degradation that removes natural detail.

## Boundary guidance
If the only evidence is that the image looks too perfect, too saturated, or too polished, use `[[GA.negative.authentic_photo]]` or a boundary node instead. If a detail-rich fish, reef, underwater, or aquarium-like image has two or more specific cues such as painted scale fields, cloned coral or anemone structures, confused fin boundaries, or impossible water interaction, do not route to `[[GA.boundary.low_quality_uncertain]]` merely because photography, CGI, or heavy editing is also possible. If a specific artifact is clearer under another node, use that node: animal paws and digits under `[[GA.visual_artifacts.anatomy.hands]]`, reflection mismatch under `[[GA.scene_geometry.inconsistent_perspective]]`, synthetic texture under `[[GA.surface_texture.plastic_skin]]`, repeated patches under `[[GA.visual_artifacts.repeated_details]]`, and low-quality uncertainty under `[[GA.boundary.low_quality_uncertain]]`.

## SME review triggers
Escalate hard cases where a real photographic explanation is plausible but not decisive, especially photorealistic wildlife portraits, animal action shots, waterfalls, lakes, forests, rolling hills, underwater scenes, aquarium scenes, reef fish, coral, anemones, and stylized fish or animal illustrations.
