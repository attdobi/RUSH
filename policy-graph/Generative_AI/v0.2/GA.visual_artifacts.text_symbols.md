<!-- GA.visual_artifacts.text_symbols.md -->
---
id: GA.visual_artifacts.text_symbols
version: Generative_AI.v0.1
title: Garbled text and symbol artifacts
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.2
coverage_target:
  easy_positive: 25
  hard_positive: 25
  easy_negative: 15
  hard_negative: 25
  platinum_min: 6
source_anchors:
  - RUSH user brief 2026-05-09 - GenAI subtype set grows over time
  - RUSH holdout review 2026-05-11 - decorative medal relief overcalled as garbled symbols
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.boundary.low_quality_uncertain}
  - {type: boundary_with, to: GA.negative.authentic_photo}
canonical_examples: []
---
# Garbled text and symbol artifacts

## Positive criteria
Use this node when text-like or symbol-like regions appear intended to be legible, branded, or semantically meaningful but show common generation failures:

1. Pseudo-letters that resemble text but do not form language.
2. Brand marks, logos, labels, captions, signs, jersey numbers, packaging, or interface text that are malformed in ways inconsistent with blur alone.
3. Repeated nonsensical typography integrated into the generated scene.
4. Inconsistent spelling, letter shapes, or logo geometry across repeated instances of the same mark.
5. Symbols that imitate known writing systems, medals, seals, emblems, or badges while failing to preserve expected structure when the image clearly intends a readable mark.

## Hard negatives
Do not classify as `gen_ai` solely because of:

- Low-resolution compression making real text unreadable.
- Motion blur, depth-of-field blur, glare, occlusion, embossing, engraving, reflections, or curved surfaces.
- Intentionally fictional signage, fantasy scripts, abstract logos, decorative medal relief, ornamental seals, trophy designs, or non-text patterns.
- Non-Latin scripts unfamiliar to the labeler; do not classify as garbled without evidence.
- Small background text where the image does not provide enough detail to determine whether the mark is malformed.

## Boundary guidance
Ask whether the region is supposed to be readable or recognizable. If it is merely decorative embossing, abstract design, a medal pattern, or a real but unfamiliar logo, use a negative or boundary node unless other positive GenAI evidence is present. If compression or blur is the best explanation, route to `[[GA.exception.compression_artifacts]]` or `[[GA.boundary.low_quality_uncertain]]`.
