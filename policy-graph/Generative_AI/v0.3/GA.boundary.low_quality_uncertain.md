---
id: GA.boundary.low_quality_uncertain
version: Generative_AI.v0.2
title: Low-quality, cropped, or insufficient-evidence cases
area: Generative_AI
node_type: boundary
parent: GA.root
polarity: negative
status: draft
coverage_weight: 1.1
coverage_target:
  easy_positive: 0
  hard_positive: 0
  easy_negative: 25
  hard_negative: 40
  platinum_min: 8
source_anchors:
  - RUSH v0.1 difficulty tiers: hard/abstain cases to SME queue
  - RUSH holdout review 2026-05-12 - detail-rich stylized fish scene incorrectly routed to uncertainty
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.visual_artifacts.anatomy.hands}
  - {type: boundary_with, to: GA.visual_artifacts.text_symbols}
canonical_examples: []
---
# Low-quality, cropped, or insufficient-evidence cases

## Decision rule
If the image is too small, blurry, cropped, compressed, occluded, or ambiguous to ground a GenAI decision, route it here instead of forcing a positive or negative call. The default label for images that land here is `abstain`. LLM labelers and human reviewers MUST NOT guess.

This node is not a catch-all for hard but visible cases. Do not use it merely because an image is saturated, stylized, macro, underwater, studio-lit, beautiful, or difficult to source when decision-relevant detail is still visible.

## Routing criteria
Use this boundary node when one or more of the following prevents a grounded GenAI determination:

1. Resolution is below 200×200 pixels, or the relevant region of interest is effectively below that size after cropping.
2. More than 60% of the image or decision-relevant subject is occluded, cropped out, blocked by overlays, or covered by watermarks.
3. Combined confidence across label and policy-graph placement is below 0.5.
4. The input is a screenshot-of-screenshot, heavily re-encoded upload, thumbnail, or other multi-generation copy where evidence is degraded.
5. Blur, compression, glare, darkness, motion, partial framing, or extreme stylization removes the visual details needed to distinguish `gen_ai` from `not_gen_ai`.

## Boundary criteria
Do not use low quality as a shortcut for suspiciousness, and do not use uncertainty to avoid classifying visible policy evidence. If decisive evidence remains visible, label the image under the appropriate positive, negative, exception, or boundary node. Detail-rich portraits, reflective water scenes, wildlife images, fish, reef, coral, aquarium scenes, and stylized illustrations should be routed to the relevant positive or negative node when their local cues are reviewable. If severe compression is the main confounder, consider `[[GA.exception.compression_artifacts]]`; if conventional editing is the main confounder, consider `[[GA.boundary.photo_editing]]`.

## SME review expectations
High-impact or sampled low-quality cases should be routed to SME review. The SME either reclassifies the image into a supported policy-graph node or confirms `abstain`. Confirmed abstains are excluded from decision-quality metric denominators so the evaluation set does not reward guessing.

## Why this node exists
Cold-start systems love overcalling blurry nonsense. This node exists to make uncertainty explicit instead of pretending every pixel is a confession.
