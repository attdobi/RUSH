---
id: GA.boundary.low_quality_uncertain
version: Generative_AI.v0.1
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
  - RUSH v0.1 difficulty tiers: hard/low-confidence cases to SME queue
edges:
  - {type: subtype_of, to: GA.root}
  - {type: boundary_with, to: GA.visual_artifacts.anatomy.hands}
  - {type: boundary_with, to: GA.visual_artifacts.text_symbols}
canonical_examples: []
---
# Low-quality, cropped, or insufficient-evidence cases

## Decision rule
If the image is too small, blurry, cropped, compressed, occluded, or ambiguous to ground a GenAI decision, cite this node — but still return a decisive label. The default for images that land here is `not_gen_ai` (evidence insufficient to establish generative provenance), with confidence set as low as the evidence warrants, difficulty `high`, and `is_boundary` true citing this node. Never output `abstain` or `unknown`: the doubt lives in the confidence score, not in a refusal.

## Routing criteria
Use this boundary node when one or more of the following prevents a grounded GenAI determination:

1. Resolution is below 200×200 pixels, or the relevant region of interest is effectively below that size after cropping.
2. More than 60% of the image or decision-relevant subject is occluded, cropped out, blocked by overlays, or covered by watermarks.
3. Combined confidence across label and policy-graph placement is below 0.5.
4. The input is a screenshot-of-screenshot, heavily re-encoded upload, thumbnail, or other multi-generation copy where evidence is degraded.
5. Blur, compression, glare, darkness, motion, partial framing, or extreme stylization removes the visual details needed to distinguish `gen_ai` from `not_gen_ai`.

## Boundary criteria
Do not use low quality as a shortcut for suspiciousness. If decisive evidence remains visible, label the image under the appropriate positive, negative, exception, or boundary node. If severe compression is the main confounder, consider `[[GA.exception.compression_artifacts]]`; if conventional editing is the main confounder, consider `[[GA.boundary.photo_editing]]`.

## SME review expectations
High-impact or sampled low-quality cases should be routed to SME review. The SME either confirms the low-confidence label or reclassifies the image into the supported policy-graph node the evidence actually grounds. Low-confidence boundary-flagged calls rank high in the re-adjudication queue, so the evaluation set is corrected by humans rather than by rewarding guessing.

## Why this node exists
Cold-start systems love overcalling blurry nonsense. This node exists to keep uncertainty explicit in confidence and difficulty — instead of pretending every pixel is a confession, or hiding behind a refusal to answer.
