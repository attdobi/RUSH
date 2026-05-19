---
id: GA.provenance.synthetic_disclosure
version: Generative_AI.v0.1
title: Explicit synthetic provenance or disclosure
area: Generative_AI
node_type: category
parent: GA.root
polarity: positive
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 20
  hard_positive: 15
  easy_negative: 20
  hard_negative: 20
  platinum_min: 5
source_anchors:
  - RUSH user brief 2026-05-09: grounded justifications and policy references
edges:
  - {type: subtype_of, to: GA.root}
canonical_examples: []
---
# Explicit synthetic provenance or disclosure

## Positive criteria
Use this node when reliable surrounding evidence indicates generation:

- Visible watermark or platform label identifying AI generation.
- Metadata, caption, or source context explicitly states the image is AI-generated.
- Creator/tool provenance is available and tied to this exact media item.

## Hard negatives
- Captions joking about AI without actual provenance.
- Reposts or memes where the source relationship is unclear.
- Watermarks from editing apps that do not imply generative synthesis.
