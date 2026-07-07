---
id: MD.boundary.8_vs_6
version: MNIST_Digits.v0.1
title: Boundary: 8 vs 6
area: MNIST_Digits
node_type: boundary
parent: MD.digit.8
polarity: mixed
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 20
  hard_positive: 20
  easy_negative: 20
  hard_negative: 20
  platinum_min: 5
source_anchors:
  - RUSH MNIST demo 2026-07-07: 8-vs-6 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.8}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Boundary: 8 vs 6

## Decision boundary
Decide this pair by whether the upper part forms a second closed chamber, not by
whether the whole glyph looks like a continuous curl.

Choose **8** when the stroke makes two stacked lobes separated by a waist/pinch
or crossing, and the upper lobe closes or visibly touches back enough to create a
second enclosed pocket. The upper loop may be small, narrow, slanted, or partly
blurred in MNIST rendering; do not ignore a tiny upper hole when a lower loop is
also present.

Choose **6** when there is exactly one enclosed region in the lower half and the
upper portion is only an open tail/curl. A 6 may bend inward above the lower
loop, but if that upper curl never reconnects, crosses, or closes into its own
chamber, it remains a 6.

## Protecting nearby classes
- Do not upgrade an open-left double-bump shape to 8 unless the lobes actually
  close; open stacked bumps remain 3.
- Do not downgrade a narrow cursive 8 to 6 solely because its lower loop is
  larger or the upper loop is tiny; the second enclosed upper pocket is decisive.
