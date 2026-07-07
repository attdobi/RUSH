---
id: MD.boundary.9_vs_0
version: MNIST_Digits.v0.1
title: Boundary: 9 vs 0
area: MNIST_Digits
node_type: boundary
parent: MD.digit.9
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
  - RUSH MNIST demo 2026-07-07: 9-vs-0 tail boundary
edges:
  - {type: confused_with, to: MD.digit.9}
  - {type: confused_with, to: MD.digit.0}
canonical_examples: []
---
# Boundary: 9 vs 0

## Decision boundary
A one-loop glyph is not automatically a 0; check for loop height and any descender.

- Label **9** when the single enclosed loop sits in the upper half and any stroke or tail descends outside the loop below it. The descender may be short, straight, curved, or attached from the right or lower side; it need not be a long clean right-side stem.
- Label **0** when the mark is a single centered oval or ellipse with no external descender, tail, or protruding lower stroke, and the contour closes smoothly at both top and bottom.
- If there is one loop plus a visible asymmetric tail extending below the loop, choose **9** over 0. If there is no tail or protrusion and the loop is centered, choose **0**.

## Protecting true 0s
A rough or slanted oval remains **0** when it has one centered hole and no separate downward tail outside the loop.
