---
id: MD.boundary.1_vs_6
version: MNIST_Digits.v0.1
title: Boundary: 1 vs 6
area: MNIST_Digits
node_type: boundary
parent: MD.digit.1
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
  - RUSH MNIST demo 2026-07-07: 1-vs-6 no-loop boundary
edges:
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Boundary: 1 vs 6

## Decision boundary
Do not infer a loop from thickness, blur, or a slight curve.

- Label **1** when the visible mark is a single unbranched slash or vertical stroke with no enclosed white interior, even if it leans diagonally or has a small hook.
- Label **6** only when there is an actual closed lower loop: a visible enclosed region in the lower half plus an open curl or tail rising above it.
- If no enclosed region is visible anywhere, the glyph is not a 6; for a single isolated stroke, choose **1**.

## Protecting true 6s
A continuous curl that clearly closes around a lower hole remains **6**, even when the loop is narrow or the upper curl is short.
