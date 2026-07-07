---
id: MD.boundary.1_vs_7
version: MNIST_Digits.v0.1
title: Boundary: 1 vs 7
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
  - RUSH MNIST demo 2026-07-07: 1-vs-7 slanted stroke boundary
edges:
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Boundary: 1 vs 7

## Decision boundary
Use stroke structure, not just slant.

- Label **1** when the glyph is one mostly straight or gently slanted continuous stroke, with at most a short top hook, flag, or serif. A short cap, angled lead-in, or small antialias blob at the top is not enough to make a 7.
- Label **7** only when there is a real top bar: a distinct left-to-right stroke that spans laterally and turns at or near its right end into a longer descending diagonal.
- If there is no clear corner separating a top bar from a diagonal, and there is no bottom base or loop, choose **1**.

## Protecting true 7s
A classic 7 with a visible horizontal top stroke and a long descending diagonal remains **7**, even if the top stroke is slightly sloped or the mark is thin.
