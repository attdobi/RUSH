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
  - RUSH MNIST demo 2026-07-07: 1-vs-7 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Boundary: 1 vs 7

## Decision boundary
Do not infer **7** from slant alone. This pair turns on whether there is a true
top bar plus diagonal construction, versus a single stroke with only a small cap
or hook.

Choose **1** when the glyph is one mostly vertical or slightly slanted stroke
with no branching structure. A short top hook, curved cap, blob, or serif is
still part of a 1 if it does not form a distinct horizontal bar spanning away
from the main stroke.

Choose **7** only when there is a clear horizontal or near-horizontal top stroke
that projects left-to-right from the top and meets a longer descending diagonal
at an angular corner. The top bar should read as its own stroke segment, not just
as the rounded beginning or small flag of a single vertical stroke.

## Quick checks
- If removing a tiny top hook would leave an ordinary single stroke, label 1.
- If the glyph has a deliberate top bar and the main body descends diagonally
  from one end of that bar, label 7.
