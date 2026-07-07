---
id: MD.boundary.1_vs_7
version: MNIST_Digits.v0.1
title: Boundary: Digit 1 vs Digit 7
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
  - RUSH MNIST demo 2026-07-07: digit-1 vs digit-7 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Boundary: Digit 1 vs Digit 7

## Use this boundary when
The glyph is a single narrow vertical or diagonal mark with possible top thickening, hook, flag, or serif, and could be read either as a plain 1 or as a minimal 7.

## Decide 1 when
- The visible skeleton is one continuous stroke from top to bottom, even if it leans strongly left or right.
- Any top feature is only a short serif, hook, taper, blob, or end-thickening attached to the same stroke.
- There is no distinct left-to-right top bar that forms a clear corner before the descending stroke.
- A lone diagonal slash with no separate bar remains a 1.

## Decide 7 when
- There is a distinct horizontal or near-horizontal top bar with meaningful left-to-right extent.
- The descending diagonal begins from the right end of that bar, creating a clear angular junction: a roof-plus-slash shape.
- Optional middle crossbar is allowed, but there is still no bottom base line or loop.

## Decisive test
Ignore blur and tiny serifs, then trace the main stroke structure. One main slash or vertical stroke with only end embellishments is 1. A true top bar joined at an angle to a long descending diagonal is 7.
