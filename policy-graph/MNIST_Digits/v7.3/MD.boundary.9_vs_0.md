---
id: MD.boundary.9_vs_0
version: MNIST_Digits.v0.1
title: Boundary: Digit 9 vs Digit 0
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
  - RUSH MNIST demo 2026-07-07: digit-9 vs digit-0 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.9}
  - {type: confused_with, to: MD.digit.0}
canonical_examples: []
---
# Boundary: Digit 9 vs Digit 0

## Use this boundary when
The glyph has one closed oval-like loop and could be read either as a plain 0 or as a 9 with a small/tucked descender.

## Decide 9 when
- The loop sits in the upper portion of the glyph and any stroke continues below the loop from its side or lower side.
- The extra stroke may be short, faint, close to the loop, or nearly vertical; it does not need to be a long clean tail.
- The overall topology is loop-plus-descender, including forms that resemble a small lowercase q or an oval with a terminal stroke attached below one side.

## Decide 0 when
- The mark is only one continuous oval/ellipse with no terminal stroke leaving the loop.
- Irregular thickness, a small gap, or a rough join stays on the oval contour rather than protruding downward as a separate tail.
- The single hole is centered within the full glyph height rather than clearly above a descender.

## Decisive test
Trace the outer contour. If after closing the loop there is a visible stroke segment that extends below the loop boundary, classify as 9. If the contour simply returns around an oval with no outside descender, classify as 0.
