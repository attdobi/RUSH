---
id: MD.boundary.8_vs_6
version: MNIST_Digits.v0.1
title: Boundary: Digit 8 vs Digit 6
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
  - RUSH MNIST demo 2026-07-07: digit-8 vs digit-6 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.8}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Boundary: Digit 8 vs Digit 6

## Use this boundary when
The glyph has a lower loop plus an upper curl or small upper chamber, and could be read either as a cursive/slanted 8 or as a 6 with an open upper tail.

## Decide 8 when
- The stroke path makes two lobes separated by a waist, pinch, crossing, or tight neck, even if the upper lobe is tiny, diagonal, faint, or partly filled by stroke thickness.
- The apparent upper curl returns back into the glyph to bound a second chamber instead of ending as a free tail.
- A small top loop or near-loop attached to a larger lower loop is still an 8 when the two parts meet at a central constriction.
- Pixel gaps, blur, or heavy stroke can make one lobe look only partly closed; classify as 8 when the intended path visibly doubles back to form the upper chamber.

## Decide 6 when
- There is exactly one enclosed chamber, located low, with an upper stroke that remains a free open curl or tail.
- The upper stroke enters or approaches the lower loop but does not cross, pinch, or return to outline a separate top lobe.
- The overall topology is loop-plus-open-curl, not two chambers stacked or diagonally connected.

## Decisive test
Trace the stroke above the lower loop. If it returns to create a second bounded lobe or clear central waist, choose 8. If it stays as a single free upper curl attached to one lower loop, choose 6.