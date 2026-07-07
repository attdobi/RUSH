---
id: MD.boundary.0_vs_6
version: MNIST_Digits.v0.1
title: Boundary: Digit 0 vs Digit 6
area: MNIST_Digits
node_type: boundary
parent: MD.digit.0
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
  - RUSH MNIST demo 2026-07-03: digit-0-vs-6 boundary refinement
edges:
  - {type: confused_with, to: MD.digit.0}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Boundary: Digit 0 vs Digit 6

## Core distinction
Both 0 and 6 may show exactly one enclosed region. Decide by whether the single
hole is the whole digit body, or whether it is a lower loop with an extra open
curl/tail above it.

- **Label 0** when the visible stroke reads as one closed oval/ellipse: the
  contour returns to itself and there is no separate stroke that protrudes from
  the loop as an entry tail.
- **Label 6** when there is a closed lower loop plus a distinct open curl/tail
  rising above or entering into that loop, so the upper part is not simply the
  perimeter of the same oval.

## Boundary rules
- Do not turn an imperfect 0 into a 6 merely because the oval is slanted,
  uneven, thickened at one side, or has a darker overlap/closure seam. MNIST
  handwriting often has visible retracing where a closed loop meets itself; that
  is still a 0 if it does not create a separate open tail.
- A suspected 6-tail must be visually separable from the loop perimeter: it
  should extend outside the oval body or enter the loop as an open curl. If the
  mark stays on the same continuous outer contour and closes cleanly, prefer 0.
- Hole position is secondary. A 6 usually has its hole low with extra stroke mass
  above it; a 0 usually has a more centered hole. But an elongated or leaning 0
  can make the hole appear slightly low. Require the actual open upper curl/tail
  before choosing 6.
- When the top of the glyph closes into the same outline as the bottom, and
  there is only one enclosed region with no interior stroke, crossbar, or second
  loop, choose 0 decisively.

## Quick checklist
Choose **0** if all are true:
1. One closed oval-like loop.
2. No distinct free tail above the loop.
3. No open curl entering the loop.
4. No secondary loop or crossbar.

Choose **6** only when the loop is paired with a clear open upper curl/tail that
makes the digit a curl-into-loop shape rather than a standalone oval.