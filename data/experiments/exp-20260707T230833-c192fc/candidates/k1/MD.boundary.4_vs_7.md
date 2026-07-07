---
id: MD.boundary.4_vs_7
version: MNIST_Digits.v0.1
title: Boundary: 4 vs 7
area: MNIST_Digits
node_type: boundary
parent: MD.digit.4
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
  - RUSH MNIST demo 2026-07-07: open-4 vs 7 angular boundary
edges:
  - {type: confused_with, to: MD.digit.4}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Boundary: 4 vs 7

## Decision boundary
Separate a 4's internal cross-structure from a 7's simple top-bar-plus-diagonal shape.

- Label **4** when an angular mark has a mid-height crossbar, side spur, or crossing stroke meeting a vertical/right stroke or diagonal support. The top may be open, compact, or slanted; a closed triangle is not required.
- Label **7** when the structure is only a top horizontal or slightly sloped bar joined at the right to one descending diagonal, with no midline crossbar, no left-side support, and no right stroke continuing through a crossbar.
- In compact noisy marks, a short stroke that intersects below the top bar, or a second supporting downward stroke, is decisive evidence for **4** over 7.

## Protecting true 7s
A clear top bar turning into a single long descending diagonal remains **7** when there is no internal crossbar or second supporting stroke.
