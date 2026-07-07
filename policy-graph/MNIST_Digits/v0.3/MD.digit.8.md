---
id: MD.digit.8
version: MNIST_Digits.v0.1
title: Digit 8
area: MNIST_Digits
node_type: digit_class
parent: MD.root
polarity: positive
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 20
  hard_positive: 20
  easy_negative: 20
  hard_negative: 20
  platinum_min: 5
source_anchors:
  - RUSH MNIST demo 2026-07-03: digit-8 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.3}
  - {type: confused_with, to: MD.digit.0}
canonical_examples: []
---
# Digit 8

## Positive criteria
Two stacked closed loops joined at a central pinch/crossing, giving two enclosed
regions (two holes) — a smaller upper loop and a usually larger lower loop. The
outline crosses or narrows in the middle.

## Distinguishing features
- Two enclosed regions is the decisive feature, versus one (0, 6, 9) or none
  (3, 5).
- Both loops close on the left; if the left side is open it is a 3.

## Hard negatives / confusions
- **3:** two right-facing bumps that stay open on the left; 8 closes both into loops.
- **0:** a single centered loop; 8 has the second loop and the central pinch.
- **6/9:** single low or high loop respectively — check for the second hole.
