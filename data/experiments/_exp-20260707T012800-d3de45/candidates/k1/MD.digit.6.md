---
id: MD.digit.6
version: MNIST_Digits.v0.1
title: Digit 6
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
  - RUSH MNIST demo 2026-07-03: digit-6 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.0}
  - {type: confused_with, to: MD.digit.5}
canonical_examples: []
---
# Digit 6

## Positive criteria
A single continuous stroke that curls down from the upper right (or top) and
closes into a single loop at the bottom. Exactly one enclosed region, located in
the lower half, with an open curl/tail rising above it.

## Distinguishing features
- The single hole sits low and there is an open descending curl above it, unlike
  0 whose loop is centered and closed at both ends.
- It is one smooth continuous curl, not a top-bar-plus-stem shape (5).

## Hard negatives / confusions
- **0:** closed at top and bottom with a centered loop; 6 has an open upper curl.
- **5:** has a flat top bar and vertical stem; 6 is a continuous curve into the loop.
- **8:** a 6 with a partially closed upper curl can approach 8 — require only one
  enclosed region.
