---
id: MD.digit.3
version: MNIST_Digits.v0.1
title: Digit 3
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
  - RUSH MNIST demo 2026-07-03: digit-3 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.8}
  - {type: confused_with, to: MD.digit.5}
canonical_examples: []
---
# Digit 3

## Positive criteria
Two stacked right-facing curves (an upper and a lower bump) that share a common
spine on the right side and are open on the left. No fully enclosed loops; the
left side stays open.

## Distinguishing features
- Both bumps open to the left; if either side closes into a loop it becomes 8.
- The upper portion is a curve, not a flat top bar + stem (that would be 5).

## Hard negatives / confusions
- **8:** the two bumps close on the left into stacked loops with a central pinch;
  3 leaves them open.
- **5:** has a straight horizontal top and a vertical stem before the lower bowl;
  3 has a rounded top bump instead.
- A tightly curled 3 can approach 8 — verify the left side is genuinely open.
