---
id: MD.digit.5
version: MNIST_Digits.v0.1
title: Digit 5
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
  - RUSH MNIST demo 2026-07-03: digit-5 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.6}
  - {type: confused_with, to: MD.digit.3}
canonical_examples: []
---
# Digit 5

## Positive criteria
A horizontal top stroke, a short vertical stem descending on the left from the
top bar, and a lower right-facing bowl/curve that closes back toward the stem.
The top is a distinct flat or near-flat bar, not a curve.

## Distinguishing features
- The straight top bar + vertical stem sequence separates 5 from the rounded
  curls of 6 and the double bumps of 3.
- The lower bowl connects back after the stem; it does not fully enclose a bottom
  loop the way 6 does.

## Hard negatives / confusions
- **6:** is a single continuous curl into a closed bottom loop with no straight
  top bar; 5 begins with a flat top and stem.
- **3:** has two right-facing bumps and a rounded top; 5 has a flat top and a
  single lower bowl.
- A rounded-top 5 can approach 6 — require the top bar/stem structure.
