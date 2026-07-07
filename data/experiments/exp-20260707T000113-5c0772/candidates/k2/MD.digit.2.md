---
id: MD.digit.2
version: MNIST_Digits.v0.1
title: Digit 2
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
  - RUSH MNIST demo 2026-07-03: digit-2 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.7}
  - {type: confused_with, to: MD.digit.3}
canonical_examples: []
---
# Digit 2

## Positive criteria
A rounded top curve (open at the lower left), a diagonal stroke descending from
upper right to lower left, and a flat horizontal base stroke at the bottom.
Often has a single small partial loop at top; no fully enclosed lower region.

## Distinguishing features
- The flat horizontal base is decisive and distinguishes 2 from 7 (which has no base).
- The curves open leftward and end in a base line, unlike 3 (two right-facing
  bumps sharing a right spine, no base).

## Hard negatives / confusions
- **7:** shares the top-right stroke and diagonal but lacks the bottom base line.
- **3:** shares upper curvature but curls back to the right twice instead of
  landing on a flat base.
- A looped/cursive 2 with a bottom curl can approach 8 or 3 — require the
  horizontal base and single top curve.
