---
id: MD.digit.1
version: MNIST_Digits.v0.1
title: Digit 1
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
  - RUSH MNIST demo 2026-07-03: digit-1 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.7}
  - {type: confused_with, to: MD.digit.2}
canonical_examples: []
---
# Digit 1

## Positive criteria
A single, mostly vertical stroke. May include a short upward-left flag/serif at
the top and/or a short horizontal base serif at the bottom. No loops and no
enclosed regions.

## Distinguishing features
- No horizontal top bar spanning to the right (that would indicate 7).
- No bottom curve or bowl and no diagonal reaching a flat base (that would
  indicate 2).

## Hard negatives / confusions
- **7:** has a distinct horizontal top stroke and a longer descending diagonal;
  a 1 with a large top flag can mimic this — check for a true horizontal bar.
- **2:** shares a diagonal descent but adds a top curve and a flat horizontal base.
- A perfectly straight thin stroke may resemble the vertical spine of other
  digits; require the absence of loops/bars.
