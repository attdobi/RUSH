---
id: MD.digit.7
version: MNIST_Digits.v0.1
title: Digit 7
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
  - RUSH MNIST demo 2026-07-03: digit-7 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.2}
canonical_examples: []
---
# Digit 7

## Positive criteria
A horizontal (or slightly downward) top stroke running left-to-right, joined at
its right end by a diagonal stroke descending toward the lower left. Optionally a
short horizontal crossbar through the middle of the diagonal (European 7). No
loops and no bottom base line.

## Distinguishing features
- A true horizontal top bar plus a long descending diagonal, with no bottom
  base, separates 7 from both 1 and 2.
- No enclosed regions anywhere.

## Hard negatives / confusions
- **1:** a 1 with a large top flag can mimic the top bar — require a clear
  horizontal top stroke and a long diagonal descent.
- **2:** shares the top-right stroke and diagonal but adds a flat horizontal base
  at the bottom; 7 has none.
