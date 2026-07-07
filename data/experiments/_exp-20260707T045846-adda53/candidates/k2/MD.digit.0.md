---
id: MD.digit.0
version: MNIST_Digits.v0.1
title: Digit 0
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
  - RUSH MNIST demo 2026-07-03: digit-0 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.6}
  - {type: confused_with, to: MD.digit.8}
canonical_examples: []
---
# Digit 0

## Positive criteria
A single closed loop forming an oval or ellipse, tallest vertically, with a
continuous outer contour and a single empty interior. No crossbar, no interior
strokes, and no secondary loop.

## Distinguishing features
- Exactly one enclosed region (one hole), unlike 8 (two holes).
- The loop closes cleanly at both top and bottom, unlike 6 (open top curl) and
  9 (open bottom tail).

## Hard negatives / confusions
- **6:** has an open curl entering the loop from the upper left; 0 has no such tail.
- **8:** has a second stacked loop and a central pinch/crossing; 0 has neither.
- A slanted, narrow 0 can resemble a 1 or a stylized 8 — check for the single
  clean interior.

<!-- dry-run experiment edit, cycle k=2 -->
