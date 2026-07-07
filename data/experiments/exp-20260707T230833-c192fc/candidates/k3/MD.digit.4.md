---
id: MD.digit.4
version: MNIST_Digits.v0.1
title: Digit 4
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
  - RUSH MNIST demo 2026-07-03: digit-4 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.9}
canonical_examples: []
---
# Digit 4

## Positive criteria
Two roughly vertical strokes joined by a horizontal crossbar: a diagonal/vertical
left stroke and a vertical right stroke, connected mid-height by a horizontal
bar. The top may be open (open-4) or meet to close a small triangle
(closed-top-4).

## Distinguishing features
- The horizontal crossbar meets a straight vertical right stroke that continues
  below the bar; the region above the bar is a triangle/wedge, not a rounded loop.
- Any enclosed top region is angular, distinguishing it from 9's rounded loop.

## Hard negatives / confusions
- **9:** has a rounded closed loop at the top and a curved/straight tail; 4's top
  region is angular and its right stroke is straight through the crossbar.
- A closed-top 4 can approach 9 — check whether the top is a sharp wedge (4) or a
  smooth loop (9).
