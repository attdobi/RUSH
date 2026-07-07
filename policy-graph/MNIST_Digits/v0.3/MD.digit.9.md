---
id: MD.digit.9
version: MNIST_Digits.v0.1
title: Digit 9
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
  - RUSH MNIST demo 2026-07-03: digit-9 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.4}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Digit 9

## Positive criteria
A rounded closed loop in the upper half with a stroke/tail descending from the
loop's right side toward the bottom (straight or slightly curved). Exactly one
enclosed region, located high.

## Distinguishing features
- The single hole sits high with a descending tail below it, mirroring 6
  (whose hole sits low).
- The top region is a smooth rounded loop, not the angular wedge/crossbar of 4.

## Hard negatives / confusions
- **4:** has an angular top wedge and a horizontal crossbar with a straight right
  stroke; 9 has a rounded top loop.
- **7:** a 9 with a weak/open top loop can approach 7 — require a genuine
  enclosed loop up top.
- **6:** same topology inverted — confirm the loop is in the upper half.
