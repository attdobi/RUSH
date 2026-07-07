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

A 6 requires **both** of these visible features:
1. a real lower enclosed hole, surrounded by stroke on all sides; and
2. an upper stroke/tail with a visible open end that feeds into that lower loop.

Do not infer a 6 from a mere curve, diagonal, or oval unless both the lower hole
and the open upper tail are present.

## Distinguishing features
- The single hole sits low and there is an open descending curl above it, unlike
  0 whose loop is centered and closed at both ends.
- It is one smooth continuous curl, not a top-bar-plus-stem shape (5).
- A lone slanted or curved stroke with **no enclosed lower hole** is not a 6;
  prefer the no-loop digit whose stroke geometry is present, such as 1 or 7.
- A lower loop plus a second closed or nearly closed upper lobe is not a 6; if a
  central pinch separates two stacked lobes, prefer 8 even when the upper lobe is
  smaller, faint, or slightly broken by handwriting noise.

## Hard negatives / confusions
- **0:** closed at top and bottom with a centered loop and no visible free tail;
  6 has an open upper tail entering a lower loop.
- **5:** has a flat top bar and vertical stem; 6 is a continuous curve into the loop.
- **7/1:** have no enclosed region. Do not hallucinate a lower loop from stroke
  thickness, blur, or a hooked diagonal.
- **8:** has two stacked lobes/holes with a waist or central pinch. A 6 has only
  one lower hole, and its upper portion remains open to the outside rather than
  closing into a second lobe.
