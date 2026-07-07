---
id: MD.boundary.0_vs_6
version: MNIST_Digits.v0.1
title: Boundary: Digit 0 vs Digit 6
area: MNIST_Digits
node_type: boundary
parent: MD.digit.0
polarity: mixed
status: draft
coverage_weight: 1.0
coverage_target:
  easy_positive: 20
  hard_positive: 20
  easy_negative: 20
  hard_negative: 20
  platinum_min: 5
source_anchors:
  - RUSH MNIST cycle 2: 0-vs-6 boundary refinements
edges:
  - {type: confused_with, to: MD.digit.0}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Boundary: Digit 0 vs Digit 6

## Core distinction
Both 0 and 6 can have exactly one enclosed region. The decisive separator is
not whether the oval is tilted, uneven, or visually bottom-heavy; it is whether
there is a distinct open curl/tail outside the loop.

- Choose **0** when the visible glyph is a single continuous closed oval/ellipse
  with one hole and no separate protruding hook, tail, or open entry stroke.
- Choose **6** only when there is a lower closed loop plus a visible open stroke
  rising above or entering into that loop, so the top of the digit is an open
  curl rather than a completed oval contour.

## Do not overcall 6
A 0 may be slanted, narrow, thick on one side, irregularly closed, or have its
ink mass slightly lower than center. These variations do **not** make it a 6 by
themselves. Do not infer an “open upper curl” from stroke thickness, tapering,
rasterization gaps, or a seam-like join if the overall contour still reads as a
closed single-loop oval.

## Evidence required for 6
Before labeling a one-hole glyph as 6, verify at least one clear 6-specific cue:

- an upper hook or tail that extends outside the closed loop;
- a visibly open top where the stroke curls down into a lower loop rather than
  completing an oval;
- a loop whose enclosed hole is clearly in the lower half **and** an attached
  open stroke continues upward beyond it.

Hole position alone is not decisive: a low-looking or bottom-heavy oval remains
0 unless the open curl/tail is actually visible.

## Practical trace test
Mentally trace the outside contour.

- If the contour returns to itself as one closed oval with no extra stroke, label
  **0**.
- If the trace forms a lower loop and then continues as an unclosed upper hook or
  tail, label **6**.

Always choose the more defensible of 0 or 6; express residual ambiguity with
confidence and difficulty, not by abstaining.