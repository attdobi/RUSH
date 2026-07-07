---
id: MD.boundary.1_vs_7
version: MNIST_Digits.v0.1
title: Boundary: Digit 1 vs Digit 7
area: MNIST_Digits
node_type: boundary
parent: MD.digit.1
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
  - RUSH MNIST demo 2026-07-07: 1-vs-7 boundary clarification
edges:
  - {type: confused_with, to: MD.digit.1}
  - {type: confused_with, to: MD.digit.7}
canonical_examples: []
---
# Boundary: Digit 1 vs Digit 7

## Scope
Use this boundary when the glyph has no loops and no bottom base, and the only
question is whether a slanted single stroke is a 1 or a 7.

## Boundary rule
Prefer **1** when the mark is one continuous mostly vertical or diagonal stroke
with one dominant direction. A slight top hook, serif, thickened cap, or tapered
endpoint does not by itself create a 7.

Prefer **7** only when there is a distinct top bar: a visible horizontal or
near-horizontal segment running left-to-right that meets a long descending
diagonal at a clear corner/angle. A 7 has two principal stroke directions; a
plain diagonal slash has only one.

## Checks
- Slant alone does not turn a 1 into a 7.
- Do not infer a 7's top bar from endpoint thickness, antialiasing, or the
  short cap at the top of a single stroke.
- A 1's top flag is short and attached to the main stroke; it does not form a
  roof-like horizontal span.
- The absence of a bottom base separates 7 from 2, but it does not decide 7 over
  1 when the top bar is missing.
