---
id: MD.root
version: MNIST_Digits.v0.1
title: MNIST Handwritten Digit Classification
area: MNIST_Digits
node_type: root
parent: null
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
  - RUSH MNIST demo 2026-07-03: cold-start handwritten digit classification
edges:
  - {type: clarifies, to: MD.digit.8}
canonical_examples: []
---
# MNIST Handwritten Digit Classification

## Decision rule
Assign exactly one label from `{0,1,2,3,4,5,6,7,8,9}` to a 28×28 grayscale
handwritten-digit image, choosing the digit whose stroke topology and geometry
best match the visible glyph.

Prefer the class whose defining features (loops, strokes, crossbars, and their
arrangement) are most fully satisfied. When two classes are plausible, use the
distinguishing feature that separates them (see each digit node's boundary
notes) rather than overall visual similarity. If the glyph is too degraded to
support any single class more than another, abstain rather than guess.

## Boundary rule for apparent single-loop glyphs
For glyphs with one enclosed region, do not decide from “one oval loop” alone.
First inspect whether the loop has an attached tail/curl and where the enclosed
region sits vertically:

- Choose **0** only when the visible mark is essentially a smooth, standalone
  oval/ellipse: no protruding endpoint, no inward hook, no upper or lower tail,
  and the hole is roughly centered within the glyph.
- Choose **6** when the single hole is in the lower half and there is any visible
  open entry stroke, hook, overlap, or tail rising above or into the loop. A 6
  may look almost closed like an oval; the low hole plus upper curl/tail is more
  important than overall ovalness.
- Choose **9** when the single hole is in the upper half and a stroke/tail
  descends from the loop.

Thus, for one-hole digits, tail/curl evidence and hole placement override a
superficial resemblance to a centered 0.

## Label hierarchy
- `digit`
  - [[MD.digit.0|0 — single closed loop]]
  - [[MD.digit.1|1 — single vertical stroke]]
  - [[MD.digit.2|2 — top curve + diagonal + flat base]]
  - [[MD.digit.3|3 — two right-facing stacked bumps]]
  - [[MD.digit.4|4 — two verticals joined by a crossbar]]
  - [[MD.digit.5|5 — top bar, stem, lower bowl]]
  - [[MD.digit.6|6 — curl into a bottom loop]]
  - [[MD.digit.7|7 — top bar + descending diagonal]]
  - [[MD.digit.8|8 — two stacked closed loops with a pinch]]
  - [[MD.digit.9|9 — top loop with a descending tail]]

## Cold-start stance
This v0.1 graph is intentionally conservative. Each digit node names the
minimal distinguishing features and the sibling digits it is most often
confused with, so examples, justifications, and SME corrections have a clear
place to attach. It is not final policy law from Mount Olympus.
