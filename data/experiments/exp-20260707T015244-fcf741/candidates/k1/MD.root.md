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

## Robust MNIST reading protocol
Treat the image as a single handwritten digit unless the mark is genuinely
unreadable or there are clearly multiple unrelated glyphs. Do not abstain merely
because a digit is small, low-contrast, slanted, compressed, teardrop-shaped,
flame-like, or letter-like; many MNIST digits are stylized. If enough stroke
topology is visible to prefer one digit, choose that digit.

Use this priority order for common hard boundaries:

1. **Count enclosed regions, then locate them.**
   - Two stacked enclosed regions with a central pinch/crossing is `8`, even if
     the glyph is thin, tilted, or one loop is much smaller.
   - One centered, self-contained loop with no protruding tail/stem is `0`;
     asymmetry or a pointed/teardrop contour does not by itself make it `6` or
     `9`.
   - One low loop plus an open upper curl/tail is `6`.
   - One high loop plus a descending tail is `9`; a lowercase-`q`-like shape is
     still a valid handwritten `9` when this topology is present.
   - A visible tail or stem rules out `0`; a second real hole rules out `0`, `6`,
     and `9`. Do not invent a second hole from a mere waist pinch, and do not
     invent a tail from ordinary oval asymmetry.

2. **Let explicit bars/stems override loop-like similarity.**
   - A distinct horizontal or near-horizontal top stroke followed by a short
     left-side descending stem and a lower right-facing bowl is `5`, even when
     the lower bowl nearly touches or briefly closes due to handwriting blur.
   - Do not relabel such a top-bar-plus-left-stem structure as `0`, `6`, or `9`
     just because part of the lower bowl is rounded or loop-like.
   - `6` lacks a flat top bar/stem sequence; `9` has a rounded upper loop with a
     tail descending from the loop, not a flat top bar with a left stem.

3. **Separate `3` from `5` by the top structure.**
   - Choose `3` when the glyph is two stacked right-facing curves open on the
     left, with a rounded upper bump.
   - Choose `5` only when there is a clear top bar plus descending left stem. A
     short locally flat segment on an otherwise rounded upper curve is not
     enough to make a `5`.

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
