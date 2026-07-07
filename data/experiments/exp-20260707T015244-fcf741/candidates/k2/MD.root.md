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

## Inspection protocol: topology before resemblance
When local stroke shape and whole-digit topology appear to conflict, decide in
this order:

1. **Trace the intended stroke.** Judge the dark handwritten mark, allowing for
   MNIST blur, antialiasing, uneven thickness, and tiny accidental gaps. Count a
   closure when the stroke visually surrounds a light interior pocket. Do not
   invent closures across clear open sides, but do count very small, skinny, or
   partially filled pockets as holes.
2. **Count enclosed regions before naming the digit.**
   - **Two vertically stacked enclosed pockets** separated by a waist, pinch, or
     crossing indicate **8**. The two holes may be unequal, skewed, or one may be
     much smaller/fainter; a continuous pen stroke, apparent tail, or locally
     flat segment does not make it 6, 9, or 5 if two closed lobes are present.
   - **Exactly one enclosed pocket** should be classified by its placement and
     attachments: centered with no tail is 0; low with an open upper curl is 6;
     high with a descending tail is 9.
   - **No enclosed pockets** shifts the decision to stroke grammar: vertical
     stroke for 1; top curve + diagonal + bottom base for 2; two open right-facing
     bumps for 3; crossbar/wedge for 4; top bar + stem + lower bowl for 5; top
     bar + descending diagonal for 7.
3. **Use decisive boundary strokes, not overall similarity.** A narrow or slanted
   closed oval remains 0 rather than 1 or 5. A rounded top plus diagonal that
   lands on a flat or near-horizontal bottom base is 2 rather than 3 or 7, even
   if the base is short or slightly curved. A true 3 has two open right-facing
   bumps and no bottom baseline.

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
