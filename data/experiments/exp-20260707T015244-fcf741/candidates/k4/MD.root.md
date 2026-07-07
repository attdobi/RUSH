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

## Rasterized handwriting safeguards
MNIST digits are low-resolution and thick strokes can create accidental contacts,
tiny gaps, dark blobs, or faint bridges. Do not decide only by a raw pixel-level
hole count. First infer the intended stroke path, terminals, and major lobes;
then treat very small gaps or contacts as noise when the surrounding geometry is
clear.

Use these boundary rules before choosing among lookalikes:

- **Trace stroke roles before naming the digit.** Identify free endpoints,
  deliberate caps/bases, major rounded lobes, central waists, and tails. Do not
  let one local feature override the whole path: a short top segment is not a 7
  unless the rest is a single diagonal with no lower bowl; a lower horizontal
  terminal is not a 2 base unless it follows a rounded top curve plus descending
  diagonal; a rounded bump is not a 9 loop unless it genuinely encloses the
  upper region.
- **One-loop boundary: 0 versus 6/9.** A 0 is a closed oval-like loop with no
  clear free tail outside the loop; slant, uneven thickness, or a dark join at
  the top/bottom is not enough to make it a 6 or 9. A 6 or 9 requires a visible
  terminal/curl/tail attached to one side of the loop: loop low with an open
  curl rising above it for 6, loop high with a tail descending below it for 9.
  When a tail is present, classify by loop placement rather than by the fact
  that there is one hole.
- **Terminal strokes beat accidental loops.** A 2 may be compact or cursive: its
  upper curve can nearly close, and nearby strokes can make a loop-like blob. If
  the stroke still reads as a top curve flowing into a descending diagonal and
  ending in a lower horizontal or near-horizontal base/terminal, prefer 2 over
  0, 3, 6, 7, or 9. A 3 lacks the flat lower base; a 7 lacks any bottom base;
  0/6/9 require the loop to be the dominant structure rather than an incidental
  closure along a 2-like path.
- **Top-bar/stem/bowl beats diagonal-only readings for 5.** A handwritten 5 may
  be angular or compressed, and its lower bowl may end in a near-horizontal
  terminal that resembles a base. Prefer 5 when the glyph has a deliberate flat
  or near-flat top cap, a drop/stem on the left side from that cap, and then a
  lower right-facing bowl or hook. Prefer 2 only when the top is a rounded curve
  flowing into a diagonal down to a bottom base, with no left stem; prefer 7
  only when there is a top bar plus one descending diagonal and no lower bowl or
  bottom terminal.
- **Open double-bump boundary: 3 versus 7/9.** A 3 is two right-facing curves
  sharing a right-side spine and staying open on the left. Do not call it 7
  merely because the upper curve has a short flat-looking top; require the 7's
  simple top-bar-plus-diagonal structure with no second lower bump. Do not call
  it 9 unless the upper portion forms a genuine enclosed loop with a descending
  tail rather than an open upper bump followed by a second lower bump.
- **Count intended lobes, not only perfect holes.** For 8, two stacked rounded
  lobes joined by a narrow waist or crossing are decisive even if one enclosed
  region is faint, pinched, partly broken, or closed only by a small contact. Do
  not downgrade to 6 or 9 merely because one lobe is weak; 6 and 9 have one loop
  plus a tail, not two stacked lobes.
- **Crossbar versus central pinch.** For 4, require a deliberate horizontal
  crossbar meeting a straight right stem and forming an angular wedge. A short
  waist, crossing, or pixel bridge between two curved lobes is the central pinch
  of an 8, not a 4 crossbar.
- **Top bar versus upper lobe.** For 5, require a deliberate flat top bar followed
  by a left stem and one lower bowl. If the apparent top bar is actually the
  curved top of a closed or nearly closed upper lobe stacked above a lower lobe,
  prefer 8.
- **Abstain only after applying near-closure reasoning.** Broken or noisy strokes
  are still classifiable when a digit's distinguishing topology can be inferred;
  abstain only when no candidate has a clear distinguishing cue.

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
