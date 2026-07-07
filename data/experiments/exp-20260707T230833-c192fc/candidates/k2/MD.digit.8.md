---
id: MD.digit.8
version: MNIST_Digits.v0.1
title: Digit 8
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
  - RUSH MNIST demo 2026-07-03: digit-8 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.3}
  - {type: confused_with, to: MD.digit.0}
canonical_examples: []
---
# Digit 8

## Positive criteria
A figure-eight glyph: two rounded lobes joined at a central pinch/crossing or
very narrow waist. In clean cases the lobes are stacked vertically and close
into two enclosed regions (two holes), with a smaller upper loop and a usually
larger lower loop. In low-resolution, small, cursive, or slanted writing, the
lobes may be diagonal/narrow and one hole may appear as only a slit or partial
closure; still require two distinct lobes organized around the central waist,
not just a single oval or a single loop with a tail.

## Distinguishing features
- Two enclosed regions, when visible, are decisive for 8 versus one-hole digits
  (0, 6, 9) or no-hole digits (3, 5, 7).
- When rasterization or pen breaks make a hole faint, use the figure-eight
  structure: a self-crossing/pinch in the middle with stroke mass curving into
  both an upper and a lower lobe. Do not reduce such a glyph to 0, 6, or 9 only
  because one closure is weak.
- A single smooth oval with no middle waist remains 0.
- A single high or low loop with an unlooped descending/ascending tail remains
  9 or 6; 8 needs a second rounded lobe on the other side of the waist.
- Both loops/lobes close or nearly close on the left; if the left side stays
  open and there is no central crossing/pinch, it is a 3.

## Hard negatives / confusions
- **3:** two right-facing bumps that stay open on the left; 8 closes or nearly
  closes both into lobes around a central waist.
- **0:** a single centered loop; 8 has a second lobe and a central pinch/crossing.
- **6/9:** single low or high loop respectively with a tail. Prefer 8 only when
  a second rounded lobe is visible, even if one hole is tiny or partly broken.
- **7:** has a top bar and descending diagonal with no lobes or waist; a knotted
  or crossed two-lobe shape is not a 7.
