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
  - {type: confused_with, to: MD.digit.6}
  - {type: confused_with, to: MD.digit.9}
canonical_examples: []
---
# Digit 8

## Positive criteria
A figure-eight glyph: two lobes or chambers separated by a central pinch, crossing, or waist. In clean writing this gives two enclosed regions (two holes), one above the other. In thick, low-resolution, cursive, or slanted MNIST writing, one lobe may be very small, diagonal, nearly closed, or partly filled by stroke thickness; still classify as 8 when the stroke path visibly makes two closed-or-pinched lobes rather than one plain oval plus a tail.

## Distinguishing features
- The decisive feature is a two-lobed topology with a waist, pinch, or crossing between lobes. The lobes may be vertically stacked or noticeably slanted/diagonal.
- Versus 0: 0 has one continuous oval with no waist; 8 has a neck/pinch dividing the outline into two chambers even if one chamber is tiny, faint, or partly filled.
- Versus 6/9: 6 and 9 have one loop plus a free external curl or tail. For 8, the apparent extra stroke curves back into the glyph to form the second lobe/chamber instead of ending as a tail.
- Versus 3: a 3 has two right-facing curves that remain open on the left; 8 closes or nearly closes both lobes around a central pinch.

## Hard negatives / confusions
- **3:** two right-facing bumps that stay open on the left; do not call it 8 unless the left side closes/near-closes into lobes.
- **0:** a single centered loop; roughness or slant alone is not an 8 without a waist/pinch/crossing.
- **6/9:** a single low or high loop with one free tail; require a second returned lobe/chamber for 8.
- A tiny, compressed, or diagonal 8 may show only one obvious hole at first glance. Inspect for a second lobe bounded by stroke and a central waist before choosing a single-loop digit.
