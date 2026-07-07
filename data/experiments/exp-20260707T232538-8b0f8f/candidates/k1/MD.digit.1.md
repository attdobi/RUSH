---
id: MD.digit.1
version: MNIST_Digits.v0.1
title: Digit 1
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
  - RUSH MNIST demo 2026-07-03: digit-1 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.7}
  - {type: confused_with, to: MD.digit.2}
  - {type: confused_with, to: MD.digit.6}
canonical_examples: []
---
# Digit 1

## Positive criteria
A single, mostly vertical or slightly slanted stroke. May include a short upward-left flag/serif at the top and/or a short horizontal base serif at the bottom. No loops and no enclosed regions.

A lone diagonal slash with no separate bar, bowl, or loop is still a 1: MNIST 1s may lean noticeably left or right and may be written as one plain stroke.

## Distinguishing features
- No horizontal top bar spanning to the right (that would indicate 7). A small cap, hook, taper, or pixel-thickening at the top of a single stroke is only a serif unless it forms a distinct left-to-right bar joined to a long descending diagonal.
- No bottom curve or bowl and no diagonal reaching a flat base (that would indicate 2).
- No enclosed white interior at all. Do not infer a 6, 0, 8, or 9 from blur, stroke thickness, or a slight hook unless a visible closed loop is actually present.

## Hard negatives / confusions
- **7:** has a distinct horizontal top stroke and a longer descending diagonal; a 1 with a large top flag can mimic this — check for a true horizontal bar rather than a short serif on a single slash.
- **2:** shares a diagonal descent but adds a top curve and a flat horizontal base.
- **6/looped digits:** require a genuine enclosed region; a simple slanted stroke or tiny end hook remains a 1.
- A perfectly straight thin stroke may resemble the vertical spine of other digits; require the absence of loops/bars.
