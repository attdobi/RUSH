---
id: MD.digit.5
version: MNIST_Digits.v0.1
title: Digit 5
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
  - RUSH MNIST demo 2026-07-03: digit-5 feature seed
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.6}
  - {type: confused_with, to: MD.digit.3}
canonical_examples: []
---
# Digit 5

## Positive criteria
A top-led glyph with an upper cap/bar, a short descending left stem or neck, and
a single lower right-facing bowl/curve that curls back toward the stem. The top
cap is often horizontal, but in handwritten MNIST it may be short, slanted,
slightly curved, or hook-like; do not require it to be ruler-straight if it still
functions as the top of a 5.

## Distinguishing features
- The decisive structure is **top cap + left stem/neck + one lower bowl**. A
  stylized 5 may look S-shaped, but its upper part is a cap or hook feeding a
  stem, not a full second rounded lobe.
- Versus **3**, look for whether the upper mark is a top cap/stem (5) or a true
  upper right-facing bump sharing a right-side spine with the lower bump (3).
  If the left side contains a descending stem/neck before the lower bowl, favor 5;
  if it is simply two open right-facing lobes with no cap/stem, favor 3.
- Versus **6**, a lower bowl that nearly closes or pinches does not by itself
  make a 6. Favor 5 when a distinct upper cap/bar or hook and left stem are
  visible above the bowl. Favor 6 only when the glyph is a continuous curl into a
  lower enclosed loop with no top-cap-plus-stem construction.

## Hard negatives / confusions
- **6:** is a single continuous curl into a closed bottom loop with no straight or
  hook-like top cap; 5 begins from a top cap and stem before forming the lower
  bowl.
- **3:** has two right-facing bumps and a rounded top; 5 has an upper cap/stem
  and a single lower bowl, even when written in a cursive S-like style.
- A rounded-top or looped-bowl 5 can approach 3 or 6 — decide by the presence of
  the top-led cap/stem structure rather than by overall visual similarity alone.
