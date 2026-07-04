# MNIST Benchmark Anchors (V0)

> Reference figures for the MNIST_Digits demo UX and scorecards.
> Version: **v0** · Companion to
> [`docs/mnist-prompt-v0.md`](mnist-prompt-v0.md).
>
> **Do not fabricate model scores.** This file lists only externally
> cited figures we can point at in the UI. RUSH's own labeler numbers
> go in `data/runs/**/scoring/decision_quality_multiclass.json`; the
> "RUSH — pending" rows here are placeholders and MUST NOT be filled
> in from imagination.

## 1. Task

- Dataset: MNIST 10k-image test set (28×28 grayscale handwritten
  digits, ten balanced classes `0`–`9`).
- Metric: top-1 accuracy on the test set, reported as either
  **accuracy %** or **error % = 100 − accuracy**.
- RUSH counterpart: macro accuracy from
  `pipeline/scoring/decision_quality_multiclass.py` on the MNIST
  multiclass task (`pipeline/scoring/tasks.py::MNIST_MULTICLASS`).

## 2. Anchor figures

| Anchor                                      | Error % | Accuracy % | Notes                                                                                           |
| ------------------------------------------- | ------- | ---------- | ----------------------------------------------------------------------------------------------- |
| Human labelers (legacy USPS/MNIST-adjacent) | 2.50 %  | 97.50 %    | Wikipedia reports a two-human average error rate from the older USPS benchmark lineage.¹         |
| Original LeCun et al. SVM baseline          | 0.80 %  | 99.20 %    | Original 1998 MNIST paper (support-vector machine).²                                            |
| Best single CNN (no augmentation)           | 0.25 %  | 99.75 %    | Reported best single-CNN error rate on MNIST as of 2016/2018 per the Wikipedia summary.³        |
| Committee of 35 CNNs (Cireşan+2012)         | 0.23 %  | 99.77 %    | "Multi-column deep neural networks for image classification" — a canonical near-human anchor.   |
| Ensemble/regularization band                | ~0.21 % | ~99.79 %   | DropConnect (2013) and 5-CNN ensembles reported about 0.21 % on the compiled table.³            |
| 2020 simple-CNN two-layer ensemble          | 0.09 %  | 99.91 %    | An et al. report up to 99.91 % top-1 test accuracy, one of the strongest published MNIST claims.⁴ |
| **RUSH v0 (this branch) — pending**         | —       | —          | Populate from the run manifest once labeling finishes; do not guess.                            |

¹ Wikipedia, ["MNIST database" § history / human error rate](https://en.wikipedia.org/wiki/MNIST_database), retrieved 2026-07-03. This is not a fresh modern crowd-human study; it is a legacy two-human benchmark reference from the USPS/MNIST lineage.

² LeCun, Bottou, Bengio, Haffner (1998), *Gradient-Based Learning Applied to Document Recognition* — the original MNIST paper. Cited via the same Wikipedia article.

³ Wikipedia, ["MNIST database" § performance table](https://en.wikipedia.org/wiki/MNIST_database), retrieved 2026-07-03. The compiled table lists a best single-CNN error of **0.25 %** (2016/2018), a **35-CNN committee at 0.23 %** (Cireşan, Meier, Schmidhuber 2012), and ≈**0.21 %** for DropConnect / 5-CNN ensembles.

⁴ Sanghyeon An, Minjun Lee, Sanglee Park, Heerin Yang, Jungmin So, ["An Ensemble of Simple Convolutional Neural Network Models for MNIST Digit Recognition"](https://arxiv.org/abs/2008.10400), arXiv:2008.10400. The abstract reports 99.87 % for majority voting across three models and up to **99.91 %** for a two-layer ensemble.

## 3. How to use these in the UX

- **Scale the y-axis honestly.** MNIST accuracy differences live in the
  0.2 %–2.5 % band; a chart that runs 0 %–100 % squashes every
  meaningful comparison. Prefer error-% on a log or clipped scale
  (e.g. 0 %–5 %).
- **Anchor lines, not bars.** Draw the legacy human-reference (2.5 %
  err), canonical CNN ensemble (~0.23 % err), and current high-water
  mark (0.09 % err) as reference lines behind whatever RUSH plots for
  its own labelers.
- **Never fabricate a "RUSH %" number.** Leave the pending row empty
  until a real scoring run writes
  `decision_quality_multiclass.json`; the demo UX should read from
  that artifact, not from this doc.
- **Cite the source in-tooltip.** When surfacing the 99.91 % SOTA
  figure, link to An et al. 2020. When surfacing the human-reference
  figure, link to the Wikipedia MNIST article and label it as a legacy
  two-human reference, not as modern consensus.

## 4. Caveats

- SOTA on MNIST is effectively saturated; treat any number below the
  ~0.2 % error band with skepticism unless a first-party paper is
  cited. Do not chase leaderboard scraps for the demo — the
  human/SOTA/RUSH gap is the interesting story.
- The 2.5 % human figure is a legacy two-labeler average, not a
  large-sample modern human study; treat it as an order-of-magnitude
  anchor, not a precise floor. Some secondary summaries describe MNIST
  saturation as "human-level" around 99.7-99.8 %, but this doc does
  not use that as a sourced benchmark row.
- These figures refer to the standard 10k MNIST test set. RUSH samples
  a 500-image evaluation subset (see
  `scripts/` MNIST sampler + `data/mnist-classification`), so RUSH's
  error bars will be wider than the leaderboard numbers imply.

## 5. Change log

- **v0 (2026-07-03)** — initial benchmark anchor set for the
  `feat/mnist-ux-polish` branch. Human figure from Wikipedia's MNIST
  article (which sources Bottou et al. 1994). Model figures pulled
  from the Wikipedia MNIST performance table plus An et al. 2020
  (retrieved 2026-07-03). RUSH's own numbers intentionally left
  pending.
