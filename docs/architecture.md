# RUSH foundation architecture

## Repository layout

```text
RUSH/
  web/                         # Static prototype UI
  policy-graph/Generative_AI/  # Obsidian-compatible policy graph versions
  data/seed/                   # Placeholder golden-set, labels, metrics, suggestions
  schemas/                     # Contract-first data objects
  docs/visuals/                # SVG diagrams
  scripts/                     # Validation and future graph compilers
```

## Core loops

1. **Inner loop per item** — image → label votes → arbiter decision → SME queue decision → tiered canonical label.
2. **Middle loop per batch** — review disagreements and stratified consensus samples; update graph metrics and SME QA.
3. **Outer loop per policy version** — propose graph/prompt refinements, evaluate on protected splits, present git-style diffs to SME.

## Contract boundaries

- `PolicyNode` Markdown frontmatter is the source of policy graph truth.
- `ImageRecord` stores media identity and split/dedupe metadata, not truth.
- `LabelVote` stores each human/LLM/arbiter decision, evidence, confidence, and prompt/policy version.
- `SMEReview` promotes or corrects labels into canonical gold/platinum records.
- `PolicyPatch` stores suggested graph changes and SME approval/rejection outcomes.
- `MetricSnapshot` stores versioned metric views by label tier, graph version, and split.

## Statistical guardrails inherited from v0.2

- Do not report final decision quality on provisional LLM-majority labels.
- Audit a stratified sample of 3/3 consensus cases.
- Keep adaptive boundary-discovery batches separate from random sentinel/prevalence batches.
- Preserve dev, validation, locked holdout, boundary holdout, and sentinel splits.
- Exclude validation/holdout examples from prompt/context packs.
