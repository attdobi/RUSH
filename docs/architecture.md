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

- `PolicyNode` Markdown frontmatter is the source of policy-node truth; `PolicyEdge` is the machine-readable edge contract.
- `ImageRecord` stores media identity and split/dedupe metadata, not truth.
- `SplitAssignment` records how an item entered development, validation, locked holdout, boundary holdout, sentinel, or adaptive batches and carries leakage guards.
- `LLMOutputFormat` defines the minimal structured model response: top-level label, L2 policy label, justification, confidence, difficulty, and boundary flag.
- `LabelVote` stores each human/LLM vote with evidence, confidence, and prompt/policy version.
- `ArbiterDecision` reconciles vote records but remains provisional until SME review.
- `SMEReview` promotes or corrects labels; `LabelTierRecord` captures canonical gold/platinum truth and tier history.
- `PolicyPatch` stores suggested graph changes and SME approval/rejection outcomes.
- `MetricSnapshot` stores versioned metric views by truth tier, graph version, split, denominators, confidence intervals, macro metrics, calibration, and graph-location metrics.
- `DecisionQuality` stores per-labeler quality against `sme_latest` truth for humans, LLMs, and majority-vote baselines.
- `ExportRecord` documents downstream exports and which tiers/splits were included or excluded.

## Statistical guardrails inherited from v0.2

- Do not report final decision quality on provisional LLM-majority, arbiter, or silver labels. Reportable truth is gold/platinum SME-reviewed data only.
- Audit a stratified sample of 3/3 consensus cases; consensus is evidence about difficulty, not proof of correctness.
- Keep adaptive boundary-discovery batches separate from random sentinel/prevalence batches so boundary mining does not contaminate prevalence estimates.
- Preserve dev, validation, locked holdout, boundary holdout, and sentinel splits with dedupe-cluster exclusivity.
- Exclude validation/holdout/sentinel examples from prompt/context packs and graph-tuning loops to avoid split leakage.
- Require denominators and uncertainty intervals next to headline metrics; hide or null metrics when sample sizes are insufficient.
- Track graph-location metrics by node/edge so policy gaps are visible without treating every model error as a new policy rule.
