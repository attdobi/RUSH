# GenAI classification image store

Local-only image workspace for the RUSH GenAI classification pilot.

## Structure

- `raw/` — immutable source images exactly as received.
- `source-datasets/` — imported local source datasets normalized to `{dataset}/{ai_generated,not_ai_generated}/`.
- `intake/` — newly collected images awaiting dedupe/metadata checks.
- `working/` — temporary review batches and relabeling queues.
- `curated/development/` — SME-reviewed development split.
- `curated/validation/` — SME-reviewed validation split.
- `curated/holdout/` — locked holdout split; avoid browsing during model/policy iteration.
- `derived/thumbnails/` — generated previews, resized copies, hashes, crops.
- `manifests/` — local CSV/JSONL manifests mapping image ids to paths, hashes, split, tier, and label state.

## Sampling

Generate deterministic development and holdout label manifests from the local source datasets:

```bash
python3 scripts/sample_genai_gold_sets.py --n-dev 100 --n-holdout 100 --seed 20260510 --force
```

The default sampling is class-balanced and source-stratified. For N=100 per split, each split contains 50 `ai_generated` and 50 `not_ai_generated` labels, distributed as 17/17/16 per class across `sdv1_4`, `midjourney`, and `wfir`.

Current source-label assumptions:

- `0_real` → `not_ai_generated`
- `1_fake` → `ai_generated`
- `sdv1.4/1_false` → `ai_generated` despite the ambiguous source directory name

## Git policy

This directory has its own `.gitignore`: images, manifests, and derived files are ignored by default. Only the folder skeleton, `.gitkeep` files, this README, and ignore rules should be committed until we intentionally add safe public fixtures.
