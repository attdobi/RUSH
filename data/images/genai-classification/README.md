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

Generate the deterministic three-way label manifests from the local source
datasets. **The defaults ARE the canonical splits** — dev_golden 2000,
holdout 1000, validation 200, seed 20260510 — so the no-argument invocation
produces byte-identical manifests on every machine with the source tree
(this is how the dev MacBook and the Mac mini stay aligned):

```bash
python3 scripts/sample_genai_gold_sets.py --force
```

The `validation` split is the FIXED cross-run benchmark (mirrors MNIST's
`bench_*` rows): the readout the experiment crank scores under the start +
final policy. It is drawn AFTER dev+holdout from the same seeded shuffle, so
re-running with the SAME seed and a larger `--n-validation` keeps every
existing dev_golden/holdout assignment identical. The web run form reads
split sizes via `GET /api/area-stats`: the benchmark checkbox enables itself
once validation rows exist, and the Test/Train defaults scale to the
dev_golden pool.

The default sampling is class-balanced and source-stratified. For N=100 per split, each split contains 50 `ai_generated` and 50 `not_ai_generated` labels, distributed as 17/17/16 per class across `sdv1_4`, `midjourney`, and `wfir`.

Current source-label assumptions:

- `0_real` → `not_ai_generated`
- `1_fake` → `ai_generated`
- `sdv1.4/1_false` → `ai_generated` despite the ambiguous source directory name

## Git policy

This directory has its own `.gitignore`: images, manifests, and derived files are ignored by default. Only the folder skeleton, `.gitkeep` files, this README, and ignore rules should be committed until we intentionally add safe public fixtures.

**Exception (committed portable fixture):** a balanced 900-image sample (150 per
dataset×class, both splits, JPEG derivatives at longest edge ≤1024px / quality
82, ~89 MiB) under `sample/`, plus
`manifests/combined_labels.portable.jsonl`, are committed so the GenAI demo runs
from a fresh clone without the ~12 GB source tree. Build it from a larger local
source manifest with:

```bash
python3 scripts/sample_genai_gold_sets.py \
  --n-dev 500 --n-holdout 500 --seed 20260510 \
  --out-dir .tmp-genai-portable-manifest --force
python3 scripts/build_portable_fixture.py \
  --manifest .tmp-genai-portable-manifest/combined_labels.jsonl \
  --max-mb 90 --per-stratum 150 \
  --encode-jpeg --jpeg-max-edge 1024 --jpeg-quality 82
```

The portable manifest hashes the committed JPEG derivatives in `sha256` and
keeps the full-source identity in `source_sha256` / `source_repo_rel_path`.

## Run / full dataset on another machine (e.g. Mac Pro)

The committed `sample/` fixture is auto-selected when the full `source-datasets/`
image tree is absent, or force it with `RUSH_PORTABLE=1`:

```bash
RUSH_PORTABLE=1 python3 scripts/run_bulk_labeling.py \
  --area Generative_AI --models openai/gpt-5.5 --split dev_golden --limit 5
```

### Where to find the full dataset

The full ~12 GB source tree (~20k raw images) is **not** in git. Byte-exact
copies live on the **Mac mini** at
`data/images/genai-classification/source-datasets/{midjourney,sdv1_4,wfir}/{ai_generated,not_ai_generated}/`.
Pull it for full parity, then regenerate the gold manifests:

```bash
rsync -avh <mac-mini-host>:/Users/sacsimoto/GitHub/RUSH/data/images/genai-classification/source-datasets/ \
  data/images/genai-classification/source-datasets/
python3 scripts/sample_genai_gold_sets.py --n-dev 100 --n-holdout 100 --seed 20260510 --force
```

Dataset identities: `midjourney` = Midjourney-generated vs real; `sdv1_4` =
Stable Diffusion v1.4 generated vs real; `wfir` = StyleGAN faces (“Which Face Is
Real”-style) vs real. No single canonical public URL is recorded in-repo; the Mac
mini tree is the source of truth.
