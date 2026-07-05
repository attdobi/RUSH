# MNIST classification image store

Local image workspace for the RUSH MNIST digit-classification demo. Mirrors the
`genai-classification/` layout, adapted for a 10-way multiclass (digits 0–9)
problem.

## Structure

- `source-datasets/mnist/<digit>/f{index}.png` — sampled source PNG payloads,
  organized by ground-truth digit (`0`..`9`). Bulk payloads are git-ignored.
- `thumbnails/` — generated previews / resized copies (git-ignored).
- `manifests/` — deterministic CSV label manifests, runner-ready
  `combined_labels.jsonl`, and `sampling_summary.json`. These are small and
  regenerable, and ARE committed for the demo.

## Source data

The upstream MNIST export lives outside the repo at `~/Downloads/mnist_png/`:

- `data/f{N}.png` — 70,000 flat PNGs, `N = 0..69999`.
- `labels_and_paths.csv` — columns `[index, label, path]`.

**Note:** the CSV `path` column is unreliable and is intentionally ignored. The
payload for a row is always resolved as `~/Downloads/mnist_png/data/f{index}.png`,
keyed by the CSV row index. Only `index` and `label` are trusted.

The split is index-based (verified): index `0–59999` = **train**,
`60000–69999` = **val**. Samples are never reshuffled across that boundary.

## Sampling / regenerate

Generate the deterministic train/val label manifests and copy the sampled PNG
payloads into `source-datasets/mnist/<digit>/`:

```bash
python3 scripts/sample_mnist_gold_sets.py \
  --n-train 2000 --n-val 500 \
  --seed 20260703 \
  --source-root ~/Downloads/mnist_png \
  --force
```

Defaults produce a class-balanced (stratified) sample: 2000 train (200 per
digit) and 500 val (50 per digit), `seed=20260703`,
`sampling_version="mnist-sampling-v1"`.

### Manifest formats

`sample_id,dataset,label,label_int,repo_rel_path,original_filename,file_ext,sha256,split,seed,sampling_version,source_index,truth_tier,policy_use`

- `label` — digit as string (`"0"`..`"9"`); `label_int` — integer `0..9`.
- `sample_id` — `train_00001` (5-digit) / `val_0001` (4-digit).
- `sha256` — hash of the copied payload (matches the upstream source byte-for-byte).
- `source_index` — original CSV/file index (`f{index}.png`).
- `policy_use` — `develop_policy` (train) / `validation_decision_quality` (val).

`combined_labels.jsonl` is the manifest consumed by
`scripts/run_bulk_labeling.py` and the scoring pipeline. It contains the same
fields as the CSV rows, one JSON object per line, with `split` mapped to the
runner's canonical selectors: `train -> dev_golden` and `val -> holdout`.
The original CSV split is preserved as `source_split`.

## Git policy

This directory has its own `.gitignore`: bulk image payloads and thumbnails are
ignored by default. Only the folder skeleton, `.gitkeep` files, this README,
ignore rules, and the deterministic `manifests/` (CSV + JSON) are committed.
Regenerate the images from the upstream export using the sampler above.
