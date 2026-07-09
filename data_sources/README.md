# Data source archives

This directory stores portable source archives that should travel with the repo.
They are not used by the normal quickstart because the unpacked portable fixture
is already committed under `data/images/`, but they preserve the source bundles
needed to rebuild or move the demos cleanly.

## Archives

- `genai-sample.zip` — the committed 132-image GenAI portable sample plus
  `data/images/genai-classification/manifests/combined_labels.portable.jsonl`.
  Extract from the repo root:

  ```bash
  unzip data_sources/genai-sample.zip -d .
  ```

- `mnist_png.zip` — the upstream-style MNIST PNG export recovered from the local
  source archive (`mnist_png/data/f{index}.png` plus labels). Extract it into a
  scratch location, then pass that folder to the sampler:

  ```bash
  unzip data_sources/mnist_png.zip -d /tmp/rush-mnist
  ./.venv/bin/python scripts/sample_mnist_gold_sets.py \
    --source-root /tmp/rush-mnist/mnist_png \
    --n-train 2000 --n-val 500 --seed 20260703 --force
  ```

For routine runs, prefer the committed fixture and `RUSH_PORTABLE=1` instructions
in the root README.
