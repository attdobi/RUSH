#!/usr/bin/env python3
"""Materialize the FIXED cross-run MNIST validation (benchmark) split.

Attila 2026-07-06: per-run test partitions are seeded from dev_golden, so
run-to-run scores are not directly comparable — and the loop adapts to its
gate set. This script fixes ONE canonical validation set for every run:

  * drawn from the canonical MNIST test images (NPZ indices 60000-69999,
    ``source_split='val'``) — never from dev_golden, so it can never overlap
    a run's train batches or gate partition;
  * disjoint from every source_index already in the manifest (the 500-image
    locked holdout also lives in that range);
  * stratified 10 x n/10, seeded once, appended to combined_labels.jsonl as
    ``split='validation'`` — LOCKED like the holdout (bulk passes require
    --allow-holdout; ``--split all`` never includes it).

Score it per run with ``run_experiment.py --validation-final``; cross-run
deltas on this split are the paper's benchmark numbers.

Idempotent: refuses to run if validation records already exist (the whole
point is that the split never changes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
MNIST_ROOT = REPO_ROOT / "data" / "images" / "mnist-classification"
DEFAULT_ARCHIVE = MNIST_ROOT / "mnist_full.npz"
DEFAULT_MANIFEST = MNIST_ROOT / "manifests" / "combined_labels.jsonl"
IMAGES_ROOT = MNIST_ROOT / "source-datasets" / "mnist"
VAL_RANGE = range(60_000, 70_000)  # canonical MNIST test set inside the NPZ
SAMPLING_VERSION = "mnist-validation-v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000,
                    help="validation set size (stratified n/10 per digit).")
    ap.add_argument("--seed", type=int, default=20260706,
                    help="fixed seed — the split must be minted exactly once.")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 10 or args.n % 10 != 0:
        print("[validation] --n must be a positive multiple of 10", file=sys.stderr)
        return 2

    manifest_lines = [line for line in args.manifest.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    existing = [json.loads(line) for line in manifest_lines]
    if any(rec.get("split") == "validation" for rec in existing):
        print("[validation] manifest already has a validation split — it is fixed by design; "
              "refusing to mint another.", file=sys.stderr)
        return 1
    used_indices = {int(rec["source_index"]) for rec in existing if "source_index" in rec}

    with np.load(args.archive) as data:
        images = data["images"]
        labels = data["labels"]

    per_digit = args.n // 10
    chosen: list[int] = []
    for digit in range(10):
        pool = [i for i in VAL_RANGE if int(labels[i]) == digit and i not in used_indices]
        if len(pool) < per_digit:
            print(f"[validation] digit {digit}: only {len(pool)} unused canonical-test "
                  f"images (< {per_digit})", file=sys.stderr)
            return 1
        rng = random.Random(f"{args.seed}:validation:{digit}")
        chosen.extend(sorted(rng.sample(pool, per_digit)))

    records = []
    for position, index in enumerate(sorted(chosen), start=1):
        digit = int(labels[index])
        out_path = IMAGES_ROOT / str(digit) / f"f{index}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic content — always (re)write atomically, so a crashed
        # previous attempt can never leave a truncated payload behind.
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        Image.fromarray(images[index], mode="L").save(tmp_path, format="PNG")
        os.replace(tmp_path, out_path)
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        records.append({
            "dataset": "mnist",
            "file_ext": "png",
            "label": str(digit),
            "label_int": digit,
            "original_filename": f"f{index}.png",
            "policy_use": "benchmark_cross_run",
            "repo_rel_path": out_path.relative_to(REPO_ROOT).as_posix(),
            "sample_id": f"bench_{position:04d}",
            "sampling_version": SAMPLING_VERSION,
            "seed": args.seed,
            "sha256": sha256,
            "source_index": index,
            "source_split": "val",
            "split": "validation",
            "truth_tier": "gold_candidate",
        })

    # Atomic manifest rewrite (existing rows + new rows, tmp + rename): a
    # crash mid-write must never leave a half-appended split that the
    # idempotency guard would then lock in.
    tmp_manifest = args.manifest.with_name(args.manifest.name + ".tmp")
    with tmp_manifest.open("w", encoding="utf-8") as fh:
        for line in manifest_lines:
            fh.write(line + "\n")
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    os.replace(tmp_manifest, args.manifest)

    print(f"[validation] minted {len(records)} fixed benchmark images "
          f"(seed {args.seed}, {per_digit}/digit, canonical MNIST test rows, "
          f"disjoint from dev_golden + holdout) -> split='validation'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
