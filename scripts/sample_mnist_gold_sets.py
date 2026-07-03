#!/usr/bin/env python3
"""Sample deterministic MNIST digit-classification train/val label manifests.

Mirrors scripts/sample_genai_gold_sets.py conventions (stdlib-only, argparse,
deterministic seed, sha256, ASSUMPTION notes, sampling_summary.json) but adapts
the binary GenAI schema to a 10-way multiclass digit problem.

The upstream MNIST export ships as flat PNGs at ``<source>/data/f{index}.png``
(N = 0..69999) plus a ``labels_and_paths.csv`` with columns [index, label, path].

ASSUMPTION: the CSV ``path`` column is unreliable and is intentionally ignored.
The real payload for a given row is always ``<source>/data/f{index}.png``, keyed
by the CSV row index. We trust the ``index`` and ``label`` columns only.

ASSUMPTION: the canonical MNIST split boundary is index-based and verified —
index 0..59999 are training samples and index 60000..69999 are validation
samples. We do not reshuffle across that boundary.

This script copies the sampled PNG payloads into the ignored RUSH image store
and writes ignored manifest files. It does not write image bytes/base64 into
manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SAMPLING_VERSION = "mnist-sampling-v1"
DATASET = "mnist"
DIGITS = tuple(str(d) for d in range(10))  # "0".."9"

# Verified index-based split boundary (see module docstring ASSUMPTION note).
TRAIN_INDEX_RANGE = (0, 59999)
VAL_INDEX_RANGE = (60000, 69999)

ASSUMPTION_NOTES = {
    "csv_path_ignored": (
        "Assumption: labels_and_paths.csv 'path' column is unreliable and ignored; "
        "payload resolved as <source>/data/f{index}.png keyed by CSV row index."
    ),
    "index_split_boundary": (
        "Assumption: verified index-based split — 0..59999 train, 60000..69999 val; "
        "no reshuffling across the boundary."
    ),
}


@dataclass(frozen=True)
class Candidate:
    source_index: int
    label: str
    label_int: int
    split: str
    src_path: Path
    original_filename: str
    file_ext: str
    sha256: str


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def stable_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def split_evenly(total: int, buckets: Iterable[str]) -> dict[str, int]:
    names = list(buckets)
    base, remainder = divmod(total, len(names))
    return {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(names)}


def split_for_index(index: int) -> str | None:
    if TRAIN_INDEX_RANGE[0] <= index <= TRAIN_INDEX_RANGE[1]:
        return "train"
    if VAL_INDEX_RANGE[0] <= index <= VAL_INDEX_RANGE[1]:
        return "val"
    return None


def collect_candidates(source_root: Path) -> dict[tuple[str, str], list[Candidate]]:
    """Read the CSV and group resolvable candidates by (split, digit)."""
    csv_path = source_root / "labels_and_paths.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")
    data_dir = source_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    grouped: dict[tuple[str, str], list[Candidate]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        for row in reader:
            if not row or len(row) < 2:
                continue
            # Column 0 = index, column 1 = label. Column 2 (path) intentionally ignored.
            index = int(row[0])
            label_int = int(row[1])
            if not 0 <= label_int <= 9:
                raise ValueError(f"Unexpected label {label_int!r} at index {index}")
            split = split_for_index(index)
            if split is None:
                raise ValueError(f"Index {index} outside known split ranges")
            label = str(label_int)
            src_path = data_dir / f"f{index}.png"  # CSV path column deliberately ignored.
            if not src_path.exists():
                raise FileNotFoundError(f"Resolved payload missing: {src_path}")
            key = (split, label)
            grouped.setdefault(key, []).append(
                Candidate(
                    source_index=index,
                    label=label,
                    label_int=label_int,
                    split=split,
                    src_path=src_path,
                    original_filename=src_path.name,
                    file_ext=src_path.suffix.lower().lstrip("."),
                    sha256=sha256_file(src_path),
                )
            )
    return grouped


def dedupe_by_hash(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for candidate in candidates:
        if candidate.sha256 in seen:
            continue
        seen.add(candidate.sha256)
        deduped.append(candidate)
    return deduped


def allocate_digit_counts(n: int) -> dict[str, int]:
    if n <= 0:
        raise ValueError("N must be positive")
    return split_evenly(n, DIGITS)


def sample_records(
    grouped: dict[tuple[str, str], list[Candidate]],
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[list[Candidate], list[Candidate], dict[str, object]]:
    rng = random.Random(seed)
    per_split_counts = {
        "train": allocate_digit_counts(n_train),
        "val": allocate_digit_counts(n_val),
    }
    picked: dict[str, list[Candidate]] = {"train": [], "val": []}
    source_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "val"):
        for digit in DIGITS:
            key = (split, digit)
            candidates = dedupe_by_hash(grouped.get(key, []))
            # Deterministic order before shuffle: sort by source_index.
            candidates.sort(key=lambda c: c.source_index)
            need = per_split_counts[split][digit]
            if len(candidates) < need:
                raise ValueError(
                    f"Not enough unique files for {split}/{digit}: need {need}, have {len(candidates)}"
                )
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            chosen = shuffled[:need]
            picked[split].extend(chosen)
            source_counts[f"{split}/{digit}"] = {
                "available_unique": len(candidates),
                "sampled": len(chosen),
            }

    # Deterministic ordering for sample_id assignment.
    for split in ("train", "val"):
        picked[split].sort(key=lambda c: (c.label_int, c.source_index))

    verify_disjoint(picked["train"], picked["val"])

    summary = {
        "sampling_version": SAMPLING_VERSION,
        "dataset": DATASET,
        "seed": seed,
        "n_train": len(picked["train"]),
        "n_val": len(picked["val"]),
        "allocation": {
            "train": per_split_counts["train"],
            "val": per_split_counts["val"],
        },
        "split_index_ranges": {
            "train": list(TRAIN_INDEX_RANGE),
            "val": list(VAL_INDEX_RANGE),
        },
        "source_counts": source_counts,
        "label_assumptions": list(ASSUMPTION_NOTES.values()),
    }
    return picked["train"], picked["val"], summary


def verify_disjoint(train: list[Candidate], val: list[Candidate]) -> None:
    train_idx = {c.source_index for c in train}
    val_idx = {c.source_index for c in val}
    train_hash = {c.sha256 for c in train}
    val_hash = {c.sha256 for c in val}
    idx_overlap = train_idx & val_idx
    hash_overlap = train_hash & val_hash
    if idx_overlap:
        raise ValueError(f"Train/val source_index overlap: {sorted(idx_overlap)[:5]}")
    if hash_overlap:
        raise ValueError(f"Train/val sha256 overlap: {sorted(hash_overlap)[:5]}")


def copy_payload(candidate: Candidate, repo_root: Path, dest_root: Path) -> str:
    """Copy the payload into source-datasets/mnist/<digit>/f{index}.png; return repo-rel path."""
    dest_dir = dest_root / candidate.label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"f{candidate.source_index}.png"
    shutil.copyfile(candidate.src_path, dest_path)
    return stable_rel(dest_path, repo_root)


def record_for(
    candidate: Candidate,
    sample_id: str,
    repo_rel_path: str,
    seed: int,
) -> dict[str, object]:
    policy_use = "develop_policy" if candidate.split == "train" else "validation_decision_quality"
    return {
        "sample_id": sample_id,
        "dataset": DATASET,
        "label": candidate.label,
        "label_int": candidate.label_int,
        "repo_rel_path": repo_rel_path,
        "original_filename": candidate.original_filename,
        "file_ext": candidate.file_ext,
        "sha256": candidate.sha256,
        "split": candidate.split,
        "seed": seed,
        "sampling_version": SAMPLING_VERSION,
        "source_index": candidate.source_index,
        "truth_tier": "gold_candidate",
        "policy_use": policy_use,
    }


CSV_FIELDNAMES = [
    "sample_id",
    "dataset",
    "label",
    "label_int",
    "repo_rel_path",
    "original_filename",
    "file_ext",
    "sha256",
    "split",
    "seed",
    "sampling_version",
    "source_index",
    "truth_tier",
    "policy_use",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2000, help="train sample size (stratified across digits)")
    parser.add_argument("--n-val", type=int, default=500, help="val sample size (stratified across digits)")
    parser.add_argument("--seed", type=int, default=20260703, help="deterministic sampling seed")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("~/Downloads/mnist_png").expanduser(),
        help="MNIST export root containing labels_and_paths.csv and data/",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dest-root", type=Path, default=None, help="source-datasets/mnist copy root")
    parser.add_argument("--force", action="store_true", help="overwrite existing manifest outputs")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    source_root = args.source_root.expanduser().resolve()
    out_dir = args.out_dir or repo_root / "data/images/mnist-classification/manifests"
    dest_root = args.dest_root or repo_root / "data/images/mnist-classification/source-datasets/mnist"

    outputs = [
        out_dir / "train_labels.csv",
        out_dir / "val_labels.csv",
        out_dir / "sampling_summary.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing manifests without --force:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 2

    grouped = collect_candidates(source_root)
    train, val, summary = sample_records(grouped, args.n_train, args.n_val, args.seed)

    train_rows: list[dict[str, object]] = []
    for i, candidate in enumerate(train, 1):
        rel = copy_payload(candidate, repo_root, dest_root)
        train_rows.append(record_for(candidate, f"train_{i:05d}", rel, args.seed))

    val_rows: list[dict[str, object]] = []
    for i, candidate in enumerate(val, 1):
        rel = copy_payload(candidate, repo_root, dest_root)
        val_rows.append(record_for(candidate, f"val_{i:04d}", rel, args.seed))

    write_csv(outputs[0], train_rows)
    write_csv(outputs[1], val_rows)
    outputs[2].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
