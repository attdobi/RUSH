#!/usr/bin/env python3
"""Sample deterministic GenAI classification golden/holdout label manifests.

This script reads local image payloads from the ignored RUSH image store and writes
ignored manifest files. It does not write image bytes/base64 into manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SAMPLING_VERSION = "genai-gold-sampling-v1"
LABELS = ("ai_generated", "not_ai_generated")
DATASETS = ("sdv1_4", "midjourney", "wfir")
SOURCE_LABEL_DIRS = {
    ("sdv1_4", "ai_generated"): "1_false",
    ("sdv1_4", "not_ai_generated"): "0_real",
    ("midjourney", "ai_generated"): "1_fake",
    ("midjourney", "not_ai_generated"): "0_real",
    ("wfir", "ai_generated"): "1_fake",
    ("wfir", "not_ai_generated"): "0_real",
}
ASSUMPTION_NOTES = {
    ("sdv1_4", "ai_generated"): "Assumption: source directory sdv1.4/1_false is treated as positive ai_generated despite ambiguous 'false' name.",
}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    label: str
    label_int: int
    repo_rel_path: str
    original_filename: str
    file_ext: str
    source_label_dir: str
    sha256: str
    assumption_note: str


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


def collect_candidates(repo_root: Path, source_root: Path) -> dict[tuple[str, str], list[Candidate]]:
    grouped: dict[tuple[str, str], list[Candidate]] = {}
    for dataset in DATASETS:
        for label in LABELS:
            class_dir = source_root / dataset / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing source directory: {class_dir}")
            files = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise ValueError(f"No image files found in {class_dir}")
            key = (dataset, label)
            grouped[key] = [
                Candidate(
                    dataset=dataset,
                    label=label,
                    label_int=1 if label == "ai_generated" else 0,
                    repo_rel_path=stable_rel(path, repo_root),
                    original_filename=path.name,
                    file_ext=path.suffix.lower().lstrip("."),
                    source_label_dir=SOURCE_LABEL_DIRS[key],
                    sha256=sha256_file(path),
                    assumption_note=ASSUMPTION_NOTES.get(key, ""),
                )
                for path in files
            ]
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


def allocate_split_counts(n: int) -> dict[str, dict[str, int]]:
    if n <= 0:
        raise ValueError("N must be positive")
    label_counts = split_evenly(n, LABELS)
    return {label: split_evenly(count, DATASETS) for label, count in label_counts.items()}


def sample_records(
    grouped: dict[tuple[str, str], list[Candidate]],
    n_dev: int,
    n_holdout: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    dev_counts = allocate_split_counts(n_dev)
    holdout_counts = allocate_split_counts(n_holdout)
    dev: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for label in LABELS:
        for dataset in DATASETS:
            key = (dataset, label)
            candidates = dedupe_by_hash(grouped[key])
            needed = dev_counts[label][dataset] + holdout_counts[label][dataset]
            if len(candidates) < needed:
                raise ValueError(
                    f"Not enough unique files for {dataset}/{label}: need {needed}, have {len(candidates)}"
                )
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            dev_take = dev_counts[label][dataset]
            dev_candidates = shuffled[:dev_take]
            holdout_candidates = shuffled[dev_take:needed]
            source_counts[f"{dataset}/{label}"] = {
                "available_unique": len(candidates),
                "dev_golden": len(dev_candidates),
                "holdout": len(holdout_candidates),
            }
            for split, rows in (("dev_golden", dev_candidates), ("holdout", holdout_candidates)):
                target = dev if split == "dev_golden" else holdout
                for idx, candidate in enumerate(rows, 1):
                    target.append(record_for(candidate, split, seed, idx))

    dev.sort(key=lambda r: (str(r["dataset"]), str(r["label"]), str(r["repo_rel_path"])))
    holdout.sort(key=lambda r: (str(r["dataset"]), str(r["label"]), str(r["repo_rel_path"])))
    for i, row in enumerate(dev, 1):
        row["sample_id"] = f"dev_golden_{i:04d}"
    for i, row in enumerate(holdout, 1):
        row["sample_id"] = f"holdout_{i:04d}"

    verify_disjoint(dev, holdout)
    summary = {
        "sampling_version": SAMPLING_VERSION,
        "seed": seed,
        "n_dev_golden": len(dev),
        "n_holdout": len(holdout),
        "allocation": {
            "dev_golden": dev_counts,
            "holdout": holdout_counts,
        },
        "source_counts": source_counts,
        "label_assumptions": [note for note in ASSUMPTION_NOTES.values()],
    }
    return dev, holdout, summary


def record_for(candidate: Candidate, split: str, seed: int, draw_index: int) -> dict[str, object]:
    return {
        "sample_id": "pending",
        "dataset": candidate.dataset,
        "source_label_dir": candidate.source_label_dir,
        "label": candidate.label,
        "label_int": candidate.label_int,
        "repo_rel_path": candidate.repo_rel_path,
        "original_filename": candidate.original_filename,
        "file_ext": candidate.file_ext,
        "sha256": candidate.sha256,
        "split": split,
        "seed": seed,
        "sampling_version": SAMPLING_VERSION,
        "draw_index_within_dataset_label": draw_index,
        "assumption_note": candidate.assumption_note,
        "truth_tier": "gold_candidate",
        "policy_use": "develop_policy" if split == "dev_golden" else "locked_holdout_decision_quality",
    }


def verify_disjoint(dev: list[dict[str, object]], holdout: list[dict[str, object]]) -> None:
    dev_paths = {str(row["repo_rel_path"]) for row in dev}
    holdout_paths = {str(row["repo_rel_path"]) for row in holdout}
    dev_hashes = {str(row["sha256"]) for row in dev}
    holdout_hashes = {str(row["sha256"]) for row in holdout}
    path_overlap = dev_paths & holdout_paths
    hash_overlap = dev_hashes & holdout_hashes
    if path_overlap:
        raise ValueError(f"Dev/holdout path overlap: {sorted(path_overlap)[:5]}")
    if hash_overlap:
        raise ValueError(f"Dev/holdout hash overlap: {sorted(hash_overlap)[:5]}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "dataset",
        "source_label_dir",
        "label",
        "label_int",
        "repo_rel_path",
        "original_filename",
        "file_ext",
        "sha256",
        "split",
        "seed",
        "sampling_version",
        "draw_index_within_dataset_label",
        "assumption_note",
        "truth_tier",
        "policy_use",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dev", type=int, default=100, help="development golden-set size")
    parser.add_argument("--n-holdout", type=int, default=100, help="locked holdout size")
    parser.add_argument("--seed", type=int, default=20260510, help="deterministic sampling seed")
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite existing manifest outputs")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    source_root = args.source_root or repo_root / "data/images/genai-classification/source-datasets"
    out_dir = args.out_dir or repo_root / "data/images/genai-classification/manifests"
    outputs = [
        out_dir / "dev_golden_labels.csv",
        out_dir / "holdout_labels.csv",
        out_dir / "combined_labels.jsonl",
        out_dir / "sampling_summary.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing manifests without --force:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 2

    grouped = collect_candidates(repo_root, source_root.resolve())
    dev, holdout, summary = sample_records(grouped, args.n_dev, args.n_holdout, args.seed)
    write_csv(outputs[0], dev)
    write_csv(outputs[1], holdout)
    write_jsonl(outputs[2], [*dev, *holdout])
    outputs[3].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
