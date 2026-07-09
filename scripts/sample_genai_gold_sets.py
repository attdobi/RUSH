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
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
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


def default_hash_workers() -> int:
    """Threads for hashing the source tree.

    Hashing the full ~12 GB / ~20k-image tree is IO-bound (per-file open +
    read dominates), and ``read()`` releases the GIL, so oversubscribing the
    core count overlaps those waits. Capped so we don't thrash the disk.
    """
    return min(16, (os.cpu_count() or 4) * 2)


def hash_files(files: list[Path], *, workers: int) -> list[str]:
    """SHA-256 every file, concurrently, PRESERVING input order.

    ``ThreadPoolExecutor.map`` yields results in the order of ``files``, so the
    hashes are byte-for-byte identical to serial hashing — the seed→manifest
    alignment contract is untouched; only the wall-clock is smaller.
    """
    if workers <= 1 or len(files) <= 1:
        return [sha256_file(path) for path in files]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(sha256_file, files))


def split_evenly(total: int, buckets: Iterable[str]) -> dict[str, int]:
    names = list(buckets)
    base, remainder = divmod(total, len(names))
    return {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(names)}


def collect_candidates(
    repo_root: Path, source_root: Path, *, hash_workers: int | None = None
) -> dict[tuple[str, str], list[Candidate]]:
    workers = default_hash_workers() if hash_workers is None else hash_workers
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
            digests = hash_files(files, workers=workers)
            grouped[key] = [
                Candidate(
                    dataset=dataset,
                    label=label,
                    label_int=1 if label == "ai_generated" else 0,
                    repo_rel_path=stable_rel(path, repo_root),
                    original_filename=path.name,
                    file_ext=path.suffix.lower().lstrip("."),
                    source_label_dir=SOURCE_LABEL_DIRS[key],
                    sha256=digest,
                    assumption_note=ASSUMPTION_NOTES.get(key, ""),
                )
                for path, digest in zip(files, digests)
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
    n_validation: int = 0,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Draw the three disjoint splits from one seeded shuffle per source group.

    The validation split (the fixed cross-run benchmark, mirroring MNIST's
    ``bench_*`` rows) is sliced AFTER dev+holdout, so re-running with the SAME
    seed and a newly non-zero ``n_validation`` keeps every existing dev_golden
    and holdout assignment identical — the benchmark can be minted later
    without invalidating labels already gathered on the other splits.
    """
    rng = random.Random(seed)
    dev_counts = allocate_split_counts(n_dev)
    holdout_counts = allocate_split_counts(n_holdout)
    validation_counts = (
        allocate_split_counts(n_validation)
        if n_validation > 0
        else {label: {dataset: 0 for dataset in DATASETS} for label in LABELS}
    )
    dev: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    source_counts: dict[str, dict[str, int]] = {}

    for label in LABELS:
        for dataset in DATASETS:
            key = (dataset, label)
            candidates = dedupe_by_hash(grouped[key])
            dev_take = dev_counts[label][dataset]
            holdout_take = holdout_counts[label][dataset]
            validation_take = validation_counts[label][dataset]
            needed = dev_take + holdout_take + validation_take
            if len(candidates) < needed:
                raise ValueError(
                    f"Not enough unique files for {dataset}/{label}: need {needed}, have {len(candidates)}"
                )
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            dev_candidates = shuffled[:dev_take]
            holdout_candidates = shuffled[dev_take:dev_take + holdout_take]
            validation_candidates = shuffled[dev_take + holdout_take:needed]
            source_counts[f"{dataset}/{label}"] = {
                "available_unique": len(candidates),
                "dev_golden": len(dev_candidates),
                "holdout": len(holdout_candidates),
                "validation": len(validation_candidates),
            }
            for split, rows, target in (
                ("dev_golden", dev_candidates, dev),
                ("holdout", holdout_candidates, holdout),
                ("validation", validation_candidates, validation),
            ):
                for idx, candidate in enumerate(rows, 1):
                    target.append(record_for(candidate, split, seed, idx))

    for rows in (dev, holdout, validation):
        rows.sort(key=lambda r: (str(r["dataset"]), str(r["label"]), str(r["repo_rel_path"])))
    for i, row in enumerate(dev, 1):
        row["sample_id"] = f"dev_golden_{i:04d}"
    for i, row in enumerate(holdout, 1):
        row["sample_id"] = f"holdout_{i:04d}"
    for i, row in enumerate(validation, 1):
        # Mirrors the MNIST benchmark split's id convention (bench_0001...).
        row["sample_id"] = f"bench_{i:04d}"

    verify_disjoint(dev, holdout, validation)
    summary = {
        "sampling_version": SAMPLING_VERSION,
        "seed": seed,
        "n_dev_golden": len(dev),
        "n_holdout": len(holdout),
        "n_validation": len(validation),
        "allocation": {
            "dev_golden": dev_counts,
            "holdout": holdout_counts,
            "validation": validation_counts,
        },
        "source_counts": source_counts,
        "label_assumptions": [note for note in ASSUMPTION_NOTES.values()],
    }
    return dev, holdout, validation, summary


_POLICY_USE_BY_SPLIT = {
    "dev_golden": "develop_policy",
    "holdout": "locked_holdout_decision_quality",
    # Same value the MNIST validation split uses: the fixed cross-run
    # benchmark scored only under the start + final policy versions.
    "validation": "benchmark_cross_run",
}


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
        "policy_use": _POLICY_USE_BY_SPLIT[split],
    }


def verify_disjoint(*split_rows: list[dict[str, object]]) -> None:
    """Every pair of splits must share no path and no content hash."""
    seen_paths: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for rows in split_rows:
        for row in rows:
            path, digest = str(row["repo_rel_path"]), str(row["sha256"])
            split = str(row["split"])
            if path in seen_paths and seen_paths[path] != split:
                raise ValueError(f"Split path overlap ({seen_paths[path]}/{split}): {path}")
            if digest in seen_hashes and seen_hashes[digest] != split:
                raise ValueError(f"Split hash overlap ({seen_hashes[digest]}/{split}): {digest}")
            seen_paths[path] = split
            seen_hashes[digest] = split


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
    # Canonical split sizes (Attila 2026-07-09): run this script with NO
    # arguments on every machine that has the source tree and the manifests
    # come out byte-identical — that IS the cross-machine alignment mechanism.
    parser.add_argument("--n-dev", type=int, default=2000, help="development golden-set size")
    parser.add_argument("--n-holdout", type=int, default=1000, help="locked holdout size")
    parser.add_argument("--n-validation", type=int, default=200,
                        help="fixed cross-run benchmark split size (0 = none). Drawn "
                             "AFTER dev+holdout from the same seeded shuffle, so adding "
                             "it later with the SAME seed keeps existing dev/holdout "
                             "assignments identical.")
    parser.add_argument("--seed", type=int, default=20260510, help="deterministic sampling seed")
    parser.add_argument("--jobs", type=int, default=None,
                        help="threads for hashing the source tree (default: auto, "
                             "min(16, 2*cpu)). Output is identical regardless — this "
                             "only trades wall-clock, since hashing dominates the run.")
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
    validation_csv = out_dir / "validation_labels.csv"
    if args.n_validation > 0:
        outputs.append(validation_csv)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing manifests without --force:", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 2

    grouped = collect_candidates(
        repo_root, source_root.resolve(), hash_workers=args.jobs
    )
    dev, holdout, validation, summary = sample_records(
        grouped, args.n_dev, args.n_holdout, args.seed, n_validation=args.n_validation
    )
    write_csv(outputs[0], dev)
    write_csv(outputs[1], holdout)
    if validation:
        write_csv(validation_csv, validation)
    write_jsonl(outputs[2], [*dev, *holdout, *validation])
    outputs[3].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
