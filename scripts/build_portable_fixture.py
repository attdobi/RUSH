#!/usr/bin/env python3
"""Build a deterministic, small committed GenAI image fixture.

The full GenAI image payload is intentionally local-only. This script copies a
budgeted subset into ``data/images/genai-classification/sample`` and writes a
matching manifest whose ``repo_rel_path`` points at that sample tree. By default
it preserves source bytes; ``--encode-jpeg`` writes deterministic resized JPEG
derivatives for a denser portable fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.io_paths import DEFAULT_SAMPLE_MANIFEST, GENAI_PORTABLE_MANIFEST, REPO_ROOT

DEFAULT_SAMPLE_ROOT = REPO_ROOT / "data" / "images" / "genai-classification" / "sample"
DEFAULT_MAX_MB = 90
DEFAULT_PER_STRATUM = 22
DEFAULT_JPEG_MAX_EDGE = 1024
DEFAULT_JPEG_QUALITY = 82
EXPECTED_STRATUM_COUNT = 12
STRATUM_FIELDS = ("dataset", "label", "split")


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    source_path: Path
    size_bytes: int
    payload: bytes | None = None
    target_filename: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def sample_id(self) -> str:
        return str(self.record["sample_id"])

    @property
    def stratum(self) -> tuple[str, str, str]:
        return tuple(str(self.record[field]) for field in STRATUM_FIELDS)  # type: ignore[return-value]


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {lineno}: invalid JSON ({exc})") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest has no rows: {path}")
    return rows


def jpeg_payload(path: Path, *, max_edge: int, quality: int) -> tuple[bytes, int, int]:
    if max_edge < 1:
        raise ValueError(f"jpeg max edge must be positive, got {max_edge}")
    if not 1 <= quality <= 95:
        raise ValueError(f"jpeg quality must be in [1, 95], got {quality}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(
            out,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        return out.getvalue(), image.width, image.height


def portable_target_filename(candidate: dict[str, Any], *, encode_jpeg: bool) -> str:
    original = str(candidate["original_filename"])
    if not encode_jpeg:
        return original
    stem = Path(original).stem
    return f"{candidate['sample_id']}_{stem}.jpg"


# The portable fixture packs dev_golden + holdout only. The manifest may also
# carry the fixed cross-run `validation` benchmark split (minted by
# sample_genai_gold_sets.py --n-validation); those rows are deliberately
# excluded here — the fixture is a size-budgeted demo corpus, and sparse
# clones keep the benchmark readout disabled until a validation split exists
# in whichever manifest resolves.
FIXTURE_SPLITS = frozenset({"dev_golden", "holdout"})


def candidates_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    encode_jpeg: bool = False,
    jpeg_max_edge: int = DEFAULT_JPEG_MAX_EDGE,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in rows:
        missing = [field for field in (*STRATUM_FIELDS, "repo_rel_path", "sample_id") if field not in row]
        if missing:
            raise ValueError(f"manifest row missing required fields {missing}: {row!r}")
        if str(row["split"]) not in FIXTURE_SPLITS:
            continue
        source_path = REPO_ROOT / str(row["repo_rel_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source image for {row['sample_id']}: {source_path}")
        payload: bytes | None = None
        width: int | None = None
        height: int | None = None
        size_bytes = source_path.stat().st_size
        if encode_jpeg:
            payload, width, height = jpeg_payload(
                source_path,
                max_edge=jpeg_max_edge,
                quality=jpeg_quality,
            )
            size_bytes = len(payload)
        candidates.append(
            Candidate(
                row,
                source_path,
                size_bytes,
                payload=payload,
                target_filename=portable_target_filename(row, encode_jpeg=encode_jpeg),
                width=width,
                height=height,
            )
        )
    return candidates


def group_by_stratum(candidates: Iterable[Candidate]) -> dict[tuple[str, str, str], list[Candidate]]:
    grouped: dict[tuple[str, str, str], list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.stratum, []).append(candidate)
    for rows in grouped.values():
        rows.sort(key=lambda candidate: (candidate.size_bytes, candidate.sample_id))
    return grouped


def select_under_budget(candidates: Iterable[Candidate], budget_bytes: int) -> list[Candidate]:
    grouped = group_by_stratum(candidates)
    if not grouped:
        raise ValueError("no strata found")

    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    total_bytes = 0

    for stratum in sorted(grouped):
        first = grouped[stratum][0]
        selected.append(first)
        selected_ids.add(first.sample_id)
        total_bytes += first.size_bytes

    if total_bytes > budget_bytes:
        raise ValueError(
            f"minimum one-image-per-stratum fixture is {total_bytes} bytes, "
            f"over budget {budget_bytes} bytes"
        )

    remaining: list[Candidate] = []
    for rows in grouped.values():
        remaining.extend(rows[1:])
    remaining.sort(key=lambda candidate: (candidate.size_bytes, candidate.sample_id))

    for candidate in remaining:
        if total_bytes + candidate.size_bytes > budget_bytes:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.sample_id)
        total_bytes += candidate.size_bytes

    # Rebuild per-stratum prefix selections to guarantee no stratum skips a
    # smaller file while keeping a larger one.
    selected_by_stratum: dict[tuple[str, str, str], set[str]] = {}
    for candidate in selected:
        selected_by_stratum.setdefault(candidate.stratum, set()).add(candidate.sample_id)
    for stratum, rows in grouped.items():
        selected_count = sum(1 for row in rows if row.sample_id in selected_by_stratum.get(stratum, set()))
        expected = {row.sample_id for row in rows[:selected_count]}
        actual = selected_by_stratum.get(stratum, set())
        if actual != expected:
            raise AssertionError(f"selection for stratum {stratum} is not a smallest-file prefix")

    selected.sort(key=lambda candidate: candidate.sample_id)
    if len(selected_ids) != len(selected):
        raise AssertionError("duplicate sample_id selected")
    return selected


def select_per_stratum(
    candidates: Iterable[Candidate],
    budget_bytes: int,
    per_stratum: int,
) -> list[Candidate]:
    if per_stratum < 1:
        raise ValueError(f"per_stratum must be positive, got {per_stratum}")

    grouped = group_by_stratum(candidates)
    if len(grouped) != EXPECTED_STRATUM_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STRATUM_COUNT} strata, found {len(grouped)}: "
            f"{sorted(grouped)}"
        )

    grouped_by_pair: dict[tuple[str, str], dict[str, list[Candidate]]] = {}
    for stratum, rows in grouped.items():
        dataset, label, split = stratum
        grouped_by_pair.setdefault((dataset, label), {})[split] = rows

    selected: list[Candidate] = []
    for pair in sorted(grouped_by_pair):
        by_split = grouped_by_pair[pair]
        splits = sorted(by_split)
        if per_stratum < len(splits):
            raise ValueError(
                f"per_stratum={per_stratum} cannot represent every split for {pair}"
            )
        pair_selected: list[Candidate] = []
        pair_selected_ids: set[str] = set()
        for split in splits:
            rows = by_split[split]
            if not rows:
                raise ValueError(f"stratum {(*pair, split)} has no candidates")
            pair_selected.append(rows[0])
            pair_selected_ids.add(rows[0].sample_id)

        remaining = sorted(
            (
                candidate
                for split in splits
                for candidate in by_split[split]
                if candidate.sample_id not in pair_selected_ids
            ),
            key=lambda candidate: (candidate.size_bytes, candidate.sample_id),
        )
        needed = per_stratum - len(pair_selected)
        if len(remaining) < needed:
            raise ValueError(
                f"dataset/label stratum {pair} has only "
                f"{len(pair_selected) + len(remaining)} candidates; need {per_stratum}"
            )
        pair_selected.extend(remaining[:needed])
        selected.extend(pair_selected)

    total_bytes = sum(candidate.size_bytes for candidate in selected)
    if total_bytes > budget_bytes:
        raise ValueError(
            f"{per_stratum}-per-stratum fixture is {total_bytes} bytes, "
            f"over budget {budget_bytes} bytes"
        )

    selected.sort(key=lambda candidate: candidate.sample_id)
    if len({candidate.sample_id for candidate in selected}) != len(selected):
        raise AssertionError("duplicate sample_id selected")
    return selected


def sample_target(candidate: Candidate, sample_root: Path) -> Path:
    dataset, label, _split = candidate.stratum
    return sample_root / dataset / label / str(candidate.target_filename or candidate.record["original_filename"])


def rebuild_sample_tree(selected: Iterable[Candidate], sample_root: Path) -> list[dict[str, Any]]:
    if sample_root.exists():
        shutil.rmtree(sample_root)
    sample_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    target_paths: set[Path] = set()
    for candidate in selected:
        target = sample_target(candidate, sample_root)
        if target in target_paths:
            raise ValueError(f"sample target collision: {target}")
        target_paths.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if candidate.payload is None:
            shutil.copy2(candidate.source_path, target)
            payload_hash = str(candidate.record["sha256"])
        else:
            target.write_bytes(candidate.payload)
            payload_hash = hashlib.sha256(candidate.payload).hexdigest()

        row = dict(candidate.record)
        if candidate.payload is not None:
            row["source_repo_rel_path"] = row["repo_rel_path"]
            row["source_sha256"] = row["sha256"]
            row["source_original_filename"] = row["original_filename"]
            row["original_filename"] = target.name
            row["file_ext"] = "jpg"
            row["sha256"] = payload_hash
            row["portable_sha256"] = payload_hash
            row["portable_byte_size"] = candidate.size_bytes
            row["portable_width"] = candidate.width
            row["portable_height"] = candidate.height
            row["portable_encoding"] = "jpeg"
        row["repo_rel_path"] = repo_rel(target)
        rows.append(row)
    rows.sort(key=lambda row: str(row["sample_id"]))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def print_summary(selected: Iterable[Candidate]) -> None:
    count_by_stratum: Counter[tuple[str, str, str]] = Counter()
    bytes_by_stratum: Counter[tuple[str, str, str]] = Counter()
    total_bytes = 0
    total_images = 0
    for candidate in selected:
        count_by_stratum[candidate.stratum] += 1
        bytes_by_stratum[candidate.stratum] += candidate.size_bytes
        total_bytes += candidate.size_bytes
        total_images += 1

    print(f"selected_images={total_images}")
    print(f"selected_bytes={total_bytes}")
    print("per_stratum:")
    for stratum in sorted(count_by_stratum):
        dataset, label, split = stratum
        print(
            f"  {dataset}/{label}/{split}: "
            f"{count_by_stratum[stratum]} images, {bytes_by_stratum[stratum]} bytes"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SAMPLE_MANIFEST,
        help=f"source JSONL manifest (default: {DEFAULT_SAMPLE_MANIFEST})",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=GENAI_PORTABLE_MANIFEST,
        help=f"portable JSONL manifest to write (default: {GENAI_PORTABLE_MANIFEST})",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=DEFAULT_SAMPLE_ROOT,
        help=f"sample image tree to rebuild (default: {DEFAULT_SAMPLE_ROOT})",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_MB,
        help=f"maximum total original image MiB to copy (default: {DEFAULT_MAX_MB})",
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=DEFAULT_PER_STRATUM,
        help=(
            "number of smallest original images to copy per dataset/label "
            "stratum while keeping every split represented "
            f"(default: {DEFAULT_PER_STRATUM})"
        ),
    )
    parser.add_argument(
        "--encode-jpeg",
        action="store_true",
        help="write portable fixture images as downscaled JPEG derivatives",
    )
    parser.add_argument(
        "--jpeg-max-edge",
        type=int,
        default=DEFAULT_JPEG_MAX_EDGE,
        help=f"maximum JPEG output width/height when --encode-jpeg is used (default: {DEFAULT_JPEG_MAX_EDGE})",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality when --encode-jpeg is used (default: {DEFAULT_JPEG_QUALITY})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    candidates = candidates_from_rows(
        rows,
        encode_jpeg=args.encode_jpeg,
        jpeg_max_edge=args.jpeg_max_edge,
        jpeg_quality=args.jpeg_quality,
    )
    budget_bytes = int(args.max_mb * 1024 * 1024)
    selected = select_per_stratum(candidates, budget_bytes, args.per_stratum)
    portable_rows = rebuild_sample_tree(selected, args.sample_root)
    write_jsonl(args.output_manifest, portable_rows)
    print_summary(selected)
    print(f"sample_root={args.sample_root}")
    print(f"output_manifest={args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
