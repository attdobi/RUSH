#!/usr/bin/env python3
"""Unpack the committed full MNIST NPZ into local PNG files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "images" / "mnist-classification" / "mnist_full.npz"
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "images" / "mnist-classification" / "source-datasets" / "mnist"
)
IMAGE_SHAPE = (28, 28)
EXPECTED_ROWS = 70_000


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        images = data["images"]
        labels = data["labels"]
        indices = data["index"]

    if images.shape != (EXPECTED_ROWS, *IMAGE_SHAPE) or images.dtype != np.uint8:
        raise ValueError(f"bad images array: shape={images.shape} dtype={images.dtype}")
    if labels.shape != (EXPECTED_ROWS,) or labels.dtype != np.uint8:
        raise ValueError(f"bad labels array: shape={labels.shape} dtype={labels.dtype}")
    if indices.shape != (EXPECTED_ROWS,) or indices.dtype != np.int32:
        raise ValueError(f"bad index array: shape={indices.shape} dtype={indices.dtype}")
    expected = np.arange(EXPECTED_ROWS, dtype=np.int32)
    if not np.array_equal(indices, expected):
        raise ValueError("index array is not ordered 0..69999")
    return images, labels, indices


def write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path)


def write_flat_labels(output_root: Path, labels: np.ndarray, indices: np.ndarray) -> None:
    csv_path = output_root / "labels_and_paths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", "label", "path"])
        for index in indices:
            label = int(labels[index])
            writer.writerow([int(index), label, f"data/{int(index)}.png"])


def unpack(archive: Path, output_root: Path, layout: str, limit: int | None) -> None:
    images, labels, indices = load_archive(archive)
    selected_indices = indices if limit is None else indices[:limit]

    for index in selected_indices:
        label = int(labels[index])
        if layout == "digit":
            out_path = output_root / str(label) / f"f{int(index)}.png"
        else:
            out_path = output_root / "data" / f"f{int(index)}.png"
        write_png(out_path, images[index])

    if layout == "flat":
        write_flat_labels(output_root, labels, selected_indices)

    print(f"unpacked {len(selected_indices)} images to {output_root} ({layout})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"MNIST NPZ archive to unpack (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--layout",
        choices=("digit", "flat"),
        default="digit",
        help="digit writes <out>/<digit>/f{index}.png; flat writes <out>/data/f{index}.png",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional smoke-test limit; omitted writes all 70000 images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    unpack(args.archive.expanduser(), args.out.expanduser(), args.layout, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
