#!/usr/bin/env python3
"""Pack the full local MNIST PNG export into a compact committed NPZ."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("~/Downloads/mnist_png").expanduser()
DEFAULT_OUTPUT = REPO_ROOT / "data" / "images" / "mnist-classification" / "mnist_full.npz"
IMAGE_SHAPE = (28, 28)
EXPECTED_ROWS = 70_000


def read_labels(csv_path: Path) -> np.ndarray:
    labels = np.full(EXPECTED_ROWS, 255, dtype=np.uint8)
    seen: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"empty labels CSV: {csv_path}")
        for row_number, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"{csv_path} line {row_number}: expected index,label,path")
            index = int(row[0])
            label = int(row[1])
            if not 0 <= index < EXPECTED_ROWS:
                raise ValueError(f"{csv_path} line {row_number}: bad index {index}")
            if not 0 <= label <= 9:
                raise ValueError(f"{csv_path} line {row_number}: bad label {label}")
            if index in seen:
                raise ValueError(f"{csv_path} line {row_number}: duplicate index {index}")
            labels[index] = label
            seen.add(index)

    if len(seen) != EXPECTED_ROWS:
        missing = sorted(set(range(EXPECTED_ROWS)) - seen)[:10]
        raise ValueError(f"expected {EXPECTED_ROWS} labels, found {len(seen)}; missing {missing}")
    return labels


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        if gray.size != IMAGE_SHAPE[::-1]:
            gray = gray.resize(IMAGE_SHAPE[::-1], Image.Resampling.LANCZOS)
        arr = np.asarray(gray, dtype=np.uint8)
    if arr.shape != IMAGE_SHAPE:
        raise ValueError(f"{path}: expected shape {IMAGE_SHAPE}, got {arr.shape}")
    return arr


def pack(source_root: Path, output: Path) -> None:
    csv_path = source_root / "labels_and_paths.csv"
    data_dir = source_root / "data"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing labels CSV: {csv_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data directory: {data_dir}")

    labels = read_labels(csv_path)
    indices = np.arange(EXPECTED_ROWS, dtype=np.int32)
    images = np.empty((EXPECTED_ROWS, *IMAGE_SHAPE), dtype=np.uint8)

    for index in range(EXPECTED_ROWS):
        image_path = data_dir / f"f{index}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing MNIST image for index {index}: {image_path}")
        images[index] = load_image(image_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=images, labels=labels, index=indices)
    print(f"wrote {output}")
    print(f"images={images.shape} {images.dtype}")
    print(f"labels={labels.shape} {labels.dtype}")
    print(f"index={indices.shape} {indices.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"MNIST PNG export root (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"NPZ output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack(args.source_root.expanduser(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
