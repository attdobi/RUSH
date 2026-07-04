#!/usr/bin/env python3
"""Build JPEG thumbnails for GenAI classification source images.

Path scheme: thumbnails mirror the source tree for easy human inspection. For
example:

  data/images/genai-classification/source-datasets/midjourney/ai_generated/image_1249.png

becomes:

  data/images/genai-classification/derived/thumbnails/midjourney/ai_generated/image_1249.jpg

Existing thumbnails are skipped when their mtime is newer than or equal to the
source image mtime.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.thumbnails import (
    DEMO_THUMBNAIL_ROOTS,
    IMAGE_SUFFIXES,
    SOURCE_ROOT_REL,
    THUMBNAIL_ROOT_REL,
    thumbnail_rel_path_for_source,
)


@dataclass
class Summary:
    scanned: int = 0
    generated: int = 0
    skipped: int = 0
    bytes_written: int = 0


def iter_source_images(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def iter_requested_images(*, repo_root: Path, source_root: Path, paths_file: Path) -> Iterable[Path]:
    """Yield explicit paths from a newline-delimited file.

    Lines may be repo-relative source paths or paths relative to ``source_root``.
    Blank lines and ``#`` comments are ignored.
    """
    source_root_resolved = source_root.resolve()
    for line_no, raw in enumerate(paths_file.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        rel = Path(text)
        if rel.is_absolute():
            raise ValueError(f"{paths_file}:{line_no}: expected a relative path")
        candidate = (repo_root / rel).resolve(strict=False)
        if not candidate.exists():
            candidate = (source_root / rel).resolve(strict=False)
        if source_root_resolved != candidate and source_root_resolved not in candidate.parents:
            raise ValueError(f"{paths_file}:{line_no}: path is outside source root: {text}")
        if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_file():
            raise ValueError(f"{paths_file}:{line_no}: not a supported image file: {text}")
        yield candidate


def should_skip(source: Path, output: Path) -> bool:
    return output.exists() and output.stat().st_mtime >= source.stat().st_mtime


def build_thumbnail(source: Path, output: Path, *, max_edge: int, quality: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if im.mode not in {"RGB", "L"}:
            background = Image.new("RGB", im.size, (255, 255, 255))
            if "A" in im.getbands():
                background.paste(im, mask=im.getchannel("A"))
            else:
                background.paste(im)
            im = background
        elif im.mode == "L":
            im = im.convert("RGB")
        im.save(output, format="JPEG", quality=quality, optimize=True)
    return output.stat().st_size


def build_all(
    *,
    repo_root: Path,
    source_root: Path,
    output_root: Path,
    max_edge: int,
    quality: int,
    limit: int | None = None,
    paths_file: Path | None = None,
) -> Summary:
    summary = Summary()
    source_iter = (
        iter_requested_images(repo_root=repo_root, source_root=source_root, paths_file=paths_file)
        if paths_file is not None
        else iter_source_images(source_root)
    )
    for source in source_iter:
        if limit is not None and summary.scanned >= limit:
            break
        summary.scanned += 1
        source_repo_rel = source.relative_to(repo_root)
        output_repo_rel = thumbnail_rel_path_for_source(
            source_repo_rel,
            source_root_rel=source_root.relative_to(repo_root),
            output_root_rel=output_root.relative_to(repo_root),
        )
        output = repo_root / output_repo_rel
        if should_skip(source, output):
            summary.skipped += 1
            continue
        summary.bytes_written += build_thumbnail(source, output, max_edge=max_edge, quality=quality)
        summary.generated += 1
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 192px JPEG thumbnails for a demo's source images.")
    parser.add_argument(
        "--demo",
        choices=sorted(DEMO_THUMBNAIL_ROOTS),
        default=None,
        help="Demo/area to build thumbnails for; sets source/output roots. "
        "Overridden by explicit --source-root/--output-root.",
    )
    parser.add_argument("--source-root", default=None, help="Repo-relative source image root")
    parser.add_argument("--output-root", default=None, help="Repo-relative thumbnail output root")
    parser.add_argument("--max-edge", type=int, default=192, help="Maximum thumbnail edge in pixels")
    parser.add_argument("--quality", type=int, default=78, help="JPEG quality")
    parser.add_argument("--limit", type=int, default=None, help="Optional max source images to scan (testing)")
    parser.add_argument("--paths-file", default=None, help="Optional newline-delimited explicit source paths to build")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    # Resolve roots: explicit flags win; else per-demo roots; else GenAI default.
    demo_source, demo_output = DEMO_THUMBNAIL_ROOTS.get(args.demo or "genai")
    source_root_rel = args.source_root or str(demo_source)
    output_root_rel = args.output_root or str(demo_output)
    source_root = (repo_root / source_root_rel).resolve()
    output_root = (repo_root / output_root_rel).resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root not found: {source_root}")
    summary = build_all(
        repo_root=repo_root,
        source_root=source_root,
        output_root=output_root,
        max_edge=args.max_edge,
        quality=args.quality,
        limit=args.limit,
        paths_file=(Path(args.paths_file).resolve() if args.paths_file else None),
    )
    print(
        "thumbnail summary: "
        f"scanned={summary.scanned} generated={summary.generated} "
        f"skipped={summary.skipped} bytes={summary.bytes_written}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
