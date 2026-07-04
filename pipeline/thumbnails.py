"""Shared thumbnail path helpers for demo source images.

Thumbnails use a human-inspectable path-mirroring scheme: a source image at
``data/images/genai-classification/source-datasets/<dataset>/<label>/foo.png``
maps to
``data/images/genai-classification/derived/thumbnails/<dataset>/<label>/foo.jpg``.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

SOURCE_ROOT_REL = Path("data/images/genai-classification/source-datasets")
THUMBNAIL_ROOT_REL = Path("data/images/genai-classification/derived/thumbnails")
SOURCE_THUMBNAIL_ROOTS: tuple[tuple[Path, Path | None], ...] = (
    (SOURCE_ROOT_REL, THUMBNAIL_ROOT_REL),
    # MNIST samples are already tiny local PNGs; no derived thumbnail tree is
    # required for the demo, but the path still needs to pass validation.
    (Path("data/images/mnist-classification/source-datasets"), None),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _repo_rel_posix(value: str | Path) -> PurePosixPath:
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid repo-relative image path: {value!r}")
    return path


def validate_source_repo_path(repo_root: Path | str, repo_rel_path: str | Path) -> Path:
    """Return a normalized source-image path under ``SOURCE_ROOT_REL``.

    Raises ``ValueError`` for absolute paths, traversal, non-source paths, and
    non-image suffixes. The returned path is repo-relative and safe to join to
    ``repo_root``.
    """
    rel_posix = _repo_rel_posix(repo_rel_path)
    allowed_root: Path | None = None
    for source_root, _thumbnail_root in SOURCE_THUMBNAIL_ROOTS:
        try:
            rel_posix.relative_to(PurePosixPath(source_root.as_posix()))
        except ValueError:
            continue
        allowed_root = source_root
        break
    if allowed_root is None:
        roots = ", ".join(root.as_posix() for root, _thumb in SOURCE_THUMBNAIL_ROOTS)
        raise ValueError(f"image path must be under one of: {roots}")
    if rel_posix.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image suffix: {rel_posix.suffix!r}")

    root = Path(repo_root).resolve()
    candidate = (root / Path(*rel_posix.parts)).resolve(strict=False)
    source_root = (root / allowed_root).resolve(strict=False)
    if source_root != candidate and source_root not in candidate.parents:
        raise ValueError(f"image path must be under {allowed_root.as_posix()}")
    return Path(*rel_posix.parts)


def thumbnail_rel_path_for_source(
    source_repo_rel_path: str | Path,
    *,
    source_root_rel: Path | str = SOURCE_ROOT_REL,
    output_root_rel: Path | str = THUMBNAIL_ROOT_REL,
) -> Path:
    """Map a source repo-relative image path to its thumbnail repo-relative path."""
    source_rel = Path(str(_repo_rel_posix(source_repo_rel_path)))
    source_root = Path(source_root_rel)
    output_root = Path(output_root_rel)
    for candidate_source_root, candidate_output_root in SOURCE_THUMBNAIL_ROOTS:
        try:
            source_rel.relative_to(candidate_source_root)
        except ValueError:
            continue
        if candidate_output_root is None:
            return source_rel
        source_root = candidate_source_root
        output_root = candidate_output_root
        break
    rel_inside_source = source_rel.relative_to(source_root)
    return (output_root / rel_inside_source).with_suffix(".jpg")


__all__ = [
    "IMAGE_SUFFIXES",
    "SOURCE_ROOT_REL",
    "SOURCE_THUMBNAIL_ROOTS",
    "THUMBNAIL_ROOT_REL",
    "thumbnail_rel_path_for_source",
    "validate_source_repo_path",
]
