"""Sample manifest loader + ground-truth join helpers (X2).

The bulk-labeling runner consumes ``combined_labels.jsonl`` (the GenAI sampler
output). This module reads it once, exposes a lightweight ``SampleRecord``
view, and provides a small ground-truth helper so X3 scoring can do the
SME-truth join without re-parsing the manifest.

Loader is stdlib-only (matches the spirit of ``scripts/validate_foundation.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

from .io_paths import DEFAULT_SAMPLE_MANIFEST, REPO_ROOT

# Maps the sampler's source-of-truth label values onto the cold-start
# bulk-labeling label vocabulary used by LLMOutput / LabelVote.
SME_LABEL_MAP: dict[str, str] = {
    "ai_generated": "gen_ai",
    "not_ai_generated": "not_gen_ai",
}

VALID_SPLITS: frozenset[str] = frozenset({"dev_golden", "holdout"})

# Splits that must never auto-run in a bulk pass without an explicit safety flag.
HOLDOUT_SPLITS: frozenset[str] = frozenset({"holdout"})


@dataclass(frozen=True)
class SampleRecord:
    """A single row from ``combined_labels.jsonl`` reduced to the fields we use."""

    sample_id: str
    repo_rel_path: str
    split: str
    sme_label_raw: str   # "ai_generated" | "not_ai_generated"
    sme_label: str       # "gen_ai" | "not_gen_ai"
    dataset: str
    sha256: str
    sampling_version: str

    @property
    def absolute_path(self) -> Path:
        return REPO_ROOT / self.repo_rel_path

    def exists(self) -> bool:
        return self.absolute_path.is_file()


def _coerce(record: dict) -> SampleRecord:
    raw_label = record.get("label", "")
    if raw_label not in SME_LABEL_MAP:
        raise ValueError(
            f"sample {record.get('sample_id')!r} has unmapped label {raw_label!r}; "
            "expected one of: " + ", ".join(sorted(SME_LABEL_MAP))
        )
    split = record.get("split", "")
    if split not in VALID_SPLITS:
        raise ValueError(
            f"sample {record.get('sample_id')!r} has unknown split {split!r}; "
            "expected one of: " + ", ".join(sorted(VALID_SPLITS))
        )
    return SampleRecord(
        sample_id=record["sample_id"],
        repo_rel_path=record["repo_rel_path"],
        split=split,
        sme_label_raw=raw_label,
        sme_label=SME_LABEL_MAP[raw_label],
        dataset=record.get("dataset", ""),
        sha256=record.get("sha256", ""),
        sampling_version=record.get("sampling_version", ""),
    )


def iter_records(path: Path | None = None) -> Iterator[SampleRecord]:
    """Yield ``SampleRecord``s from a JSONL manifest in file order."""
    manifest = path or DEFAULT_SAMPLE_MANIFEST
    with manifest.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest} line {lineno}: invalid JSON ({exc})"
                ) from exc
            yield _coerce(obj)


def load_records(path: Path | None = None) -> list[SampleRecord]:
    return list(iter_records(path))


def select_samples(
    records: Iterable[SampleRecord],
    *,
    split: str | None = None,
    limit: int | None = None,
    sample_ids: Iterable[str] | None = None,
) -> list[SampleRecord]:
    """Filter + deterministically order records.

    Order: by ``sample_id`` ascending (matches §5.6 determinism rule).
    """
    if split is not None and split not in VALID_SPLITS and split != "all":
        raise ValueError(
            f"unknown split {split!r}; use one of: all, " + ", ".join(sorted(VALID_SPLITS))
        )

    explicit_ids: set[str] | None = None
    if sample_ids is not None:
        explicit_ids = {s for s in sample_ids if s}

    keep: list[SampleRecord] = []
    for rec in records:
        if split and split != "all" and rec.split != split:
            continue
        if explicit_ids is not None and rec.sample_id not in explicit_ids:
            continue
        keep.append(rec)

    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

    keep.sort(key=lambda r: r.sample_id)

    if limit is not None:
        if split == "all" and explicit_ids is None:
            # Demo batches should be a real N-per-portion pass, not a single
            # N-sized slice that happens to include whichever split sorts first.
            # With the bundled manifest this means N dev_golden + N holdout.
            per_split: list[SampleRecord] = []
            for split_name in sorted(VALID_SPLITS):
                rows = [rec for rec in keep if rec.split == split_name]
                per_split.extend(rows[:limit])
            keep = sorted(per_split, key=lambda r: r.sample_id)
        else:
            keep = keep[:limit]
    return keep


def build_ground_truth(records: Iterable[SampleRecord]) -> dict[str, str]:
    """Return ``{sample_id: sme_label}`` for the SME truth join (X3)."""
    return {rec.sample_id: rec.sme_label for rec in records}


def load_policy_markdown(policy_dir: Path) -> str:
    """Concatenate ``*.md`` policy files in lexical order into a single string.

    The runner injects this string into ``LabelRequest.policy_markdown``.
    Files are joined with two newlines and a header line so providers can see
    boundaries between nodes without us shipping an extra index file.
    """
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"policy directory not found: {policy_dir}")
    chunks: list[str] = []
    for md in sorted(policy_dir.glob("*.md")):
        chunks.append(f"<!-- {md.name} -->\n{md.read_text(encoding='utf-8').rstrip()}\n")
    if not chunks:
        raise FileNotFoundError(f"no policy *.md files under {policy_dir}")
    return "\n".join(chunks)


__all__ = [
    "SampleRecord",
    "SME_LABEL_MAP",
    "VALID_SPLITS",
    "HOLDOUT_SPLITS",
    "iter_records",
    "load_records",
    "select_samples",
    "build_ground_truth",
    "load_policy_markdown",
]
