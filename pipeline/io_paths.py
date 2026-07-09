"""Path conventions and run_id minting for the bulk-labeling pipeline (X2).

Centralising path layout means scoring (X3) and web exporters (X3/X4) can read
back artifacts without guessing where they live.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import secrets

# Repo root is two levels up from this file (pipeline/io_paths.py -> RUSH/).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Default input manifest (the 200-row dev_golden + holdout sample).
DEFAULT_SAMPLE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "images"
    / "genai-classification"
    / "manifests"
    / "combined_labels.jsonl"
)

GENAI_PORTABLE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "images"
    / "genai-classification"
    / "manifests"
    / "combined_labels.portable.jsonl"
)

GENAI_SOURCE_DATASETS_ROOT = (
    REPO_ROOT
    / "data"
    / "images"
    / "genai-classification"
    / "source-datasets"
)

MNIST_SAMPLE_MANIFEST = (
    REPO_ROOT
    / "data"
    / "images"
    / "mnist-classification"
    / "manifests"
    / "combined_labels.jsonl"
)

# Default output root (gitignored).
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "runs"

# Default policy graph directory bundled into LabelRequest.policy_markdown.
DEFAULT_POLICY_GRAPH_DIR = REPO_ROOT / "policy-graph" / "Generative_AI" / "v0.1"
DEFAULT_POLICY_GRAPH_VERSION = "v0.1"

RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")

_GENAI_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _genai_source_tree_has_images(root: Path = GENAI_SOURCE_DATASETS_ROOT) -> bool:
    if not root.is_dir():
        return False
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if Path(filename).suffix.lower() in _GENAI_IMAGE_SUFFIXES:
                return True
    return False


def genai_manifest_default() -> Path:
    """Return the default GenAI manifest for this checkout.

    Set ``RUSH_PORTABLE=1`` (also accepts ``true`` or ``yes``, case-insensitive)
    to force the committed portable 72-row fixture. Without that override, the
    full 200-row GenAI manifest is used only when the local
    ``source-datasets`` image tree exists and contains at least one image file;
    sparse/portable clones fall back to the committed portable manifest.
    """
    portable_env = os.environ.get("RUSH_PORTABLE", "").strip().lower()
    if portable_env in {"1", "true", "yes"}:
        return GENAI_PORTABLE_MANIFEST
    if not _genai_source_tree_has_images():
        return GENAI_PORTABLE_MANIFEST
    if not DEFAULT_SAMPLE_MANIFEST.exists():
        # Source images are present but the gold manifests haven't been
        # minted yet (mid-rsync from the Mac mini, or a fresh 10 GB copy
        # before running scripts/sample_genai_gold_sets.py). Keep serving the
        # committed portable fixture instead of pointing every GenAI surface
        # at a manifest that does not exist.
        return GENAI_PORTABLE_MANIFEST
    return DEFAULT_SAMPLE_MANIFEST


def mint_run_id(now: datetime | None = None, *, rng: object | None = None) -> str:
    """Return a fresh ``run_id`` of the form ``YYYYMMDDTHHMMSS-xxxxxxxx`` (UTC).

    ``rng`` is accepted only for test injection: any object with ``.token_hex(4)``
    works (default: ``secrets``). The default uses ``secrets.token_hex`` for
    cryptographic-quality uniqueness across parallel runs.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token_source = rng if rng is not None else secrets
    short = token_source.token_hex(4)  # type: ignore[attr-defined]
    return f"{stamp}-{short}"


def is_valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_PATTERN.match(run_id))


@dataclass(frozen=True)
class RunPaths:
    """All per-run output paths under ``data/runs/<run_id>/``."""

    run_id: str
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def label_votes(self) -> Path:
        return self.root / "label_votes.jsonl"

    @property
    def llm_outputs(self) -> Path:
        return self.root / "llm_outputs.jsonl"

    @property
    def errors(self) -> Path:
        return self.root / "errors.jsonl"

    @property
    def costs(self) -> Path:
        """Durable per-image-per-model cost ledger (analysis-ready)."""
        return self.root / "costs.jsonl"

    @property
    def model_speed_summary(self) -> Path:
        """Per-model speed/cost rollup for frontend and offline analysis."""
        return self.root / "model_speed_summary.json"

    @property
    def scoring_dir(self) -> Path:
        return self.root / "scoring"

    @property
    def policy_patches(self) -> Path:
        return self.root / "policy_patches.jsonl"

    @property
    def policy_pdf(self) -> Path:
        return self.root / "policy.pdf"

    @property
    def web_dir(self) -> Path:
        return self.root / "web"

    def ensure(self) -> None:
        """Create the run output tree (idempotent)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.scoring_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)


def run_paths(run_id: str, runs_root: Path | None = None) -> RunPaths:
    if not is_valid_run_id(run_id):
        raise ValueError(f"invalid run_id format: {run_id!r}")
    root = (runs_root or DEFAULT_RUNS_ROOT) / run_id
    return RunPaths(run_id=run_id, root=root)


__all__ = [
    "REPO_ROOT",
    "DEFAULT_SAMPLE_MANIFEST",
    "GENAI_PORTABLE_MANIFEST",
    "GENAI_SOURCE_DATASETS_ROOT",
    "MNIST_SAMPLE_MANIFEST",
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_POLICY_GRAPH_DIR",
    "DEFAULT_POLICY_GRAPH_VERSION",
    "RUN_ID_PATTERN",
    "RunPaths",
    "genai_manifest_default",
    "mint_run_id",
    "is_valid_run_id",
    "run_paths",
]
