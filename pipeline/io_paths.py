"""Path conventions and run_id minting for the bulk-labeling pipeline (X2).

Centralising path layout means scoring (X3) and web exporters (X3/X4) can read
back artifacts without guessing where they live.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

# Default output root (gitignored).
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "runs"

# Default policy graph directory bundled into LabelRequest.policy_markdown.
DEFAULT_POLICY_GRAPH_DIR = REPO_ROOT / "policy-graph" / "Generative_AI" / "v0.1"
DEFAULT_POLICY_GRAPH_VERSION = "v0.1"

RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")


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
    "DEFAULT_RUNS_ROOT",
    "DEFAULT_POLICY_GRAPH_DIR",
    "DEFAULT_POLICY_GRAPH_VERSION",
    "RUN_ID_PATTERN",
    "RunPaths",
    "mint_run_id",
    "is_valid_run_id",
    "run_paths",
]
