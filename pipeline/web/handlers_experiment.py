"""API handlers for the experiment crank (seeded PPO policy-iteration runs).

File-based like the rest of the web layer: everything is served from the
portable ``data/experiments/<id>/experiment.json`` the driver rewrites
atomically — a fresh clone demos with no database. SME gate reviews are the
one write path; they update the JSON and best-effort mirror to Postgres
(``rush.gate_review``) as future RLHF data for the critic agent.

Endpoints (dispatched from handlers_runs.handle_api):
    GET  /api/experiments                     — newest-first summaries
    GET  /api/experiments/{id}                — full state (the UI poll target)
    POST /api/experiments/start               — spawn run_experiment.py (registry job)
    POST /api/experiments/{id}/review         — {k, verdict, reviewer?, comment?}
    GET  /api/adjudication[?area=...]         — cross-run SME re-adjudication queue
    GET  /api/area-stats[?demo=|area=]        — the area's manifest split sizes
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline import experiment as exp

from ._safety import APIError


def handle_list_experiments(repo_root: Path | str) -> tuple[int, dict[str, Any]]:
    return 200, {"experiments": exp.list_experiments(repo_root)}


def _genai_manifests_dir() -> Path:
    from pipeline.io_paths import REPO_ROOT

    return REPO_ROOT / "data" / "images" / "genai-classification" / "manifests"


def _genai_mint_marker() -> Path:
    return _genai_manifests_dir() / ".mint_in_progress.json"


def _mint_status() -> dict[str, Any] | None:
    """The in-flight split mint, or None. A dead pid clears the marker."""
    import json as _json
    import os as _os

    marker = _genai_mint_marker()
    if not marker.exists():
        return None
    try:
        info = _json.loads(marker.read_text(encoding="utf-8"))
        pid = int(info.get("pid") or 0)
        _os.kill(pid, 0)  # signal 0: existence probe only
        return info
    except (ValueError, TypeError, _json.JSONDecodeError, ProcessLookupError):
        marker.unlink(missing_ok=True)
        return None
    except PermissionError:
        # Pid exists but is another user's — treat as alive to stay safe.
        return info


def handle_mint_splits(
    repo_root: Path | str, body: dict[str, Any] | None, registry: Any
) -> tuple[int, dict[str, Any]]:
    """POST /api/genai/splits/mint — re-mint the GenAI split manifests.

    Runs ``scripts/sample_genai_gold_sets.py --seed S --n-dev D --n-holdout H
    --n-validation V --force`` detached (hashing the 20k-image tree takes a
    minute or two); the UI polls /api/area-stats until ``mint_running`` clears
    and the split sizes land. GenAI-only by design: MNIST's splits are
    committed; GenAI's are minted per machine — the SEED is the alignment
    contract between machines, which is exactly why it is user-visible.

    Guards: the source tree must exist (nothing to sample otherwise), no
    labeling/experiment job may be running (re-assigning splits under a live
    run corrupts its premise), and one mint at a time.
    """
    import subprocess
    import sys as _sys

    from pipeline.io_paths import _genai_source_tree_has_images
    from pipeline.web.run_registry import utcnow_iso

    body = body or {}

    def _int_field(name: str, default: int, lo: int, hi: int) -> int:
        raw = body.get(name, default)
        if raw is None:
            raw = default
        if not isinstance(raw, int) or isinstance(raw, bool) or not lo <= raw <= hi:
            raise APIError(400, "validation_error",
                           f"{name} must be an integer in [{lo}, {hi}]",
                           details={"field": name})
        return raw

    seed = _int_field("seed", 20260510, 0, 2**32 - 1)
    n_dev = _int_field("n_dev", 2000, 10, 10000)
    n_holdout = _int_field("n_holdout", 1000, 10, 10000)
    n_validation = _int_field("n_validation", 200, 0, 10000)

    if not _genai_source_tree_has_images():
        raise APIError(
            409, "source_tree_missing",
            "the GenAI source-datasets tree has no images on this machine — "
            "splits can only be minted where the data lives",
        )
    running = registry.list_jobs(running_only=True) if registry is not None else []
    if running:
        raise APIError(
            409, "run_in_flight",
            "a labeling/experiment job is running — re-assigning splits under "
            "a live run would corrupt it; wait or cancel first",
        )
    if _mint_status() is not None:
        raise APIError(409, "mint_in_progress", "a split mint is already running")

    root = Path(repo_root)
    venv_python = root / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else _sys.executable
    argv = [
        python, "scripts/sample_genai_gold_sets.py",
        "--seed", str(seed),
        "--n-dev", str(n_dev),
        "--n-holdout", str(n_holdout),
        "--n-validation", str(n_validation),
        "--force",
    ]
    manifests_dir = _genai_manifests_dir()
    manifests_dir.mkdir(parents=True, exist_ok=True)
    log_path = manifests_dir / ".mint.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=str(root), stdout=log, stderr=log)
    params = {"seed": seed, "n_dev": n_dev, "n_holdout": n_holdout,
              "n_validation": n_validation}
    _genai_mint_marker().write_text(
        json.dumps({"pid": proc.pid, "started_at": utcnow_iso(), **params}),
        encoding="utf-8",
    )
    return 202, {"status": "minting", **params}


def handle_area_stats(
    query: dict[str, list[str]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Split sizes of the active area's manifest, for demo-aware form sizing.

    The run form uses this to size Test/Train defaults to the dev_golden pool
    and to enable the benchmark readout only when a fixed ``validation`` split
    actually exists (the GenAI area ships without one until it is minted with
    ``scripts/sample_genai_gold_sets.py --n-validation ...``). Counts reflect
    whichever manifest :func:`genai_manifest_default` resolves on THIS host —
    portable sample on sparse clones, the full manifest where the source
    image tree exists.
    """
    from pipeline.io_paths import MNIST_SAMPLE_MANIFEST, genai_manifest_default
    from pipeline.manifest import load_records

    from .demo_area import MNIST_POLICY_AREA, area_from_query

    try:
        area = area_from_query(query or {})
    except ValueError as exc:
        raise APIError(400, "validation_error", str(exc)) from exc
    manifest = (
        MNIST_SAMPLE_MANIFEST if area == MNIST_POLICY_AREA else genai_manifest_default()
    )
    splits: dict[str, int] = {"dev_golden": 0, "holdout": 0, "validation": 0}
    try:
        for record in load_records(manifest):
            if record.split in splits:
                splits[record.split] += 1
    except (OSError, ValueError) as exc:
        raise APIError(503, "manifest_unavailable",
                       f"cannot read the {area} manifest: {exc}") from exc
    # The sampling seed is the cross-machine alignment contract for minted
    # (GenAI) splits — surface it so the UI can display and re-mint with it.
    sampling_seed: int | None = None
    if area != MNIST_POLICY_AREA:
        summary_path = _genai_manifests_dir() / "sampling_summary.json"
        try:
            sampling_seed = json.loads(summary_path.read_text(encoding="utf-8")).get("seed")
        except (OSError, ValueError):
            sampling_seed = None
    return 200, {
        "area": area,
        "manifest": str(manifest),
        "splits": splits,
        "total": sum(splits.values()),
        "sampling_seed": sampling_seed,
        "mint_running": (area != MNIST_POLICY_AREA) and _mint_status() is not None,
    }


def handle_get_experiment(
    repo_root: Path | str, experiment_id: str
) -> tuple[int, dict[str, Any]]:
    try:
        exp.validate_experiment_id(experiment_id)
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    try:
        state = exp.load_state(repo_root, experiment_id)
    except FileNotFoundError as excinfo:
        raise APIError(
            404, "not_found", f"unknown experiment: {experiment_id}"
        ) from excinfo
    return 200, state


def handle_adjudication_queue(
    repo_root: Path | str, query: dict[str, list[str]] | None = None
) -> tuple[int, dict[str, Any]]:
    """The running cross-run list of items flagged for SME re-adjudication,
    aggregated at read time from every experiment.json (dry runs excluded —
    their deterministic fake votes are not human work)."""
    area = ((query or {}).get("area") or [""])[0].strip() or None
    return 200, exp.aggregate_readjudication(repo_root, area=area)


def handle_adjudication_review(
    repo_root: Path | str, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    """Record an SME's confirm / overturn / uncertain verdict on a queue item,
    then return the refreshed queue for the item's area."""
    body = body or {}
    area = str(body.get("area") or "").strip()
    key = str(body.get("key") or "").strip()
    verdict = str(body.get("verdict") or "").strip()
    if not area or not key:
        raise APIError(400, "validation_error", "area and key are required",
                       details={"field": "key" if area else "area"})
    if verdict not in exp.VERDICTS:
        raise APIError(400, "validation_error",
                       "verdict must be one of: " + ", ".join(exp.VERDICTS),
                       details={"field": "verdict"})
    new_label = body.get("new_label")
    if verdict == "overturn" and not str(new_label or "").strip():
        raise APIError(400, "validation_error", "overturn requires a new_label",
                       details={"field": "new_label"})
    try:
        record = exp.record_adjudication(
            repo_root, area=area, key=key,
            image_id=str(body.get("image_id") or "") or None,
            verdict=verdict, prior_truth=str(body.get("prior_truth") or "") or None,
            new_label=str(new_label or "") or None,
            reviewer=str(body.get("reviewer") or "sme"),
            comment=str(body.get("comment") or ""),
        )
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    return 200, {"recorded": record,
                 "queue": exp.aggregate_readjudication(repo_root, area=area)}


def handle_gate_review(
    repo_root: Path | str, experiment_id: str, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    body = body or {}
    try:
        exp.validate_experiment_id(experiment_id)
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    k = body.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise APIError(
            400, "validation_error", "k must be a positive integer",
            details={"field": "k"},
        )
    verdict = str(body.get("verdict") or "")
    reviewer = str(body.get("reviewer") or "sme")[:80]
    comment = str(body.get("comment") or "")[:2000]
    try:
        review = exp.record_gate_review(
            repo_root, experiment_id, k,
            verdict=verdict, reviewer=reviewer, comment=comment,
        )
    except FileNotFoundError as excinfo:
        raise APIError(
            404, "not_found", f"unknown experiment: {experiment_id}"
        ) from excinfo
    except KeyError as excinfo:
        raise APIError(404, "not_found", str(excinfo)) from excinfo
    except ValueError as excinfo:
        raise APIError(400, "validation_error", str(excinfo)) from excinfo
    return 200, {"experiment_id": experiment_id, "k": k, "review": review}
