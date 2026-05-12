"""Pure aggregation helpers for the web decision-quality and insights APIs.

All reads are bounded to a caller-provided runs root. The module never writes
under ``data/runs``; handlers can safely call these functions on live run data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from itertools import combinations
from pathlib import Path
from typing import Any

RUNS_ROOT = Path("data/runs")
_MAX_ITEMS = 50


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_under(path: Path, runs_root: Path) -> Path:
    root = runs_root.resolve()
    resolved = path.resolve()
    if not _is_relative_to(resolved, root):
        raise ValueError(f"Path escapes runs root: {path}")
    return resolved


def _read_json(path: Path, runs_root: Path) -> dict[str, Any]:
    safe = _resolve_under(path, runs_root)
    with safe.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _read_json_if_exists(path: Path, runs_root: Path) -> dict[str, Any] | None:
    safe = _resolve_under(path, runs_root)
    if not safe.exists():
        return None
    with safe.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _read_jsonl_if_exists(path: Path, runs_root: Path) -> list[dict[str, Any]]:
    safe = _resolve_under(path, runs_root)
    if not safe.exists():
        return []
    rows: list[dict[str, Any]] = []
    with safe.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{safe.name}:{line_no} invalid JSON: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _scored_run_dirs(runs_root: Path) -> list[Path]:
    root = runs_root.resolve()
    if not root.exists():
        return []
    out: list[Path] = []
    for candidate in sorted(root.iterdir(), key=lambda p: p.name):
        run_dir = _resolve_under(candidate, root)
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if (run_dir / "run_manifest.json").exists() and (
            run_dir / "scoring" / "decision_quality.json"
        ).exists():
            out.append(run_dir)
    return out


def _latest_flip_rate_dir(runs_root: Path) -> Path | None:
    root = runs_root.resolve()
    flip_root = _resolve_under(root / "_flip_rate", root)
    if not flip_root.exists():
        return None
    dirs = [
        _resolve_under(path, root)
        for path in flip_root.iterdir()
        if path.is_dir()
    ]
    return sorted(dirs, key=lambda p: p.name)[-1] if dirs else None


def _latest_flip_rate_summary(runs_root: Path) -> dict[str, Any] | None:
    latest = _latest_flip_rate_dir(runs_root)
    if latest is None:
        return None
    return _read_json_if_exists(latest / "flip_rate_summary.json", runs_root)


def _latest_flip_rate_records(runs_root: Path) -> list[dict[str, Any]]:
    latest = _latest_flip_rate_dir(runs_root)
    if latest is None:
        return []
    return _read_jsonl_if_exists(latest / "flip_rate.jsonl", runs_root)


def _policy_version(manifest: dict[str, Any], dq: dict[str, Any]) -> str | None:
    return (
        dq.get("policy_graph_version")
        or manifest.get("policy_graph_version")
        or manifest.get("policy_version")
    )


def _n_images(manifest: dict[str, Any], consensus: dict[str, Any] | None, borderline: dict[str, Any] | None) -> int:
    sample_ids = manifest.get("sample_ids")
    if isinstance(sample_ids, list):
        return len(sample_ids)
    if consensus:
        summary = consensus.get("summary", {})
        if isinstance(summary, dict) and isinstance(summary.get("n_images_total"), int):
            return int(summary["n_images_total"])
    if borderline:
        summary = borderline.get("summary", {})
        if isinstance(summary, dict) and isinstance(summary.get("total_images"), int):
            return int(summary["total_images"])
    return 0


def _boundary_rate(consensus: dict[str, Any] | None, borderline: dict[str, Any] | None) -> float | None:
    if borderline:
        summary = borderline.get("summary", {})
        if isinstance(summary, dict):
            total = summary.get("total_images") or 0
            boundary = summary.get("borderline_images") or 0
            if total:
                return round(float(boundary) / float(total), 6)
    if consensus:
        summary = consensus.get("summary", {})
        if isinstance(summary, dict):
            total = summary.get("n_images_total") or 0
            boundary = summary.get("n_images_with_boundary_flag") or 0
            if total:
                return round(float(boundary) / float(total), 6)
    return None


def _split_labelers(labelers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    majority: dict[str, Any] | None = None
    for row in labelers:
        if row.get("labeler_id") == "majority_vote":
            majority = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else row
            break
    return labelers, majority


def aggregate_decision_quality(
    runs_root: Path,
    run_id: str | None = None,
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Aggregate scored run decision-quality snapshots for the web API."""
    root = runs_root.resolve()
    flip_summary = _latest_flip_rate_summary(root)
    runs: list[dict[str, Any]] = []
    discovered_policy_versions: set[str] = set()

    for run_dir in _scored_run_dirs(root):
        manifest = _read_json(run_dir / "run_manifest.json", root)
        dq = _read_json(run_dir / "scoring" / "decision_quality.json", root)
        rid = str(manifest.get("run_id") or run_dir.name)
        version = _policy_version(manifest, dq)
        manifest_version = manifest.get("policy_graph_version")
        if version:
            discovered_policy_versions.add(str(version))
        if run_id is not None and rid != run_id:
            continue
        if policy_version is not None and policy_version not in {str(version), str(manifest_version)}:
            continue

        consensus = _read_json_if_exists(run_dir / "scoring" / "consensus.json", root)
        borderline = _read_json_if_exists(run_dir / "scoring" / "borderline.json", root)
        labelers, majority_vote = _split_labelers(list(dq.get("labelers", [])))
        runs.append(
            {
                "run_id": rid,
                "started_at": manifest.get("started_at"),
                "policy_graph_version": version,
                "n_images": _n_images(manifest, consensus, borderline),
                "labelers": labelers,
                "majority_vote": majority_vote,
                "cost": dq.get("cost") if isinstance(dq.get("cost"), dict) else None,
                "consensus_summary": (consensus or {}).get("summary", {}),
                "boundary_rate": _boundary_rate(consensus, borderline),
                "flip_rate_summary": flip_summary,
            }
        )

    runs.sort(key=lambda row: (str(row.get("started_at") or ""), str(row.get("run_id") or "")))
    return {"runs": runs, "policy_versions": sorted(discovered_policy_versions)}


def _to_vote_list(voters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "labeler_id": voter.get("labeler_id") or voter.get("model_id"),
            "model_id": voter.get("model_id"),
            "label": voter.get("label"),
            "confidence": voter.get("confidence"),
            "is_boundary": voter.get("is_boundary"),
        }
        for voter in voters
    ]


def _add_repo_rel_path(row: dict[str, Any], image_paths: dict[str, str]) -> dict[str, Any]:
    image_id = row.get("image_id")
    if image_id is None:
        return row
    repo_rel_path = image_paths.get(str(image_id))
    if repo_rel_path:
        row["repo_rel_path"] = repo_rel_path
    return row


def _borderline_group_records(borderline: dict[str, Any]) -> list[dict[str, Any]]:
    groups = borderline.get("groups", {})
    if isinstance(groups, dict):
        buckets = groups.values()
    elif isinstance(groups, list):
        buckets = groups
    else:
        return []

    records: list[dict[str, Any]] = []
    for bucket in buckets:
        if isinstance(bucket, list):
            records.extend(record for record in bucket if isinstance(record, dict))
        elif isinstance(bucket, dict):
            if bucket.get("image_id") is not None:
                records.append(bucket)
            else:
                for value in bucket.values():
                    if isinstance(value, list):
                        records.extend(record for record in value if isinstance(record, dict))
    return records


def _image_repo_rel_paths(misalignment: dict[str, Any], borderline: dict[str, Any]) -> dict[str, str]:
    image_paths: dict[str, str] = {}

    def remember(record: dict[str, Any]) -> None:
        image_id = record.get("image_id")
        repo_rel_path = record.get("repo_rel_path")
        if image_id is not None and repo_rel_path:
            image_paths.setdefault(str(image_id), str(repo_rel_path))

    records = misalignment.get("records", [])
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict):
                remember(record)
    for record in _borderline_group_records(borderline):
        remember(record)
    return image_paths


def _policy_clarity_hot_spots(run_id: str, runs_root: Path, image_paths: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for record in _latest_flip_rate_records(runs_root):
        run_ids = record.get("run_ids")
        if isinstance(run_ids, list) and run_id not in {str(r) for r in run_ids}:
            continue
        if isinstance(run_ids, list) or record.get("first_seen_run_id") == run_id or record.get("last_seen_run_id") == run_id:
            rows.append(
                _add_repo_rel_path(
                    {
                        "image_id": record.get("image_id"),
                        "model_id": record.get("model_id"),
                        "flip_rate": record.get("flip_rate"),
                        "flip_count": record.get("flip_count"),
                        "n_runs": record.get("n_runs"),
                        "labels_observed": record.get("labels_observed", []),
                    },
                    image_paths,
                )
            )
    rows.sort(
        key=lambda row: (
            -float(row.get("flip_rate") or 0),
            -int(row.get("flip_count") or 0),
            str(row.get("image_id") or ""),
            str(row.get("model_id") or ""),
        )
    )
    return rows[:_MAX_ITEMS]


def _majority_wrong(consensus: dict[str, Any], misalignment: dict[str, Any], image_paths: dict[str, str]) -> list[dict[str, Any]]:
    by_image = {row.get("image_id"): row for row in misalignment.get("records", [])}
    rows: list[dict[str, Any]] = []
    for record in consensus.get("records", []):
        majority_label = record.get("majority_label")
        if majority_label is None:
            continue
        joined = by_image.get(record.get("image_id"), {})
        sme_truth = joined.get("sme_truth")
        if sme_truth is None or majority_label == sme_truth:
            continue
        rows.append(
            _add_repo_rel_path(
                {
                    "image_id": record.get("image_id"),
                    "sme_truth": sme_truth,
                    "majority_label": majority_label,
                    "majority_count": record.get("majority_count"),
                    "majority_fraction": record.get("majority_fraction"),
                    "votes": joined.get("votes") or _to_vote_list(record.get("voters", [])),
                },
                image_paths,
            )
        )
    return rows[:_MAX_ITEMS]


def _model_disagreement(consensus: dict[str, Any], image_paths: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for record in consensus.get("records", []):
        if not record.get("is_split"):
            continue
        rows.append(
            _add_repo_rel_path(
                {
                    "image_id": record.get("image_id"),
                    "vote_distribution": record.get("vote_distribution", {}),
                    "votes": _to_vote_list(record.get("voters", [])),
                },
                image_paths,
            )
        )
    return rows[:_MAX_ITEMS]


def _boundary_concentration(borderline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = borderline.get("groups", {})
    if not isinstance(groups, dict):
        return rows
    for l0_bucket, records in groups.items():
        if not isinstance(records, list):
            continue
        l2_counts: Counter[str] = Counter()
        image_ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            image_id = record.get("image_id")
            if image_id:
                image_ids.add(str(image_id))
            for vote in record.get("votes", []):
                l2 = vote.get("l2_label") if isinstance(vote, dict) else None
                if l2:
                    l2_counts[str(l2)] += 1
        rows.append(
            {
                "l0_bucket": l0_bucket,
                "n_images": len(image_ids),
                "top_l2_nodes": [name for name, _ in l2_counts.most_common(10)],
            }
        )
    rows.sort(key=lambda row: (-int(row.get("n_images") or 0), str(row.get("l0_bucket") or "")))
    return rows[:_MAX_ITEMS]


def _consistent_pair_disagreement(consensus: dict[str, Any]) -> list[dict[str, Any]]:
    pair_disagreements: Counter[tuple[str, str]] = Counter()
    pair_totals: Counter[tuple[str, str]] = Counter()
    for record in consensus.get("records", []):
        labels: dict[str, str] = {}
        for voter in record.get("voters", []):
            labeler = voter.get("labeler_id") or voter.get("model_id")
            label = voter.get("label")
            if labeler is None or label is None:
                continue
            labels[str(labeler)] = str(label)
        for left, right in combinations(sorted(labels), 2):
            pair = (left, right)
            pair_totals[pair] += 1
            if labels[left] != labels[right]:
                pair_disagreements[pair] += 1
    rows: list[dict[str, Any]] = []
    for pair, n_disagreements in pair_disagreements.items():
        total = pair_totals[pair]
        rows.append(
            {
                "pair": list(pair),
                "n_disagreements": n_disagreements,
                "fraction": round(n_disagreements / total, 6) if total else None,
            }
        )
    rows.sort(key=lambda row: (-int(row["n_disagreements"]), str(row["pair"][0]), str(row["pair"][1])))
    return rows[:_MAX_ITEMS]


def compute_insights(run_dir: Path) -> dict[str, Any]:
    """Compute insight worklists for a single scored run directory."""
    resolved_run_dir = run_dir.resolve()
    runs_root = resolved_run_dir.parent.resolve()
    resolved_run_dir = _resolve_under(resolved_run_dir, runs_root)
    run_id = resolved_run_dir.name

    consensus = _read_json(resolved_run_dir / "scoring" / "consensus.json", runs_root)
    misalignment = _read_json(resolved_run_dir / "scoring" / "misalignment.json", runs_root)
    borderline = _read_json(resolved_run_dir / "scoring" / "borderline.json", runs_root)

    image_paths = _image_repo_rel_paths(misalignment, borderline)

    return {
        "run_id": run_id,
        "policy_clarity_hot_spots": _policy_clarity_hot_spots(run_id, runs_root, image_paths),
        "majority_wrong": _majority_wrong(consensus, misalignment, image_paths),
        "model_disagreement": _model_disagreement(consensus, image_paths),
        "boundary_concentration": _boundary_concentration(borderline),
        "consistent_pair_disagreement": _consistent_pair_disagreement(consensus),
    }
