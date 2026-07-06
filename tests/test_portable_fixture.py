from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pipeline import io_paths
from pipeline.io_paths import (
    DEFAULT_SAMPLE_MANIFEST,
    GENAI_PORTABLE_MANIFEST,
    REPO_ROOT,
    genai_manifest_default,
)
from pipeline.manifest import load_records
from pipeline.web import run_registry as run_registry_mod
from pipeline.web.run_registry import RunRegistry
from pipeline.web._safety import safe_static_path


def _source_tree_available() -> bool:
    return io_paths._genai_source_tree_has_images()  # type: ignore[attr-defined]


def test_portable_fixture_builder_is_deterministic_and_balanced() -> None:
    if not DEFAULT_SAMPLE_MANIFEST.exists() or not _source_tree_available():
        pytest.skip("full GenAI source tree is required for builder determinism")

    from scripts import build_portable_fixture as builder

    rows = builder.read_jsonl(DEFAULT_SAMPLE_MANIFEST)
    candidates = builder.candidates_from_rows(rows)
    # The committed fixture is built with --max-mb 50 --per-stratum 12.
    selected_once = builder.select_per_stratum(candidates, 50 * 1024 * 1024, 12)
    selected_twice = builder.select_per_stratum(candidates, 50 * 1024 * 1024, 12)

    assert [c.sample_id for c in selected_once] == [c.sample_id for c in selected_twice]

    def portable_manifest_text(selected: list[builder.Candidate]) -> str:
        out: list[str] = []
        for candidate in selected:
            row = dict(candidate.record)
            row["repo_rel_path"] = builder.repo_rel(
                builder.sample_target(candidate, builder.DEFAULT_SAMPLE_ROOT)
            )
            out.append(json.dumps(row, ensure_ascii=True) + "\n")
        return "".join(out)

    manifest_once = portable_manifest_text(selected_once)
    manifest_twice = portable_manifest_text(selected_twice)
    assert manifest_once == manifest_twice
    assert manifest_once == GENAI_PORTABLE_MANIFEST.read_text(encoding="utf-8")

    strata = Counter(candidate.stratum for candidate in selected_once)
    assert len(strata) == 12
    assert {label for _dataset, label, _split in strata} == {
        "ai_generated",
        "not_ai_generated",
    }
    assert {split for _dataset, _label, split in strata} == {"dev_golden", "holdout"}
    assert len(selected_once) == 72
    assert sum(candidate.size_bytes for candidate in selected_once) <= 50 * 1024 * 1024


def test_portable_manifest_integrity() -> None:
    records = load_records(GENAI_PORTABLE_MANIFEST)

    assert len(records) == 72
    for record in records:
        assert record.repo_rel_path.startswith("data/images/genai-classification/sample/")
        assert (REPO_ROOT / record.repo_rel_path).is_file()


def test_genai_manifest_default_env_force(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUSH_PORTABLE", "TrUe")

    assert genai_manifest_default() == GENAI_PORTABLE_MANIFEST


def test_genai_manifest_default_uses_full_manifest_when_source_tree_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _source_tree_available():
        pytest.skip("full GenAI source tree is absent in this checkout")
    monkeypatch.delenv("RUSH_PORTABLE", raising=False)

    assert genai_manifest_default() == DEFAULT_SAMPLE_MANIFEST


def test_genai_manifest_default_uses_portable_when_source_tree_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUSH_PORTABLE", raising=False)
    monkeypatch.setattr(io_paths, "_genai_source_tree_has_images", lambda: False)

    assert genai_manifest_default() == GENAI_PORTABLE_MANIFEST


def test_run_bulk_labeling_plan_only_uses_portable_manifest_with_env() -> None:
    expected = sum(1 for record in load_records(GENAI_PORTABLE_MANIFEST) if record.split == "dev_golden")
    env = os.environ.copy()
    env["RUSH_PORTABLE"] = "1"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_bulk_labeling.py",
            "--models",
            "openai/gpt-5.5",
            "--split",
            "dev_golden",
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["n_samples"] == expected == 40


def test_web_start_job_uses_portable_manifest_with_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list["FakePopen"] = []

    class EmptyStdout:
        def __iter__(self):
            return iter(())

    class FakePopen:
        def __init__(self, argv, **kwargs):  # noqa: ANN001
            self.argv = argv
            self.pid = 24680
            self.stdout = EmptyStdout()
            created.append(self)

        def poll(self):  # noqa: ANN201
            return 0

        def wait(self):  # noqa: ANN201
            return 0

    monkeypatch.setenv("RUSH_PORTABLE", "1")
    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", FakePopen)

    state = RunRegistry(tmp_path).start_job(
        {
            "models": ["openai/gpt-5.5"],
            "split": "dev_golden",
            "limit": 3,
            "sample_ids": None,
            "policy_version": "v0.1",
            "mode": "cold_start",
            "reasoning_effort": "high",
            "allow_spend": True,
            "allow_holdout": False,
            "concurrency": 1,
        }
    )

    argv = created[0].argv
    assert argv[argv.index("--manifest") + 1] == str(GENAI_PORTABLE_MANIFEST)
    assert state["sample_manifest_path"] == str(GENAI_PORTABLE_MANIFEST)


def test_sample_static_allowlist_serves_portable_images() -> None:
    first = load_records(GENAI_PORTABLE_MANIFEST)[0]
    resolved = safe_static_path(REPO_ROOT, "/" + first.repo_rel_path)

    assert resolved == (REPO_ROOT / first.repo_rel_path).resolve()
