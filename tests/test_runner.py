"""Unit tests for the bulk-labeling runner (X2).

Covers:
* sample manifest loader + ground-truth join,
* deterministic ordering of (sample, model) dispatch,
* atomic JSONL append + schema validation in persistence,
* full run end-to-end with the deterministic fake client (no network),
* propagation of prepared-image audit metadata into both label_votes and
  llm_outputs records (the Pista 2026-05-10 correction),
* run-manifest schema validation (start state and finished state),
* holdout split guard,
* CLI plan-only mode.

All tests are offline. They use a temp ``runs_root`` so nothing lands in
``data/runs/``. Compatible with both ``pytest`` and ``python -m unittest``.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

# Repo root on sys.path so ``import pipeline`` works under both runners.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline import persistence
from pipeline.io_paths import (
    DEFAULT_POLICY_GRAPH_DIR,
    DEFAULT_SAMPLE_MANIFEST,
    is_valid_run_id,
    mint_run_id,
    run_paths,
)
from pipeline.manifest import (
    SME_LABEL_MAP,
    build_ground_truth,
    iter_records,
    load_policy_markdown,
    load_records,
    select_samples,
)
from pipeline.providers.base import LabelRequest, LabelResponse
from pipeline.runner import (
    DEFAULT_PROMPT_VERSION,
    DeterministicFakeClient,
    IMAGE_PREP_FORMAT,
    IMAGE_PREP_LONGEST_EDGE_PX,
    IMAGE_PREP_QUALITY,
    ModelSpec,
    deterministic_fake_factory,
    run_labeling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stable_factory(_spec):
    """Always return a fresh deterministic fake (avoids cross-test state)."""
    return DeterministicFakeClient(model_id=_spec.model_id)


# ---------------------------------------------------------------------------
# io_paths
# ---------------------------------------------------------------------------


class TestIoPaths(unittest.TestCase):
    def test_run_id_format(self):
        rid = mint_run_id()
        self.assertTrue(is_valid_run_id(rid), f"bad run_id: {rid}")

    def test_run_id_rejected(self):
        for bad in ["", "20260510-abc", "abc", "20260510T1234-abcdef00"]:
            with self.assertRaises(ValueError):
                run_paths(bad)

    def test_run_paths_layout(self):
        with TemporaryDirectory() as tmp:
            rid = mint_run_id()
            paths = run_paths(rid, runs_root=Path(tmp))
            paths.ensure()
            self.assertTrue(paths.root.is_dir())
            self.assertTrue(paths.scoring_dir.is_dir())
            self.assertTrue(paths.web_dir.is_dir())
            # Property paths are stable strings under root.
            self.assertEqual(paths.label_votes.name, "label_votes.jsonl")
            self.assertEqual(paths.llm_outputs.name, "llm_outputs.jsonl")
            self.assertEqual(paths.errors.name, "errors.jsonl")
            self.assertEqual(paths.manifest.name, "run_manifest.json")


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class TestManifestLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(DEFAULT_SAMPLE_MANIFEST)

    def test_loader_yields_200(self):
        self.assertEqual(len(self.records), 200)

    def test_label_mapping_complete(self):
        for rec in self.records:
            self.assertIn(rec.sme_label_raw, SME_LABEL_MAP)
            self.assertEqual(rec.sme_label, SME_LABEL_MAP[rec.sme_label_raw])
            self.assertIn(rec.split, {"dev_golden", "holdout"})

    def test_select_samples_deterministic(self):
        a = select_samples(self.records, split="dev_golden", limit=5)
        b = select_samples(self.records, split="dev_golden", limit=5)
        self.assertEqual([r.sample_id for r in a], [r.sample_id for r in b])
        # Sorted ascending by sample_id (§5.6 determinism).
        self.assertEqual(
            [r.sample_id for r in a],
            sorted(r.sample_id for r in a),
        )
        self.assertEqual(len(a), 5)

    def test_select_by_explicit_ids(self):
        target = {"dev_golden_0001", "dev_golden_0002", "holdout_0001"}
        sel = select_samples(self.records, sample_ids=target)
        self.assertEqual({r.sample_id for r in sel}, target)

    def test_ground_truth_dict(self):
        sel = select_samples(self.records, split="dev_golden", limit=3)
        gt = build_ground_truth(sel)
        for rec in sel:
            self.assertEqual(gt[rec.sample_id], rec.sme_label)

    def test_load_policy_markdown_concatenates(self):
        text = load_policy_markdown(DEFAULT_POLICY_GRAPH_DIR)
        self.assertIn("GA.root.md", text)
        # File header markers separate nodes.
        self.assertGreater(text.count("<!-- GA."), 5)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = run_paths(mint_run_id(), runs_root=Path(self.tmp.name))
        self.paths.ensure()

    def _valid_vote(self) -> dict:
        return {
            "run_id": self.paths.run_id,
            "image_id": "dev_golden_0001",
            "labeler_type": "llm",
            "labeler_id": "openai/gpt-5.5",
            "model_id": "openai/gpt-5.5",
            "label": "gen_ai",
            "node_ids": [],
            "confidence": 0.92,
            "justification": "Looks like an obvious GenAI render.",
            "policy_graph_version": "v0.1",
            "prompt_version": "v0.1",
            "label_tier": "provisional",
            "is_boundary": False,
            "difficulty": "low",
            "prepared_image_sha256": "a" * 64,
            "prepared_image_width": 1024,
            "prepared_image_height": 1024,
            "prepared_image_mime_type": "image/jpeg",
            "prepared_image_byte_size": 84321,
            "latency_ms": 1234,
            "attempts": 1,
        }

    def test_append_label_vote_validates_and_persists(self):
        persistence.append_label_vote(self.paths, self._valid_vote())
        rows = _read_jsonl(self.paths.label_votes)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prepared_image_sha256"], "a" * 64)
        self.assertEqual(rows[0]["prepared_image_byte_size"], 84321)

    def test_invalid_label_vote_routed_to_errors(self):
        bad = self._valid_vote()
        bad["label"] = "definitely_not_an_enum_value"
        with self.assertRaises(persistence.PersistenceError):
            persistence.append_label_vote(self.paths, bad)
        # Nothing should land in label_votes...
        self.assertFalse(self.paths.label_votes.exists())
        # ...but errors.jsonl should have a row.
        errors = _read_jsonl(self.paths.errors)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "label_vote_validation")

    def test_append_llm_output_includes_prepared_image_fields(self):
        out = {
            "label": "gen_ai",
            "l2_label": "",
            "justification": "Smooth plastic skin and broken hand.",
            "confidence": 0.81,
            "difficulty": "medium",
            "is_boundary": False,
            "prepared_image_sha256": "b" * 64,
            "prepared_image_width": 1024,
            "prepared_image_height": 768,
            "prepared_image_mime_type": "image/jpeg",
            "prepared_image_byte_size": 51200,
        }
        persistence.append_llm_output(
            self.paths, out, image_id="dev_golden_0001", model_id="openai/gpt-5.5"
        )
        rows = _read_jsonl(self.paths.llm_outputs)
        self.assertEqual(len(rows), 1)
        envelope = rows[0]
        self.assertEqual(envelope["image_id"], "dev_golden_0001")
        self.assertEqual(envelope["model_id"], "openai/gpt-5.5")
        self.assertEqual(envelope["output"]["prepared_image_sha256"], "b" * 64)
        self.assertEqual(envelope["output"]["prepared_image_height"], 768)

    def test_run_manifest_validation(self):
        manifest = {
            "run_id": self.paths.run_id,
            "started_at": "2026-05-10T17:00:00Z",
            "finished_at": None,
            "sample_manifest_path": "data/images/genai-classification/manifests/combined_labels.jsonl",
            "sample_ids": ["dev_golden_0001"],
            "models": [{"model_id": "openai/gpt-5.5"}],
            "policy_graph_version": "v0.1",
            "prompt_version": "v0.1",
            "sampling_version": "genai-gold-sampling-v1",
            "concurrency": 1,
            "image_prep": {
                "longest_edge_px": IMAGE_PREP_LONGEST_EDGE_PX,
                "format": IMAGE_PREP_FORMAT,
                "quality": IMAGE_PREP_QUALITY,
                "helper_module": "pipeline.labeling.image_prep",
            },
            "totals": {"expected_calls": 1, "completed_calls": 0, "errored_calls": 0},
            "dry_run": True,
        }
        persistence.write_run_manifest(self.paths, manifest)
        loaded = json.loads(self.paths.manifest.read_text())
        self.assertEqual(loaded["run_id"], self.paths.run_id)
        self.assertIsNone(loaded["finished_at"])

    def test_strip_image_bytes_defence_in_depth(self):
        nested = {
            "messages": [
                {
                    "image": "AAAAAA",  # short -> kept as-is by base, but our scrub keys-by-name
                    "image_url": "data:image/jpeg;base64,XXXXX",
                    "inline_data": {"mime_type": "image/jpeg", "data": "yyyy"},
                }
            ],
            "ok": True,
        }
        scrubbed = persistence._strip_image_bytes(nested)
        msg = scrubbed["messages"][0]
        self.assertEqual(msg["image"], "<image-bytes-omitted>")
        self.assertEqual(msg["image_url"], "<image-bytes-omitted>")
        self.assertEqual(msg["inline_data"]["data"], "<image-bytes-omitted>")
        self.assertEqual(scrubbed["ok"], True)


# ---------------------------------------------------------------------------
# Deterministic fake client
# ---------------------------------------------------------------------------


class TestDeterministicFakeClient(unittest.TestCase):
    def test_response_is_stable(self):
        a = DeterministicFakeClient("openai/gpt-5.5")
        b = DeterministicFakeClient("openai/gpt-5.5")
        req = LabelRequest(
            image_path=Path("/dev/null"),
            image_id="dev_golden_0001",
            policy_markdown="...",
            policy_graph_version="v0.1",
            prompt_version="v0.1",
            model_id="openai/gpt-5.5",
        )
        ra = a.label(req)
        rb = b.label(req)
        self.assertEqual(ra.label, rb.label)
        self.assertEqual(ra.confidence, rb.confidence)
        self.assertEqual(ra.prepared_image_sha256, rb.prepared_image_sha256)

    def test_response_has_audit_metadata(self):
        client = DeterministicFakeClient("anthropic/claude-opus-4-6")
        req = LabelRequest(
            image_path=Path("/dev/null"),
            image_id="holdout_0042",
            policy_markdown="...",
            policy_graph_version="v0.1",
            prompt_version="v0.1",
            model_id="anthropic/claude-opus-4-6",
        )
        resp = client.label(req)
        self.assertEqual(len(resp.prepared_image_sha256), 64)
        self.assertEqual(resp.prepared_image_mime_type, "image/jpeg")
        self.assertEqual(resp.prepared_image_width, IMAGE_PREP_LONGEST_EDGE_PX)
        self.assertGreater(resp.prepared_image_byte_size, 0)
        self.assertIsNone(resp.input_tokens)
        self.assertIsNone(resp.output_tokens)
        self.assertIsNone(resp.cost_usd)
        self.assertIn(resp.label, {"gen_ai", "not_gen_ai", "abstain"})


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


class TestRunLabelingE2E(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs_root = Path(self.tmp.name)

    def _run(self, *, models, limit=3, concurrency=1, allow_holdout=False, split="dev_golden"):
        return run_labeling(
            models=models,
            split=split,
            limit=limit,
            runs_root=self.runs_root,
            client_factory=_stable_factory,
            concurrency=concurrency,
            allow_holdout=allow_holdout,
        )

    def test_smoke_3_samples_x_2_models(self):
        summary = self._run(models=["openai/gpt-5.5", "anthropic/claude-opus-4-6"], limit=3)
        self.assertEqual(summary.expected_calls, 6)
        self.assertEqual(summary.completed_calls, 6)
        self.assertEqual(summary.errored_calls, 0)

        # Manifest exists and was finalized.
        manifest = json.loads(summary.paths.manifest.read_text())
        self.assertIsNotNone(manifest["finished_at"])
        self.assertEqual(manifest["totals"]["completed_calls"], 6)
        self.assertEqual(manifest["totals"]["errored_calls"], 0)
        self.assertEqual(
            sorted(m["model_id"] for m in manifest["models"]),
            ["anthropic/claude-opus-4-6", "openai/gpt-5.5"],
        )
        self.assertEqual(manifest["image_prep"]["longest_edge_px"], 1024)
        self.assertEqual(manifest["image_prep"]["format"], "JPEG")
        self.assertEqual(manifest["image_prep"]["quality"], 85)

        # Both JSONL files have 6 rows each.
        votes = _read_jsonl(summary.paths.label_votes)
        outputs = _read_jsonl(summary.paths.llm_outputs)
        self.assertEqual(len(votes), 6)
        self.assertEqual(len(outputs), 6)

        # Every row carries the prepared-image audit metadata.
        for v in votes:
            self.assertIn("prepared_image_sha256", v)
            self.assertEqual(len(v["prepared_image_sha256"]), 64)
            self.assertEqual(v["prepared_image_width"], 1024)
            self.assertEqual(v["prepared_image_mime_type"], "image/jpeg")
            self.assertGreater(v["prepared_image_byte_size"], 0)
            self.assertNotIn("input_tokens", v)
            self.assertNotIn("output_tokens", v)
            self.assertNotIn("cost_usd", v)
        for envelope in outputs:
            out = envelope["output"]
            self.assertIn("prepared_image_sha256", out)
            self.assertEqual(len(out["prepared_image_sha256"]), 64)
            self.assertNotIn("input_tokens", out)
            self.assertNotIn("output_tokens", out)
            self.assertNotIn("cost_usd", out)

    def test_run_is_byte_stable_across_repeats(self):
        s1 = self._run(models=["openai/gpt-5.5"], limit=2)
        s2 = self._run(models=["openai/gpt-5.5"], limit=2)
        v1 = _read_jsonl(s1.paths.label_votes)
        v2 = _read_jsonl(s2.paths.label_votes)
        # Strip the run_id (per-run unique) before comparing.
        for row in v1 + v2:
            row.pop("run_id", None)
        # latency_ms is 0 for the dry-run client; image audit fields are stable.
        self.assertEqual(v1, v2)

    def test_holdout_requires_allow_flag(self):
        with self.assertRaises(PermissionError):
            self._run(models=["openai/gpt-5.5"], split="holdout", limit=1)

    def test_holdout_allowed_with_flag(self):
        summary = self._run(
            models=["openai/gpt-5.5"], split="holdout", limit=1, allow_holdout=True
        )
        self.assertEqual(summary.completed_calls, 1)

    def test_failing_client_writes_error(self):
        class BoomClient:
            model_id = "openai/gpt-5.5"

            def label(self, request):  # noqa: D401
                raise RuntimeError("simulated provider boom")

        summary = run_labeling(
            models=["openai/gpt-5.5"],
            split="dev_golden",
            limit=2,
            runs_root=self.runs_root,
            client_factory=lambda spec: BoomClient(),
            concurrency=1,
        )
        self.assertEqual(summary.completed_calls, 0)
        self.assertEqual(summary.errored_calls, 2)
        errors = _read_jsonl(summary.paths.errors)
        self.assertEqual(len(errors), 2)
        for err in errors:
            self.assertEqual(err["stage"], "provider_call")
            self.assertIn("simulated provider boom", err["reason"])


# ---------------------------------------------------------------------------
# CLI plan-only path
# ---------------------------------------------------------------------------


class TestPlanOnlyCli(unittest.TestCase):
    def test_plan_only_exits_clean(self):
        from scripts.run_bulk_labeling import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main([
                "--models", "openai/gpt-5.5,anthropic/claude-opus-4-6",
                "--split", "dev_golden",
                "--limit", "4",
                "--plan-only",
            ])
        self.assertEqual(rc, 0)
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan["n_samples"], 4)
        self.assertEqual(plan["n_calls"], 8)
        self.assertEqual(plan["models"], ["openai/gpt-5.5", "anthropic/claude-opus-4-6"])
        self.assertTrue(plan["dry_run"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
