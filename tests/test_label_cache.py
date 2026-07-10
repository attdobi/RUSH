"""Tests for the cross-run label cache (pipeline/label_cache.py).

Unit layer: prompt-fingerprint sensitivity (the make-or-break identity
contract), the temp==0 → 1 round rule, majority-vote + flip-rate math, and
the serve-time token/cost strip.

Integration layer: ``run_labeling(label_cache=True)`` against an in-memory
fake store (same API as ``LabelCache``, no Postgres) — misses label live and
store; a warm cache serves votes that carry the ``label_cache`` marker, cost
zero, and no cost-ledger rows; abstain sentinels are never stored; dry runs
never construct a cache at all.

All tests are offline; nothing touches a database.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline import runner as runner_mod
from pipeline.label_cache import (
    CACHE_SAMPLES,
    majority_vote,
    prompt_fingerprint,
    required_samples,
    serve_payload,
)
from pipeline.manifest import SampleRecord
from pipeline.runner import ModelSpec, run_labeling


def _fp(**overrides):
    base = dict(
        area="Generative_AI",
        prompt_version="v0.1",
        system_prompt="SYSTEM",
        user_instructions="USER",
        response_schema={"type": "object"},
        policy_markdown="# policy v0.1\nrule one.",
        max_image_size=(1024, 1024),
        jpeg_quality=85,
        temperature=0.1,
        reasoning_effort="xhigh",
        model_params={},
    )
    base.update(overrides)
    return prompt_fingerprint(**base)


class TestPromptFingerprint(unittest.TestCase):
    def test_stable_for_identical_inputs(self):
        self.assertEqual(_fp(), _fp())

    def test_every_prompt_shaping_input_changes_the_hash(self):
        baseline = _fp()
        # Each of these is a real drift mode we have lived through: the r50
        # no-abstain prompt rewrite (system prompt), policy edits, compressed
        # vs full renders (policy_markdown), temperature/effort changes.
        variants = [
            _fp(system_prompt="SYSTEM v2"),
            _fp(user_instructions="USER v2"),
            _fp(response_schema={"type": "object", "required": ["label"]}),
            _fp(policy_markdown="# policy v0.1\nrule one, tightened."),
            _fp(max_image_size=(112, 112)),
            _fp(jpeg_quality=90),
            _fp(temperature=0.0),
            _fp(reasoning_effort="high"),
            _fp(model_params={"max_completion_tokens": 6000}),
            _fp(prompt_version="v0.2"),
            _fp(area="MNIST_Digits"),
        ]
        self.assertEqual(len(set(variants)), len(variants))
        for variant in variants:
            self.assertNotEqual(baseline, variant)

    def test_version_name_is_not_part_of_the_identity(self):
        # Same bytes under a different accepted-version NAME must still hit:
        # the fingerprint has no policy_graph_version input at all.
        self.assertEqual(
            _fp(policy_markdown="# same bytes"),
            _fp(policy_markdown="# same bytes"),
        )


class TestSamplingRule(unittest.TestCase):
    def test_temp_zero_needs_one_round(self):
        self.assertEqual(required_samples(0.0), 1)

    def test_nondeterministic_needs_cache_samples(self):
        self.assertEqual(required_samples(0.1), CACHE_SAMPLES)
        self.assertEqual(required_samples(1.0), CACHE_SAMPLES)
        # None = temperature unsupported/omitted (gpt-5.5 reasoning family):
        # provider-default decoding is nondeterministic.
        self.assertEqual(required_samples(None), CACHE_SAMPLES)


def _sample(idx: int, label: str, **response) -> dict:
    response.setdefault("label", label)
    response.setdefault("justification", f"sample {idx}")
    return {"sample_idx": idx, "label": label, "response": response}


class TestMajorityVote(unittest.TestCase):
    def test_unanimous(self):
        voted = majority_vote(
            [_sample(1, "gen_ai"), _sample(2, "gen_ai"), _sample(3, "gen_ai")]
        )
        self.assertEqual(voted["majority_label"], "gen_ai")
        self.assertEqual(voted["flip_rate"], 0.0)
        self.assertEqual(voted["n_samples"], 3)

    def test_two_one_split_serves_latest_majority_consistent_response(self):
        voted = majority_vote(
            [_sample(1, "gen_ai"), _sample(2, "not_gen_ai"), _sample(3, "gen_ai")]
        )
        self.assertEqual(voted["majority_label"], "gen_ai")
        self.assertAlmostEqual(voted["flip_rate"], round(1 / 3, 4))
        # The justification served must argue for the label served.
        self.assertEqual(voted["response"]["justification"], "sample 3")

    def test_full_tie_resolves_to_latest(self):
        voted = majority_vote(
            [_sample(1, "0"), _sample(2, "7"), _sample(3, "1")]
        )
        self.assertEqual(voted["majority_label"], "1")
        self.assertAlmostEqual(voted["flip_rate"], round(2 / 3, 4))

    def test_serve_payload_strips_token_and_cost_fields(self):
        voted = majority_vote(
            [
                _sample(
                    1,
                    "gen_ai",
                    input_tokens=1000,
                    output_tokens=50,
                    cached_input_tokens=900,
                    cache_creation_input_tokens=0,
                    cost_usd=0.01,
                    latency_ms=1234,
                    confidence=0.9,
                )
            ]
        )
        served = serve_payload(voted)
        for key in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cost_usd",
            "latency_ms",
        ):
            self.assertNotIn(key, served)
        self.assertEqual(served["confidence"], 0.9)
        self.assertEqual(served["label"], "gen_ai")


# ---------------------------------------------------------------------------
# Runner integration against an in-memory fake store
# ---------------------------------------------------------------------------


class FakeLabelCache:
    """LabelCache API over a shared in-memory dict — no Postgres."""

    shared: dict = {}

    def __init__(self, **_kwargs):
        self.enabled = True
        self.error = None
        self.stats = {"hits": 0, "misses": 0, "stored": 0}

    def lookup(self, image_sha256, prompt_sha256, model_id, *, required):
        rows = self.shared.get((image_sha256, prompt_sha256, model_id), [])
        if len(rows) < max(1, required):
            self.stats["misses"] += 1
            return None
        samples = [
            {"sample_idx": i + 1, "label": r["label"], "response": r}
            for i, r in enumerate(rows)
        ]
        self.stats["hits"] += 1
        return majority_vote(samples)

    def store(self, image_sha256, prompt_sha256, model_id, response, **_meta):
        self.shared.setdefault(
            (image_sha256, prompt_sha256, model_id), []
        ).append(dict(response))
        self.stats["stored"] += 1
        return True

    def manifest_block(self, *, samples_required):
        return {
            "enabled": self.enabled,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "stored": self.stats["stored"],
            "samples_required": dict(samples_required),
        }


def _records(n: int) -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id=f"train_{i:05d}",
            repo_rel_path=f"data/fake/img_{i}.jpg",
            split="dev_golden",
            sme_label_raw="ai_generated",
            sme_label="gen_ai",
            dataset="fake",
            sha256=f"{i:064x}",
            sampling_version="test-sampling-v1",
        )
        for i in range(1, n + 1)
    ]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestRunnerIntegration(unittest.TestCase):
    def setUp(self):
        FakeLabelCache.shared = {}
        self._real_cache = runner_mod.LabelCache
        runner_mod.LabelCache = FakeLabelCache

    def tearDown(self):
        runner_mod.LabelCache = self._real_cache

    def _run(self, runs_root: Path, **kwargs):
        # resolved_temperature=0.0 → required_samples == 1, so the second
        # pass hits without needing three warm-up runs.
        return run_labeling(
            models=[ModelSpec(model_id="openai/fake-judge", resolved_temperature=0.0)],
            samples=_records(4),
            split="dev_golden",
            runs_root=runs_root,
            dry_run=False,
            label_cache=True,
            **kwargs,
        )

    def test_cold_then_warm_run(self):
        with TemporaryDirectory() as tmp:
            runs_root = Path(tmp)

            cold = self._run(runs_root)
            self.assertEqual(cold.label_cache_hits, 0)
            # The fake client abstains on some hash buckets; abstains are
            # never stored, everything else is.
            cold_votes = _read_jsonl(cold.paths.root / "label_votes.jsonl")
            n_decisive = sum(1 for v in cold_votes if v["label"] != "abstain")
            self.assertEqual(cold.label_cache_stored, n_decisive)
            self.assertTrue(all("label_cache" not in v for v in cold_votes))

            warm = self._run(runs_root)
            self.assertEqual(warm.label_cache_hits, n_decisive)
            # Hits are not re-stored, and the abstains that re-ran live are
            # never stored — a warm pass writes nothing new.
            self.assertEqual(warm.label_cache_stored, 0)

            warm_votes = _read_jsonl(warm.paths.root / "label_votes.jsonl")
            served = [v for v in warm_votes if "label_cache" in v]
            self.assertEqual(len(served), n_decisive)
            for vote in served:
                self.assertTrue(vote["label_cache"]["hit"])
                self.assertEqual(vote["label_cache"]["n_samples"], 1)
                self.assertEqual(vote["label_cache"]["flip_rate"], 0.0)
                self.assertEqual(vote["cost_usd"], 0.0)
                self.assertNotIn("input_tokens", vote)

            # Served votes never land in the cost ledger.
            warm_costs = _read_jsonl(warm.paths.root / "costs.jsonl")
            served_ids = {v["image_id"] for v in served}
            self.assertTrue(
                served_ids.isdisjoint({c["image_id"] for c in warm_costs})
            )

            manifest = json.loads(
                (warm.paths.root / "run_manifest.json").read_text(encoding="utf-8")
            )
            block = manifest["label_cache"]
            self.assertTrue(block["enabled"])
            self.assertEqual(block["hits"], n_decisive)
            self.assertEqual(
                block["samples_required"], {"openai/fake-judge": 1}
            )

    def test_nondeterministic_judge_takes_three_rounds(self):
        with TemporaryDirectory() as tmp:
            runs_root = Path(tmp)
            spec = ModelSpec(model_id="openai/fake-judge", resolved_temperature=0.7)
            for expected_hits in (0, 0, 0):
                summary = run_labeling(
                    models=[spec],
                    samples=_records(2),
                    split="dev_golden",
                    runs_root=runs_root,
                    dry_run=False,
                    label_cache=True,
                )
                self.assertEqual(summary.label_cache_hits, expected_hits)
            # Fourth round: three samples stored per decisive key → majority served.
            summary = run_labeling(
                models=[spec],
                samples=_records(2),
                split="dev_golden",
                runs_root=runs_root,
                dry_run=False,
                label_cache=True,
            )
            self.assertGreater(summary.label_cache_hits, 0)
            votes = _read_jsonl(summary.paths.root / "label_votes.jsonl")
            served = [v for v in votes if "label_cache" in v]
            for vote in served:
                self.assertEqual(vote["label_cache"]["n_samples"], 3)
                # The deterministic fake never flips, so majority is clean.
                self.assertEqual(vote["label_cache"]["flip_rate"], 0.0)

    def test_dry_run_never_constructs_a_cache(self):
        constructed = []

        class ExplodingCache(FakeLabelCache):
            def __init__(self, **kwargs):
                constructed.append(1)
                super().__init__(**kwargs)

        runner_mod.LabelCache = ExplodingCache
        with TemporaryDirectory() as tmp:
            summary = run_labeling(
                models=["openai/fake-judge"],
                samples=_records(2),
                split="dev_golden",
                runs_root=Path(tmp),
                dry_run=True,
                label_cache=True,
            )
            self.assertEqual(constructed, [])
            manifest = json.loads(
                (summary.paths.root / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["label_cache"]["enabled"])
            self.assertIn("dry runs", manifest["label_cache"]["reason"])


if __name__ == "__main__":
    unittest.main()
