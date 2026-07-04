from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pipeline.manifest import SampleRecord
from pipeline.providers.base import LabelRequest, LabelResponse
from pipeline.runner import DeterministicFakeClient, run_labeling


SAMPLES = [
    SampleRecord(
        sample_id=f"dev_golden_{idx:04d}",
        repo_rel_path=f"data/images/test/dev_golden_{idx:04d}.jpg",
        split="dev_golden",
        sme_label_raw="ai_generated" if idx % 2 else "not_ai_generated",
        sme_label="gen_ai" if idx % 2 else "not_gen_ai",
        dataset="test",
        sha256=f"{idx:064x}"[-64:],
        sampling_version="test-sampling-v1",
    )
    for idx in range(1, 8)
]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CountingBatchClient(DeterministicFakeClient):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.label_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        self.batch_models: list[set[str]] = []

    def label(self, request: LabelRequest) -> LabelResponse:
        self.label_calls += 1
        return super().label(request)

    def batch_label(self, requests: list[LabelRequest]) -> list[LabelResponse]:
        self.batch_calls += 1
        self.batch_sizes.append(len(requests))
        self.batch_models.append({request.model_id for request in requests})
        return super().batch_label(requests)


class LabelOnlyClient(DeterministicFakeClient):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.label_calls = 0

    def label(self, request: LabelRequest) -> LabelResponse:
        self.label_calls += 1
        return super().label(request)

    def batch_label(self, requests: list[LabelRequest]) -> list[LabelResponse]:  # pragma: no cover
        raise AssertionError("batch_label must not be used for batch_size=1")


def _vote_key(row: dict) -> tuple[str, str]:
    return (row["image_id"], row["labeler_id"])


def test_batch_size_rejects_values_below_one() -> None:
    with TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            run_labeling(
                models=["openai/gpt-5.5"],
                split="dev_golden",
                limit=1,
                samples=SAMPLES,
                runs_root=Path(tmp),
                batch_size=0,
            )


def test_batch_size_greater_than_one_reduces_distinct_label_calls() -> None:
    with TemporaryDirectory() as tmp:
        clients: dict[str, CountingBatchClient] = {}

        def factory(spec):
            client = CountingBatchClient(spec.model_id)
            clients[spec.model_id] = client
            return client

        summary = run_labeling(
            models=["openai/gpt-5.5"],
            split="dev_golden",
            limit=5,
            samples=SAMPLES,
            runs_root=Path(tmp),
            client_factory=factory,
            concurrency=1,
            batch_size=3,
        )

        client = clients["openai/gpt-5.5"]
        assert summary.expected_calls == 5
        assert summary.completed_calls == 5
        assert summary.effective_batches == 2
        assert client.label_calls == 0
        assert client.batch_calls == 2
        assert client.batch_sizes == [3, 2]

        manifest = json.loads(summary.paths.manifest.read_text())
        assert manifest["batch_size"] == 3
        assert manifest["effective_batches"] == 2
        assert len(_read_jsonl(summary.paths.label_votes)) == 5


def test_batching_is_per_model_one_provider_call_per_model_batch() -> None:
    with TemporaryDirectory() as tmp:
        clients: dict[str, CountingBatchClient] = {}

        def factory(spec):
            client = CountingBatchClient(spec.model_id)
            clients[spec.model_id] = client
            return client

        summary = run_labeling(
            models=["openai/gpt-5.5", "anthropic/claude-opus-4-6"],
            split="dev_golden",
            limit=6,
            samples=SAMPLES,
            runs_root=Path(tmp),
            client_factory=factory,
            concurrency=1,
            batch_size=3,
        )

        assert summary.expected_calls == 12
        assert summary.completed_calls == 12
        assert summary.effective_batches == 4
        for model_id, client in clients.items():
            assert client.label_calls == 0
            assert client.batch_calls == 2
            assert client.batch_sizes == [3, 3]
            assert client.batch_models == [{model_id}, {model_id}]


def test_local_models_stay_single_image_when_api_models_batch() -> None:
    with TemporaryDirectory() as tmp:
        clients: dict[str, CountingBatchClient] = {}

        def factory(spec):
            client = CountingBatchClient(spec.model_id)
            clients[spec.model_id] = client
            return client

        summary = run_labeling(
            models=["local/gemma-4-26b-a4b-qat", "openai/gpt-5.5"],
            split="dev_golden",
            limit=5,
            samples=SAMPLES,
            runs_root=Path(tmp),
            client_factory=factory,
            concurrency=1,
            batch_size=3,
        )

        local_client = clients["local/gemma-4-26b-a4b-qat"]
        api_client = clients["openai/gpt-5.5"]
        assert summary.expected_calls == 10
        assert summary.completed_calls == 10
        assert summary.effective_batches == 7
        assert local_client.label_calls == 5
        assert local_client.batch_calls == 0
        assert api_client.label_calls == 0
        assert api_client.batch_calls == 2
        assert api_client.batch_sizes == [3, 2]
        assert api_client.batch_models == [{"openai/gpt-5.5"}, {"openai/gpt-5.5"}]


def test_batched_outputs_match_single_mode_labels() -> None:
    models = ["openai/gpt-5.5", "anthropic/claude-opus-4-6"]
    with TemporaryDirectory() as tmp_single, TemporaryDirectory() as tmp_batched:
        single = run_labeling(
            models=models,
            split="dev_golden",
            limit=4,
            samples=SAMPLES,
            runs_root=Path(tmp_single),
            client_factory=lambda spec: DeterministicFakeClient(spec.model_id),
            concurrency=1,
            batch_size=1,
        )
        batched = run_labeling(
            models=models,
            split="dev_golden",
            limit=4,
            samples=SAMPLES,
            runs_root=Path(tmp_batched),
            client_factory=lambda spec: DeterministicFakeClient(spec.model_id),
            concurrency=1,
            batch_size=3,
        )

        single_votes = {_vote_key(row): row for row in _read_jsonl(single.paths.label_votes)}
        batched_votes = {_vote_key(row): row for row in _read_jsonl(batched.paths.label_votes)}
        assert set(batched_votes) == set(single_votes)
        for key, batched_row in batched_votes.items():
            single_row = single_votes[key]
            assert batched_row["label"] == single_row["label"]
            assert batched_row["l2_label"] == single_row["l2_label"]
            assert batched_row["confidence"] == single_row["confidence"]
            assert batched_row["prepared_image_sha256"] == single_row["prepared_image_sha256"]


def test_batch_size_one_preserves_single_label_path_outputs() -> None:
    with TemporaryDirectory() as tmp:
        clients: dict[str, LabelOnlyClient] = {}

        def factory(spec):
            client = LabelOnlyClient(spec.model_id)
            clients[spec.model_id] = client
            return client

        summary = run_labeling(
            models=["openai/gpt-5.5"],
            split="dev_golden",
            limit=3,
            samples=SAMPLES,
            runs_root=Path(tmp),
            client_factory=factory,
            concurrency=1,
            batch_size=1,
        )

        assert clients["openai/gpt-5.5"].label_calls == 3
        assert summary.effective_batches == 3
        votes = _read_jsonl(summary.paths.label_votes)
        for row in votes:
            row.pop("run_id", None)

        with TemporaryDirectory() as tmp_baseline:
            baseline = run_labeling(
                models=["openai/gpt-5.5"],
                split="dev_golden",
                limit=3,
                samples=SAMPLES,
                runs_root=Path(tmp_baseline),
                client_factory=lambda spec: DeterministicFakeClient(spec.model_id),
                concurrency=1,
                batch_size=1,
            )
            baseline_votes = _read_jsonl(baseline.paths.label_votes)
            for row in baseline_votes:
                row.pop("run_id", None)

        assert votes == baseline_votes
