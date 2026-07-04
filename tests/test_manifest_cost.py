"""Per-batch + per-image cost ledger in the run manifest (X1 backend)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

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
    for idx in range(1, 7)
]


class UsageClient(DeterministicFakeClient):
    """Fake client that attaches deterministic token usage per response."""

    def _response_for(self, request: LabelRequest) -> LabelResponse:
        resp = super()._response_for(request)
        resp.input_tokens = 1000
        resp.output_tokens = 200
        return resp


def _load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text())


def test_manifest_records_priced_cost_block():
    with TemporaryDirectory() as tmp:
        summary = run_labeling(
            models=["openai/gpt-5.5"],
            samples=SAMPLES,
            split="dev_golden",
            runs_root=Path(tmp),
            client_factory=lambda spec: UsageClient(spec.model_id),
            batch_size=3,
            dry_run=True,
        )
        manifest = _load_manifest(summary.paths.root)
        cost = manifest["cost"]
        # 6 images / batch_size 3 => 2 batches.
        assert len(cost["per_batch"]) == 2
        assert cost["priced_images"] == 6
        # gpt-5.5: 1.25/Mtok in, 10/Mtok out. Per image = 1.25e-3 + 2e-3.
        per_image = 1.25 * 1000 / 1e6 + 10.0 * 200 / 1e6
        assert abs(cost["total_cost_usd"] - per_image * 6) < 1e-9
        assert abs(cost["cost_per_image_usd"] - per_image) < 1e-9
        for batch in cost["per_batch"]:
            assert batch["images"] == 3
            assert abs(batch["cost_usd"] - per_image * 3) < 1e-9
        assert abs(summary.total_cost_usd - per_image * 6) < 1e-9


def test_local_models_are_free():
    with TemporaryDirectory() as tmp:
        summary = run_labeling(
            models=["local/qwen3.6-27b"],
            samples=SAMPLES,
            split="dev_golden",
            runs_root=Path(tmp),
            client_factory=lambda spec: UsageClient(spec.model_id),
            batch_size=20,
            dry_run=True,
        )
        manifest = _load_manifest(summary.paths.root)
        assert manifest["cost"]["total_cost_usd"] == 0.0
