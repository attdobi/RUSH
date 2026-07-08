"""Lane-parallelism / anti-starvation tests for the runner dispatch.

These assert the structural property the per-lane executor design guarantees:
a slow/blocked local lane must NEVER hold worker slots belonging to another
lane, and distinct local models must dispatch concurrently (one per GPU card).

We drive this with an instrumented fake client that blocks on a controllable
barrier so we can observe true in-flight overlap across lanes rather than
relying on wall-clock timing.
"""
from __future__ import annotations

import threading
import time

from pipeline.manifest import SampleRecord
from pipeline.runner import ModelSpec, run_labeling
from pipeline.providers.base import LabelClient, LabelRequest, LabelResponse


def _samples(n: int) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for i in range(n):
        out.append(
            SampleRecord(
                sample_id=f"dev_golden_{i:04d}",
                repo_rel_path=f"data/images/test/dev_golden_{i:04d}.jpg",
                split="dev_golden",
                sme_label_raw="ai_generated",
                sme_label="gen_ai",
                dataset="test",
                sha256=f"{i:064x}"[-64:],
                sampling_version="test-sampling-v1",
            )
        )
    return out


class _InstrumentedClient(LabelClient):
    """Records concurrent in-flight windows per model_id, no network.

    ``block_models`` names models whose calls park on ``release_event`` until
    the test releases them, letting us prove other lanes progress meanwhile.
    """

    def __init__(
        self,
        model_id: str,
        *,
        state: dict,
        lock: threading.Lock,
        block_models: set[str],
        release_event: threading.Event,
    ) -> None:
        self.model_id = model_id
        self._state = state
        self._lock = lock
        self._block_models = block_models
        self._release = release_event

    def _fake(self, request: LabelRequest) -> LabelResponse:
        with self._lock:
            cur = self._state["inflight"].get(self.model_id, 0) + 1
            self._state["inflight"][self.model_id] = cur
            self._state["max_inflight"][self.model_id] = max(
                self._state["max_inflight"].get(self.model_id, 0), cur
            )
            # Snapshot the set of models running concurrently right now.
            active = {m for m, c in self._state["inflight"].items() if c > 0}
            self._state["co_active"].append(frozenset(active))
        try:
            if self.model_id in self._block_models:
                # Park this lane until released (simulates a slow local card).
                self._release.wait(timeout=10.0)
            else:
                time.sleep(0.01)
            return LabelResponse(
                image_id=request.image_id,
                model_id=request.model_id,
                label="abstain",
                l2_label="",
                justification="instrumented",
                confidence=0.5,
                difficulty="low",
                is_boundary=False,
                raw_provider_payload={"instrumented": True},
                error=None,
                latency_ms=1,
                attempts=1,
                prepared_image_sha256="0" * 64,
                prepared_image_width=112,
                prepared_image_height=112,
                prepared_image_mime_type="image/jpeg",
                prepared_image_byte_size=1234,
            )
        finally:
            with self._lock:
                self._state["inflight"][self.model_id] -= 1

    def label(self, request: LabelRequest) -> LabelResponse:
        return self._fake(request)

    def batch_label(self, requests: list[LabelRequest]) -> list[LabelResponse]:
        return [self._fake(r) for r in requests]


def _make_factory(state, lock, block_models, release_event):
    def factory(spec: ModelSpec) -> LabelClient:
        return _InstrumentedClient(
            spec.model_id,
            state=state,
            lock=lock,
            block_models=block_models,
            release_event=release_event,
        )

    return factory


def test_hosted_lane_not_starved_behind_slow_local_lane(tmp_path) -> None:
    """A blocked local lane must not prevent the hosted lane from running.

    gemma (local) parks all its calls; openai (hosted) and qwen (local) must
    still accrue calls concurrently. If the old shared-pool + in-worker-sem
    design were in place, the hosted lane would be starved to 0.
    """
    state = {"inflight": {}, "max_inflight": {}, "co_active": []}
    lock = threading.Lock()
    release = threading.Event()

    # >= old-pool worker count (concurrency * #providers = 4*2 = 8) of blocking
    # gemma singleton batches, so the OLD shared-pool design fully occupies
    # every worker with parked gemma calls and starves openai + qwen to 0.
    models = ["openai/gpt-5.5", "local/gemma-4-26b", "local/qwen3.6-27b"]
    samples = _samples(12)

    result_box: dict = {}

    def _run() -> None:
        result_box["summary"] = run_labeling(
            models=models,
            samples=samples,
            client_factory=_make_factory(state, lock, {"local/gemma-4-26b"}, release),
            concurrency=4,
            batch_size=20,
            dry_run=False,
            runs_root=tmp_path,
            allow_holdout=True,
        )

    t = threading.Thread(target=_run)
    t.start()

    # While gemma is parked, poll until openai AND qwen have both had at least
    # one call in flight (proving neither is starved behind gemma).
    deadline = time.time() + 8.0
    progressed = False
    while time.time() < deadline:
        with lock:
            openai_seen = state["max_inflight"].get("openai/gpt-5.5", 0) > 0
            qwen_seen = state["max_inflight"].get("local/qwen3.6-27b", 0) > 0
            gemma_inflight = state["inflight"].get("local/gemma-4-26b", 0) > 0
        if openai_seen and qwen_seen and gemma_inflight:
            progressed = True
            break
        time.sleep(0.02)

    # Release gemma so the run can finish, then join.
    release.set()
    t.join(timeout=15.0)

    assert progressed, (
        "hosted (openai) and second local (qwen) lanes did not progress while "
        "the gemma local lane was blocked -> head-of-line starvation regression"
    )
    summary = result_box["summary"]
    assert summary.completed_calls == len(samples) * len(models)


def test_two_distinct_local_models_dispatch_concurrently(tmp_path) -> None:
    """gemma and qwen (distinct local models = distinct GPU cards) must be able
    to run at the same instant, not serialized one after the other."""
    state = {"inflight": {}, "max_inflight": {}, "co_active": []}
    lock = threading.Lock()
    release = threading.Event()
    release.set()  # nobody blocks; we only need overlap observation

    models = ["local/gemma-4-26b", "local/qwen3.6-27b"]
    samples = _samples(8)

    run_labeling(
        models=models,
        samples=samples,
        client_factory=_make_factory(state, lock, set(), release),
        concurrency=4,
        batch_size=20,
        dry_run=False,
        runs_root=tmp_path,
        allow_holdout=True,
    )

    # At least one observed instant had BOTH local models in flight together.
    both_together = any(
        {"local/gemma-4-26b", "local/qwen3.6-27b"} <= set(active)
        for active in state["co_active"]
    )
    assert both_together, (
        "gemma and qwen never overlapped -> local lanes ran sequentially "
        "instead of concurrently across cards"
    )
    # And each local model still serialized its own calls (max 1 per card).
    assert state["max_inflight"].get("local/gemma-4-26b", 0) == 1
    assert state["max_inflight"].get("local/qwen3.6-27b", 0) == 1


def test_interleave_lane_batches_round_robins_models() -> None:
    """Model-major input comes out A0,B0,A1,B1,… with per-model order kept."""
    from pipeline.runner import _build_work_batches, _interleave_lane_batches

    specs = [ModelSpec("openai/gpt-5.5"), ModelSpec("openai/gpt-5.5-low")]
    batches = _build_work_batches(_samples(4), specs, batch_size=2)
    entries = list(enumerate(batches))

    ordered = _interleave_lane_batches(entries)

    models = [batch[0][1].model_id for _, batch in ordered]
    assert models == [
        "openai/gpt-5.5",
        "openai/gpt-5.5-low",
        "openai/gpt-5.5",
        "openai/gpt-5.5-low",
    ]
    # Batch indices (the cost-ledger key) survive, ascending per model.
    for model_id in {"openai/gpt-5.5", "openai/gpt-5.5-low"}:
        idx = [i for i, batch in ordered if batch[0][1].model_id == model_id]
        assert idx == sorted(idx)
    assert sorted(i for i, _ in ordered) == list(range(len(batches)))


def test_two_hosted_models_same_provider_overlap(tmp_path) -> None:
    """Two judges of ONE hosted provider must run side by side.

    Regression (exp-20260708T150915): batches queued model-major into the
    shared openai lane made the second judge's first call start the instant
    the first judge's last call finished — head-to-tail, not parallel. With
    round-robin interleave + per-model lane sizing they must be observed in
    flight together.
    """
    state = {"inflight": {}, "max_inflight": {}, "co_active": []}
    lock = threading.Lock()
    release = threading.Event()
    release.set()  # nobody blocks; we only need overlap observation

    models = ["openai/gpt-5.5", "openai/gpt-5.5-low"]
    samples = _samples(12)

    summary = run_labeling(
        models=models,
        samples=samples,
        client_factory=_make_factory(state, lock, set(), release),
        concurrency=2,
        batch_size=2,
        dry_run=False,
        runs_root=tmp_path,
        allow_holdout=True,
    )

    both_together = any(
        {"openai/gpt-5.5", "openai/gpt-5.5-low"} <= set(active)
        for active in state["co_active"]
    )
    assert both_together, (
        "the two openai judges never overlapped -> the shared provider lane "
        "is serializing models again (model-major starvation regression)"
    )
    assert summary.completed_calls == len(samples) * len(models)
