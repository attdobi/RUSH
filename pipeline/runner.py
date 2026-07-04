"""Bulk-labeling run orchestrator (X2).

Walks ``N images × M models`` deterministically, writes a validated
``run_manifest.json``, calls each provider client (X1) per image, and persists
``label_votes.jsonl`` + ``llm_outputs.jsonl`` + ``errors.jsonl``.

Image bytes never leave X1's shared ``image_prep`` helper. The runner only
records the audit metadata (``prepared_image_*`` fields) returned on each
``LabelResponse``.

Design choices
--------------
* **Determinism first.** Sample IDs and (image, model) pairs are sorted before
  dispatch (§5.6). Tests run with ``concurrency=1`` to keep ordering exact.
* **Concurrency.** Per-provider semaphores cap in-flight calls at
  ``concurrency`` (default 4 per §5.5). Provider buckets run in parallel via
  one shared ``ThreadPoolExecutor`` whose worker count = ``concurrency``
  per provider.
* **No tight loops.** Retries are X1's responsibility inside the client.
* **Dry runs.** A built-in ``DeterministicFakeClient`` (used by tests + the
  CLI when ``--dry-run`` is set) produces stable, no-network responses with
  fake but well-formed prepared-image audit metadata.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import threading
from pathlib import Path
from typing import Callable, Iterable

from . import persistence
from .providers._config import resolve_temperature
from .providers.base import LabelClient, LabelRequest, LabelResponse
from .providers.pricing import compute_call_cost
from .io_paths import (
    DEFAULT_POLICY_GRAPH_DIR,
    DEFAULT_POLICY_GRAPH_VERSION,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SAMPLE_MANIFEST,
    REPO_ROOT,
    RunPaths,
    mint_run_id,
    run_paths,
)
from .manifest import (
    HOLDOUT_SPLITS,
    SampleRecord,
    load_policy_markdown,
    load_records,
    select_samples,
)
from .web.demo_area import area_from_policy_version

# Shared image-prep defaults (mirrored in run_manifest.image_prep).
IMAGE_PREP_LONGEST_EDGE_PX = 1024
IMAGE_PREP_FORMAT = "JPEG"
IMAGE_PREP_QUALITY = 85
IMAGE_PREP_HELPER = "pipeline.labeling.image_prep"

DEFAULT_PROMPT_VERSION = "v0.1"


@dataclass(frozen=True)
class ModelSpec:
    """Lightweight runner-side view of a configured model.

    The runtime registry lives in X1's ``pipeline.providers.registry``; the
    runner only needs the id and (optionally) the phase/params for the manifest.
    """

    model_id: str
    phase: str | None = None
    params: dict | None = None
    resolved_temperature: float | None = None

    @property
    def provider(self) -> str:
        """Provider prefix used to bucket per-provider concurrency caps."""
        return self.model_id.split("/", 1)[0] if "/" in self.model_id else self.model_id


ClientFactory = Callable[[ModelSpec], LabelClient]


@dataclass
class RunSummary:
    """Returned to the CLI / tests after a run completes."""

    run_id: str
    paths: RunPaths
    expected_calls: int
    completed_calls: int = 0
    errored_calls: int = 0
    started_at: str = ""
    finished_at: str = ""
    dry_run: bool = False
    batch_size: int = 20
    effective_batches: int = 0


# ---------------------------------------------------------------------------
# Built-in deterministic fake client (offline tests + --dry-run)
# ---------------------------------------------------------------------------


class DeterministicFakeClient(LabelClient):
    """Offline-only client used by tests and ``--dry-run``.

    Produces fully schema-valid responses whose values are a deterministic
    function of ``(model_id, image_id)`` so test assertions are stable.
    Does not read image bytes; fabricates plausible prepared-image metadata.
    Subclasses X1's :class:`pipeline.providers.base.LabelClient` so it
    satisfies the same contract real provider clients do.
    """

    provider_id = "dryrun"

    def __init__(
        self,
        model_id: str,
        *,
        label_strategy: Callable[[str, str], str] | None = None,
    ) -> None:
        # Bypass the ABC's ClientConfig requirement: the dry-run client
        # carries only a model_id and an optional label strategy.
        self.model_id = model_id
        self.label_strategy = label_strategy

    def _digest(self, image_id: str, salt: str = "") -> str:
        return hashlib.sha256(f"{self.model_id}|{image_id}|{salt}".encode("utf-8")).hexdigest()

    def _fake_label(self, image_id: str) -> str:
        if self.label_strategy is not None:
            return self.label_strategy(self.model_id, image_id)
        # Cycle deterministically so test fixtures get a mix of labels.
        bucket = int(self._digest(image_id, "label")[:2], 16) % 5
        if bucket == 0:
            return "abstain"
        return "gen_ai" if bucket % 2 == 0 else "not_gen_ai"

    def _response_for(self, request: LabelRequest) -> LabelResponse:
        digest = self._digest(request.image_id)
        return LabelResponse(
            image_id=request.image_id,
            model_id=request.model_id,
            label=self._fake_label(request.image_id),
            l2_label="",  # cold start: no L2 yet
            justification=(
                "Deterministic dry-run response: this image was NOT sent to a "
                "provider. Generated for offline schema validation."
            ),
            confidence=round((int(digest[:4], 16) % 1000) / 1000.0, 3),
            difficulty=("low", "medium", "high")[int(digest[4:6], 16) % 3],
            is_boundary=(int(digest[6:8], 16) % 4 == 0),
            raw_provider_payload={"dry_run": True, "model_id": request.model_id},
            error=None,
            latency_ms=0,
            attempts=1,
            prepared_image_sha256=digest,  # fake but well-formed sha256
            prepared_image_width=IMAGE_PREP_LONGEST_EDGE_PX,
            prepared_image_height=IMAGE_PREP_LONGEST_EDGE_PX,
            prepared_image_mime_type="image/jpeg",
            prepared_image_byte_size=int(digest[8:14], 16) % 200_000 + 10_000,
        )

    def label(self, request: LabelRequest) -> LabelResponse:
        return self._response_for(request)

    def batch_label(self, requests: list[LabelRequest]) -> list[LabelResponse]:
        return [self._response_for(request) for request in requests]


def deterministic_fake_factory(_spec: ModelSpec) -> LabelClient:
    return DeterministicFakeClient(model_id=_spec.model_id)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_request(
    sample: SampleRecord,
    model_spec: ModelSpec,
    *,
    policy_markdown: str,
    policy_graph_version: str,
    prompt_version: str,
    area: str,
) -> LabelRequest:
    return LabelRequest(
        image_path=sample.absolute_path,
        image_id=sample.sample_id,
        policy_markdown=policy_markdown,
        policy_graph_version=policy_graph_version,
        prompt_version=prompt_version,
        model_id=model_spec.model_id,
        area=area,
    )


def _coerce_optional_confidence(value: object) -> float | None:
    """Schema allows confidence: number | null. Coerce safely so providers
    that legitimately propagate `None` (e.g. missing/malformed fields) do not
    crash the runner. Anything that can't become a float is treated as null.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_llm_output(response: LabelResponse) -> dict:
    """Project a LabelResponse onto the llm-output.schema.json shape."""
    output: dict = {
        "label": response.label,
        "l2_label": response.l2_label,
        "justification": response.justification,
        "confidence": _coerce_optional_confidence(response.confidence),
        "difficulty": response.difficulty,
        "is_boundary": bool(response.is_boundary),
    }
    if response.is_boundary_between:
        output["is_boundary_between"] = list(response.is_boundary_between)
    if response.prepared_image_sha256:
        output["prepared_image_sha256"] = response.prepared_image_sha256
    if response.prepared_image_width:
        output["prepared_image_width"] = int(response.prepared_image_width)
    if response.prepared_image_height:
        output["prepared_image_height"] = int(response.prepared_image_height)
    if response.prepared_image_mime_type:
        output["prepared_image_mime_type"] = response.prepared_image_mime_type
    if response.prepared_image_byte_size:
        output["prepared_image_byte_size"] = int(response.prepared_image_byte_size)
    if response.input_tokens is not None:
        output["input_tokens"] = int(response.input_tokens)
    if response.output_tokens is not None:
        output["output_tokens"] = int(response.output_tokens)
    if response.cost_usd is not None:
        output["cost_usd"] = float(response.cost_usd)
    return output


def _build_label_vote(
    response: LabelResponse,
    *,
    run_id: str,
    policy_graph_version: str,
    prompt_version: str,
) -> dict:
    """Project a LabelResponse onto label-vote.schema.json (cold-start)."""
    vote: dict = {
        "run_id": run_id,
        "image_id": response.image_id,
        "labeler_type": "llm",
        "labeler_id": response.model_id,
        "model_id": response.model_id,
        "label": response.label,
        "node_ids": [response.l2_label] if response.l2_label else [],
        "confidence": _coerce_optional_confidence(response.confidence),
        "justification": response.justification or "(no justification provided)",
        "policy_graph_version": policy_graph_version,
        "prompt_version": prompt_version,
        "label_tier": "provisional",
        "l2_label": response.l2_label,
        "is_boundary": bool(response.is_boundary),
        "difficulty": response.difficulty,
        "latency_ms": int(response.latency_ms),
        "attempts": int(max(1, response.attempts)),
    }
    if response.is_boundary_between:
        vote["is_boundary_between"] = list(response.is_boundary_between)
    if response.prepared_image_sha256:
        vote["prepared_image_sha256"] = response.prepared_image_sha256
    if response.prepared_image_width:
        vote["prepared_image_width"] = int(response.prepared_image_width)
    if response.prepared_image_height:
        vote["prepared_image_height"] = int(response.prepared_image_height)
    if response.prepared_image_mime_type:
        vote["prepared_image_mime_type"] = response.prepared_image_mime_type
    if response.prepared_image_byte_size:
        vote["prepared_image_byte_size"] = int(response.prepared_image_byte_size)
    if response.input_tokens is not None:
        vote["input_tokens"] = int(response.input_tokens)
    if response.output_tokens is not None:
        vote["output_tokens"] = int(response.output_tokens)
    if response.cost_usd is not None:
        vote["cost_usd"] = float(response.cost_usd)
    return vote


def _model_runtime_config(
    models: list[ModelSpec],
    *,
    reasoning_effort: str | None = None,
) -> dict[str, dict]:
    """Return auditable per-model runtime knobs that affect provider behavior."""
    out: dict[str, dict] = {}
    for model in models:
        params = dict(model.params or {})
        if model.model_id == "openai/gpt-5.5" and reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort
        runtime: dict = {}
        if params.get("reasoning_effort") is not None:
            runtime["reasoning_effort"] = params["reasoning_effort"]
        if params.get("thinking_budget_tokens") is not None:
            runtime["thinking_budget_tokens"] = params["thinking_budget_tokens"]
        if runtime:
            out[model.model_id] = runtime
    return out


def _initial_manifest(
    *,
    run_id: str,
    started_at: str,
    sample_manifest_rel: str,
    sample_ids: list[str],
    models: list[ModelSpec],
    policy_graph_version: str,
    prompt_version: str,
    sampling_version: str,
    split: str | None,
    limit: int | None,
    concurrency: int,
    expected_calls: int,
    dry_run: bool,
    batch_size: int = 20,
    effective_batches: int = 0,
    reasoning_effort: str | None = None,
) -> dict:
    manifest: dict = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "sample_manifest_path": sample_manifest_rel,
        "sample_ids": sample_ids,
        "model_runtime_config": _model_runtime_config(
            models,
            reasoning_effort=reasoning_effort,
        ),
        "models": [
            {
                k: v
                for k, v in {
                    "model_id": m.model_id,
                    "phase": m.phase,
                    "params": m.params,
                    "resolved_temperature": (
                        m.resolved_temperature
                        if m.resolved_temperature is not None
                        else resolve_temperature(m.model_id)
                    ),
                }.items()
                if v is not None or k in {"model_id", "resolved_temperature"}
            }
            for m in models
        ],
        "policy_graph_version": policy_graph_version,
        "prompt_version": prompt_version,
        "sampling_version": sampling_version,
        "concurrency": concurrency,
        "batch_size": batch_size,
        "effective_batches": effective_batches,
        "image_prep": {
            "longest_edge_px": IMAGE_PREP_LONGEST_EDGE_PX,
            "format": IMAGE_PREP_FORMAT,
            "quality": IMAGE_PREP_QUALITY,
            "helper_module": IMAGE_PREP_HELPER,
        },
        "totals": {
            "expected_calls": expected_calls,
            "completed_calls": 0,
            "errored_calls": 0,
        },
        "dry_run": dry_run,
    }
    if split is not None:
        manifest["split"] = split
    if limit is not None:
        manifest["limit"] = limit
    return manifest


def _coerce_models(models: Iterable[str | ModelSpec]) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for m in models:
        spec = m if isinstance(m, ModelSpec) else ModelSpec(model_id=m)
        if spec.model_id in seen:
            continue
        seen.add(spec.model_id)
        out.append(spec)
    out.sort(key=lambda s: s.model_id)
    return out


def _chunked(
    items: list[tuple[SampleRecord, ModelSpec]],
    size: int,
) -> list[list[tuple[SampleRecord, ModelSpec]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_work_batches(
    selected: list[SampleRecord],
    model_specs: list[ModelSpec],
    *,
    batch_size: int,
) -> list[list[tuple[SampleRecord, ModelSpec]]]:
    """Build deterministic logical provider batches.

    ``batch_size=1`` intentionally preserves the historical sample-major
    dispatch order byte-for-byte. Larger batches group by model because one
    provider batch call can only target one concrete model. Local models keep
    singleton dispatch even when API models are batched.
    """
    if batch_size == 1:
        return [[(s, m)] for s in selected for m in model_specs]

    batches: list[list[tuple[SampleRecord, ModelSpec]]] = []
    for spec in model_specs:
        model_pairs = [(sample, spec) for sample in selected]
        spec_batch_size = 1 if spec.model_id.startswith("local/") else batch_size
        batches.extend(_chunked(model_pairs, spec_batch_size))
    return batches


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_labeling(
    *,
    models: Iterable[str | ModelSpec],
    sample_manifest_path: Path | None = None,
    samples: Iterable[SampleRecord] | None = None,
    split: str | None = None,
    limit: int | None = None,
    sample_ids: Iterable[str] | None = None,
    runs_root: Path | None = None,
    policy_graph_dir: Path | None = None,
    policy_graph_version: str = DEFAULT_POLICY_GRAPH_VERSION,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    sampling_version: str = "genai-gold-sampling-v1",
    client_factory: ClientFactory = deterministic_fake_factory,
    concurrency: int = 1,
    batch_size: int = 20,
    allow_holdout: bool = False,
    run_id: str | None = None,
    dry_run: bool = True,
    reasoning_effort: str | None = None,
) -> RunSummary:
    """Execute one labeling run.

    Defaults are intentionally safe: ``client_factory`` is the deterministic
    fake and ``dry_run=True``. The CLI flips ``dry_run=False`` only after
    real provider clients are wired by X1 and Pista approves spend.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    model_specs = _coerce_models(models)
    if not model_specs:
        raise ValueError("no models supplied")

    # 1. Resolve samples (deterministic ordering applied inside select_samples).
    if samples is None:
        records = load_records(sample_manifest_path)
    else:
        records = list(samples)
    selected = select_samples(records, split=split, limit=limit, sample_ids=sample_ids)
    if not selected:
        raise ValueError("no samples matched the selection (split/limit/sample_ids)")

    selected_holdout_splits = sorted({r.split for r in selected if r.split in HOLDOUT_SPLITS})
    if selected_holdout_splits and not allow_holdout:
        raise PermissionError(
            "refusing to run against holdout split(s) "
            f"{', '.join(selected_holdout_splits)!r} without allow_holdout=True"
        )

    # 2. Resolve paths + run_id + policy markdown.
    rid = run_id or mint_run_id()
    paths = run_paths(rid, runs_root=runs_root)
    paths.ensure()

    policy_dir = policy_graph_dir or DEFAULT_POLICY_GRAPH_DIR
    policy_markdown = load_policy_markdown(policy_dir)
    # Derive the demo/policy area (selects the per-project ontology used by
    # every provider client) from the policy graph version, which encodes the
    # area as a prefix (e.g. "MNIST_Digits.v1" vs "Generative_AI.v1"/"v0.1").
    run_area = area_from_policy_version(policy_graph_version)

    sample_manifest_path = sample_manifest_path or DEFAULT_SAMPLE_MANIFEST
    try:
        sample_manifest_rel = str(sample_manifest_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        sample_manifest_rel = str(sample_manifest_path)

    sample_ids_sorted = [r.sample_id for r in selected]
    expected = len(sample_ids_sorted) * len(model_specs)
    batches = _build_work_batches(selected, model_specs, batch_size=batch_size)
    effective_batches = len(batches)
    started_at = _utcnow_iso()

    # 3. Write initial run manifest (validated).
    manifest = _initial_manifest(
        run_id=rid,
        started_at=started_at,
        sample_manifest_rel=sample_manifest_rel,
        sample_ids=sample_ids_sorted,
        models=model_specs,
        policy_graph_version=policy_graph_version,
        prompt_version=prompt_version,
        sampling_version=sampling_version,
        split=split,
        limit=limit,
        concurrency=concurrency,
        batch_size=batch_size,
        effective_batches=effective_batches,
        expected_calls=expected,
        dry_run=dry_run,
        reasoning_effort=reasoning_effort,
    )
    persistence.write_run_manifest(paths, manifest)

    # 4. Dispatch logical batches deterministically.
    summary = RunSummary(
        run_id=rid,
        paths=paths,
        expected_calls=expected,
        started_at=started_at,
        dry_run=dry_run,
        batch_size=batch_size,
        effective_batches=effective_batches,
    )

    # Per-provider client cache + semaphore (cap in-flight per provider).
    client_cache: dict[str, LabelClient] = {}
    provider_locks: dict[str, threading.Semaphore] = {}
    cache_lock = threading.Lock()

    def _client_for(spec: ModelSpec) -> LabelClient:
        with cache_lock:
            if spec.model_id not in client_cache:
                client_cache[spec.model_id] = client_factory(spec)
            return client_cache[spec.model_id]

    def _provider_sem(provider: str) -> threading.Semaphore:
        with cache_lock:
            if provider not in provider_locks:
                provider_locks[provider] = threading.Semaphore(concurrency)
            return provider_locks[provider]

    write_lock = threading.Lock()  # keeps JSONL appends from interleaving

    def _persist_response(response: LabelResponse) -> bool:
        if response.cost_usd is None:
            response.cost_usd = compute_call_cost(
                response.model_id,
                response.input_tokens,
                response.output_tokens,
                image_count=1,
            )

        # If the client returned a populated `error`, persist as failure.
        if response.error:
            with write_lock:
                persistence.append_error(
                    paths,
                    stage="provider_call",
                    image_id=response.image_id,
                    model_id=response.model_id,
                    reason=response.error,
                    attempts=max(1, response.attempts),
                )
            return False

        try:
            llm_output = _build_llm_output(response)
            label_vote = _build_label_vote(
                response,
                run_id=rid,
                policy_graph_version=policy_graph_version,
                prompt_version=prompt_version,
            )
            with write_lock:
                persistence.append_llm_output(
                    paths,
                    llm_output,
                    image_id=response.image_id,
                    model_id=response.model_id,
                )
                persistence.append_label_vote(paths, label_vote)
            return True
        except persistence.PersistenceError:
            return False

    def _process_batch(batch: list[tuple[SampleRecord, ModelSpec]]) -> tuple[int, int]:
        # Every batch is homogeneous by model/provider (except historical
        # singleton batches, where homogeneity is trivially true).
        _, spec = batch[0]
        sem = _provider_sem(spec.provider)
        requests = [
            _build_request(
                sample,
                model_spec,
                policy_markdown=policy_markdown,
                policy_graph_version=policy_graph_version,
                prompt_version=prompt_version,
                area=run_area,
            )
            for sample, model_spec in batch
        ]
        with sem:
            try:
                client = _client_for(spec)
                if len(requests) == 1:
                    responses = [client.label(requests[0])]
                else:
                    batch_method = getattr(client, "batch_label", None)
                    if callable(batch_method):
                        responses = list(batch_method(requests))
                    else:
                        responses = [client.label(request) for request in requests]
            except Exception as exc:  # client raised; treat each image as a hard error.
                with write_lock:
                    for sample, model_spec in batch:
                        persistence.append_error(
                            paths,
                            stage="provider_call",
                            image_id=sample.sample_id,
                            model_id=model_spec.model_id,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                return 0, len(batch)

        if len(responses) != len(requests):
            with write_lock:
                for sample, model_spec in batch:
                    persistence.append_error(
                        paths,
                        stage="provider_call",
                        image_id=sample.sample_id,
                        model_id=model_spec.model_id,
                        reason=(
                            "provider returned mismatched batch size: "
                            f"expected {len(requests)}, got {len(responses)}"
                        ),
                    )
            return 0, len(batch)

        completed = 0
        errored = 0
        for response in responses:
            if _persist_response(response):
                completed += 1
            else:
                errored += 1
        return completed, errored

    if concurrency == 1:
        for batch in batches:
            completed, errored = _process_batch(batch)
            summary.completed_calls += completed
            summary.errored_calls += errored
    else:
        # ThreadPoolExecutor with provider-bound semaphores keeps per-provider
        # in-flight logical batches capped at `concurrency`.
        providers = {batch[0][1].provider for batch in batches}
        with ThreadPoolExecutor(max_workers=concurrency * max(1, len(providers))) as pool:
            futures: list[Future] = [pool.submit(_process_batch, batch) for batch in batches]
            for fut in futures:
                completed, errored = fut.result()
                summary.completed_calls += completed
                summary.errored_calls += errored

    # 5. Finalize manifest.
    summary.finished_at = _utcnow_iso()
    final_manifest = dict(manifest)
    final_manifest["finished_at"] = summary.finished_at
    final_manifest["totals"] = {
        "expected_calls": expected,
        "completed_calls": summary.completed_calls,
        "errored_calls": summary.errored_calls,
    }
    persistence.write_run_manifest(paths, final_manifest)
    return summary


__all__ = [
    "ModelSpec",
    "RunSummary",
    "DeterministicFakeClient",
    "deterministic_fake_factory",
    "run_labeling",
    "IMAGE_PREP_LONGEST_EDGE_PX",
    "IMAGE_PREP_FORMAT",
    "IMAGE_PREP_QUALITY",
    "IMAGE_PREP_HELPER",
    "DEFAULT_PROMPT_VERSION",
]
