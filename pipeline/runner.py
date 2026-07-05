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
* **Concurrency.** One dedicated ``ThreadPoolExecutor`` per lane — a lane is
  a hosted provider bucket (sized ``concurrency``) or a distinct local model
  (sized ``LOCAL_MODEL_MAX_CONCURRENCY``, one GPU card). Lanes run fully in
  parallel; a batch only ever occupies a worker in its own lane, so a slow
  local lane can never starve a hosted lane (no head-of-line blocking). At
  ``concurrency=1`` dispatch stays sequential and sample-major for exact
  deterministic ordering.
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
from .providers.pricing import compute_call_cost, PRICING_VERSION
from .providers.ontology import get_ontology
from .scoring.cost_ledger import (
    build_cost_row,
    build_model_speed_summary,
    build_per_model_timing_block,
    rollup_cost_rows,
)
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
from .web.demo_area import normalize_policy_area

# Shared image-prep defaults (mirrored in run_manifest.image_prep).
IMAGE_PREP_LONGEST_EDGE_PX = 1024
IMAGE_PREP_FORMAT = "JPEG"
IMAGE_PREP_QUALITY = 85
IMAGE_PREP_HELPER = "pipeline.labeling.image_prep"

DEFAULT_PROMPT_VERSION = "v0.1"

# Provider tag for local GPU models (see pipeline/providers/registry.py).
# Each local model runs on its OWN dedicated GPU card (Attila's rig: two RTX
# 3090s, one model per card), so LM Studio can serve them simultaneously. Thus
# local semaphores are keyed by MODEL_ID (each card = its own lock domain),
# letting distinct local models run in parallel across cards. A 27B on a 24GB
# card is near capacity, so each local model serializes its own calls
# (LOCAL_MODEL_MAX_CONCURRENCY, default 1 — tunable if a card handles more).
# Hosted providers are keyed by PROVIDER (shared API rate limit) at size
# `concurrency`.
LOCAL_PROVIDER_TAG = "local"
LOCAL_MODEL_MAX_CONCURRENCY = 1


def _sem_key_and_size(
    provider: str, model_id: str, concurrency: int
) -> tuple[str, int]:
    """Resolve the semaphore lock-domain key and in-flight size for a spec.

    - Local provider: key on ``model_id`` (each local model = its own GPU
      card), size ``LOCAL_MODEL_MAX_CONCURRENCY``. Distinct local models get
      distinct semaphores → they run in parallel across cards; each serializes
      its own calls.
    - Hosted providers: key on ``provider`` (shared API rate limit), size
      ``concurrency``.

    Keyed on the provider tag so any future local model is covered without
    hardcoding model ids.
    """
    if provider == LOCAL_PROVIDER_TAG:
        return model_id, LOCAL_MODEL_MAX_CONCURRENCY
    return provider, concurrency


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
    total_cost_usd: float = 0.0
    per_batch_costs: list[dict] = field(default_factory=list)
    per_model_expected: dict[str, int] = field(default_factory=dict)
    per_model_completed: dict[str, int] = field(default_factory=dict)
    per_model_errored: dict[str, int] = field(default_factory=dict)
    fatal_error: str | None = None


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

    def _fake_label(self, request: LabelRequest) -> str:
        if self.label_strategy is not None:
            return self.label_strategy(self.model_id, request.image_id)
        if request.area == "MNIST_Digits":
            # MNIST dry-runs must exercise the same 10-class pipeline as live
            # runs. Prefer the sample id's truth-ish digit prefix when present
            # only through a deterministic hash, never by reading filenames.
            bucket = int(self._digest(request.image_id, "mnist-label")[:2], 16) % 11
            return "abstain" if bucket == 10 else str(bucket)
        # Cycle deterministically so test fixtures get a mix of labels.
        bucket = int(self._digest(request.image_id, "label")[:2], 16) % 5
        if bucket == 0:
            return "abstain"
        return "gen_ai" if bucket % 2 == 0 else "not_gen_ai"

    def _response_for(self, request: LabelRequest) -> LabelResponse:
        digest = self._digest(request.image_id)
        label = self._fake_label(request)
        is_boundary = int(digest[6:8], 16) % 4 == 0
        boundary_pair: list[str] = []
        if request.area == "MNIST_Digits" and is_boundary and label != "abstain":
            other = str((int(label) + 1) % 10)
            boundary_pair = sorted([label, other])
        return LabelResponse(
            image_id=request.image_id,
            model_id=request.model_id,
            label=label,
            l2_label=(
                f"MD.digit.{label}"
                if request.area == "MNIST_Digits" and label != "abstain"
                else ""
            ),
            justification=(
                "Deterministic dry-run response: this image was NOT sent to a "
                "provider. Generated for offline schema validation."
            ),
            confidence=round((int(digest[:4], 16) % 1000) / 1000.0, 3),
            difficulty=("low", "medium", "high")[int(digest[4:6], 16) % 3],
            is_boundary=is_boundary,
            raw_provider_payload={"dry_run": True, "model_id": request.model_id},
            error=None,
            latency_ms=0,
            attempts=1,
            prepared_image_sha256=digest,  # fake but well-formed sha256
            prepared_image_width=IMAGE_PREP_LONGEST_EDGE_PX,
            prepared_image_height=IMAGE_PREP_LONGEST_EDGE_PX,
            prepared_image_mime_type="image/jpeg",
            prepared_image_byte_size=int(digest[8:14], 16) % 200_000 + 10_000,
            is_boundary_between=boundary_pair,
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
    # Area-aware downsampler cap: MNIST prepares tiny (~112px) images so image
    # token cost stays low; GenAI keeps the 1024px baseline. Threaded from the
    # ontology down to prepare_image_for_labeling via LabelRequest.max_image_size.
    max_image_size = get_ontology(area).max_image_size
    return LabelRequest(
        image_path=sample.absolute_path,
        image_id=sample.sample_id,
        policy_markdown=policy_markdown,
        policy_graph_version=policy_graph_version,
        prompt_version=prompt_version,
        model_id=model_spec.model_id,
        area=area,
        max_image_size=max_image_size,
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
        "latency_ms": int(max(0, response.latency_ms)),
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
    area: str = "Generative_AI",
    policy_version: str = "v0.1",
) -> dict:
    manifest: dict = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
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
        "area": area,
        "policy_version": policy_version,
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


def _fatal_error_reason(summary: RunSummary) -> str | None:
    """Return a run-level failure reason for completed runs without usable coverage."""
    if summary.expected_calls > 0 and summary.completed_calls == 0:
        return "all calls failed"

    failed_models = [
        model_id
        for model_id, expected in sorted(summary.per_model_expected.items())
        if expected > 0
        and summary.per_model_completed.get(model_id, 0) == 0
        and summary.per_model_errored.get(model_id, 0) >= expected
    ]
    if failed_models:
        return "all calls failed for model(s): " + ", ".join(failed_models)
    return None


def run_completed_with_results(summary: RunSummary) -> bool:
    """True when the pass produced at least one usable label and has no fatal gap."""
    return summary.completed_calls > 0 and _fatal_error_reason(summary) is None


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
    area: str | None = None,
    policy_version: str | None = None,
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
    selected_sampling_versions = sorted({r.sampling_version for r in selected if r.sampling_version})
    if len(selected_sampling_versions) == 1:
        sampling_version = selected_sampling_versions[0]

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

    run_area = normalize_policy_area(area) if area is not None else area_from_policy_version(policy_graph_version)
    manifest_policy_version = policy_version or (
        policy_graph_version.removeprefix(f"{run_area}.")
        if policy_graph_version.startswith(f"{run_area}.")
        else policy_graph_version
    )

    policy_dir = policy_graph_dir or DEFAULT_POLICY_GRAPH_DIR
    policy_markdown = load_policy_markdown(policy_dir)

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
        area=run_area,
        policy_version=manifest_policy_version,
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
        per_model_expected={spec.model_id: len(selected) for spec in model_specs},
        per_model_completed={spec.model_id: 0 for spec in model_specs},
        per_model_errored={spec.model_id: 0 for spec in model_specs},
    )

    # Per-batch cost ledger (batch_index -> record). Keyed by index so the
    # manifest stays deterministic regardless of concurrent completion order.
    batch_costs: dict[int, dict] = {}
    cost_lock = threading.Lock()

    # Per-provider client cache. Concurrency is enforced structurally by the
    # per-lane executors below (one executor per hosted provider bucket / per
    # distinct local model), each sized to its own cap. This replaces the old
    # single-pool + in-worker-semaphore design, which let a slow local lane
    # park worker slots and starve hosted lanes (head-of-line blocking).
    client_cache: dict[str, LabelClient] = {}
    cache_lock = threading.Lock()

    def _client_for(spec: ModelSpec) -> LabelClient:
        with cache_lock:
            if spec.model_id not in client_cache:
                client_cache[spec.model_id] = client_factory(spec)
            return client_cache[spec.model_id]

    write_lock = threading.Lock()  # keeps JSONL appends from interleaving

    # Durable per-image cost ledger rows (X1). Written to costs.jsonl with
    # LIVE registry rates + pricing_version so future analysis has current,
    # self-describing cost data (not stale historical cost_usd).
    cost_rows: list[dict] = []

    def _persist_response(response: LabelResponse, batch_index: int = 0) -> bool:
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
                # Durable, analysis-ready cost row: LIVE registry rates +
                # pricing_version (recomputed, not the possibly-stale
                # provider/response cost).
                cost_row = build_cost_row(
                    run_id=rid,
                    batch_index=batch_index,
                    image_id=response.image_id,
                    model_id=response.model_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    recorded_at=_utcnow_iso(),
                )
                persistence.append_cost_row(paths, cost_row)
                cost_rows.append(cost_row)
            return True
        except persistence.PersistenceError:
            return False

    def _record_batch_cost(
        batch_index: int,
        model_id: str,
        images: int,
        cost: float | None,
    ) -> None:
        with cost_lock:
            batch_costs[batch_index] = {
                "batch_index": batch_index,
                "model_id": model_id,
                "images": images,
                "cost_usd": cost,
                "cost_per_image_usd": (
                    (cost / images) if (cost is not None and images > 0) else None
                ),
            }

    def _retry_parse_failed_responses(
        client: LabelClient,
        requests: list[LabelRequest],
        responses: list[LabelResponse],
    ) -> list[LabelResponse]:
        """Give JSON-format flakes one extra single-image attempt."""
        retried = list(responses)
        for idx, response in enumerate(responses):
            if response.error != "parse_failed":
                continue
            original_attempts = max(1, response.attempts)
            try:
                retry = client.label(requests[idx])
            except Exception:
                response.attempts = original_attempts + 1
                continue
            retry.attempts = original_attempts + max(1, retry.attempts)
            retried[idx] = retry
        return retried

    def _process_batch(
        batch_index: int,
        batch: list[tuple[SampleRecord, ModelSpec]],
    ) -> tuple[str, int, int]:
        # Every batch is homogeneous by model/provider (except historical
        # singleton batches, where homogeneity is trivially true). No semaphore
        # here: the owning lane executor caps this batch's concurrency, so a
        # blocked call only ties up its own lane's worker(s), never another
        # lane's.
        _, spec = batch[0]
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
            _record_batch_cost(batch_index, spec.model_id, len(batch), None)
            return spec.model_id, 0, len(batch)

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
            _record_batch_cost(batch_index, spec.model_id, len(batch), None)
            return spec.model_id, 0, len(batch)

        responses = _retry_parse_failed_responses(client, requests, responses)

        completed = 0
        errored = 0
        batch_cost: float | None = None
        for response in responses:
            if _persist_response(response, batch_index=batch_index):
                completed += 1
                if response.cost_usd is not None:
                    batch_cost = (batch_cost or 0.0) + float(response.cost_usd)
            else:
                errored += 1
        _record_batch_cost(batch_index, spec.model_id, len(batch), batch_cost)
        return spec.model_id, completed, errored

    if concurrency == 1:
        for batch_index, batch in enumerate(batches):
            model_id, completed, errored = _process_batch(batch_index, batch)
            summary.completed_calls += completed
            summary.errored_calls += errored
            summary.per_model_completed[model_id] += completed
            summary.per_model_errored[model_id] += errored
    else:
        # Per-lane executors: one dedicated ThreadPoolExecutor per lane, where a
        # lane is a hosted provider bucket (key=provider, size=concurrency) or a
        # distinct local model (key=model_id, size=LOCAL_MODEL_MAX_CONCURRENCY).
        # Every lane runs concurrently with every other lane, and each lane is
        # sized to its own cap. Because a batch only ever occupies a worker in
        # its OWN lane's executor, a slow/blocked local lane can never hold
        # worker slots belonging to another lane — no head-of-line blocking, no
        # cross-lane starvation. This structurally replaces the old shared-pool
        # + in-worker-semaphore design.
        lanes: dict[str, list[tuple[int, list[tuple[SampleRecord, ModelSpec]]]]] = {}
        lane_size: dict[str, int] = {}
        for batch_index, batch in enumerate(batches):
            spec = batch[0][1]
            key, size = _sem_key_and_size(spec.provider, spec.model_id, concurrency)
            lanes.setdefault(key, []).append((batch_index, batch))
            lane_size[key] = size

        executors: list[ThreadPoolExecutor] = []
        futures: list[Future] = []
        try:
            for key, lane_batches in lanes.items():
                pool = ThreadPoolExecutor(
                    max_workers=max(1, lane_size[key]),
                    thread_name_prefix=f"lane-{key}",
                )
                executors.append(pool)
                for batch_index, batch in lane_batches:
                    futures.append(pool.submit(_process_batch, batch_index, batch))
            for fut in futures:
                model_id, completed, errored = fut.result()
                summary.completed_calls += completed
                summary.errored_calls += errored
                summary.per_model_completed[model_id] += completed
                summary.per_model_errored[model_id] += errored
        finally:
            for pool in executors:
                pool.shutdown(wait=True)

    # 5. Finalize manifest (including per-batch + per-image cost ledger).
    summary.finished_at = _utcnow_iso()
    per_batch_costs = [batch_costs[i] for i in sorted(batch_costs)]
    summary.per_batch_costs = per_batch_costs
    known_costs = [b["cost_usd"] for b in per_batch_costs if b["cost_usd"] is not None]
    total_cost = float(sum(known_costs)) if known_costs else 0.0
    total_images = sum(
        b["images"] for b in per_batch_costs if b["cost_usd"] is not None
    )
    summary.total_cost_usd = total_cost
    # Per-LLM breakdown + pricing_version stamp from the durable ledger rows.
    ledger = rollup_cost_rows(cost_rows)
    model_speed_summary = build_model_speed_summary(cost_rows)
    per_model_timing = build_per_model_timing_block(cost_rows)
    cost_block = {
        "total_cost_usd": total_cost,
        "cost_per_image_usd": (total_cost / total_images) if total_images else None,
        "priced_images": total_images,
        "batches_with_unknown_cost": sum(
            1 for b in per_batch_costs if b["cost_usd"] is None
        ),
        "per_batch": per_batch_costs,
        # Per-LLM breakdown (analysis-ready) recomputed from LIVE registry.
        "per_model": ledger["per_model"],
        "pricing_version": PRICING_VERSION,
        "pricing_versions_present": ledger["pricing_versions"],
    }
    persistence.write_model_speed_summary(
        paths,
        {
            "run_id": rid,
            "generated_at": summary.finished_at,
            "models": model_speed_summary,
            "per_model_timing": per_model_timing,
        },
    )
    final_manifest = dict(manifest)
    summary.fatal_error = _fatal_error_reason(summary)
    completed_with_errors = (
        summary.fatal_error is None
        and summary.completed_calls > 0
        and summary.errored_calls > 0
    )
    final_manifest["finished_at"] = summary.finished_at
    final_manifest["status"] = "failed" if summary.fatal_error else "completed"
    final_manifest["completed_with_errors"] = completed_with_errors
    if summary.fatal_error:
        final_manifest["abort_reason"] = summary.fatal_error
    final_manifest["totals"] = {
        "expected_calls": expected,
        "completed_calls": summary.completed_calls,
        "errored_calls": summary.errored_calls,
    }
    final_manifest["cost"] = cost_block
    final_manifest["per_model_timing"] = per_model_timing
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
    "LOCAL_PROVIDER_TAG",
    "LOCAL_MODEL_MAX_CONCURRENCY",
    "_sem_key_and_size",
    "_fatal_error_reason",
    "run_completed_with_results",
]
