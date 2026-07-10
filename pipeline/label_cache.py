"""Cross-run label cache — never pay for the same (image, prompt, model) twice.

Motivation (Attila, 2026-07-09): every benchmark re-labels the v0.1 baseline
with byte-identical prompts. After ``CACHE_SAMPLES`` live rounds per
(image, prompt, model) the verdict is served from Postgres as a majority vote
over the stored samples; the disagreement across those rounds is recorded as
an **intra-rater flip rate** — a per-model reliability signal the panel's
inter-rater disagreement cannot see.

Identity contract
-----------------
* ``image_sha256``  — the SOURCE-file sha256 from the sample manifest (the
  label store's ``entity_id``). Stable across manifests, renames, re-samples.
* ``prompt_sha256`` — content-derived, never version-named. It hashes every
  input that shapes the model's answer: system prompt, user instructions,
  response schema, the exact policy render the judge saw (full vs compressed
  digests hash differently for free), image-prep knobs, temperature, and
  runtime params. Any prompt drift — a `_prompts.py` rewrite, a policy edit,
  a reasoning-effort change — auto-invalidates; there is no manual busting.
* ``model_id``      — the registry id; provider message shaping is constant
  per model, so it never needs hashing.

Sampling rule: deterministic decoding (``temperature == 0``) caches after one
round; anything else takes ``CACHE_SAMPLES`` independent rounds first so the
flip rate means something. A served label is a *denoised* majority — every
cached vote carries a ``label_cache`` marker so benchmarks stay auditable.

Safety stance: the cache NEVER breaks a run. Any database problem (Postgres
down, table missing on a fresh mac mini before setup, bad URL) disables the
cache for the rest of the run and the reason lands in the run manifest.
Dry runs never construct a cache at all — no DB writes, per the standing
dry-run contract. Connection comes from ``RUSH_DB_URL`` like the label store
(default ``postgresql:///adobi``); the table self-provisions on first use.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any

from .labelstore import db_url as labelstore_db_url

# Live rounds required before a nondeterministic (temp != 0) judge's verdict
# is served from cache. 3 = the smallest N where "majority" and "flip" are
# distinct concepts; temp == 0 needs only 1 (Attila: "if temp=0 then stick
# to 1 round").
CACHE_SAMPLES = 3

DEFAULT_TABLE = "rush.label_cache"

# Fingerprint payload version — bump when the *composition* of the payload
# changes (not when prompt contents change; those re-hash by themselves).
FINGERPRINT_VERSION = 1

# Stored responses keep full token/cost provenance from the live call, but a
# served vote must not re-report them: the cached call cost nothing and took
# no time, and summing original tokens would double-count spend in rollups.
_SERVE_STRIP_KEYS = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "cost_usd",
    "latency_ms",
)

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")

DDL_TEMPLATE = """
CREATE SCHEMA IF NOT EXISTS {schema};
CREATE TABLE IF NOT EXISTS {table} (
  image_sha256         TEXT NOT NULL,
  prompt_sha256        TEXT NOT NULL,
  model_id             TEXT NOT NULL,
  sample_idx           INT  NOT NULL,
  label                TEXT NOT NULL,
  response             JSONB NOT NULL,
  area                 TEXT,
  policy_graph_version TEXT,
  prompt_version       TEXT,
  policy_render        TEXT,
  run_id               TEXT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (image_sha256, prompt_sha256, model_id, sample_idx)
);
CREATE INDEX IF NOT EXISTS label_cache_model_idx ON {table} (model_id);
"""


def prompt_fingerprint(
    *,
    area: str,
    prompt_version: str,
    system_prompt: str,
    user_instructions: str,
    response_schema: dict[str, Any],
    policy_markdown: str,
    max_image_size: tuple[int, int] | list[int],
    jpeg_quality: int,
    temperature: float | None,
    reasoning_effort: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> str:
    """sha256 over every prompt-shaping input, canonically serialized.

    Deliberately content-derived: two policy versions with identical bytes
    hash the same (legitimate hit), while an invisible-to-version-names
    change — the r50 no-abstain prompt rewrite, a compressed vs full render,
    a temperature tweak — hashes differently (legitimate miss).
    """
    payload = {
        "fp_version": FINGERPRINT_VERSION,
        "area": area,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "user_instructions": user_instructions,
        "response_schema": response_schema,
        "policy_markdown": policy_markdown,
        "max_image_size": list(max_image_size),
        "jpeg_quality": int(jpeg_quality),
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "model_params": model_params or {},
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def required_samples(temperature: float | None) -> int:
    """Rounds needed before serving: 1 at temp==0, else CACHE_SAMPLES.

    ``None`` (temperature unsupported/omitted — e.g. gpt-5.5 reasoning
    models decode at their provider default) counts as nondeterministic.
    """
    if temperature is not None and float(temperature) == 0.0:
        return 1
    return CACHE_SAMPLES


def majority_vote(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve stored samples (sample_idx order) into one served verdict.

    Majority label; ties resolve to the LATEST sample among the tied labels
    (mirrors ``labelstore.resolve_golden``). The served response is the most
    recent sample that voted with the majority, so the justification always
    argues for the label being served. ``flip_rate = 1 - majority_n / n``:
    0.0 = perfectly self-consistent, 2/3 max at n=3 all-distinct.
    """
    if not samples:
        raise ValueError("majority_vote needs at least one sample")
    labels = [str(s["label"]) for s in samples]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts.values())
    winners = {label for label, c in counts.items() if c == best}
    majority = next(label for label in reversed(labels) if label in winners)
    chosen = next(s for s in reversed(samples) if str(s["label"]) == majority)
    return {
        "majority_label": majority,
        "labels": labels,
        "n_samples": len(samples),
        "flip_rate": round(1.0 - counts[majority] / len(labels), 4),
        "response": dict(chosen["response"]),
    }


def serve_payload(voted: dict[str, Any]) -> dict[str, Any]:
    """Project a majority-vote result onto the response dict a run persists."""
    response = {
        k: v for k, v in voted["response"].items() if k not in _SERVE_STRIP_KEYS
    }
    response["label"] = voted["majority_label"]
    return response


class LabelCache:
    """Thread-safe, fail-open Postgres access for the label cache.

    One shared connection guarded by a lock — volume is one lookup + at most
    one insert per (image, model), never hot. The first database error of any
    kind flips ``enabled`` off for the rest of the run and keeps the reason
    for the manifest; labeling proceeds live as if no cache existed.
    """

    def __init__(self, *, db_url: str | None = None, table: str = DEFAULT_TABLE):
        if not _TABLE_RE.match(table):
            raise ValueError(f"label cache table must be schema.table: {table!r}")
        self._db_url = db_url or labelstore_db_url()
        self._table = table
        self._conn = None
        self._lock = threading.Lock()
        self.enabled = True
        self.error: str | None = None
        self.stats = {"hits": 0, "misses": 0, "stored": 0}

    # -- internals ----------------------------------------------------------

    def _connect(self):
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self._db_url, autocommit=True)
            schema = self._table.split(".", 1)[0]
            self._conn.execute(
                DDL_TEMPLATE.format(schema=schema, table=self._table)
            )
        return self._conn

    def _disable(self, exc: Exception) -> None:
        self.enabled = False
        self.error = f"{type(exc).__name__}: {exc}"
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # -- public api ---------------------------------------------------------

    def lookup(
        self,
        image_sha256: str,
        prompt_sha256: str,
        model_id: str,
        *,
        required: int,
    ) -> dict[str, Any] | None:
        """Return the majority-vote payload, or None below ``required`` rounds."""
        if not self.enabled or not image_sha256:
            return None
        with self._lock:
            try:
                conn = self._connect()
                rows = conn.execute(
                    f"SELECT sample_idx, label, response FROM {self._table} "
                    "WHERE image_sha256 = %s AND prompt_sha256 = %s "
                    "AND model_id = %s ORDER BY sample_idx",
                    (image_sha256, prompt_sha256, model_id),
                ).fetchall()
            except Exception as exc:  # noqa: BLE001 — fail-open by design
                self._disable(exc)
                return None
        if len(rows) < max(1, required):
            self.stats["misses"] += 1
            return None
        samples = [
            {"sample_idx": idx, "label": label, "response": response}
            for idx, label, response in rows
        ]
        voted = majority_vote(samples)
        self.stats["hits"] += 1
        return voted

    def store(
        self,
        image_sha256: str,
        prompt_sha256: str,
        model_id: str,
        response: dict[str, Any],
        *,
        area: str,
        policy_graph_version: str,
        prompt_version: str,
        policy_render: str,
        run_id: str,
    ) -> bool:
        """Append one live sample. sample_idx is assigned atomically per key;
        a concurrent-run collision retries once with the next index."""
        if not self.enabled or not image_sha256:
            return False
        label = str(response.get("label", "")).strip()
        if not label:
            return False
        response_json = json.dumps(response, sort_keys=True, ensure_ascii=False)
        with self._lock:
            try:
                conn = self._connect()
                for _attempt in range(2):
                    cur = conn.execute(
                        f"INSERT INTO {self._table} "
                        "(image_sha256, prompt_sha256, model_id, sample_idx, "
                        " label, response, area, policy_graph_version, "
                        " prompt_version, policy_render, run_id) "
                        "SELECT %s, %s, %s, COALESCE(MAX(sample_idx), 0) + 1, "
                        "       %s, %s::jsonb, %s, %s, %s, %s, %s "
                        f"FROM {self._table} "
                        "WHERE image_sha256 = %s AND prompt_sha256 = %s "
                        "AND model_id = %s "
                        "ON CONFLICT DO NOTHING",
                        (
                            image_sha256, prompt_sha256, model_id,
                            label, response_json, area, policy_graph_version,
                            prompt_version, policy_render, run_id,
                            image_sha256, prompt_sha256, model_id,
                        ),
                    )
                    if cur.rowcount:
                        self.stats["stored"] += 1
                        return True
                return False
            except Exception as exc:  # noqa: BLE001 — fail-open by design
                self._disable(exc)
                return False

    def manifest_block(self, *, samples_required: dict[str, int]) -> dict[str, Any]:
        """The auditable record for run_manifest.json."""
        block: dict[str, Any] = {
            "enabled": self.enabled,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "stored": self.stats["stored"],
            "samples_required": dict(sorted(samples_required.items())),
        }
        if self.error:
            block["error"] = self.error
        return block


__all__ = [
    "CACHE_SAMPLES",
    "DEFAULT_TABLE",
    "FINGERPRINT_VERSION",
    "LabelCache",
    "majority_vote",
    "prompt_fingerprint",
    "required_samples",
    "serve_payload",
]
