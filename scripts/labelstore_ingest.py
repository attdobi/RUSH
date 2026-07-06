#!/usr/bin/env python3
"""Ingest sample manifests and recorded runs into the Postgres label store.

Files stay the per-run provenance; this backfills/refreshes the cross-run
store (rush schema, see pipeline/labelstore/schema.sql):

* ``item``            — one row per unique image (entity_id = source sha256)
* ``label_event``     — one SEED human-label event per item (idempotent:
                        re-ingest never duplicates seeds; the loop appends
                        real SME events later)
* ``golden_label``    — materialized from the event history (§4.1 tiers)
* ``generator_version`` — one row per policy_graph_version seen in runs
* ``llm_label``       — every model verdict, deduped on
                        (entity_id, generator_id, model_id, judge_index)

Usage::

    scripts/labelstore_ingest.py --init            # create schema only
    scripts/labelstore_ingest.py --all             # manifests + every run
    scripts/labelstore_ingest.py --run 20260706T042415-1b258772
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import labelstore  # noqa: E402
from pipeline.io_paths import MNIST_SAMPLE_MANIFEST, genai_manifest_default  # noqa: E402
from pipeline.manifest import load_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "data" / "runs"

MNIST_AREA = "MNIST_Digits"
GENAI_AREA = "Generative_AI"


def _manifest_records(area: str):
    path = MNIST_SAMPLE_MANIFEST if area == MNIST_AREA else genai_manifest_default()
    try:
        return load_records(path)
    except FileNotFoundError:
        return []


def _seed_rater(area: str) -> str:
    return f"seed:{'mnist-dataset' if area == MNIST_AREA else 'genai-combined-labels'}"


def ingest_manifest(conn, area: str) -> dict:
    """Upsert items + one idempotent seed label_event each; materialize golden."""
    records = _manifest_records(area)
    rater = _seed_rater(area)
    n_items = n_seeds = 0
    with conn.cursor() as cur:
        for rec in records:
            cur.execute(
                """
                INSERT INTO rush.item (entity_id, sample_id, content_uri, media_type, source, area)
                VALUES (%s, %s, %s, 'image', %s, %s)
                ON CONFLICT (entity_id) DO NOTHING
                """,
                (rec.sha256, rec.sample_id, rec.repo_rel_path, rec.dataset or "seed_gds", area),
            )
            n_items += cur.rowcount
            # Seed event: exactly one per item, ever. The partial unique index
            # label_event_seed_once makes this concurrency-safe; the EXISTS
            # guard on (entity_id, area) keeps a sha that appears in TWO areas'
            # manifests from seeding a foreign-domain label onto the item the
            # first area already owns.
            cur.execute(
                """
                INSERT INTO rush.label_event (entity_id, label, source_type, rater_id, comment)
                SELECT %s, %s, 'SME', %s, 'seed label imported from sample manifest'
                WHERE EXISTS (
                  SELECT 1 FROM rush.item WHERE entity_id = %s AND area = %s
                )
                ON CONFLICT (entity_id, rater_id) WHERE rater_id LIKE 'seed:%%' DO NOTHING
                """,
                (rec.sha256, rec.sme_label, rater, rec.sha256, area),
            )
            n_seeds += cur.rowcount
        # Materialize golden_label for every item in this area from its events.
        cur.execute(
            """
            SELECT i.entity_id,
                   array_agg(e.label ORDER BY e.label_event_id) AS labels,
                   max(e.cycle_id) AS last_epoch
            FROM rush.item i
            JOIN rush.label_event e ON e.entity_id = i.entity_id
            WHERE i.area = %s AND e.source_type = 'SME'
            GROUP BY i.entity_id
            """,
            (area,),
        )
        rows = cur.fetchall()
        for entity_id, labels, last_epoch in rows:
            golden = labelstore.golden_row(list(labels), seed_source="sme_single", last_epoch=last_epoch)
            cur.execute(
                """
                INSERT INTO rush.golden_label
                  (entity_id, current_label, seed_source, num_sme_labels,
                   num_sme_agree_current, confidence_tier, at_cap, last_epoch, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (entity_id) DO UPDATE SET
                  current_label = EXCLUDED.current_label,
                  num_sme_labels = EXCLUDED.num_sme_labels,
                  num_sme_agree_current = EXCLUDED.num_sme_agree_current,
                  confidence_tier = EXCLUDED.confidence_tier,
                  at_cap = EXCLUDED.at_cap,
                  last_epoch = EXCLUDED.last_epoch,
                  updated_at = now()
                """,
                (
                    entity_id,
                    golden["current_label"],
                    golden["seed_source"],
                    golden["num_sme_labels"],
                    golden["num_sme_agree_current"],
                    golden["confidence_tier"],
                    golden["at_cap"],
                    golden["last_epoch"],
                ),
            )
    conn.commit()
    return {"area": area, "manifest_records": len(records), "new_items": n_items, "new_seed_events": n_seeds}


def _entity_lookup(conn, area: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT sample_id, entity_id FROM rush.item WHERE area = %s", (area,))
        return dict(cur.fetchall())


def ingest_run(conn, run_dir: Path) -> dict | None:
    manifest_path = run_dir / "run_manifest.json"
    votes_path = run_dir / "label_votes.jsonl"
    if not manifest_path.exists() or not votes_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    area = manifest.get("area") or GENAI_AREA
    generator_id = manifest.get("policy_graph_version") or manifest.get("policy_version") or "unknown"
    run_id = manifest.get("run_id") or run_dir.name

    lookup = _entity_lookup(conn, area)
    inserted = skipped_dupe = unmatched = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rush.generator_version (generator_id, diff_text, gate_status)
            VALUES (%s, '', 'accepted')
            ON CONFLICT (generator_id) DO NOTHING
            """,
            (generator_id,),
        )
        for raw in votes_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            vote = json.loads(raw)
            entity_id = lookup.get(str(vote.get("image_id") or ""))
            if not entity_id:
                unmatched += 1
                continue
            cur.execute(
                """
                INSERT INTO rush.llm_label
                  (entity_id, generator_id, model_id, judge_index, decision, l2_label,
                   is_boundary, difficulty, confidence, justification, prompt_version,
                   run_id, latency_ms, input_tokens, output_tokens, cost_usd)
                VALUES (%s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT llm_label_dedup DO NOTHING
                """,
                (
                    entity_id,
                    generator_id,
                    str(vote.get("model_id") or vote.get("labeler_id") or "unknown"),
                    str(vote.get("label") or "abstain"),
                    vote.get("l2_label"),
                    bool(vote.get("is_boundary")),
                    str(vote.get("difficulty") or "low"),
                    vote.get("confidence"),
                    vote.get("justification"),
                    vote.get("prompt_version"),
                    run_id,
                    vote.get("latency_ms"),
                    vote.get("input_tokens"),
                    vote.get("output_tokens"),
                    vote.get("cost_usd"),
                ),
            )
            if cur.rowcount:
                inserted += 1
            else:
                skipped_dupe += 1
    conn.commit()
    return {
        "run_id": run_id,
        "area": area,
        "generator_id": generator_id,
        "inserted": inserted,
        "skipped_duplicates": skipped_dupe,
        "unmatched_sample_ids": unmatched,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="create the rush schema and exit")
    ap.add_argument("--all", action="store_true", help="ingest manifests + every non-archived run")
    ap.add_argument("--run", default=None, help="ingest one run id")
    args = ap.parse_args(argv)
    if not (args.init or args.all or args.run):
        ap.error("pick one of --init / --all / --run <run_id>")

    conn = labelstore.connect()
    try:
        labelstore.init_schema(conn)
        if args.init and not (args.all or args.run):
            print(f"schema ready in {labelstore.db_url()} (rush.*)")
            return 0

        for area in (MNIST_AREA, GENAI_AREA):
            stats = ingest_manifest(conn, area)
            print(f"[manifest] {stats}")

        run_dirs: list[Path] = []
        if args.run:
            run_dir = RUNS_ROOT / args.run
            if not (run_dir / "run_manifest.json").exists() or not (run_dir / "label_votes.jsonl").exists():
                print(f"error: no ingestable run at {run_dir} "
                      "(need run_manifest.json + label_votes.jsonl)", file=sys.stderr)
                return 2
            run_dirs = [run_dir]
        elif args.all:
            run_dirs = sorted(
                d for d in RUNS_ROOT.iterdir()
                if d.is_dir() and not d.name.startswith("_") and (d / "label_votes.jsonl").exists()
            )
        total_inserted = total_skipped = 0
        for run_dir in run_dirs:
            stats = ingest_run(conn, run_dir)
            if stats:
                total_inserted += stats["inserted"]
                total_skipped += stats["skipped_duplicates"]
                print(f"[run] {stats}")
        if run_dirs:
            print(f"\nTOTAL: {total_inserted} verdicts stored, {total_skipped} cross-run duplicates skipped")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
