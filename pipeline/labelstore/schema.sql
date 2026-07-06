-- RUSH label store — implements §4 "Data model" of the re-adjudication loop
-- MVP (rush_readjudication_loop_mvp.docx): an append-only event log
-- (label_event) plus materialized state (golden_label), everything joined on
-- entity_id. Lives in its own schema so `DROP SCHEMA rush CASCADE` removes it
-- cleanly from the shared database.
--
-- Deviations from the reference DDL, all additive and noted inline:
--   * item.area          — one store serves multiple demo areas (MNIST_Digits,
--                          Generative_AI); label domains are validated
--                          app-side against the area ontology instead of the
--                          doc's binary CHECK.
--   * entity_id          — the source-file sha256 from the sample manifest:
--                          stable across manifests, renames, and re-samples.
--   * llm_label          — carries model/run/cost provenance, and the dedup
--                          contract: UNIQUE (entity_id, generator_id,
--                          model_id, judge_index). Same image x same policy
--                          version x same judge is stored exactly once.
--   * generator_version  — cycle_id nullable and gate fields optional so
--                          pre-loop policy versions can be backfilled.

CREATE SCHEMA IF NOT EXISTS rush;

CREATE TABLE IF NOT EXISTS rush.item (
  entity_id      TEXT PRIMARY KEY,            -- source-file sha256
  sample_id      TEXT NOT NULL,               -- manifest id (train_00001, ...)
  content_uri    TEXT NOT NULL,               -- repo-relative path
  media_type     TEXT NOT NULL DEFAULT 'image',
  source         TEXT NOT NULL,               -- dataset / seed_gds / production_sample
  area           TEXT NOT NULL,               -- MNIST_Digits | Generative_AI | ...
  l2_category    TEXT,                        -- reserved for later stratification
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS item_area_sample_idx ON rush.item (area, sample_id);

CREATE TABLE IF NOT EXISTS rush.label_event (   -- append-only, never updated
  label_event_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  entity_id       TEXT NOT NULL REFERENCES rush.item(entity_id),
  label           TEXT NOT NULL,
  source_type     TEXT NOT NULL CHECK (source_type IN ('SME','BPO','CLICK')),
  rater_id        TEXT,
  cycle_id        INT,                        -- the epoch; NULL for seed labels
  saw_llm_context BOOLEAN NOT NULL DEFAULT FALSE,
  escalation_flag BOOLEAN NOT NULL DEFAULT FALSE,
  comment         TEXT,
  duration_ms     INT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS label_event_entity_idx ON rush.label_event (entity_id);
-- Concurrency-safe idempotency for imported SEED events: the ingest guard is
-- WHERE NOT EXISTS, which two parallel ingests can both pass under READ
-- COMMITTED; this partial unique index makes the second writer a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS label_event_seed_once
  ON rush.label_event (entity_id, rater_id) WHERE rater_id LIKE 'seed:%';

CREATE TABLE IF NOT EXISTS rush.golden_label (  -- materialized from label_event
  entity_id             TEXT PRIMARY KEY REFERENCES rush.item(entity_id),
  current_label         TEXT NOT NULL,
  seed_source           TEXT NOT NULL,        -- sme_single|bpo_single|bpo_maj_3|click
  num_sme_labels        INT NOT NULL DEFAULT 0,
  num_sme_agree_current INT NOT NULL DEFAULT 0,
  confidence_tier       TEXT NOT NULL,        -- §4.1 tiers
  at_cap                BOOLEAN NOT NULL DEFAULT FALSE,
  persistent_misaligned BOOLEAN NOT NULL DEFAULT FALSE,
  last_epoch            INT,
  updated_at            TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS rush.cycle (
  cycle_id        INT PRIMARY KEY,
  split_seed      BIGINT NOT NULL,
  gds_epoch_in    INT NOT NULL,
  gds_epoch_out   INT,
  start_generator TEXT,
  final_generator TEXT,
  status          TEXT NOT NULL DEFAULT 'open',
  started_at      TIMESTAMPTZ,
  closed_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS rush.generator_version (  -- prompt version control
  generator_id  TEXT PRIMARY KEY,             -- policy_graph_version, e.g. 'MNIST_Digits.v0.1'
  cycle_id      INT REFERENCES rush.cycle(cycle_id),  -- NULL for backfilled pre-loop versions
  minibatch_k   INT,
  parent_id     TEXT REFERENCES rush.generator_version(generator_id),
  diff_text     TEXT NOT NULL DEFAULT '',     -- the single trackable edit
  prompt_len    INT NOT NULL DEFAULT 0,
  gate_status   TEXT NOT NULL DEFAULT 'accepted' CHECK (gate_status IN ('accepted','rejected')),
  f1_val_before NUMERIC,
  f1_val_after  NUMERIC,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rush.split_assignment (   -- re-rolled every cycle
  cycle_id      INT  NOT NULL REFERENCES rush.cycle(cycle_id),
  entity_id     TEXT NOT NULL REFERENCES rush.item(entity_id),
  split         TEXT NOT NULL CHECK (split IN ('train','val','test')),
  sample_weight NUMERIC NOT NULL DEFAULT 1.0,
  PRIMARY KEY (cycle_id, entity_id)
);

CREATE TABLE IF NOT EXISTS rush.llm_label (
  llm_label_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  entity_id     TEXT NOT NULL REFERENCES rush.item(entity_id),
  generator_id  TEXT NOT NULL REFERENCES rush.generator_version(generator_id),
  model_id      TEXT NOT NULL,
  judge_index   INT NOT NULL DEFAULT 1,       -- >1 only for 3x ensembles
  decision      TEXT NOT NULL,                -- includes 'abstain'
  l2_label      TEXT,
  is_boundary   BOOLEAN NOT NULL DEFAULT FALSE,
  difficulty    TEXT NOT NULL DEFAULT 'low',
  confidence    NUMERIC,
  justification TEXT,
  prompt_version TEXT,
  run_id        TEXT,                         -- provenance: first run that produced this row
  latency_ms    INT,
  input_tokens  INT,
  output_tokens INT,
  cost_usd      NUMERIC,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- The dedup contract: one stored verdict per image x policy version x judge.
  CONSTRAINT llm_label_dedup UNIQUE (entity_id, generator_id, model_id, judge_index)
);
CREATE INDEX IF NOT EXISTS llm_label_generator_idx ON rush.llm_label (generator_id);

CREATE TABLE IF NOT EXISTS rush.misalignment (       -- cycle-close snapshot + queue state
  cycle_id     INT  NOT NULL REFERENCES rush.cycle(cycle_id),
  entity_id    TEXT NOT NULL REFERENCES rush.item(entity_id),
  generator_id TEXT NOT NULL,
  golden_label TEXT NOT NULL,
  llm_label    TEXT NOT NULL,
  split        TEXT NOT NULL,
  priority     NUMERIC NOT NULL,
  queue_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (queue_status IN ('pending','queued','adjudicated','skipped_cap','audit')),
  PRIMARY KEY (cycle_id, entity_id)
);
