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
  -- Human-label confidence (Attila 2026-07-06): p = 1 - 1/(m + 0.2) where
  -- m = number of human labels AGREEING with the current resolved label.
  -- m=1 -> 0.167, m=2 -> 0.545, m=3 -> 0.688 — a lone human label is weak
  -- evidence ("the golden set is not so golden"); agreement compounds it.
  human_confidence      NUMERIC,
  at_cap                BOOLEAN NOT NULL DEFAULT FALSE,
  persistent_misaligned BOOLEAN NOT NULL DEFAULT FALSE,
  last_epoch            INT,
  updated_at            TIMESTAMPTZ NOT NULL
);
-- Migration for stores created before human_confidence existed.
ALTER TABLE rush.golden_label ADD COLUMN IF NOT EXISTS human_confidence NUMERIC;

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

-- ---------------------------------------------------------------------------
-- Derived per-sample gradient signals ("Notation and the per-sample gradient",
-- Attila's policy-optimization note, 2026-07-06). For judge j on item i with
-- self-reported confidence c in its OWN predicted label:
--   p    = c            if prediction == golden label   (prob. on true class)
--   p    = 1 - c        otherwise                        (binary approximation)
--   |g|  = 1 - p        gradient magnitude: confident-wrong ~1 (most
--                       informative error), confident-correct ~0 (ignore)
--   h    = c(1 - c)     curvature/uncertainty: max at c=0.5, correctness-blind
--   loss = -ln(p)       per-sample cross-entropy
-- Views (not tables): always current w.r.t. golden_label overturns, nothing
-- to re-materialize. Abstains and NULL confidences are excluded.
CREATE OR REPLACE VIEW rush.sample_gradient AS
SELECT
  l.llm_label_id,
  l.entity_id,
  l.generator_id,
  l.model_id,
  l.decision,
  l.confidence,
  g.current_label,
  g.human_confidence,
  (l.decision = g.current_label)                          AS is_correct,
  CASE WHEN l.decision = g.current_label
       THEN l.confidence ELSE 1 - l.confidence END        AS p_true,
  1 - CASE WHEN l.decision = g.current_label
       THEN l.confidence ELSE 1 - l.confidence END        AS grad_magnitude,
  l.confidence * (1 - l.confidence)                       AS hessian_uncertainty,
  -ln(GREATEST(CASE WHEN l.decision = g.current_label
       THEN l.confidence ELSE 1 - l.confidence END, 1e-6)) AS loss_ce,
  l.is_boundary,
  l.difficulty
FROM rush.llm_label l
JOIN rush.golden_label g USING (entity_id)
WHERE l.decision <> 'abstain' AND l.confidence IS NOT NULL;

-- Panel-level rollup per (item, policy version): the candidate-selection
-- substrate for policy-gradient experiments. avg_grad_magnitude is the
-- "average of confidence scores across the multiple LLMs" signal, expressed
-- in gradient form; is_split / any_boundary / difficulty counts feed the
-- other selection strategies.
CREATE OR REPLACE VIEW rush.panel_signal AS
SELECT
  entity_id,
  generator_id,
  count(*)                                        AS n_judges,
  avg(confidence)                                 AS avg_confidence,
  avg(p_true)                                     AS avg_p_true,
  avg(grad_magnitude)                             AS avg_grad_magnitude,
  max(grad_magnitude)                             AS max_grad_magnitude,
  avg(hessian_uncertainty)                        AS avg_hessian,
  avg(loss_ce)                                    AS avg_loss,
  count(DISTINCT decision)                        AS distinct_decisions,
  (count(DISTINCT decision) > 1)                  AS is_split,
  bool_or(is_boundary)                            AS any_boundary,
  count(*) FILTER (WHERE NOT is_correct)          AS n_wrong,
  count(*) FILTER (WHERE difficulty = 'high')     AS n_difficulty_high,
  count(*) FILTER (WHERE difficulty = 'medium')   AS n_difficulty_medium,
  max(human_confidence)                           AS human_confidence
FROM rush.sample_gradient
GROUP BY entity_id, generator_id;

-- ---------------------------------------------------------------------------
-- Experiment crank (Attila 2026-07-06): each demo is a seeded, numbered
-- experiment run — a PPO-style loop of k_max cycles where a mini-batch of N
-- train images drives a clipped (1..max_changes) policy edit, the fixed
-- seeded test partition gates acceptance on system macro-F1, and the locked
-- holdout is scored only at the start/final versions. Everything below is
-- dual-written: data/experiments/<id>/ JSON stays the portable per-run truth
-- (a fresh clone demos with no DB); these tables are the cross-experiment
-- analysis layer for the paper.

-- generator_version predates the crank: widen the gate vocabulary so
-- candidate policy versions that were evaluated but not accepted persist
-- ('skipped'), and staged-but-ungated ones are representable ('pending').
ALTER TABLE rush.generator_version DROP CONSTRAINT IF EXISTS generator_version_gate_status_check;
ALTER TABLE rush.generator_version ADD CONSTRAINT generator_version_gate_status_check
  CHECK (gate_status IN ('accepted','rejected','skipped','pending'));
ALTER TABLE rush.generator_version ADD COLUMN IF NOT EXISTS experiment_id TEXT;
ALTER TABLE rush.generator_version ADD COLUMN IF NOT EXISTS n_changes INT;
-- When a candidate is accepted it becomes a real policy-graph version; the
-- candidate row keeps its evaluation verdicts (llm_label FKs) and points at
-- the accepted generator_id here.
ALTER TABLE rush.generator_version ADD COLUMN IF NOT EXISTS accepted_as TEXT;

CREATE TABLE IF NOT EXISTS rush.experiment (
  experiment_id  TEXT PRIMARY KEY,             -- exp-YYYYMMDDTHHMMSS-<hex6>
  run_number     INT,                          -- human-friendly sequence per area
  area           TEXT NOT NULL,
  seed           BIGINT NOT NULL,              -- master RNG seed: test partition,
                                               -- per-cycle train batches, anchors
  k_max          INT NOT NULL,
  batch_n        INT NOT NULL,                 -- train images per cycle
  test_n         INT NOT NULL,                 -- fixed gate partition size
  judge_models   JSONB NOT NULL,               -- the panel, e.g. ["openai/gpt-5.4-mini-low", ...]
  gate_model     TEXT NOT NULL,
  drafter_model  TEXT NOT NULL,
  strategy       TEXT NOT NULL DEFAULT 'random_misalignment',  -- S1; S2-S5 later
  max_changes    INT NOT NULL DEFAULT 5 CHECK (max_changes BETWEEN 1 AND 5),
  epsilon        NUMERIC NOT NULL DEFAULT 0,   -- accept iff f1_after > f1_before + epsilon
  base_generator TEXT,                         -- policy version at k=0 (e.g. 'MNIST_Digits.v0.1')
  config         JSONB NOT NULL DEFAULT '{}'::jsonb,
  status         TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','completed','failed','stopped')),
  started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS rush.experiment_cycle (
  experiment_id       TEXT NOT NULL REFERENCES rush.experiment(experiment_id),
  k                   INT  NOT NULL,           -- 0 = baseline eval, 1..k_max = crank turns
  cycle_seed          BIGINT,                  -- derived from (seed, k); drives sampling
  generator_before    TEXT,                    -- policy version entering the cycle
  candidate_generator TEXT,                    -- minted candidate id (NULL if no edit proposed)
  generator_after     TEXT,                    -- == candidate on accept, == before on skip
  train_ids           JSONB,                   -- sample_ids labeled this cycle
  n_misaligned        INT,
  anchor_ids          JSONB,                   -- S1 random anchor sample_ids
  n_changes_proposed  INT,                     -- what the drafter emitted
  n_changes_applied   INT,                     -- after the 1..max_changes clip
  proposal_id         TEXT,                    -- data/policy_proposals/<id>
  train_run_id        TEXT,                    -- child labeling runs (data/runs/<id>)
  candidate_run_id    TEXT,
  status              TEXT NOT NULL DEFAULT 'open',
  error               TEXT,
  started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at           TIMESTAMPTZ,
  PRIMARY KEY (experiment_id, k)
);
-- Status vocabulary lives in a re-runnable named constraint so it can widen
-- (baseline = the k=0 eval row; stopped = operator interrupted mid-cycle).
ALTER TABLE rush.experiment_cycle DROP CONSTRAINT IF EXISTS experiment_cycle_status_check;
ALTER TABLE rush.experiment_cycle ADD CONSTRAINT experiment_cycle_status_check
  CHECK (status IN ('open','baseline','accepted','skipped','no_misalignments','failed','stopped'));

-- Attila's tracking spec verbatim: "track accuracy, f1, precision, recall,
-- fpr, fnr at each cycle for each model and the system of judges" — on both
-- train and test. scorer = a judge model_id or 'system' (majority vote).
-- Candidate evaluations land under the candidate's generator_id, so one
-- cycle can carry test metrics for both baseline and candidate.
CREATE TABLE IF NOT EXISTS rush.experiment_metric (
  experiment_id   TEXT NOT NULL,
  k               INT  NOT NULL,
  split           TEXT NOT NULL CHECK (split IN ('train','test','holdout')),
  scorer          TEXT NOT NULL,               -- model_id | 'system'
  generator_id    TEXT NOT NULL,
  n               INT NOT NULL,
  n_abstained     INT NOT NULL DEFAULT 0,
  accuracy        NUMERIC,
  macro_f1        NUMERIC,
  macro_precision NUMERIC,
  macro_recall    NUMERIC,
  macro_fpr       NUMERIC,
  macro_fnr       NUMERIC,
  micro_f1        NUMERIC,
  micro_precision NUMERIC,
  micro_recall    NUMERIC,
  micro_fpr       NUMERIC,
  micro_fnr       NUMERIC,
  per_class       JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (experiment_id, k, split, scorer, generator_id),
  FOREIGN KEY (experiment_id, k)
    REFERENCES rush.experiment_cycle(experiment_id, k)
);

-- One row per gate evaluation. The deterministic rule (metric_pass) is the
-- trust region hard wall; the gate agent (gpt-5.5 default) may veto a
-- metric-passed candidate but can never force-accept a metric-failed one.
CREATE TABLE IF NOT EXISTS rush.gate_decision (
  gate_decision_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  experiment_id    TEXT NOT NULL,
  k                INT  NOT NULL,
  baseline_generator  TEXT NOT NULL,
  candidate_generator TEXT NOT NULL,
  gate_model       TEXT NOT NULL,
  metric           TEXT NOT NULL DEFAULT 'test_system_macro_f1',
  value_before     NUMERIC,
  value_after      NUMERIC,
  metric_pass      BOOLEAN NOT NULL,
  decision         TEXT NOT NULL CHECK (decision IN ('accept','skip')),
  decided_by       TEXT NOT NULL
    CHECK (decided_by IN ('metric_rule','gate_agent','gate_agent_veto','human','override_guard','gate_off')),
  rationale        TEXT,
  raw_response     TEXT,
  cost_usd         NUMERIC,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (experiment_id, k)
    REFERENCES rush.experiment_cycle(experiment_id, k)
);
-- decided_by vocabulary is re-runnable so it can widen (gate_off = the
-- acceptance gate disabled for the run; every clipped edit lands, the
-- metric is recorded but never enforced).
ALTER TABLE rush.gate_decision DROP CONSTRAINT IF EXISTS gate_decision_decided_by_check;
ALTER TABLE rush.gate_decision ADD CONSTRAINT gate_decision_decided_by_check
  CHECK (decided_by IN ('metric_rule','gate_agent','gate_agent_veto','human','override_guard','gate_off'));

-- The human "critic of the critic": SME review of gate decisions, deferred to
-- the end of the iteration cycle so the loop stays automated. Recorded for
-- future RLHF of the gate agent. Keyed by (experiment_id, k) because the UI
-- reads the portable JSON, not gate_decision_id.
CREATE TABLE IF NOT EXISTS rush.gate_review (
  gate_review_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  experiment_id  TEXT NOT NULL,
  k              INT  NOT NULL,
  reviewer       TEXT NOT NULL,
  verdict        TEXT NOT NULL CHECK (verdict IN ('correct','incorrect','unsure')),
  comment        TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (experiment_id, k)
    REFERENCES rush.experiment_cycle(experiment_id, k)
);
CREATE INDEX IF NOT EXISTS gate_review_exp_idx ON rush.gate_review (experiment_id, k);
