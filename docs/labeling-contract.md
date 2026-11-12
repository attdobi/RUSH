# Labeling workspace implementation contract

This additive prototype uses the repository's standard-library Python and plain JavaScript stack. The standalone service owns a SQLite review store. The static preview uses explicitly labeled synthetic examples and browser-local state. It never launches a paid RUSH experiment or claims a production write succeeded.

## Files and ownership

- `labeling/`: Python service, SQLite lifecycle, adapters (backend).
- `web/labeling/`: frontend and a matching browser-demo adapter (frontend).
- `web/labeling/fixtures.json`, `web/labeling/assets/`: synthetic examples (fixtures).
- `tests/test_labeling_store.py`, `tests/test_labeling_http.py`, `tests/test_labeling_demo.mjs`: lifecycle and integration tests.
- `docs/labeling-platform.md`: running and integration instructions.

## Shared item schema

```json
{
  "id": "turn-1042", "project": "ConvoAI", "type": "assistant",
  "title": "A refusal with a harmful request", "description": "A synthetic boundary case.",
  "category": "harassment", "policy_version": "7.2", "revision": 1,
  "status": "open", "split": "train", "assignee": null,
  "created_at": "2026-11-12T16:00:00Z",
  "source": {"name": "Synthetic production example", "sampling": "apo_disagreement", "run_id": "example-run-018"},
  "priority": {"rank": 1, "score": 0.5066667, "gradient": 0.90, "disagreement": 1, "consensus": 1, "reason": "Boundary ambiguity", "tier": "T1"},
  "route": {"nodes": ["HAR-ROOT", "HAR-2"], "version": "router-demo-3", "confidence": 0.98, "fallback": false},
  "content": {"image": "assets/conversation.svg", "title": "Example", "description": "Example description", "messages": [{"role": "user", "text": "Example request"}, {"role": "assistant", "text": "Example reply"}]},
  "targets": [
    {"id": "input", "name": "User input", "category": "harassment", "choices": ["No violation", "Violation", "Uncertain"], "ai": {"label": "Violation", "confidence": 0.91, "confidence_kind": "uncalibrated", "model": "example-judge", "prompt_version": "18", "rationale": "Policy-grounded example rationale", "policy_refs": ["HAR-2"], "evidence": "Quoted content"}, "gold": {"label": "No violation", "revision": 1}, "staged_label": null},
    {"id": "output", "name": "AI response", "category": "harassment", "choices": ["No violation", "Violation", "Uncertain"], "ai": null, "gold": null, "staged_label": null}
  ],
  "events": [], "independent_labels": null,
  "guidance_visible": false
}
```

Assistant items require both `input` and `output` targets. Image/pin items use `content`. Search relevance uses a `relevance` target with choices `1` through `5` and `Uncertain`. Target and category identities survive every event and snapshot. Fixtures may contain `seed_independent_labels` for the demo reviewer to make the first selected item an adjudication-ready example. All other items start blind. UI must not render hidden AI/current-gold guidance until independent review is submitted. API must redact AI and current gold values from item responses for blind reviewers, not merely hide with CSS. Summary results never include those values.

## Fixture file

Top-level object: `initial_epoch: 42`, `viewer: {id: "demo-policy", name: "Alex Rivera", role: "policy"}`, `categories: [{id,name,color}]`, `projects: [string]`, `policies: [{id,category,title,body,version,parent}]`, `items: [item]`. All examples, ranking signals, scores and policy text are synthetic. Keep at least one protected evaluation item (`split: "holdout"`) visibly protected.

The fixture also supplies a versioned `graph` with stable node IDs, typed edges,
root IDs and a content hash. Policy nodes include type, body hash, provenance
and locked/pinned context. The optional historical `route.trace` records the
router/model version, policy version, graph hash, requested/effective mode,
proposed (`selected_ids`), pinned, served and cited nodes, and node decisions.
Those AI trace fields are redacted before independent review and omitted from
queue summaries. A human route correction preserves the historical trace and
marks it stale; it does not rewrite what the judge originally received.
Proposed context may differ from served context in shadow mode. Known nodes
cited without being served, or missing pinned context, remain visible as audit
diagnostics instead of being silently repaired or discarded.

Routing relevance, retained score mass and a recall target are separate from
measured recall. Priority is supplied upstream; the labeling service does not
replace RUSH's ranking algorithm. Human-review priority, policy anchor value,
SME agreement, panel consensus and gradient proxy are distinct fields.

## HTTP API

All paths start `/api/labeling`. Errors: `{error: {code, message}}` with appropriate 400/401/403/404/409/422 status. Mutations accept `Idempotency-Key` (same key + same body returns stored result, different body returns 409). The server derives actor and role from its session/auth configuration, never from submitted JSON. The local demo viewer is explicit. In token mode require configured bearer tokens, roles and reviewer IDs, with no default identity fallback.

- `GET /bootstrap` → `{viewer, mode, epoch, stats, categories, projects, policies, epochs, runs, integrations}`. `stats`: total, open, escalated, approved, published, gold_items. `integrations`: `{gds: "local", apo: "outbox"}`.
- `GET /items` → `{items, total}`. Filter query parameters: `q`, `type`, `category`, `status`, `project`. Sort by priority score descending by default. Summaries include core metadata, priority, source, route, content thumbnail and target count, but no gold/AI decisions.
- `GET /items/{id}` → full item with `events`, `independent_labels`, `guidance_visible`. Return current viewer's first-pass choices in independent_labels when available.
- `POST /items/{id}/claim` body `{base_revision}` → updated full item. Reject another active assignee (409).
- `POST /items/{id}/reviews` body `{base_revision, phase: "independent"|"adjudication", labels: {target_id: label}, rationale, policy_refs: [string], decision: "save"|"approve"|"escalate"}` → updated full item. All required targets need valid choices. Independent/save records first pass and reveals guidance. Adjudication/approve stages labels, status approved, no current-gold mutation. Adjudication/escalate sets status escalated and an escalation owner, preserves old gold. Only SME/policy role can adjudicate. Uncertain cannot be approved. Holdout cannot be approved via this workflow. Require rationale. Record phase, AI-visible flag, actor/role, target/category, timestamp, before/after and policy version. Check base revision transactionally.
- `POST /items/{id}/route` body `{base_revision, nodes: [policy_id], reason}` → updated full item. Require valid nodes and reason, append event; broad/fallback review is allowed.
- `POST /epochs` body `{expected_epoch, note}` → epoch record `{epoch, created_at, created_by, item_count, changed_count, digest, status: "local_published"}`. Policy role only. Atomically publish all staged approved training items, append publication events, preserve snapshots, update effective local gold. Reject no changes (409) and stale expected epoch (409). Export is not a production writeback acknowledgment.
- `GET /epochs/{epoch}/export` → immutable manifest `{epoch, policy_versions, created_at, digest, items}`. Items are optimization-eligible approved gold records with target/category identity. Never include holdout rows. Pin policy/content versions and complete input/output labels.
- `POST /reruns` body `{epoch, note}` → `{id, epoch, digest, created_at, status: "awaiting_connector"}`. Policy role only; epoch must exist, handoff pins exact snapshot digest. Persist to outbox. No automatic experiment launch, claimed optimization, artificial quality uplift or promotion.
- `GET /events` → `{events}` in reverse chronological order, containing audit metadata and guidance-safe labels (may omit labels globally).
- `POST /items/import` body `{items: [item]}` → `{imported, skipped}`. Policy role only. Validate IDs, targets, split, choices, safe local asset or HTTPS image reference, known policy refs/categories. Existing IDs must not overwrite human history. Never trust imported status, staged labels, actor, review events or approved-state fields.

`runtime.js` is static with `window.RUSH_LABELING = {mode: "demo"}`. The Python server overrides this resource with `{mode: "api", baseUrl: "/api/labeling"}`. Shared frontend chooses one client. Use the same response shapes and lifecycle in both modes. No framework/build step required.

## Review UX

Use the actual RUSH navy, blue and mint visual language. A three-column queue / original content / human-decision workspace is the primary screen. Make image, pin and input/output examples visually distinct. Support search, filter, item selection, independent label submission then AI reveal, confirm/change labels, explicit Policy escalation, routing correction, full history and keyboard navigation. Secondary views show gold epochs/export, Policy insights from actual open escalations, and activity. Publication and queued APO handoff must be distinct actions with clear status. Every visible control works or has an explicit disabled reason. Responsive layout and keyboard labels/focus are required.

## Acceptance boundaries

Protect original RUSH run/spend gates. The existing RUSH review endpoint only writes a JSONL log and its experiment endpoint supports image-based demos, not ConvoAI labels. Provide a read-only queue import adapter plus a durable outbox/export contract for broader integration, documented honestly. No existing main application route or data is replaced. Tests exercise independent input/output persistence, blind reveal, idempotent retries, stale revisions, roles, holdout exclusion, immutable epochs and version-pinned rerun requests.
