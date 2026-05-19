# Decision-quality demo QA notes

Attila's demo has two headline objectives:

1. Grow the GenAI policy graph from SME/LLM evidence.
2. Prove that growth improves decision quality rather than just adding graph nodes.

The web flow should therefore read as: sample → label → score → **quality gate** → insights → grow/review proposal. The Quality section now frames itself as that gate and surfaces the decision readout before the detailed labeler table.

## UX checks added

- Quality summary now includes scored runs, images scored, policy-version count, ensemble accuracy, ensemble lift/drop vs the best individual labeler, split/boundary review load, and total cost.
- A short “Demo read” banner tells the presenter how to interpret the quality section before accepting policy growth.
- Ensemble rows are explicitly tagged so the majority-vote decision is not confused with an individual model.
- Proposal actions are state-aware: only pending proposals can be accepted, and pending/parse-error proposals can be rejected. The diff viewer shows the proposal state before the file diffs.

## Broken / timeout-prone flows found

- The Insights “Score this run” button called `/api/runs/<run_id>/compute`, but the server only routed `/compute-now`. The UI now calls `/compute-now`, and the server keeps `/compute` as a compatibility alias. Covered by `test_compute_alias_covers_insights_score_button`.
- The following proposal endpoints may be 524-prone behind a reverse proxy/tunnel because they synchronously call LLMs or build artifacts:
  - `POST /api/policy/cold-start`
  - `POST /api/policy/grow-batch`
  - `POST /api/policy/propose-diff`
  - `POST /api/policy/build-pdf`

For demo QA, hit validation failures first to confirm routing is healthy without spending or waiting on providers:

```bash
curl -sS -X POST http://127.0.0.1:8766/api/policy/grow-batch \
  -H 'content-type: application/json' \
  -d '{}' | python3 -m json.tool
```

Expected: fast `400` with a `run_id is required` error. If this hangs or returns proxy HTML, the issue is routing/proxy, not the LLM provider.

## Automated smoke coverage

Run:

```bash
python3 -m pytest tests/test_handlers_growth.py tests/test_web_policy_flow.py -q
node --check web/decision-quality.js
node --check web/insights.js
node --check web/policy-diff.js
node --check web/policy-grow.js
```

The web-policy flow test verifies:

- `/api/runs/<id>/compute-now` and legacy `/compute` both trigger scoring.
- Cold-start and grow-batch policy routes dispatch through the web server.
- Proposal get/accept/reject routes are wired for review states.
