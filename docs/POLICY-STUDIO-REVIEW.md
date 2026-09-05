# Policy Studio — review guide

Branch: `feat/policy-intelligence-executive-demo`  
Baseline reviewed: RUSH `172dc0febc6063aff08a95bb68cd69935be89c2e`  
Review date: 2026-09-05

## Run without touching the current deployment

Use the Python environment already used by the existing RUSH server. A separate
worktree and port keep the current checkout and running service unchanged:

```sh
cd ~/RUSH
git fetch origin
git worktree add ../RUSH-vp-demo origin/feat/policy-intelligence-executive-demo
cd ../RUSH-vp-demo
RUSH_STUDIO_DATA_ROOT="$HOME/RUSH" python3 scripts/rush_web_server.py --port 8767
```

Open `http://127.0.0.1:8767/`. The optional `RUSH_STUDIO_DATA_ROOT` points the
**read-only studio API** at the existing local policy and experiment history.
It does not redirect any write handler. Omit it to inspect only the worktree's
own artifacts. No label records, credentials or provider calls are needed for
the studio. The worktree does not automatically inherit the original `.env`.
The existing mutation-capable research lab is at `/lab.html#loop`; use it only
when intentionally running an experiment, with the desired environment configured.

## Five-minute presentation

Start at the new landing page: **Learn the rule. Not the row.** Choose GenAI or
MNIST. A successful local-history connection selects a recorded run with explicit
lineage when one is available. A snapshot catalog is clearly marked as not a
lineage. Older dry runs are not used as evidence.

Select **Illustrative walkthrough** for a predictable presentation. It is labeled
on every frame and contains no measured performance claims. Play the timeline,
pause on a boundary, inspect a rule, compare its prior wording, then show the
rejected step: the graph does not grow just because a proposal exists. Added,
changed and retired nodes are distinguished. Zoom, pan, keyboard inspection,
search and an expanded view support exploration.

Open **Decision paths**. The typed shadow program is separate from the knowledge
graph. Try clear evidence, conflicting evidence and missing evidence. Unknown or
untrusted observations route to review. GenAI does not equate missing metadata
with a real-image label. Export the program, supplied facts and resulting trace
for inspection. A recorded policy without an attached executable program gets
an honest empty state, never a guessed rule compiler.

Finish at **Method & evidence**. It distinguishes implementation from hypotheses
and explains why improvements must be measured on protected, representative data.

## What changed

- New dependency-free executive landing and graph explorer in `web/index.html`,
  `studio.css`, `studio.js`, `studio-core.js` and `studio-fixtures.js`.
- The prior landing/experiment UI is preserved byte-for-byte as `web/lab.html`;
  its shared About module now supplies corrected methodology and adjusts several
  misleading legacy presentation statements without removing controls.
- `pipeline/web/studio.py` supplies bounded read-only snapshot and history APIs.
  Version directories and explicit run records determine what is shown. Body
  SHA-256 identifies content changes, not merely version-frontmatter changes.
  Path traversal and escaping symlinks are rejected. Missing lineage is disclosed.
- `pipeline/web/server.py` serves both entry points with build-aware asset
  versions and routes `/api/studio/*` separately. Studio POST requests return 405.
- The existing multiclass public scoring API delegates to a dependency-free,
  count-based implementation. F1 is `2TP / (2TP + FP + FN)`: errors with zero true
  positives yield zero, not a missing value omitted from macro-F1. Entirely absent
  classes remain undefined under the documented convention. Ensemble ties are
  counted as abstentions rather than silently omitted from the coverage count.

## Important scientific boundaries

The current optimization loop edits policy text. It is not model-weight PPO or
GRPO training, and a file-count edit cap is not the PPO clipped objective. Human
reviews form feedback data; a recorded review is not proof of critic training.
The supplied names “GRPA” and “policy-opt” did not uniquely resolve to a method.
Specific primary references are linked in the UI instead of claiming equivalence:
PPO (1707.06347), DeepSeekMath/GRPO (2402.03300), GEPA (2507.19457), Modular Prompt
Optimization (2601.04055) and RLMOpt (2608.10471).

The repeatedly consulted gate partition is development validation. Gate gains,
resampling, limited edits and consensus do not prove generalization or eliminate
overfitting. Judge extraction remains probabilistic even when downstream routing
is deterministic. Source tags in the shadow facts are caller assertions, not
cryptographic or authenticated provenance. This branch does not automatically
compile natural-language rules, run shadow routes in the production judges,
train weights, or demonstrate measured gains from deterministic execution.

**Historical metrics:** the prior implementation omitted zero-F1 classes and
could inflate macro-F1. Existing stored scores and gate decisions are left intact
as an audit record, not silently rewritten. Re-score saved predictions with the
new definition and run a fresh baseline before comparing or resuming optimization.
The executive studio deliberately does not promote those old scores as current
performance evidence. An abstention-excluding score must be interpreted with its
coverage; it is not accuracy over all incoming content.

## d-ai-trader review and transfer

The reviewed `docs/POLICY_GRAPH.md`, `policy_graph/paths.py` and graph-related
source separate context-based assembly from observed route/citation/outcome
statistics. RUSH borrows the separation between guideline identity, explicit
routing, and auditable traces—not trading rewards or execution behavior.
An ordered gate written in a prompt is still interpreted by the model. A citation
is not causal proof, and co-cited guidelines cannot each claim additive P&L
credit. No d-ai-trader files, trading settings or market actions were changed.

## Validation and limits

Run the dependency-free targeted tests from the repository root:

```sh
node --test tests/studio-core.test.cjs
python3 -m unittest discover -s tests -p test_studio_contracts.py -v
```

Optional in-memory browser smoke tests (Playwright and Chromium required):

```sh
python3 tests/studio-browser-smoke.py --chromium /path/to/chromium
```

The browser harness loads the actual HTML, CSS and JavaScript with explicitly
mocked offline/recorded API responses. It exercises source labeling, both demos,
keyboard inspection, unknown paths, source escaping, missing-snapshot behavior,
method navigation and responsive widths. It does not call models or a live server.
Desktop and mobile screenshots were also inspected during implementation.

The full repository suite, live provider integrations and the operator's private
LAN deployment were not executed from the editing environment. Public/private
live-site access failed; the review used repository source. A localhost smoke
check against your actual history is the next review step. Do not treat the
new shadow evaluator as production enforcement or the presentation as a claim
of statistically demonstrated decision-quality gains.
