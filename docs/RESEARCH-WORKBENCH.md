# RUSH research workbench

## Scope of this revision

The research lab is the default application again. The original `web/index.html`
DOM is restored from the pre-studio lab, preserving the existing experiment,
judge-panel, labeling, adjudication, benchmark, gate-review, learning-curve,
confusion, and node-change controls. The RUSH server progressively adds a new
research workbench without replacing those controls or their data contracts.

The previous presentation is retained at `/studio.html` as a secondary shadow
path sandbox. Its illustrative scenarios are not silently substituted for
recorded research evidence. `main` and the running local service are not changed
by checking out this branch in a separate worktree.

## Reviewing locally

For the worktree created during the first review, stop only the preview server
on port 8767, then run:

```bash
cd ~/RUSH-vp-demo
git status --short
git fetch origin
git switch --detach origin/feat/policy-intelligence-executive-demo
RUSH_STUDIO_DATA_ROOT="$HOME/RUSH" \
  python3 scripts/rush_web_server.py --port 8767
```

Do not discard local edits if `git status` reports any. Use the Python environment
that already runs RUSH. Open `http://127.0.0.1:8767/#loop` and
`http://127.0.0.1:8767/#about`.

`RUSH_STUDIO_DATA_ROOT` is an operator-selected, **read-only evidence root** for
the new workbench. Native lab actions still target the actual worktree and its
configured environment; this setting does not redirect writes, provider calls,
labeling, or adjudication into the other checkout. When the evidence root is
external, the workbench explicitly discloses that boundary and does not synchronize
its run selection into the native action controls. No live run is launched on load.

For end-to-end experiments, use a checkout with the desired local datasets,
provider configuration, and dependencies. This revision does not copy private
data or credentials between worktrees. Use the RUSH server, not a generic static
server, to receive the progressive lab assets and evidence APIs.

## Policy dynamics

The first panel is a read-only investigation surface, not a marketing walkthrough.
It follows recorded `generator_before` / `generator_after` transitions for a
selected experiment. A rejected or skipped candidate retains the incumbent
policy graph while its candidate measurements are clearly identified separately.
A broken parent chain, invalid cycle order, or missing accepted snapshot stops
replay at the last verified transition. A version catalog is labeled as a catalog,
not presented as an ancestral chain.

Network and hierarchy views share explicit artifact edges. Zoom, pan, fit,
full-screen, search, keyboard selection, type/degree sizing, and cross-link/label
switches support inspection. Added, edited, and retired nodes have distinct visual
markers. The inspector shows the actual rule text, its previous wording, explicit
relations, and ancestry. Neither visual position nor node size claims decision
quality or causal importance. Missing edges are disclosed, not fabricated to
make the graph appear connected.

The cycle rail supports scrubbing and replay. Optional follow mode refreshes saved
updates; it does not execute experiments. Latest-request-wins guards prevent an
older snapshot response from replacing a newly selected one. Loading failures
clear stale graph, metrics, version, and export state. There is no synthetic
fallback in the research workbench.

## Measurements and evidence

The panel reports macro FPR, macro FNR, decision coverage, and sample count where
those quantities are present. Its small trajectory display plots **evaluated
candidate** rates, with non-promoted candidates distinguished; it is not a claimed
incumbent learning curve. The native learning-curve and judge-level views remain
available immediately below, alongside the full gate ledger and confusion view.

Valid stored confusion counts support on-read diagnostic recomputation. Supported
zero-F1 classes remain zero rather than disappearing from a macro average.
Otherwise the workbench labels diagnostics as stored measurements. Historical gate
verdicts are not recomputed or retroactively changed. Old F1 artifacts still need
re-scoring before comparison with the corrected definition. Missing abstention
counts produce unknown coverage, not an invented 100 percent.

The sample-ID audit checks only recorded train/gate identifier overlap in a fixed
partition when sufficient identifiers exist. Unknown or resampled partitions do
not receive a fabricated clean audit. The check does not detect duplicate content,
across-run reuse, contaminated labels, or adaptive use of a benchmark. A zero
overlap count is not a certificate of generalization.

Evidence export includes the selected frame, current and comparison snapshots,
and the bounded public experiment record. Strings are escaped in the UI. The
adapter validates policy area, run ID, path containment, symlink containment,
file size, cycle ordering, and explicit run status; dry and unverified runs are
not exposed as measured evidence. It uses GET-only `/api/studio/*` endpoints.

## Methods notebook

The default About tab is a research notebook covering the graph-edit mechanism,
count definitions, selective risk and coverage, reference-label provenance,
consensus versus truth, golden-set coverage, gate modes, adaptive validation,
new-theme evaluation, and a falsifiable research agenda.

An interactive two-sided 95 percent Wilson calculator demonstrates uncertainty
for a single binomial rate. It is explicitly not an experiment result and does
not claim to solve clustering, distribution shift, or repeated selection. A draft
ablation-plan export captures paired seeds, a primary endpoint, current form
selections, and missing identities to pin before execution. It never launches
runs and includes `execution_authorized: false`.

Related-work references distinguish textual prompt optimization from PPO/GRPO
weight training. Deterministic retrieval, model-extracted observations, and typed
decision execution are separate mechanisms. The existing path examples remain
shadow demonstrations; no automatic Markdown compiler, production predicate
executor, critic weight training, or measured routing-quality improvement is
introduced in this revision.

## Validation performed for this revision

```bash
node --test tests/research-core.test.cjs
python tests/test_research_contracts.py
python tests/research-browser-smoke.py
```

- 30 JavaScript unit tests passed.
- 28 Python contract tests passed.
- 22 Chromium component/API-fixture checks passed.

The browser checks cover explicit accepted/rejected lineage, body changes,
keyboard inspection, text escaping, layout switching, search, evidence export,
native-selector synchronization, the methods calculator, ablation validation,
read-only requests, stale-response races, missing measurements, failed refresh,
offline behavior, policy-pin mismatch, retired-rule labeling, external-root isolation,
MNIST, reduced motion, and mobile
horizontal overflow. Their page is a controlled legacy-DOM/API fixture, not a
claim that live models ran or that the full legacy application was exercised.

The full repository suite, live provider calls, private LAN deployment, Safari,
and end-to-end native lab jobs were **not run** for this revision. The older studio
browser test now points to the relocated studio files; that older test was not
rerun in this revision's partial test environment. Fixture screenshots are labeled
as such and are not evidence of model performance.
