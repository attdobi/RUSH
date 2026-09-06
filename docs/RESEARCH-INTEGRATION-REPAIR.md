# Native research integration repair

This revision supersedes the disconnected workbench described in RESEARCH-WORKBENCH.md.

## What went wrong

The previous revision added a parallel /api/studio history reader and instructed
the reviewer to point it at a guessed $HOME/RUSH filesystem root. The working
application already used /api/experiments, /api/policy/graph, native proposal
artifacts and /api/thumbnail. Those sources were not unified. An empty result
from the new discovery path was therefore not evidence that the working
application had no data. The supplied screenshots demonstrate the mismatch.
The previous browser screenshots were controlled fixtures, not the user's runs.

## Restoration and repair

web/about.js is restored byte-for-byte to original blob
013af218b8a8b053aaea5263fc03684ddd8c7a24. The architecture, equations, research
questions, label-noise discussion and control reference are not rewritten.
research-addendum.js appends explicit amendments after that original document.

The lab no longer injects research.js or uses its /api/studio discovery path.
The native graph, run selector, policy-version controls, cycle chips, learning
curves, image evidence drawer, optimizer controls and gate ledger remain the
application. lab-evidence.js adds a linked anchor -> proposed node edit ->
measured-response view and replay driven by the native cycle controls.
Accepted edits and unaccepted proposals have different node markers. All values
are saved measurements; no new learning curve or synthetic fallback is created.
The native graph is not cleared if the additive evidence panel fails.

## Review using the working data server

From the preview checkout, after updating the branch:

```bash
python3 scripts/rush_preview_server.py \
  --upstream https://rush.attiladobi.com --port 8767
```

Open http://127.0.0.1:8767/?demo=genai#loop or change demo to mnist.
The original About is at /#about. The same preview can use an existing local
RUSH origin with --upstream http://127.0.0.1:8766 when that is the actual port.

The preview serves the branch's web assets and proxies native API, policy-file
and media GET requests to that single operator-configured origin. It binds only
to loopback, rejects all write verbs, strips incoming credentials, rejects
cross-site browser reads, and does not follow upstream redirects. It does not
copy data, migrate the database, initialize a schema or launch model calls.
An unavailable upstream returns an explicit error, not an empty history.

A direct branch deployment using scripts/rush_web_server.py retains the native
write workflows and uses that checkout's data. The read-only preview is for
review, not running new optimization jobs.

## Where SQL and files fit

pipeline/labelstore/__init__.py resolves RUSH_DB_URL, defaulting to
postgresql:///adobi (local PostgreSQL; schema rush). This is a source-code
configuration, not confirmation of the private host's running connection.

- Native experiment pages: data/experiments/<id>/experiment.json.
- Native graph: policy-graph/<area>/<version>/*.md and edges.json.
- Cross-run SQL mirror: rush.experiment, experiment_cycle, experiment_metric,
  gate_decision, gate_review and generator_version.
- Live human-label history / resolved golden labels: rush.label_event and
  rush.golden_label. These are not reconstructed from summary metrics.

Run scripts/rush_source_audit.py --repo-root /actual/working/RUSH --database on
the data host using its configured Python environment to inspect the actual
connection, data directory where permitted, rush table names and estimated
row counts. It uses read-only transactions and never prints the DSN/password
or calls the schema initializer. Without --database it only inventories files
and identifies the configuration source. No private SQL connection was made
from the assistant's container.

## Research implications for the KDD study

The central object is co-development of reusable policy and reference evidence,
not just an unconstrained prompt search. Preserve two independently versioned
objects: G, the policy graph, and L, the golden-label snapshot.

For any score Q, a before/after change decomposes exactly as:

Q(G1,L1)-Q(G0,L0) = [Q(G1,L1)-Q(G0,L1)] + [Q(G0,L1)-Q(G0,L0)].

The first term compares policies on one fixed revised reference; the second is
the changed reference's effect at the incumbent. This is an accounting identity,
not a causal identification result. A protected audit set and blinded review are
still needed, and a different decomposition order changes the interpretation
when effects interact. Never treat a score shift caused only by relabeling as
policy-learning lift.

For MNIST, use untouched confusion-pair and unaffected-digit slices, with
source-level separation before transformations. The original p=c when correct,
otherwise 1-c expression is an error/confidence ranking heuristic for multiclass
labels, not the true-class probability without a full class distribution.

For GenAI, first pin the operational definition of generated/edited/hybrid/CGI
content. Provenance of generation and visual signs are different observations.
Test held-out generator families, source families and time windows. Ambiguous or
unidentifiable cases need a review/abstention protocol; improved artifactual-cue
recognition alone does not establish provenance accuracy.

Use a frozen-policy control, flat-text optimization, graph-local edits, random
versus ranked anchors, and later routed versus full-policy judging under matched
budgets. Byte-identical flat and graph-rendered prompts are a fidelity control;
the storage representation alone cannot demonstrate judge-quality improvement.
To evaluate a node's contribution, define a held-out intervention/ablation rather
than treating its citation count as a causal effect. Separate observed facts,
served rules, reported citations and executed predicates.

The existing golden-label confidence formula is an operational weighting rule
until independently calibrated; reviewer confirmation after seeing a panel
answer is not an independent vote. Seed-to-seed policy embedding spread is not,
by itself, a Lyapunov exponent. These qualifications are appended to the original
About rather than deleting its research program.

Related comparator: GEPA, https://arxiv.org/abs/2507.19457. Match evidence access,
model capability, evaluation cost and budget when comparing methods; no published
GEPA result is presented as a RUSH result.

## Verification

Locally executed: 13 JavaScript contract tests and 13 Python bridge/configuration
tests passed. These include native new_version compatibility, rejected-step
handling, missing coverage, origin validation, identical native routes/queries,
write blocking, no credential forwarding, redirect rejection and traversal guards.

The new tests/native_review_e2e.py runs the COMPLETE native application against
an actual RUSH server and committed repository artifacts, verifies both demos,
asserts the original About blob hash and checks that proxy graph JSON equals
the native response. It does not mock API responses. The accompanying GitHub
Actions workflow captures results and browser images. Its result must be checked;
its presence is not a claim of a passing run. It does not access the user's
private SQL server or execute new model-labeling experiments.
