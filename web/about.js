// About this demo — the metric formalism, project-agnostic. Explains the
// per-judge gradient (p, |g|), the two multi-LLM alignment signals (SME
// agreement vs LLM consensus), the four-tier importance that ranks the
// adjudication queue and the policy-learning anchors, and how the human-label
// confidence fades re-adjudication. Kept in sync with pipeline/experiment
// panel_signal / importance_scores.
(() => {
  const $ = (sel) => document.querySelector(sel);

  const HTML = `
  <div class="about-doc">
    <div class="section-head compact">
      <p class="eyebrow">About this demo</p>
      <h2>How RUSH scores, ranks, and learns — the metrics behind every number.</h2>
      <p>This applies to every project (MNIST digits, GenAI images, or a real policy). A panel of
      cheap LLM judges labels each item against a versioned policy graph; a human SME owns the golden
      label. The numbers on the other tabs all come from the definitions below.</p>
    </div>

    <h2>The three roles</h2>
    <ul>
      <li><strong>Judges</strong> — the 2–5 cheap panel models. They label every item and produce
      <em>every</em> decision-quality metric. No expensive model ever scores quality.</li>
      <li><strong>Drafter</strong> — one model (cheap or frontier) that, each cycle, reads the most
      instructive anchors (the misaligned images themselves, plus a sample of correctly-classified
      ones) and writes a single policy edit of ≤5 node files. It drafts; it never judges.</li>
      <li><strong>Gate</strong> — a deterministic rule: accept the edit only if the panel's
      test-partition macro-F1 strictly improves. An optional agent may <em>veto</em> a suspicious
      win, never force one.</li>
    </ul>

    <h2>Per-judge gradient: p and |g|</h2>
    <p>Each judge <em>j</em> returns a label <code>ŷ</code> and a self-reported confidence
    <code>c ∈ [0,1]</code> in <em>its own</em> label. Against the SME truth <code>y</code> we map
    that to a probability on the true class (a binary approximation), and to a gradient magnitude —
    how much this judgment could teach:</p>
    <span class="about-formula">p   = c        if ŷ = y   (correct)
p   = 1 − c    if ŷ ≠ y   (wrong)

|g| = 1 − p                 gradient magnitude — how informative the error is
h   = c·(1 − c)             curvature / uncertainty (max at c = 0.5, blind to correctness)
loss= −ln p                 per-sample cross-entropy</span>
    <p>So the four corners the panel can be in:</p>
    <ul>
      <li><strong>Confident &amp; right</strong> (c high, ŷ=y): p≈1, <code>|g|≈0</code> — nothing to learn.</li>
      <li><strong>Confident &amp; wrong</strong> (c high, ŷ≠y): p≈0, <code>|g|≈1</code> — the most
      informative error; either the policy is failing or the golden label is wrong.</li>
      <li><strong>Unsure</strong> (c≈0.5, either way): p≈0.5, <code>|g|≈0.5</code> — a moderate signal,
      the item is genuinely ambiguous.</li>
    </ul>
    <p>Uncertainty lives in <code>c</code> and the difficulty rating — the judges are instructed to
    always return a label, never abstain.</p>

    <h2>Two alignment signals — do not conflate them</h2>
    <p>A multi-LLM panel gives two <em>different</em> agreement numbers, and RUSH keeps them separate:</p>
    <ul>
      <li><strong>SME agreement</strong> <code>a = (# judges whose ŷ = y) / N</code> — LLM↔human.
      This is graded (e.g. 3/4, 2/4, 1/4). Misalignment is <code>m = 1 − a</code>.</li>
      <li><strong>LLM consensus</strong> <code>κ = (# judges on the modal label) / N</code> — LLM↔LLM,
      computed <em>without</em> looking at the SME label. High κ means the judges agree <em>with each
      other</em>, whether or not they're right.</li>
    </ul>
    <p>The two come apart exactly where it matters: the panel can be <em>unanimous</em> (κ = 1) and
    <em>entirely wrong</em> (a = 0). For accounting we collapse the panel to its <strong>majority
    vote</strong> and compare that to the SME — but to stack-rank items we use the full graded
    signals, plus the boundary rate <code>b = (# judges flagging is_boundary) / N</code>.</p>

    <h2>The four-tier hierarchy</h2>
    <p>Consensus <em>flips its meaning</em> with alignment. When the panel is misaligned, agreeing
    with each other makes it worse (a systematic, confident error). When it's aligned, agreeing makes
    it better (the ideal state). That gives four tiers, ranked by how much a human should care:</p>
    <table class="about-tier-table">
      <thead><tr><th>Tier</th><th>Alignment</th><th>LLM consensus</th><th>Meaning</th></tr></thead>
      <tbody>
        <tr><td><span class="adjudicate-tier adjudicate-tier--1">T1</span></td><td>misaligned</td><td>high</td><td><strong>The worst.</strong> Unanimous &amp; wrong — most valuable for re-adjudication and (if the label holds) for policy learning.</td></tr>
        <tr><td><span class="adjudicate-tier adjudicate-tier--2">T2</span></td><td>misaligned</td><td>low</td><td>Split &amp; wrong — the panel argued and still missed.</td></tr>
        <tr><td><span class="adjudicate-tier adjudicate-tier--3">T3</span></td><td>aligned</td><td>low</td><td>Right, but the panel argued — still instructive for the boundary.</td></tr>
        <tr><td><span class="adjudicate-tier adjudicate-tier--4">T4</span></td><td>aligned</td><td>high</td><td>Unanimous &amp; right — the ideal state, lowest priority.</td></tr>
      </tbody>
    </table>
    <p>A single continuous score reproduces that ordering and interpolates the graded signals:</p>
    <span class="about-formula">I_base = ( m + κ·(2m − 1) + 1 ) / 3          ∈ [0, 1]

   misaligned (m→1):  I_base rises with κ   (consensus makes it worse)
   aligned    (m→0):  I_base falls with κ   (consensus makes it better)</span>

    <h2>Two derived scores: anchor value and re-adjudication priority</h2>
    <p>The base score is amplified by the panel's confidence (mean <code>|g|</code>) and its boundary
    rate, because a confident, boundary-flagged error teaches more:</p>
    <span class="about-formula">amp            = (1 + mean|g|) · (1 + 0.5·b)

anchor value   = I_base · amp                      → ranks policy-learning anchors
re-adjudication= I_base · amp · (1 − p_human)       → ranks the human queue</span>
    <p><strong>Anchor value</strong> drives the <code>top_importance</code> selection strategy —
    which misalignments the drafter studies. <strong>Re-adjudication priority</strong> is the same
    score, but faded by how confident we already are in the golden label:</p>

    <h3>Human-label confidence</h3>
    <p>The golden label carries its own confidence, growing with the number of SME confirmations
    <code>m<sub>SME</sub></code>:</p>
    <span class="about-formula">p_human = 1 − 1 / (m_SME + 0.2)
          m=1 → 0.167 (default)   m=2 → 0.545   m=3 → 0.688</span>
    <p>Re-adjudication priority is multiplied by <code>(1 − p_human)</code>, so once two or three SMEs
    have re-confirmed a label, its weight for going back to a human vanishes — the human queue keeps
    flowing toward genuinely unresolved cases. (Policy anchor value is <em>not</em> faded: the policy
    still has to learn even a certain label.)</p>

    <h2>Where these show up</h2>
    <ul>
      <li><strong>Run summary</strong> — every judge's full response per image: label, <code>c</code>,
      difficulty, is_boundary + the confusion pair, citations, quotes, tokens, cost.</li>
      <li><strong>Adjudicate</strong> — the cross-run queue, one row per image, every column
      click-sortable: Tier, SME agree (a), LLM consensus (κ), Avg conf, Difficulty, Boundary rate (b),
      <code>|g|</code>, and the composite Importance (the default sort).</li>
      <li><strong>Run the loop</strong> — the anchor selection strategy (<code>random</code>,
      <code>top_gradient</code>, or <code>top_importance</code>) decides which of these the drafter sees.</li>
    </ul>
    <p class="hint">All formulas above are implemented verbatim in <code>pipeline/experiment</code>
    (<code>panel_signal</code>, <code>importance_scores</code>, <code>human_confidence</code>) and mirror
    the <code>rush.sample_gradient</code> / <code>rush.panel_signal</code> SQL views.</p>

    <h2>Open research questions — is textual policy-gradient descent sound?</h2>
    <p>RUSH treats the policy prompt as the parameter and runs textual gradient descent on it. The crank
    is the harness for testing whether that optimizer is sound. The methodological spine: <strong>every
    gradient / stack-ranked strategy is compared against random selection</strong> on the same seed —
    random is the null hypothesis the gradient has to beat. Full writeup in
    <code>docs/RESEARCH.md</code>.</p>
    <ul>
      <li><strong>Overfitting / generalization</strong> — does an edit learn a general rule ("fire is
      hot") or a hyper-specific one ("the blue [stove] ring is hot", which fails on the red ring, or
      memorizes the training image)? Measured by the train → test → holdout → benchmark generalization
      gap; the fixed cross-run benchmark exists for exactly this. Regularizers: the ≤5-change clip, the
      gate's trust region, the "no per-image answers" drafter constraint, and the aligned anchors.</li>
      <li><strong>Convergence, two senses</strong> — (a) does DQ plateau over accepted steps on the
      <em>honest</em> holdout/benchmark curve (not the winner's-curse-biased gate set), and does the SME
      queue shrink to a trickle? (b) the chaos sense: run the same config under different seeds — do the
      final policy documents converge, or does a positive Lyapunov exponent send them to wildly different
      policies (measured as spread in policy-embedding space)?</li>
      <li><strong>Random vs stack-ranked selection</strong> — the central ablation: does ranking anchors
      by the four-tier importance converge faster / higher / with fewer human touches than random
      sampling? If not, the gradient formalism isn't earning its complexity — itself a result.</li>
      <li><strong>Prompt-tuning architectures</strong> — the drafter is one optimizer; the crank A/Bs it
      against reflective / GEPA-style / node-statistic alternatives on the same seeds and splits.</li>
    </ul>
    <p class="hint">Known bias to fix before publishing: the gate's winner's curse (one noisy eval,
    ε=0, inherited baseline). Mitigations to A/B — ε&gt;0, paired incumbent re-eval, N-consecutive-wins —
    and always report lift from the holdout/benchmark, never the gate set alone.</p>
  </div>`;

  function init() {
    const host = $('#aboutContent');
    if (host) host.innerHTML = HTML;
  }

  if (typeof window.rushApiOnReady === 'function') window.rushApiOnReady(() => init());
  else document.addEventListener('DOMContentLoaded', init);
})();
