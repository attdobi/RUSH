// About this demo — the metric formalism, project-agnostic. Explains every
// config knob on the Run tab, what one optimization cycle actually does (what
// the drafter sees, what the gate computes), the per-judge gradient (p, |g|),
// the two multi-LLM alignment signals (SME agreement vs LLM consensus), the
// four-tier importance that ranks the adjudication queue and the
// policy-learning anchors, and how human-label confidence fades
// re-adjudication. Kept in sync with pipeline/experiment panel_signal /
// importance_scores and scripts/run_experiment.py. Sections are wrapped in
// .about-section so they flow into balanced columns without a definition
// splitting mid-way; formulas use the .about-eq display-math grid.
(() => {
  const $ = (sel) => document.querySelector(sel);

  // eq(rows, footnote): one display-math block. Each row is [lhs, op, rhs, note].
  const eq = (rows, span) => `<div class="about-eq">${rows.map(([l, o, r, n]) =>
    `<span class="eq-l">${l}</span><span class="eq-op">${o}</span><span class="eq-r">${r}</span><span class="eq-note">${n || ''}</span>`).join('')}${span ? `<span class="about-eq-span">${span}</span>` : ''}</div>`;

  const HTML = `
  <div class="about-doc">
    <div class="section-head compact about-intro">
      <p class="eyebrow">About this demo</p>
      <h2>How RUSH scores, ranks, and learns — every knob and every number.</h2>
      <p>This applies to every project (MNIST digits, GenAI images, or a real policy). A panel of
      cheap LLM judges labels each item against a versioned policy graph; a human SME owns the golden
      label. The numbers on the other tabs all come from the definitions below.</p>
    </div>

    <div class="about-columns">
    <section class="about-section">
      <h2>The three roles</h2>
      <ul>
        <li><strong>Judges</strong> — the 2–5 cheap panel models. They label every item and produce
        <em>every</em> decision-quality metric. No expensive model ever scores quality.</li>
        <li><strong>Drafter</strong> — one model (cheap or frontier) that, each cycle, reads the most
        instructive anchors (the misaligned images themselves, plus a sample of correctly-classified
        ones) and writes a single policy edit of ≤5 node files. It drafts; it never judges.</li>
        <li><strong>Gate</strong> — a deterministic rule: accept the edit only if the panel's
        test-partition macro-F1 strictly improves. An optional <strong>gate agent</strong> is a
        <em>subtractive</em> soundness check on top of that rule: it can veto a metric-passing edit
        it judges unsound — one that <strong>overfits to named examples instead of stating a general
        rule</strong>, leaks the golden answer, games one judge's quirks, tells judges to abstain, or
        dumps pair-specific rules into the root instead of the owning node. It never forces or accepts;
        the metric stays the hard gate. That over-specificity veto is the crank's fourth overfitting
        guard, alongside the ≤5-change trust region, the no-per-image-answers drafter rule, and the
        aligned anchors.</li>
      </ul>
    </section>

    <section class="about-section">
      <h2>Every knob on the Run tab</h2>
      <p>The full hyperparameter surface of one run. Negatives/positives = the misaligned/aligned
      anchor images the drafter studies.</p>
      <table class="about-knob-table">
        <thead><tr><th>Knob</th><th>Default</th><th>What it does</th></tr></thead>
        <tbody>
          <tr><td>Judges</td><td>2–5 models</td><td>The panel. Labels everything, scores everything; each judge also returns confidence, difficulty, is_boundary and a policy-cited justification.</td></tr>
          <tr><td>Cycles <var>k</var></td><td>5</td><td>Optimization steps per run — each cycle proposes at most one policy edit. Accepted edits mint version <code>v&lt;run&gt;.&lt;k&gt;</code>.</td></tr>
          <tr><td>Train batch <var>N</var></td><td>20</td><td>Fresh seeded mini-batch labeled every cycle. Its misalignments are the raw gradient signal.</td></tr>
          <tr><td>Test size <var>T</var></td><td>100</td><td>The run's fixed test partition, drawn once at k=0 and reused all run — the gate's yardstick.</td></tr>
          <tr><td>Seed</td><td>13</td><td>Fixes the partition and every batch draw. Same seed ⇒ same data path (reproducibility, and the handle for the chaos/Lyapunov ablation).</td></tr>
          <tr><td>Drafter</td><td>gpt-5.5</td><td>The model that writes the edit. It sees anchors + the current policy; it never scores. A cheaper drafter is a legitimate config. Its per-cycle spend is recorded on each cycle and shown in the gate ledger.</td></tr>
          <tr><td>Input</td><td>text only</td><td>What the drafter gets per anchor. <code>text only</code> (default): every judge's full text output — label, confidence, difficulty, boundary flag, justification — plus the SME truth. The justifications describe what the judges saw, so this is usually enough, and it keeps optimizer cost flat as anchor counts grow. <code>images + text</code>: additionally attaches the anchor image bytes so the drafter can inspect the pixels itself — stronger evidence on visual boundary cases (a 9 with a broken loop, a plastic-skin artifact) at extra input-token cost per cycle.</td></tr>
          <tr><td>Anchors (method)</td><td>random (S1)</td><td>How misalignments are picked for the drafter: <code>random</code> = the null hypothesis every gradient must beat; <code>top |g|</code> = confident-wrong first; <code>top importance</code> = the four-tier rank below.</td></tr>
          <tr><td>Misaligned</td><td>15</td><td>The <strong>negatives</strong>: how many misaligned images (pixels included) go to the drafter each cycle.</td></tr>
          <tr><td>Aligned</td><td>5</td><td>The <strong>positives</strong>: correctly-labeled images sent alongside, so the drafter sees what already works and does not over-correct (0 = off).</td></tr>
          <tr><td>Max changes</td><td>3</td><td>The edit clip: at most 3 node files touched per proposal (hard cap 5) — the trust region that keeps every step human-reviewable.</td></tr>
          <tr><td>Gate mode</td><td>metric rule</td><td>Four modes: <strong>metric rule</strong> (the default — accept only on strict panel macro-F1 improvement); + agent veto (rule stays the hard wall, agent can only reject); critic agent only (the agent's verdict decides, metric recorded as advisory, never enforced; agent failure falls back to the rule); OFF (accept every edit — the unfiltered-drift demo).</td></tr>
          <tr><td>Gate persona</td><td>lenient</td><td>The critic's stance, appended to its system prompt. <code>lenient</code>: a flat metric on a small test partition is sampling noise — skip only clear defects or large multi-judge regressions. <code>moderate</code>: weigh measured movement and structural value together. <code>strict</code>: any regression or unmeasured claimed value skips.</td></tr>
          <tr><td><var>ε</var> (epsilon)</td><td>0</td><td>Extra margin the candidate must clear. ε&gt;0 is the first winner's-curse mitigation on the research list.</td></tr>
          <tr><td>Benchmark readout</td><td>on</td><td>Scores the fixed 1,000-image cross-run validation split under the start and final policy — the honest cross-run comparison. Costs two extra panel passes.</td></tr>
          <tr><td>Parallelism</td><td>4</td><td>Concurrent labeling calls per judge; hosted judges of one provider run side by side in a shared, per-model-sized pool.</td></tr>
        </tbody>
      </table>
    </section>

    <section class="about-section">
      <h2>One cycle, end to end</h2>
      <p>What actually happens each cycle <var>k</var> — including exactly what information the
      optimizer is given:</p>
      <ol class="about-flow">
        <li><strong>Label.</strong> The panel labels a fresh <var>N</var>-image train batch under the
        current policy <var>G<sub>k</sub></var>. Every judge returns label <var>ŷ</var>, confidence
        <var>c</var>, difficulty, is_boundary (+ the confusion pair), and a justification citing
        policy nodes.</li>
        <li><strong>Select anchors.</strong> Misaligned images are ranked by the chosen method
        (random / top&nbsp;<var>|g|</var> / top&nbsp;importance); the top ≤15 misaligned + ≤5 aligned
        become the anchor set (both counts are knobs).</li>
        <li><strong>Draft.</strong> The drafter receives: the current policy graph (the
        <em>generator</em> — the exact prompt the judges run), and per anchor the SME golden label
        plus each judge's full text output (label, confidence, difficulty, boundary flag,
        justification). With the Input knob on <code>images + text</code> the anchor image pixels
        are also attached (<code>text only</code> is the default — the justifications usually carry
        the visual evidence in words). It returns <em>one</em> edit touching ≤3 nodes (the clip
        knob). Its token usage and cost are recorded per cycle.</li>
        <li><strong>Score.</strong> The panel relabels the fixed test partition under the candidate
        policy <var>G<sub>k</sub></var> ⊕ <var>e</var>.</li>
        <li><strong>Gate.</strong> A deterministic comparison of two panel scores — the expensive
        model never computes decision quality:</li>
      </ol>
      ${eq([
        ['accept(<var>e</var>)', '⇔', 'F1<sub>test</sub>(<var>G</var> ⊕ <var>e</var>) &gt; F1<sub>test</sub>(<var>G</var>) + <var>ε</var>', 'and no gate-agent veto'],
      ], 'F1 = the judge panel’s macro-F1 on the run’s fixed test partition. Accepted ⇒ the policy becomes v&lt;run&gt;.&lt;k&gt;; skipped ⇒ the incumbent stays.')}
    </section>

    <section class="about-section">
      <h2>Per-judge gradient: p and |g|</h2>
      <p>Each judge <em>j</em> returns a label <var>ŷ</var> and a self-reported confidence
      <var>c</var> ∈ [0,1] in <em>its own</em> label. Against the SME truth <var>y</var> we map that
      to a probability on the true class (a binary approximation), and to a gradient magnitude — how
      much this judgment could teach:</p>
      ${eq([
        ['<var>p</var>', '=', '<var>c</var>', 'judge correct (<var>ŷ</var> = <var>y</var>)'],
        ['<var>p</var>', '=', '1 − <var>c</var>', 'judge wrong (<var>ŷ</var> ≠ <var>y</var>)'],
        ['|<var>g</var>|', '=', '1 − <var>p</var>', 'gradient magnitude — how informative'],
        ['<var>h</var>', '=', '<var>c</var>·(1 − <var>c</var>)', 'curvature — peaks at <var>c</var> = 0.5, blind to correctness'],
        ['loss', '=', '−ln <var>p</var>', 'per-sample cross-entropy'],
      ])}
      <p>So the four corners the panel can be in:</p>
      <ul>
        <li><strong>Confident &amp; right</strong> (c high, ŷ=y): p≈1, <code>|g|≈0</code> — nothing to learn.</li>
        <li><strong>Confident &amp; wrong</strong> (c high, ŷ≠y): p≈0, <code>|g|≈1</code> — the most
        informative error; either the policy is failing or the golden label is wrong.</li>
        <li><strong>Unsure</strong> (c≈0.5, either way): p≈0.5, <code>|g|≈0.5</code> — a moderate signal,
        the item is genuinely ambiguous.</li>
      </ul>
      <p>Uncertainty lives in <var>c</var> and the difficulty rating — the judges are instructed to
      always return a label, never abstain.</p>
    </section>

    <section class="about-section">
      <h2>Two alignment signals — do not conflate them</h2>
      <p>A multi-LLM panel gives two <em>different</em> agreement numbers, and RUSH keeps them separate:</p>
      ${eq([
        ['<var>a</var>', '=', '(# judges with <var>ŷ</var> = <var>y</var>) / <var>N</var>', '<strong>SME agreement</strong> — LLM↔human, graded (3/4, 2/4, …)'],
        ['<var>m</var>', '=', '1 − <var>a</var>', 'misalignment'],
        ['<var>κ</var>', '=', '(# judges on the modal label) / <var>N</var>', '<strong>LLM consensus</strong> — LLM↔LLM, computed SME-blind'],
        ['<var>b</var>', '=', '(# judges flagging boundary) / <var>N</var>', 'boundary rate'],
      ])}
      <p>The two come apart exactly where it matters: the panel can be <em>unanimous</em> (κ = 1) and
      <em>entirely wrong</em> (a = 0). For accounting we collapse the panel to its <strong>majority
      vote</strong> and compare that to the SME — but to stack-rank items we use the full graded
      signals.</p>
    </section>

    <section class="about-section">
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
      ${eq([
        ['<var>I</var><sub>base</sub>', '=', '( <var>m</var> + <var>κ</var>·(2<var>m</var> − 1) + 1 ) / 3', '∈ [0, 1]'],
      ], 'misaligned (m→1): rises with κ — consensus makes it worse · aligned (m→0): falls with κ — consensus makes it better')}
    </section>

    <section class="about-section">
      <h2>Two derived scores: anchor value and re-adjudication priority</h2>
      <p>The base score is amplified by the panel's confidence (mean <var>|g|</var>) and its boundary
      rate, because a confident, boundary-flagged error teaches more:</p>
      ${eq([
        ['amp', '=', '(1 + mean|<var>g</var>|) · (1 + ½·<var>b</var>)', ''],
        ['anchor value', '=', '<var>I</var><sub>base</sub> · amp', 'ranks policy-learning anchors'],
        ['re-adjudication', '=', '<var>I</var><sub>base</sub> · amp · (1 − <var>p</var><sub>human</sub>)', 'ranks the human queue'],
      ])}
      <p><strong>Anchor value</strong> drives the <code>top_importance</code> selection strategy —
      which misalignments the drafter studies. <strong>Re-adjudication priority</strong> is the same
      score, but faded by how confident we already are in the golden label:</p>
      <h3>Human-label confidence</h3>
      <p>The golden label carries its own confidence, growing with the number of SME confirmations
      <var>m</var><sub>SME</sub>:</p>
      ${eq([
        ['<var>p</var><sub>human</sub>', '=', '1 − 1 / (<var>m</var><sub>SME</sub> + 0.2)', 'm=1 → 0.17 (default) · m=2 → 0.55 · m=3 → 0.69'],
      ])}
      <p>Re-adjudication priority is multiplied by <code>(1 − p_human)</code>, so once two or three SMEs
      have re-confirmed a label, its weight for going back to a human vanishes — the human queue keeps
      flowing toward genuinely unresolved cases. (Policy anchor value is <em>not</em> faded: the policy
      still has to learn even a certain label.)</p>
    </section>

    <section class="about-section">
      <h2>The Adjudicate columns</h2>
      <p>Each row is one image; every column is click-sortable, and hovering a header shows the same
      short definition. <strong>Importance is the default sort — it is the re-adjudication priority
      score above (<code>anchor value × (1 − p_human)</code>), recomputed after any SME action.</strong>
      A high Importance means "a human should look here first."</p>
      <table class="about-tier-table">
        <thead><tr><th>Column</th><th>What it means</th></tr></thead>
        <tbody>
          <tr><td><strong>Tier</strong></td><td>The four-tier bucket (T1 worst → T4 ideal). After an overturn, re-scored against the new label.</td></tr>
          <tr><td><strong>SME truth</strong></td><td>The human (golden) label.</td></tr>
          <tr><td><strong>SME agree</strong></td><td>LLM↔human: fraction of judges matching the SME label (<code>a = m/N</code>). Low = misaligned.</td></tr>
          <tr><td><strong>LLM consensus</strong></td><td>LLM↔LLM, SME-blind: fraction on the modal label (<code>κ</code>). High + misaligned = systematic.</td></tr>
          <tr><td><strong>Avg conf</strong></td><td>Mean self-reported judge confidence <code>c</code>.</td></tr>
          <tr><td><strong>Difficulty</strong></td><td>low=0, medium=0.5, high=1, averaged across judges.</td></tr>
          <tr><td><strong>Boundary</strong></td><td>Fraction of judges flagging a documented confusion boundary (<code>b</code>).</td></tr>
          <tr><td><strong>|g|</strong></td><td>Gradient magnitude <code>1 − p</code>; confident-wrong ≈ 1, confident-right ≈ 0.</td></tr>
          <tr><td><strong>Importance</strong></td><td><strong>The default rank.</strong> Re-adjudication priority = <code>I_base(misalignment×consensus) × (1+|g|) × (1+½·boundary) × (1 − p_human)</code>. Fades as SMEs confirm.</td></tr>
          <tr><td><strong>Status</strong></td><td>SME verdict: open · confirmed ×N · overturned X→Y · uncertain. Resolved = two or more SMEs agree.</td></tr>
        </tbody>
      </table>
    </section>

    <section class="about-section">
      <h2>Where else these show up</h2>
      <ul>
        <li><strong>Run summary</strong> — every judge's full response per image: label, <code>c</code>,
        difficulty, is_boundary + the confusion pair, citations, quotes, tokens, cost.</li>
        <li><strong>Run the loop</strong> — the anchor selection strategy (<code>random</code>,
        <code>top_gradient</code>, or <code>top_importance</code>) decides which of these the drafter sees.
        <code>top_importance</code> ranks by the same score as the Adjudicate Importance column.</li>
      </ul>
      <p class="hint">All formulas above are implemented verbatim in <code>pipeline/experiment</code>
      (<code>panel_signal</code>, <code>importance_scores</code>, <code>human_confidence</code>) and mirror
      the <code>rush.sample_gradient</code> / <code>rush.panel_signal</code> SQL views.</p>
    </section>

    <section class="about-section">
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
    </section>
    </div>
  </div>`;

  function init() {
    const host = $('#aboutContent');
    if (host) host.innerHTML = HTML;
  }

  if (typeof window.rushApiOnReady === 'function') window.rushApiOnReady(() => init());
  else document.addEventListener('DOMContentLoaded', init);
})();
