// About this demo — the metric formalism, project-agnostic. Opens with the
// MAS architecture diagram (four independent agents: judge panel = MAS
// labelers, drafter = policy-iteration agent, gate + acceptance critic, SME =
// human principal) and closes with the PPO/GEPA/VISTA comparison (distilled
// from docs/TECHNICAL-REPORT.md §7). In between: every config knob on the Run
// tab, what one optimization cycle actually does (what
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

    <section class="about-section about-arch">
      <h2>The architecture — one cycle around the loop</h2>
      <div class="arch-scroll">
      <svg class="arch-svg" viewBox="0 0 1200 545" role="img"
           aria-label="RUSH multi-agent architecture: policy graph, judge panel, anchor selection, drafter, candidate eval, gate, SME">
        <defs>
          <marker id="archArrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/>
          </marker>
          <marker id="archArrowGreen" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="var(--green)"/>
          </marker>
        </defs>

        <!-- Policy graph: the parameter -->
        <rect x="430" y="16" width="340" height="80" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--blue)" stroke-width="1.6"/>
        <text x="600" y="46" text-anchor="middle" fill="var(--text)" font-size="15" font-weight="800">POLICY GRAPH G<tspan baseline-shift="sub" font-size="11">k</tspan> — the parameter</text>
        <text x="600" y="68" text-anchor="middle" fill="var(--muted)" font-size="11">versioned markdown KG · the exact prompt the judges run</text>
        <text x="600" y="84" text-anchor="middle" fill="var(--muted)" font-size="11">accepted edits mint v&lt;run&gt;.&lt;k&gt;</text>

        <!-- Data splits -->
        <rect x="28" y="150" width="200" height="150" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--line)" stroke-width="1.4"/>
        <text x="128" y="176" text-anchor="middle" fill="var(--text)" font-size="12.5" font-weight="800">DATA — seeded splits</text>
        <text x="44" y="200" fill="var(--muted)" font-size="11">train batch N · fresh per cycle</text>
        <text x="44" y="220" fill="var(--muted)" font-size="11">test T · fixed at k=0 (gate)</text>
        <text x="44" y="240" fill="var(--muted)" font-size="11">holdout · end-of-run only</text>
        <text x="44" y="260" fill="var(--muted)" font-size="11">benchmark · fixed cross-run</text>
        <text x="44" y="285" fill="var(--muted)" font-size="10" font-style="italic">same seed ⇒ same data path</text>

        <!-- Judge panel -->
        <rect x="272" y="150" width="240" height="150" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--blue)" stroke-width="1.6"/>
        <text x="392" y="176" text-anchor="middle" fill="var(--text)" font-size="13" font-weight="800">JUDGE PANEL</text>
        <text x="392" y="194" text-anchor="middle" fill="var(--muted)" font-size="11">MAS labelers — 2–5 independent mLLMs</text>
        <g font-size="10.5" text-anchor="middle">
          <rect x="290" y="208" width="46" height="22" rx="7" fill="rgba(130,181,255,.12)" stroke="var(--blue)" stroke-width="1"/>
          <text x="313" y="223" fill="var(--text)">J1</text>
          <rect x="342" y="208" width="46" height="22" rx="7" fill="rgba(130,181,255,.12)" stroke="var(--blue)" stroke-width="1"/>
          <text x="365" y="223" fill="var(--text)">J2</text>
          <rect x="394" y="208" width="46" height="22" rx="7" fill="rgba(130,181,255,.12)" stroke="var(--blue)" stroke-width="1"/>
          <text x="417" y="223" fill="var(--text)">J3</text>
          <rect x="446" y="208" width="46" height="22" rx="7" fill="rgba(130,181,255,.12)" stroke="var(--blue)" stroke-width="1"/>
          <text x="469" y="223" fill="var(--text)">J4</text>
        </g>
        <text x="392" y="252" text-anchor="middle" fill="var(--muted)" font-size="11">ŷ, confidence, difficulty,</text>
        <text x="392" y="268" text-anchor="middle" fill="var(--muted)" font-size="11">boundary flag, justification</text>
        <text x="392" y="289" text-anchor="middle" fill="var(--muted)" font-size="10" font-style="italic">label only · never draft · never gate</text>

        <!-- Anchor selection / stack ranking -->
        <rect x="556" y="150" width="200" height="150" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--line)" stroke-width="1.4"/>
        <text x="656" y="176" text-anchor="middle" fill="var(--text)" font-size="12.5" font-weight="800">STACK RANK &amp; SELECT</text>
        <text x="656" y="196" text-anchor="middle" fill="var(--muted)" font-size="11">anchors: ≤15 misaligned + ≤5 aligned</text>
        <text x="656" y="216" text-anchor="middle" fill="var(--muted)" font-size="11">random / top |g| / importance</text>
        <text x="656" y="236" text-anchor="middle" fill="var(--muted)" font-size="11">judge votes + SME truth</text>
        <text x="656" y="256" text-anchor="middle" fill="var(--muted)" font-size="11">(+ images, if Input allows)</text>
        <text x="656" y="285" text-anchor="middle" fill="var(--muted)" font-size="10" font-style="italic">random = the null hypothesis</text>

        <!-- Drafter -->
        <rect x="800" y="150" width="240" height="150" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--purple)" stroke-width="1.6"/>
        <text x="920" y="176" text-anchor="middle" fill="var(--text)" font-size="13" font-weight="800">DRAFTER</text>
        <text x="920" y="194" text-anchor="middle" fill="var(--muted)" font-size="11">the policy-iteration agent (optimizer)</text>
        <text x="920" y="222" text-anchor="middle" fill="var(--muted)" font-size="11">one edit per cycle · ≤5 node files</text>
        <text x="920" y="242" text-anchor="middle" fill="var(--muted)" font-size="11">no per-image answers · no reword</text>
        <text x="920" y="262" text-anchor="middle" fill="var(--muted)" font-size="11">grows KG sub-nodes, not the root</text>
        <text x="920" y="289" text-anchor="middle" fill="var(--muted)" font-size="10" font-style="italic">drafts only · never scores</text>

        <!-- Candidate eval -->
        <rect x="800" y="380" width="240" height="110" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--line)" stroke-width="1.4"/>
        <text x="920" y="408" text-anchor="middle" fill="var(--text)" font-size="12.5" font-weight="800">CANDIDATE EVAL</text>
        <text x="920" y="432" text-anchor="middle" fill="var(--muted)" font-size="11">the same judge panel re-labels</text>
        <text x="920" y="450" text-anchor="middle" fill="var(--muted)" font-size="11">the fixed test T under G ⊕ e</text>
        <text x="920" y="472" text-anchor="middle" fill="var(--muted)" font-size="11">→ F1 before vs after</text>

        <!-- Gate -->
        <rect x="430" y="380" width="300" height="110" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--gold)" stroke-width="1.6"/>
        <text x="580" y="406" text-anchor="middle" fill="var(--text)" font-size="13" font-weight="800">GATE</text>
        <text x="580" y="428" text-anchor="middle" fill="var(--muted)" font-size="11">metric rule (default): accept ⇔ F1 improves + ε</text>
        <text x="580" y="446" text-anchor="middle" fill="var(--muted)" font-size="11">optional gate agent — the acceptance critic</text>
        <text x="580" y="462" text-anchor="middle" fill="var(--muted)" font-size="11">(persona: lenient · moderate · strict)</text>
        <text x="580" y="481" text-anchor="middle" fill="var(--muted)" font-size="10" font-style="italic">verdicts only · never edits</text>

        <!-- SME -->
        <rect x="28" y="380" width="340" height="110" rx="14" fill="rgba(16,26,49,.85)" stroke="var(--green)" stroke-width="1.6"/>
        <text x="198" y="406" text-anchor="middle" fill="var(--text)" font-size="13" font-weight="800">SME — the human principal</text>
        <text x="198" y="430" text-anchor="middle" fill="var(--muted)" font-size="11">owns the golden labels y · works the re-adjudication queue</text>
        <text x="198" y="448" text-anchor="middle" fill="var(--muted)" font-size="11">reviews gate verdicts — the critic-of-the-critic</text>
        <text x="198" y="472" text-anchor="middle" fill="var(--muted)" font-size="10" font-style="italic">the only human · the only source of truth</text>

        <!-- Flows -->
        <g fill="none" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#archArrow)">
          <path d="M470,96 C440,118 410,130 396,146"/>
          <path d="M228,225 L268,225"/>
          <path d="M512,225 L552,225"/>
          <path d="M756,225 L796,225"/>
          <path d="M730,96 C800,112 870,128 912,146"/>
          <path d="M920,300 L920,376"/>
          <path d="M800,435 L734,435"/>
          <path d="M310,380 C420,344 540,330 640,304"/>
        </g>
        <!-- Post-run flow (dashed): only AFTER the last cycle closes does the
             driver stack-rank the residual misalignments under the final
             policy into the SME re-adjudication queue. Routed around the page
             edge to read as outside the per-cycle loop. -->
        <path d="M426,56 L18,56 L18,430 L24,430" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#archArrow)"/>
        <g fill="none" stroke="var(--green)" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#archArrowGreen)">
          <path d="M368,435 L426,435"/>
        </g>
        <path d="M534,380 L534,233" fill="none" stroke="var(--green)" stroke-width="1.8"/>
        <path d="M534,217 L534,100" fill="none" stroke="var(--green)" stroke-width="1.8" marker-end="url(#archArrowGreen)"/>

        <!-- Flow labels -->
        <text x="256" y="132" fill="var(--muted)" font-size="10.5">policy = the judges’ prompt</text>
        <text x="795" y="110" fill="var(--muted)" font-size="10.5" text-anchor="end">drafter reads the full policy</text>
        <text x="930" y="340" fill="var(--muted)" font-size="10.5">candidate edit e (≤5 files)</text>
        <text x="365" y="356" fill="var(--muted)" font-size="10.5">golden truth y → misalignment</text>
        <text x="18" y="30" fill="var(--muted)" font-size="10.5">AFTER THE RUN — residuals still misaligned under the final policy</text>
        <text x="18" y="46" fill="var(--muted)" font-size="10.5">→ re-adjudication queue: misaligned + high LLM consensus (T1) first</text>
        <text x="544" y="330" fill="var(--green)" font-size="10.5" font-weight="700">accept ⇒ G ⊕ e mints v&lt;run&gt;.&lt;k&gt;</text>
        <text x="544" y="348" fill="var(--red)" font-size="10.5">skip ⇒ the incumbent stays</text>

        <text x="600" y="530" text-anchor="middle" fill="var(--muted)" font-size="11" font-style="italic">One cycle k (solid): label → stack rank &amp; select → draft → eval → gate. After the last cycle (dashed): residuals → SME queue. No agent approves its own work.</text>
      </svg>
      </div>
    </section>

    <div class="about-columns">
    <section class="about-section">
      <h2>The cast — four independent agents</h2>
      <p>RUSH is a multi-agent system with a strict separation of powers: no agent both proposes
      and approves, no LLM owns the ground truth, and each role can run a different model. The
      canonical names, as they appear in the code, the ledgers, and this UI:</p>
      <ul>
        <li><strong>Judges</strong> (<em>the judge panel</em> — the MAS labeling layer) — 2–5 cheap
        mLLMs that label every item independently and produce <em>every</em> decision-quality
        metric. They label only: they never draft, never gate, and never see the golden label while
        judging (LLM consensus κ is computed SME-blind). No expensive model ever scores quality.
        The panel is also what makes <strong>policy blame</strong> measurable: several independent
        labelers citing the same clause while voting wrong indicts the policy text itself — a
        signal a single-labeler system cannot produce (see the derived-scores section).</li>
        <li><strong>Drafter</strong> (<em>the policy-iteration agent</em>; "the optimizer" in the
        cost ledger and on this page) — one model (cheap or frontier) that, each cycle, reads the
        most instructive anchors and writes a single policy edit of ≤5 node files. It drafts; it
        never judges. Its <strong>no-reword rule</strong> forbids paraphrase-only churn: a sentence
        may be touched only to change its semantic meaning, tighten a decision boundary, or clarify
        an objective fact — everything else stays byte-for-byte intact so every diff is real.
        Its packet includes <strong>policy blame</strong> — the nodes most often cited by
        <em>wrong</em> votes across ≥2 different judges (every judge cites the node it applied, so
        wrong votes name the clause that misled them). Model-agnostic by construction: one judge's
        quirks never steer the policy, but a clause that misleads several judges gets fixed once and
        helps them all. And its <strong>edit repertoire is full</strong>: it may narrow or delete
        clauses, remove entire nodes, and simplify the graph — including repairing an implicated
        root clause — not just append and clarify; the gate is told evidenced removals are
        legitimate.</li>
        <li><strong>Gate</strong> — a deterministic rule: accept the edit only if the panel's
        test-partition macro-F1 strictly improves. An optional <strong>gate agent</strong>
        (<em>the acceptance critic</em>, ledgered as <code>gate_agent</code>) is a
        <em>subtractive</em> soundness check on top of that rule: it can veto a metric-passing edit
        it judges unsound — one that <strong>overfits to named examples instead of stating a general
        rule</strong>, leaks the golden answer, games one judge's quirks, tells judges to abstain,
        dumps pair-specific rules into the root instead of the owning node, or <strong>merely
        rewords existing sentences</strong> without changing their meaning (a no-op edit can pass
        the metric on small-partition noise). It never forces or accepts, and it never writes edits;
        the metric stays the hard gate. The over-specificity veto is the crank's fourth overfitting
        guard, alongside the ≤5-change trust region, the no-per-image-answers drafter rule, and the
        aligned anchors.</li>
        <li><strong>SME</strong> (<em>the human principal</em>) — owns the golden labels, works the
        re-adjudication queue, and reviews gate verdicts (the recorded critic-of-the-critic, future
        RLHF data for the gate agent). The only human in the loop, and the only source of truth.</li>
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
          <tr><td>Prompt compression</td><td>Off (On for qwen-7B)</td><td><strong>Per-judge</strong>, on each judge's row in the panel picker. <strong>Off</strong>: the judge labels under the complete policy bundle. <strong>On</strong>: it labels under the <em>deterministic structural digest</em> — rationale ("why this node exists"), SME-workflow, and dataset-curation sections dropped whole; every node id, edge, and decision rule kept byte-for-byte (a projection, never a paraphrase — no compression agent, nothing to audit). Why it exists: the bundle is the judge's entire context, and prompt mass measurably drowns small judges — qwen-7B went 0/6 detected under the full ~25k-char GenAI bundle and 8/8 under a two-line prompt on the same images, while 26B gemma kept discriminating. The digest is the production artifact a lightweight labeler would ship with; view it via the "compressed render" link by the panel picker. Recorded on every run + run manifest, so <strong>policy length × judge capacity</strong> is a first-class research axis (see below).</td></tr>
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
        <em>generator</em> — the exact prompt the judges run), per anchor the SME golden label
        plus each judge's full text output (label, confidence, difficulty, boundary flag,
        justification), and the cycle's <strong>policy-blame table</strong> — nodes cited by wrong
        votes across ≥2 judges, with right-vote counts for calibration (a node also carrying
        correct decisions wants narrowing; one cited almost only in error is a removal candidate).
        With the Input knob on <code>images + text</code> the anchor image pixels
        are also attached (<code>text only</code> is the default — the justifications usually carry
        the visual evidence in words). It returns <em>one</em> edit touching ≤3 nodes (the clip
        knob) — add, amend, narrow, or remove. Its token usage and cost are recorded per cycle.</li>
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
      <p><strong>amp is the amplifier</strong> on the base misalignment×consensus score — how much
      this item can <em>teach</em>. Three multiplicative factors, each in [1, 1+w]: the panel's
      confidence (mean <var>|g|</var> — a confident-wrong error teaches more than a hesitant one),
      its boundary rate (<var>b</var> — documented confusion cases teach more), and the
      <strong>policy-blame share</strong> (<var>s</var><sub>blame</sub> — the fraction of this
      image's wrong votes that cited an <em>indicted</em> policy clause; see below):</p>
      ${eq([
        ['amp', '=', '(1 + mean|<var>g</var>|) · (1 + ½·<var>b</var>) · (1 + <var>s</var><sub>blame</sub>)', ''],
        ['<var>s</var><sub>blame</sub>', '=', '(# wrong votes citing a blamed node) / (# wrong votes)', 'anchor-selection side only'],
        ['anchor value', '=', '<var>I</var><sub>base</sub> · amp', 'ranks policy-learning anchors'],
        ['re-adjudication', '=', '<var>I</var><sub>base</sub> · amp · (1 − <var>p</var><sub>human</sub>)', 'ranks the human queue (s_blame = 0 here)'],
      ])}
      <h3>Policy blame — the panel indicts the clause, not the model</h3>
      <p>Every judge cites the policy node it applied, so a <em>wrong</em> vote names the clause
      that misled it. Each cycle aggregates these into a per-node table — wrong vs right citation
      counts, <code>wrong_share</code>, and an advisory edit-type <code>hint</code>
      (<em>remove_or_narrow</em>: cited mostly in error, the clause misleads ·
      <em>split_or_tighten</em>: mixed at volume, the node conflates two patterns ·
      <em>clarify</em>: mostly right, occasional misleads) — recorded on the cycle
      (<code>policy_blame</code>) and fed to the drafter and gate. Only nodes wrong-cited by
      <strong>≥2 distinct judges</strong> reach the agents or the amplifier: one judge's quirks
      never steer the policy, but a clause that misleads several gets fixed once and helps them
      all. <strong>This signal only exists because the labeling layer is a multi-agent panel</strong>
      — with a single labeler, "the model is weak" and "the policy misleads" are indistinguishable;
      with several independent labelers converging on the same cited clause, the policy text itself
      is indicted. An image whose errors are policy-attributable outranks an idiosyncratic one
      (via <var>s</var><sub>blame</sub>) precisely because fixing the clause fixes every judge it
      misleads.</p>
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
      <h2>How RUSH differs from PPO, GEPA, VISTA</h2>
      <p>Same family — iterate a policy against feedback with bounded steps — but RUSH makes two
      choices most neighbors don't: the reward is <em>human</em> (SME golden labels with a live
      correction channel), and the search state is <em>one auditable incumbent</em>, not a
      candidate pool.</p>
      <ul>
        <li><strong>PPO</strong> (Schulman et al. 2017, arXiv:1707.06347) — RUSH borrows the
        discipline, not the math. The ≤5-file clip plus the strict-improvement gate bound each
        step the way PPO's clipped surrogate bounds a policy ratio — but here the parameter is
        text, the "gradient" is a drafted diff, and acceptance is closer to a line search than a
        ratio clip. The reward's human provenance follows RLHF (Ouyang et al. 2022,
        arXiv:2203.02155).</li>
        <li><strong>GEPA</strong> (Agrawal et al. 2025, arXiv:2507.19457) — the nearest optimizer
        family: reflective LLM mutation of prompts with an acceptance test. GEPA keeps an
        instance-wise Pareto <em>pool</em> of candidate prompts and optimizes the system's own
        metric; RUSH keeps exactly one versioned incumbent — an enterprise policy must be a
        single reviewable document — and optimizes against external SME truth that
        re-adjudication can correct mid-run.</li>
        <li><strong>VISTA</strong> (Long et al. 2025, arXiv:2510.15831) — the same loop shape
        (multi-agent, test-time self-improvement) with the opposite trust model: VISTA's judges
        and reward are model-internal; RUSH's reward is human-anchored. That provenance
        difference is exactly why the SME sits <em>inside</em> the loop, not above it.</li>
        <li><strong>Adjacent</strong> — TextGrad (arXiv:2406.07496) backpropagates textual
        feedback through a computation graph; OPRO (arXiv:2309.03409) prompts an optimizer LLM
        with a (solution, score) history; MIPROv2 / DSPy (arXiv:2406.11695) searches
        instructions + demonstrations. All optimize a single artifact in-process; none separate
        labeler / drafter / critic / human into independent agents around a versioned
        incumbent.</li>
      </ul>
      <p class="hint">Full comparison with the formalism and citations:
      <code>docs/TECHNICAL-REPORT.md</code> §7.</p>
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
        <li><strong>Policy length × judge capacity</strong> — the policy is a growing textual
        parameter, and each judge has a capacity budget it must fit inside. Measured (2026-07-09):
        under the full ~25k-char GenAI bundle a 7B judge collapsed to the policy's default branch
        on every call (0/6 generated images detected) yet scored 8/8 on the same images under a
        two-line prompt, while a 26B judge kept discriminating — prompt drowning, not capability.
        The per-judge <em>policy render</em> knob makes this a two-way ablation the crank can run:
        every cycle records the bundle size (the parameter-count analog), every run records which
        judges labeled under the compressed render, and the fixed benchmark scores both. Nobody in
        the PPO/GEPA/VISTA lineage has measured this axis.</li>
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
