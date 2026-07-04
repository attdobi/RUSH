// RUSH demo registry.
// A single source of truth for which classification demo the web UI is running.
// Each entry describes its label space, policy-graph location, manifest URLs,
// thumbnail directory, and hero copy. app.js and the module scripts read the
// ACTIVE demo config instead of hardcoding GenAI-specific constants.
window.RUSH_DEMOS = {
  genai: {
    id: 'genai', title: 'GenAI Image Classification', kind: 'binary',
    classes: ['gen_ai', 'not_gen_ai'], positiveClass: 'gen_ai',
    policyGraph: { area: 'Generative_AI', version: 'v0.3', rootId: 'GA.root', path: 'policy-graph/Generative_AI/v0.3' },
    manifests: {
      dev: '../data/images/genai-classification/manifests/dev_golden_labels.csv',
      holdout: '../data/images/genai-classification/manifests/holdout_labels.csv',
      summary: '../data/images/genai-classification/manifests/sampling_summary.json'
    },
    thumbnailsDir: 'data/images/genai-classification/thumbnails',
    heroCopy: {
      eyebrow: 'Bulk LLM image audit with SME policy',
      h1: 'RUSH turns SME policy into a repeatable image-audit loop.',
      cta: 'Start GenAI demo'
    }
  },
  mnist: {
    id: 'mnist', title: 'MNIST Digit Classification', kind: 'multiclass',
    classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], positiveClass: null,
    policyGraph: { area: 'MNIST_Digits', version: 'v0.1', rootId: 'MD.root', path: 'policy-graph/MNIST_Digits/v0.1' },
    manifests: {
      dev: '../data/images/mnist-classification/manifests/train_labels.csv',
      holdout: '../data/images/mnist-classification/manifests/val_labels.csv',
      summary: '../data/images/mnist-classification/manifests/sampling_summary.json'
    },
    thumbnailsDir: 'data/images/mnist-classification/source-datasets/mnist',
    // Confusion pairs surfaced from the v0.1 MD frontmatter (confused_with edges).
    // Order = the pairs the demo copy calls out; do not reorder without also
    // updating hero/sample copy and the confusion-pair strip.
    confusionPairs: [
      { pair: ['4', '9'], id: '4-9', label: '4 vs 9', reason: 'Angular top wedge vs a closed rounded loop.' },
      { pair: ['3', '5'], id: '3-5', label: '3 vs 5', reason: 'Two open right-facing bumps vs a top bar + lower bowl.' },
      { pair: ['7', '1'], id: '7-1', label: '7 vs 1', reason: 'Horizontal top bar + long diagonal vs a single vertical stroke.' },
      { pair: ['8', '3'], id: '8-3', label: '8 vs 3', reason: 'Two closed stacked loops vs open right-facing bumps.' }
    ],
    // Map a digit class label to the corresponding policy-graph node id.
    classNodeId: cls => `MD.digit.${cls}`,
    // Tiny distinguishing-feature summary used by sample cards. Sourced from
    // policy-graph/MNIST_Digits/v0.1/MD.digit.*.md "Positive criteria".
    classHint: {
      '0': 'single closed loop, no crossbar',
      '1': 'single vertical stroke (optional top flag / base)',
      '2': 'top curve + descending diagonal + flat base',
      '3': 'two stacked right-facing bumps, open on the left',
      '4': 'two verticals joined by a horizontal crossbar',
      '5': 'flat top bar, stem, lower right-facing bowl',
      '6': 'left-leaning stroke curling into a closed bottom loop',
      '7': 'horizontal top bar + long descending diagonal',
      '8': 'two stacked closed loops meeting at a pinch',
      '9': 'closed top loop with a descending tail'
    },
    // Consensus filter options for §4. Each entry ends up as an <option>.
    // pair:X-Y filters to images whose SME truth is X or Y (surfaces confusion pairs).
    consensusFilters: [
      { value: 'all', label: 'All images' },
      { value: 'unanimous', label: 'Unanimous only' },
      { value: 'split', label: 'Split only' },
      { value: 'boundary', label: 'Boundary-flagged' },
      { value: 'pair:4-9', label: 'Pair 4 vs 9' },
      { value: 'pair:3-5', label: 'Pair 3 vs 5' },
      { value: 'pair:7-1', label: 'Pair 7 vs 1' },
      { value: 'pair:8-3', label: 'Pair 8 vs 3' }
    ],
    // Text overrides applied by app.js applyDemoChrome() — leave English strings
    // stable; ids that receive them are defined in web/index.html.
    sectionCopy: {
      sampleH2: 'Sample the MNIST golden set.',
      sampleSub: 'Start with a balanced 10-class slice: 200 train + 50 holdout images per digit, sourced from the local MNIST manifests.',
      growSub: 'Seed the MNIST Generator Prompt V0 as a per-digit policy graph → label a batch → propose prompt updates → accept → label again. Each accepted SME update creates the next generator prompt version.',
      growLoopStep2Body: 'Jump to §3 for the default 20-image train+holdout batch, labeled with the selected generator prompt version.',
      labelH2: 'Run the next digit-labeling round.',
      labelSub: 'Default: 20 MNIST digits drawn from both training and holdout splits, scored against SME digit truth.',
      labelStartButton: 'Start 20-image train+holdout run',
      labelDefaultBatch: 'N=20 · train + holdout',
      labelDefaultDetail: 'Uses split <code>all</code>, latest selected policy version, and true batched labeling of MNIST digits.',
      scoreSub: 'Consensus, misalignment, and confusion-pair views share the selected run and stay in one digit-audit surface.',
      consensusH3: 'What did the panel decide about each digit?',
      consensusSub: 'Every model votes 0–9 per image. Unanimous, majority, tie, and majority-vs-SME misses are flagged fast.',
      misalignmentH3: 'Model vs SME digit disagreement, ranked for review.',
      misalignmentSub: 'Per-image SME digit truth vs model labels. Confusion between the seed pairs (4/9, 3/5, 7/1, 8/3) rises to the top; policy-node citations link straight to the digit node whose criteria were cited.',
      borderlineH3: 'Where the panel hedges between digits, SMEs decide.',
      borderlineSub: 'Hard cases grouped by the confused digit (e.g. a 4 that could be a 9, a 3 that could be a 5). Will fan out as the graph grows.',
      qualityH2: 'Digit-classification quality is the gate for policy growth.',
      qualitySub: 'Compare per-digit accuracy, macro F1, per-class precision/recall, review burden, and cost by labeler, run, and policy version.',
      insightsH2: 'Where the panel disagreed with SME digit truth.',
      insightsSub: 'Start with majority-wrong digits. Open More cuts for model disagreement, confusion-pair concentration (4/9, 3/5, 7/1, 8/3), and recurring pair disagreement.',
      policyGraphTitle: 'MNIST Generator Prompt v0.1 (policy graph)',
      policyGraphBlurb: 'This graph is the current MNIST generator prompt version: ten digit nodes (0–9) under a root, plus confused_with edges for the boundary pairs the demo optimizes. Hover to trace neighbors; click to drill into a digit\u2019s criteria.'
    },
    heroCopy: {
      eyebrow: 'Multiclass digit audit with SME policy',
      h1: 'RUSH turns SME digit criteria into a repeatable MNIST audit loop.',
      lede: 'Sample MNIST digits, seed Generator Prompt V0 as a per-digit policy graph, run a 20-image LLM panel, then use disagreement — especially confusion pairs 4/9, 3/5, 7/1, 8/3 — to propose SME-reviewed updates.',
      cta: 'Start MNIST demo'
    }
  }
};

window.RUSH_ACTIVE_DEMO = 'genai'; // default; overridden by ?demo= or selector

// Resolve the active demo id from (in priority order): ?demo= query param,
// localStorage, then the default. Persists whatever it resolves so the header
// selector and query param stay in sync. Falls back to 'genai' on anything odd.
window.rushResolveActiveDemoId = function rushResolveActiveDemoId() {
  const demos = window.RUSH_DEMOS || {};
  const fallback = 'genai';
  let resolved = window.RUSH_ACTIVE_DEMO || fallback;
  try {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get('demo');
    const fromStorage = window.localStorage.getItem('rush_active_demo');
    if (fromQuery && demos[fromQuery]) {
      resolved = fromQuery;
    } else if (fromStorage && demos[fromStorage]) {
      resolved = fromStorage;
    }
  } catch (error) {
    // Query/storage access can throw in sandboxed contexts; keep the default.
  }
  if (!demos[resolved]) resolved = fallback;
  window.RUSH_ACTIVE_DEMO = resolved;
  try { window.localStorage.setItem('rush_active_demo', resolved); } catch (error) { /* ignore */ }
  return resolved;
};

// Return the active demo config object.
window.rushActiveDemo = function rushActiveDemo() {
  const demos = window.RUSH_DEMOS || {};
  const id = window.RUSH_ACTIVE_DEMO && demos[window.RUSH_ACTIVE_DEMO]
    ? window.RUSH_ACTIVE_DEMO
    : window.rushResolveActiveDemoId();
  return demos[id] || demos.genai;
};

// Resolve the id once at load so consumers can rely on window.RUSH_ACTIVE_DEMO.
window.rushResolveActiveDemoId();
