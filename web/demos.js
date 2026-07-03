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
    heroCopy: {
      eyebrow: 'Multiclass digit audit with SME policy',
      h1: 'RUSH scales SME digit-classification policy.',
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
