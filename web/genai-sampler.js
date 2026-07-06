/*
 * RUSH GenAI synthetic gold-set sampler.
 * Dependency-free browser/Node module for the VC demo reset path. It never reads
 * image bytes, invokes Python, or writes/embeds manifests.
 *
 * API:
 *   window.RushGenaiSampler.runDemoReset({ seed, nDev, nHoldout, mode, humanLabelOverrides })
 *     -> { devGolden, holdout, combined, summary, leakageChecks, assumptions }
 *
 * Item shape includes: sample_id, dataset, source_label_dir, label, label_int,
 * synthetic_repo_rel_path, sha256, split, seed, sampling_version, truth_tier,
 * policy_use, human_override_label:null, human_override_note:''.
 */
(function attach(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RushGenaiSampler = factory();
})(typeof globalThis !== 'undefined' ? globalThis : window, function factory() {
  'use strict';

  const SAMPLING_VERSION = 'genai-web-synthetic-sampling-v1';
  const DEFAULT_SEED = 20260510;
  const DEFAULT_SAMPLE_N = 100;
  const DEFAULT_ITEMS_PER_BUCKET = 240;
  const DATASETS = Object.freeze(['sdv1_4', 'midjourney', 'wfir']);
  const LABELS = Object.freeze(['ai_generated', 'not_ai_generated']);
  const HUMAN_OVERRIDE_LABELS = Object.freeze(['ai_generated', 'not_ai_generated', 'needs_review']);
  const MODES = Object.freeze(['cold_start', 'warm_start']);
  const SOURCE_LABEL_DIRS = Object.freeze({
    sdv1_4: Object.freeze({ ai_generated: '1_false', not_ai_generated: '0_real' }),
    midjourney: Object.freeze({ ai_generated: '1_fake', not_ai_generated: '0_real' }),
    wfir: Object.freeze({ ai_generated: '1_fake', not_ai_generated: '0_real' })
  });
  const ASSUMPTION_NOTE = 'sdv1_4/1_false is treated as positive ai_generated; 0_real is negative real/not_ai.';
  const ASSUMPTIONS = Object.freeze([ASSUMPTION_NOTE]);

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) throw new Error(`${name} must be a non-negative integer`);
  }
  function labelInt(label) {
    if (label === 'ai_generated') return 1;
    if (label === 'not_ai_generated') return 0;
    throw new Error(`Unknown label: ${label}`);
  }
  function sourceLabelDir(dataset, label) {
    const dir = SOURCE_LABEL_DIRS[dataset] && SOURCE_LABEL_DIRS[dataset][label];
    if (!dir) throw new Error(`No source_label_dir mapping for ${dataset}/${label}`);
    return dir;
  }
  function assertKnownMode(mode) {
    if (!MODES.includes(mode)) throw new Error(`Unknown mode: ${mode}. Expected ${MODES.join(', ')}`);
  }

  // Deterministic PRNG/hash helpers; not cryptographic, only for stable demo sampling.
  function fnv1a32(input) {
    let hash = 0x811c9dc5;
    for (const char of String(input)) {
      hash ^= char.charCodeAt(0);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash >>> 0;
  }
  function mulberry32(seed) {
    let state = seed >>> 0;
    return function next() {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function stableSyntheticSha256(input) {
    const chunks = [];
    let state = fnv1a32(`rush:${input}`);
    for (let i = 0; i < 8; i += 1) {
      state = fnv1a32(`${state}:${i}:${input}`);
      chunks.push(state.toString(16).padStart(8, '0'));
    }
    return chunks.join('');
  }
  function shuffleDeterministically(values, seedKey) {
    const rng = mulberry32(fnv1a32(seedKey));
    const copy = values.slice();
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }

  function splitEvenly(total, buckets) {
    assertNonNegativeInteger(total, 'total');
    const base = Math.floor(total / buckets.length);
    const remainder = total % buckets.length;
    return buckets.reduce((acc, bucket, index) => {
      acc[bucket] = base + (index < remainder ? 1 : 0);
      return acc;
    }, {});
  }
  function allocateSplitCounts(total) {
    const labelCounts = splitEvenly(total, LABELS);
    return LABELS.reduce((acc, label) => {
      acc[label] = splitEvenly(labelCounts[label], DATASETS);
      return acc;
    }, {});
  }

  function repoRelPathFor(dataset, label, index) {
    const padded = String(index).padStart(5, '0');
    return `data/images/genai-classification/source-datasets/${dataset}/${label}/${sourceLabelDir(dataset, label)}_synthetic_${padded}.jpg`;
  }
  function generateSyntheticCandidateCatalog(options = {}) {
    const itemsPerBucket = options.itemsPerBucket ?? DEFAULT_ITEMS_PER_BUCKET;
    assertNonNegativeInteger(itemsPerBucket, 'itemsPerBucket');
    if (itemsPerBucket === 0) throw new Error('itemsPerBucket must be greater than zero');
    const candidates = [];
    for (const dataset of DATASETS) {
      for (const label of LABELS) {
        for (let index = 1; index <= itemsPerBucket; index += 1) {
          const synthetic_repo_rel_path = repoRelPathFor(dataset, label, index);
          const source_label_dir = sourceLabelDir(dataset, label);
          candidates.push({
            candidate_id: `${dataset}:${label}:${String(index).padStart(5, '0')}`,
            dataset,
            source_label_dir,
            label,
            label_int: labelInt(label),
            synthetic_repo_rel_path,
            original_filename: synthetic_repo_rel_path.split('/').pop(),
            file_ext: 'jpg',
            sha256: stableSyntheticSha256(synthetic_repo_rel_path),
            source_structure: `${dataset}/${label}/${source_label_dir}`,
            is_synthetic_demo_candidate: true,
            assumption_note: dataset === 'sdv1_4' && label === 'ai_generated' ? ASSUMPTION_NOTE : ''
          });
        }
      }
    }
    return candidates;
  }
  function normalizeCandidateCatalog(catalog) {
    if (!Array.isArray(catalog)) throw new Error('catalog must be an array');
    return catalog.map((candidate, index) => {
      const { dataset, label } = candidate;
      if (!DATASETS.includes(dataset)) throw new Error(`Unknown dataset at catalog[${index}]: ${dataset}`);
      if (!LABELS.includes(label)) throw new Error(`Unknown label at catalog[${index}]: ${label}`);
      const synthetic_repo_rel_path = candidate.synthetic_repo_rel_path || candidate.repo_rel_path;
      if (!synthetic_repo_rel_path) throw new Error(`Missing synthetic_repo_rel_path at catalog[${index}]`);
      return {
        ...candidate,
        dataset,
        label,
        label_int: candidate.label_int ?? labelInt(label),
        source_label_dir: candidate.source_label_dir || sourceLabelDir(dataset, label),
        synthetic_repo_rel_path,
        sha256: candidate.sha256 || stableSyntheticSha256(synthetic_repo_rel_path),
        is_synthetic_demo_candidate: candidate.is_synthetic_demo_candidate ?? true
      };
    });
  }
  function groupCandidates(catalog) {
    const grouped = new Map(DATASETS.flatMap(dataset => LABELS.map(label => [`${dataset}/${label}`, []])));
    for (const candidate of normalizeCandidateCatalog(catalog)) grouped.get(`${candidate.dataset}/${candidate.label}`).push(candidate);
    return grouped;
  }
  function dedupeByPathAndHash(candidates) {
    const paths = new Set();
    const hashes = new Set();
    return candidates.filter(candidate => {
      if (paths.has(candidate.synthetic_repo_rel_path) || hashes.has(candidate.sha256)) return false;
      paths.add(candidate.synthetic_repo_rel_path);
      hashes.add(candidate.sha256);
      return true;
    });
  }
  function recordFor(candidate, split, seed, mode, drawIndexWithinDatasetLabel) {
    return {
      sample_id: 'pending',
      dataset: candidate.dataset,
      source_label_dir: candidate.source_label_dir,
      label: candidate.label,
      label_int: candidate.label_int,
      synthetic_repo_rel_path: candidate.synthetic_repo_rel_path,
      original_filename: candidate.original_filename || candidate.synthetic_repo_rel_path.split('/').pop(),
      file_ext: candidate.file_ext || 'jpg',
      sha256: candidate.sha256,
      split,
      seed,
      sampling_mode: mode,
      sampling_version: SAMPLING_VERSION,
      truth_tier: 'gold_candidate',
      policy_use: split === 'dev_golden' ? (mode === 'warm_start' ? 'policy_refinement' : 'develop_policy') : 'locked_holdout_decision_quality',
      human_override_label: null,
      human_override_label_int: null,
      human_override_note: '',
      human_override_applied: false,
      llm_status: 'not_started_no_model_outputs',
      draw_index_within_dataset_label: drawIndexWithinDatasetLabel,
      source_structure: candidate.source_structure || `${candidate.dataset}/${candidate.label}/${candidate.source_label_dir}`,
      is_synthetic_demo_candidate: true,
      assumption_note: candidate.assumption_note || ''
    };
  }
  function sortRecords(records) {
    return records.slice().sort((a, b) => a.dataset.localeCompare(b.dataset) || a.label.localeCompare(b.label) || a.synthetic_repo_rel_path.localeCompare(b.synthetic_repo_rel_path));
  }
  function assignSampleIds(records, prefix) {
    return records.map((record, index) => ({ ...record, sample_id: `${prefix}_${String(index + 1).padStart(4, '0')}` }));
  }

  function countBy(records, keyFn) {
    return records.reduce((acc, record) => {
      const key = keyFn(record);
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
  }
  function nestedSplitCounts(records, keyFn) {
    const result = { dev_golden: {}, holdout: {} };
    for (const record of records) {
      const key = keyFn(record);
      result[record.split][key] = (result[record.split][key] || 0) + 1;
    }
    return result;
  }
  function splitAndCombinedCounts(records, keyFn) {
    return {
      dev_golden: countBy(records.filter(row => row.split === 'dev_golden'), keyFn),
      holdout: countBy(records.filter(row => row.split === 'holdout'), keyFn),
      combined: countBy(records, keyFn)
    };
  }
  function buildSummary({ combined, seed, mode, nDev, nHoldout, allocation, sourceAvailability }) {
    const splitCounts = countBy(combined, row => row.split);
    const classBalance = splitAndCombinedCounts(combined, row => row.label);
    const sourceBalance = splitAndCombinedCounts(combined, row => row.dataset);
    const sourceLabelBalance = splitAndCombinedCounts(combined, row => `${row.dataset}/${row.source_label_dir}`);
    return {
      sampling_version: SAMPLING_VERSION,
      synthetic_demo: true,
      seed,
      mode,
      n_dev_golden: splitCounts.dev_golden || 0,
      n_holdout: splitCounts.holdout || 0,
      requested: { nDev, nHoldout },
      class_balance: classBalance,
      source_balance: sourceBalance,
      source_label_balance: sourceLabelBalance,
      architecture_note: 'Client-side synthetic sampler reset only: no Python invocation, no LLM calls, and no image bytes/base64/manifests in the browser demo.',
      counts: {
        total: combined.length,
        by_split: splitCounts,
        by_class: countBy(combined, row => row.label),
        by_dataset: countBy(combined, row => row.dataset),
        by_source: countBy(combined, row => `${row.dataset}/${row.source_label_dir}`),
        by_split_class: nestedSplitCounts(combined, row => row.label),
        by_split_dataset: nestedSplitCounts(combined, row => row.dataset),
        by_split_source: nestedSplitCounts(combined, row => `${row.dataset}/${row.source_label_dir}`)
      },
      allocation,
      source_availability: sourceAvailability,
      assumptions: ASSUMPTIONS.slice(),
      source_label_dir_mapping: JSON.parse(JSON.stringify(SOURCE_LABEL_DIRS)),
      note: 'Static web demo samples synthetic paths only. Real local image manifests are produced by scripts/sample_genai_gold_sets.py.'
    };
  }
  function buildLeakageChecks(devGolden, holdout) {
    const overlap = (left, right) => [...left].filter(value => right.has(value)).sort();
    const pathOverlap = overlap(new Set(devGolden.map(row => row.synthetic_repo_rel_path)), new Set(holdout.map(row => row.synthetic_repo_rel_path)));
    const hashOverlap = overlap(new Set(devGolden.map(row => row.sha256)), new Set(holdout.map(row => row.sha256)));
    const combined = devGolden.concat(holdout);
    const duplicates = (keyFn) => Object.entries(countBy(combined, keyFn)).filter(([, count]) => count > 1).map(([key]) => key).sort();
    const duplicateSampleIds = duplicates(row => row.sample_id);
    const duplicateSyntheticPaths = duplicates(row => row.synthetic_repo_rel_path);
    const duplicateSha256 = duplicates(row => row.sha256);
    const ok = pathOverlap.length === 0 && hashOverlap.length === 0 && duplicateSampleIds.length === 0 && duplicateSyntheticPaths.length === 0 && duplicateSha256.length === 0;
    return {
      ok,
      devHoldoutDisjoint: pathOverlap.length === 0 && hashOverlap.length === 0,
      dev_holdout_disjoint_by_synthetic_path: pathOverlap.length === 0,
      dev_holdout_disjoint_by_sha256: hashOverlap.length === 0,
      no_duplicate_sample_ids: duplicateSampleIds.length === 0,
      no_duplicate_synthetic_paths: duplicateSyntheticPaths.length === 0,
      no_duplicate_sha256: duplicateSha256.length === 0,
      pathOverlap,
      hashOverlap,
      note: ok ? 'Synthetic dev golden and locked holdout samples are disjoint by path and hash.' : 'Sampler leakage check found overlapping synthetic paths or hashes.',
      path_overlap: pathOverlap,
      hash_overlap: hashOverlap,
      duplicate_sample_ids: duplicateSampleIds,
      duplicate_synthetic_paths: duplicateSyntheticPaths,
      duplicate_sha256: duplicateSha256
    };
  }

  function normalizeOverrideValue(value) {
    if (typeof value === 'string') return { label: value, note: '' };
    if (!value || typeof value !== 'object') throw new Error('Human overrides must be strings or objects');
    return { label: value.human_override_label || value.label, note: value.human_override_note || value.note || '' };
  }
  function normalizeHumanOverrides(humanLabelOverrides) {
    if (!humanLabelOverrides) return [];
    if (Array.isArray(humanLabelOverrides)) {
      return humanLabelOverrides.map((entry, index) => {
        const normalized = normalizeOverrideValue(entry);
        const key = entry.sample_id || entry.sha256 || entry.synthetic_repo_rel_path;
        if (!key) throw new Error(`Human override at index ${index} needs sample_id, sha256, or synthetic_repo_rel_path`);
        if (!HUMAN_OVERRIDE_LABELS.includes(normalized.label)) throw new Error(`Invalid human override label at index ${index}: ${normalized.label}`);
        return { ...normalized, key };
      });
    }
    if (typeof humanLabelOverrides === 'object') {
      return Object.entries(humanLabelOverrides).map(([key, value]) => {
        const normalized = normalizeOverrideValue(value);
        if (!HUMAN_OVERRIDE_LABELS.includes(normalized.label)) throw new Error(`Invalid human override label for ${key}: ${normalized.label}`);
        return { ...normalized, key };
      });
    }
    throw new Error('humanLabelOverrides must be an array, object map, or omitted');
  }
  function applyHumanOverrides(records, humanLabelOverrides) {
    const overrides = normalizeHumanOverrides(humanLabelOverrides);
    if (!overrides.length) return records.map(record => ({ ...record }));
    return records.map(record => {
      const override = overrides.find(entry => entry.key === record.sample_id || entry.key === record.sha256 || entry.key === record.synthetic_repo_rel_path);
      return override ? {
        ...record,
        human_override_label: override.label,
        human_override_label_int: override.label === 'needs_review' ? null : labelInt(override.label),
        human_override_note: override.note,
        human_override_applied: true
      } : { ...record };
    });
  }

  function runDemoReset(options = {}) {
    const seed = options.seed ?? DEFAULT_SEED;
    const nDev = options.nDev ?? options.n_dev ?? DEFAULT_SAMPLE_N;
    const nHoldout = options.nHoldout ?? options.n_holdout ?? DEFAULT_SAMPLE_N;
    const mode = options.mode || 'cold_start';
    assertNonNegativeInteger(nDev, 'nDev');
    assertNonNegativeInteger(nHoldout, 'nHoldout');
    assertKnownMode(mode);

    const minimumPerBucket = Math.ceil((nDev + nHoldout) / (DATASETS.length * LABELS.length)) + 12;
    const catalog = options.catalog || generateSyntheticCandidateCatalog({ itemsPerBucket: Math.max(DEFAULT_ITEMS_PER_BUCKET, minimumPerBucket) });
    const grouped = groupCandidates(catalog);
    const devAllocation = allocateSplitCounts(nDev);
    const holdoutAllocation = allocateSplitCounts(nHoldout);
    const sourceAvailability = {};
    const devRows = [];
    const holdoutRows = [];

    for (const label of LABELS) {
      for (const dataset of DATASETS) {
        const key = `${dataset}/${label}`;
        const candidates = dedupeByPathAndHash(grouped.get(key) || []);
        const devNeeded = devAllocation[label][dataset];
        const holdoutNeeded = holdoutAllocation[label][dataset];
        const needed = devNeeded + holdoutNeeded;
        if (candidates.length < needed) throw new Error(`Not enough unique synthetic candidates for ${key}: need ${needed}, have ${candidates.length}`);
        const shuffled = shuffleDeterministically(candidates, `${SAMPLING_VERSION}:${mode}:${seed}:${key}`);
        const devCandidates = shuffled.slice(0, devNeeded);
        const holdoutCandidates = shuffled.slice(devNeeded, needed);
        sourceAvailability[key] = { available_unique: candidates.length, dev_golden: devCandidates.length, holdout: holdoutCandidates.length, source_label_dir: sourceLabelDir(dataset, label) };
        devCandidates.forEach((candidate, index) => devRows.push(recordFor(candidate, 'dev_golden', seed, mode, index + 1)));
        holdoutCandidates.forEach((candidate, index) => holdoutRows.push(recordFor(candidate, 'holdout', seed, mode, index + 1)));
      }
    }

    const devGolden = assignSampleIds(sortRecords(devRows), 'dev_golden');
    const holdout = assignSampleIds(sortRecords(holdoutRows), 'holdout');
    const combined = applyHumanOverrides(devGolden.concat(holdout), options.humanLabelOverrides || options.human_overrides);
    const finalDevGolden = combined.filter(row => row.split === 'dev_golden');
    const finalHoldout = combined.filter(row => row.split === 'holdout');
    const leakageChecks = buildLeakageChecks(finalDevGolden, finalHoldout);
    const summary = buildSummary({
      combined,
      seed,
      mode,
      nDev,
      nHoldout,
      allocation: { dev_golden: devAllocation, holdout: holdoutAllocation },
      sourceAvailability
    });
    return { devGolden: finalDevGolden, holdout: finalHoldout, combined, summary, leakageChecks, assumptions: ASSUMPTIONS.slice() };
  }

  return Object.freeze({
    SAMPLING_VERSION,
    DEFAULT_SAMPLE_N,
    DATASETS,
    LABELS,
    HUMAN_OVERRIDE_LABELS,
    MODES,
    SOURCE_LABEL_DIRS,
    ASSUMPTIONS,
    allocateSplitCounts,
    applyHumanOverrides,
    buildLeakageChecks,
    generateSyntheticCandidateCatalog,
    normalizeCandidateCatalog,
    runDemoReset,
    sampleCatalog: runDemoReset,
    stableSyntheticSha256
  });
});
