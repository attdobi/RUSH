#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const sampler = require('../web/genai-sampler.js');

function allEqual(values, expected, message) {
  for (const value of values) assert.equal(value, expected, message);
}

function assertBalancedBucketCounts(result, split, expectedPerBucket) {
  const byClass = result.summary.counts.by_split_class[split];
  const byDataset = result.summary.counts.by_split_dataset[split];
  const bySource = result.summary.counts.by_split_source[split];
  assert.equal(Object.values(byClass).reduce((sum, value) => sum + value, 0), result.summary.counts.by_split[split]);
  assert.equal(Object.values(byDataset).reduce((sum, value) => sum + value, 0), result.summary.counts.by_split[split]);
  assert.equal(Object.values(bySource).reduce((sum, value) => sum + value, 0), result.summary.counts.by_split[split]);

  const allocationCounts = [];
  for (const label of sampler.LABELS) {
    for (const dataset of sampler.DATASETS) {
      allocationCounts.push(result.summary.allocation[split][label][dataset]);
    }
  }
  allEqual(allocationCounts, expectedPerBucket, `${split} should be evenly allocated across class/source buckets`);
}


const defaults = sampler.runDemoReset();
assert.equal(defaults.devGolden.length, 100, 'default dev golden count should be 100');
assert.equal(defaults.holdout.length, 100, 'default holdout count should be 100');
assert.equal(defaults.combined.length, 200, 'default combined count should be 200');
assert.equal(defaults.summary.n_dev_golden, 100, 'default summary dev count should be 100');
assert.equal(defaults.summary.n_holdout, 100, 'default summary holdout count should be 100');
assert.equal(sampler.DEFAULT_SAMPLE_N, 100, 'exported default sample N should be 100');

const options = { seed: 8675309, nDev: 18, nHoldout: 12, mode: 'cold_start' };
const first = sampler.runDemoReset(options);
const second = sampler.runDemoReset(options);
const differentSeed = sampler.runDemoReset({ ...options, seed: options.seed + 1 });

assert.deepEqual(first, second, 'same inputs should produce byte-stable deterministic output');
assert.notDeepEqual(
  first.combined.map(row => row.sha256),
  differentSeed.combined.map(row => row.sha256),
  'different seed should produce a different sample order/selection'
);

assert.equal(first.devGolden.length, options.nDev, 'dev golden count should match request');
assert.equal(first.holdout.length, options.nHoldout, 'holdout count should match request');
assert.equal(first.combined.length, options.nDev + options.nHoldout, 'combined total should match request');
assert.equal(first.summary.counts.by_split.dev_golden, options.nDev, 'summary dev count should match');
assert.equal(first.summary.counts.by_split.holdout, options.nHoldout, 'summary holdout count should match');
assert.equal(first.summary.n_dev_golden, options.nDev, 'UI-compatible n_dev_golden should match');
assert.equal(first.summary.n_holdout, options.nHoldout, 'UI-compatible n_holdout should match');
assert.ok(first.summary.architecture_note.includes('no Python'), 'summary should explain client-side/no-Python demo path');
assert.equal(first.summary.class_balance.dev_golden.ai_generated, 9, 'UI class balance should be available');
assert.equal(first.summary.source_balance.combined.sdv1_4, 10, 'UI source balance should be available');
assert.ok(first.summary.source_label_balance.combined['sdv1_4/1_false'], 'UI source-label balance should be available');
assert.equal(first.leakageChecks.ok, true, 'leakage checks should pass');
assert.equal(first.leakageChecks.devHoldoutDisjoint, true, 'UI-compatible leakage flag should pass');
assert.equal(first.leakageChecks.dev_holdout_disjoint_by_synthetic_path, true, 'dev/holdout paths must be disjoint');
assert.equal(first.leakageChecks.dev_holdout_disjoint_by_sha256, true, 'dev/holdout hashes must be disjoint');
assert.deepEqual(first.leakageChecks.pathOverlap, [], 'UI-compatible path overlap list should be empty');
assert.deepEqual(first.leakageChecks.hashOverlap, [], 'UI-compatible hash overlap list should be empty');
assertBalancedBucketCounts(first, 'dev_golden', 3);
assertBalancedBucketCounts(first, 'holdout', 2);

assert.ok(
  first.assumptions.includes('sdv1_4/1_false is treated as positive ai_generated; 0_real is negative real/not_ai.'),
  'required source-label assumption should be present'
);
assert.equal(sampler.SOURCE_LABEL_DIRS.sdv1_4.ai_generated, '1_false', 'sdv1_4 positive mapping must be 1_false');
assert.equal(sampler.SOURCE_LABEL_DIRS.sdv1_4.not_ai_generated, '0_real', 'sdv1_4 negative mapping must be 0_real');
assert.equal(first.devGolden[0].human_override_label, null, 'records should default to no human override');
assert.equal(first.devGolden[0].llm_status, 'not_started_no_model_outputs', 'records should explicitly avoid fake LLM status');
assert.ok(first.devGolden[0].original_filename.endsWith('.jpg'), 'records should include UI-compatible synthetic filename');
assert.ok(!('model_label' in first.devGolden[0]), 'sampler must not emit fake model labels');
assert.ok(!('image_bytes' in first.devGolden[0]), 'sampler must not emit image bytes');
assert.ok(!('base64' in first.devGolden[0]), 'sampler must not emit base64 payloads');

const overrideTarget = first.devGolden[0];
const overridden = sampler.runDemoReset({
  ...options,
  humanLabelOverrides: [{ sample_id: overrideTarget.sample_id, label: 'not_ai_generated', note: 'SME demo correction' }]
});
const overriddenRecord = overridden.combined.find(row => row.sample_id === overrideTarget.sample_id);
assert.equal(overriddenRecord.human_override_label, 'not_ai_generated', 'sample_id override should attach human label');
assert.equal(overriddenRecord.human_override_label_int, 0, 'human override label_int should be normalized');
assert.equal(overriddenRecord.human_override_note, 'SME demo correction', 'human override note should be retained');
assert.equal(overriddenRecord.human_override_applied, true, 'human override should be flagged');
assert.equal(overridden.leakageChecks.ok, true, 'overrides should not affect leakage checks');

const warm = sampler.runDemoReset({ seed: 42, nDev: 6, nHoldout: 6, mode: 'warm_start' });
assert.equal(warm.summary.mode, 'warm_start', 'warm_start mode should be represented in summary');
assert.equal(warm.devGolden[0].policy_use, 'policy_refinement', 'warm_start dev records should be policy refinement inputs');
assert.equal(warm.holdout[0].policy_use, 'locked_holdout_decision_quality', 'holdout should remain locked for decision quality');

console.log('validate_web_sampler.js: ok');
