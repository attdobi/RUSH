/* Pure contracts shared by the executive UI and dependency-free Node tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RushStudioCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  const scalar = v => typeof v === 'boolean' || typeof v === 'string' || (typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= Number.MAX_SAFE_INTEGER);
  const plain = v => v !== null && typeof v === 'object' && !Array.isArray(v);
  const text = (v, name) => { if (typeof v !== 'string' || !v.trim() || v.length > 500) throw new Error(`Invalid ${name}`); return v; };
  const SOURCES = new Set(['sme', 'dataset', 'system', 'model']);
  function naturalVersions(values) {
    return [...new Set(values.map(v => typeof v === 'string' ? v : v?.version).filter(v => typeof v === 'string' && v))]
      .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
  }
  function normalizeGraph(payload) {
    if (!plain(payload) || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) throw new Error('Invalid graph response: expected nodes and edges arrays.');
    if (!payload.nodes.length || payload.nodes.length > 2000 || payload.edges.length > 10000) throw new Error('Graph size is outside the supported range.');
    const ids = new Set();
    const nodes = payload.nodes.map(n => {
      if (!plain(n)) throw new Error('Invalid graph node');
      text(n.id, 'node id');
      if (ids.has(n.id)) throw new Error(`Duplicate node: ${n.id}`);
      ids.add(n.id);
      return { ...n, title: String(n.title || n.id), node_type: String(n.node_type || 'guideline') };
    });
    const warnings = [], seen = new Set(), edges = [];
    for (const e of payload.edges) {
      if (!plain(e)) { warnings.push('Ignored invalid edge'); continue; }
      const source = e.source_node_id ?? e.source?.id ?? e.source;
      const target = e.target_node_id ?? e.target?.id ?? e.target ?? e.to;
      const type = String(e.edge_type ?? e.type ?? 'related_to');
      if (!ids.has(source) || !ids.has(target)) { warnings.push(`Unresolved edge: ${String(source)} → ${String(target)}`); continue; }
      const key = JSON.stringify([source, target, type, Boolean(e.synthetic)]);
      if (seen.has(key)) continue;
      seen.add(key); edges.push({ ...e, source, target, type, synthetic: Boolean(e.synthetic) });
    }
    // No invented semantic edges to make an orphan appear connected.
    return { ...payload, nodes, edges, warnings };
  }
  function fingerprint(node) {
    return JSON.stringify(['title', 'node_type', 'polarity', 'parent', 'body', 'content_hash', 'status'].map(k => [k, node[k] ?? null]));
  }
  function graphDiff(before, after) {
    const old = new Map((before?.nodes || []).map(n => [n.id, n]));
    const current = new Map(after.nodes.map(n => [n.id, n]));
    return {
      added: after.nodes.filter(n => !old.has(n.id)).map(n => n.id),
      changed: after.nodes.filter(n => old.has(n.id) && fingerprint(old.get(n.id)) !== fingerprint(n)).map(n => n.id),
      removed: [...old.keys()].filter(id => !current.has(id))
    };
  }
  function validateProgram(p) {
    if (!plain(p) || p.schema_version !== 1 || p.mode !== 'shadow') throw new Error('Only schema v1 shadow programs are supported.');
    ['id', 'policy_area', 'policy_version', 'entry'].forEach(k => text(p[k], k));
    if (!Array.isArray(p.nodes) || !p.nodes.length || p.nodes.length > 128) throw new Error('A program needs 1–128 nodes.');
    const byId = new Map();
    for (const n of p.nodes) {
      if (!plain(n)) throw new Error('Invalid decision node');
      text(n.id, 'decision id');
      if (byId.has(n.id)) throw new Error(`Duplicate decision node: ${n.id}`);
      byId.set(n.id, n);
      if (n.kind === 'action') {
        text(n.action, 'action');
        if (n.when != null || n.next != null) throw new Error('Action nodes cannot have predicates or branches.');
      } else if (n.kind === 'rule') {
        text(n.policy_node_id, 'policy node reference');
        if (!plain(n.when) || !plain(n.next)) throw new Error('Rules require predicates and branches.');
        text(n.when.field, 'field');
        if (['__proto__', 'constructor', 'prototype'].includes(n.when.field)) throw new Error('Reserved field');
        if (!['eq', 'gte', 'lte', 'in'].includes(n.when.op)) throw new Error('Unsupported operator');
        const v = n.when.value;
        if (n.when.op === 'in' ? (!Array.isArray(v) || !v.length || v.length > 100 || !v.every(scalar)) : !scalar(v)) throw new Error('Invalid predicate value');
        if (['gte', 'lte'].includes(n.when.op) && (typeof v !== 'number' || !Number.isFinite(v))) throw new Error('Numeric predicates require finite numbers');
        if (!Array.isArray(n.when.sources) || !n.when.sources.length || n.when.sources.some(s => !SOURCES.has(s))) throw new Error('Rules must declare trusted observation sources');
        if (n.when.min_confidence !== undefined && (typeof n.when.min_confidence !== 'number' || !Number.isFinite(n.when.min_confidence) || n.when.min_confidence < 0 || n.when.min_confidence > 1)) throw new Error('Invalid confidence threshold');
        if (Object.keys(n.next).sort().join(',') !== 'false,true,unknown') throw new Error('Every rule requires true, false and unknown branches');
        for (const outcome of ['true', 'false', 'unknown']) text(n.next[outcome], 'branch target');
      } else throw new Error('Unknown decision node kind');
    }
    if (!byId.has(p.entry)) throw new Error('Entry node not found');
    for (const n of p.nodes) if (n.kind === 'rule') for (const target of Object.values(n.next)) if (!byId.has(target)) throw new Error(`Dangling decision edge: ${target}`);
    const active = new Set(), visited = new Set();
    function visit(id) {
      if (active.has(id)) throw new Error('Decision programs must be acyclic');
      if (visited.has(id)) return;
      active.add(id);
      const n = byId.get(id);
      if (n.kind === 'rule') Object.values(n.next).forEach(visit);
      active.delete(id); visited.add(id);
    }
    visit(p.entry);
    if (visited.size !== byId.size) throw new Error('Unreachable decision nodes');
    for (const n of p.nodes) if (n.kind === 'rule') {
      const target = byId.get(n.next.unknown);
      if (target.kind !== 'action' || target.action !== 'sme_review') throw new Error('Unknown evidence must route directly to SME review');
    }
    return byId;
  }
  function predicate(when, facts) {
    if (!plain(facts) || !own(facts, when.field)) return 'unknown';
    const observation = facts[when.field];
    if (!plain(observation) || !own(observation, 'value') || !own(observation, 'source') || !when.sources.includes(observation.source) || !scalar(observation.value)) return 'unknown';
    if (when.min_confidence !== undefined && (!own(observation, 'confidence') || typeof observation.confidence !== 'number' || !Number.isFinite(observation.confidence) || observation.confidence < when.min_confidence || observation.confidence > 1)) return 'unknown';
    const value = observation.value, expected = when.value;
    if (when.op === 'in') {
      if (!expected.some(x => typeof x === typeof value)) return 'unknown';
      return expected.includes(value) ? 'true' : 'false';
    }
    if (typeof value !== typeof expected) return 'unknown';
    if (when.op === 'eq') return value === expected ? 'true' : 'false';
    if (typeof value !== 'number') return 'unknown';
    return (when.op === 'gte' ? value >= expected : value <= expected) ? 'true' : 'false';
  }
  function evaluateProgram(program, facts, context) {
    const nodes = validateProgram(program);
    if (context?.policy_area !== program.policy_area || context?.policy_version !== program.policy_version) throw new Error('Policy pin mismatch: refusing to evaluate a stale program');
    if (!Array.isArray(context.policy_node_ids)) throw new Error('Policy node ids required');
    for (const n of nodes.values()) if (n.kind === 'rule' && !context.policy_node_ids.includes(n.policy_node_id)) throw new Error('Decision rule references a missing policy node');
    const trace = [];
    let id = program.entry;
    for (let step = 0; step <= nodes.size; step++) {
      const n = nodes.get(id);
      if (n.kind === 'action') return { schema_version: 1, program_id: program.id, policy_area: program.policy_area, policy_version: program.policy_version, mode: 'shadow', action: n.action, terminal_id: n.id, trace };
      const outcome = predicate(n.when, facts);
      trace.push({ node_id: n.id, policy_node_id: n.policy_node_id, field: n.when.field, outcome, next: n.next[outcome], source: own(facts || {}, n.when.field) ? facts[n.when.field]?.source ?? null : null });
      id = n.next[outcome];
    }
    throw new Error('Decision step limit exceeded');
  }
  return { naturalVersions, normalizeGraph, graphDiff, validateProgram, predicate, evaluateProgram };
});
