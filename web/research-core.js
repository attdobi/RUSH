/* Pure research-view contracts. No network, DOM, model calls, or mutations. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.RushResearchCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  'use strict';
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const ratio = (a, b) => b > 0 ? a / b : null;
  const mean = values => { const xs = values.filter(finite); return xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : null; };
  function graph(raw) {
    if (!raw || !Array.isArray(raw.nodes) || !Array.isArray(raw.edges) || !raw.nodes.length || raw.nodes.length > 2000) throw new Error('Invalid or oversized graph');
    const ids = new Set(), warnings = [...(Array.isArray(raw.warnings) ? raw.warnings : [])];
    const nodes = raw.nodes.map(n => {
      if (!n || typeof n.id !== 'string' || !n.id || ids.has(n.id)) throw new Error('Missing or duplicate node id');
      ids.add(n.id);
      return { ...n, title: String(n.title || n.id), body: String(n.body ?? n.markdown ?? '') };
    });
    const seen = new Set(), edges = [];
    for (const e of raw.edges) {
      if (!e || typeof e !== 'object') { warnings.push('Invalid edge omitted'); continue; }
      const source = e.source_node_id ?? e.source?.id ?? e.source;
      const target = e.target_node_id ?? e.target?.id ?? e.target ?? e.to;
      const type = String(e.edge_type ?? e.type ?? 'related_to');
      if (!ids.has(source) || !ids.has(target)) { warnings.push(`Unresolved edge: ${String(source)} → ${String(target)}`); continue; }
      const key = JSON.stringify([source, target, type, Boolean(e.synthetic)]);
      if (!seen.has(key)) { edges.push({ ...e, source, target, type, key }); seen.add(key); }
    }
    // Graph topology comes only from the artifacts; visual proximity is not an edge.
    return { ...raw, nodes, edges, warnings };
  }
  function signature(n) { return JSON.stringify([n.title, n.node_type, n.parent, n.polarity, n.body, n.content_hash]); }
  function difference(before, after) {
    const old = new Map((before?.nodes || []).map(n => [n.id, n]));
    const now = new Map(after.nodes.map(n => [n.id, n]));
    return { added: after.nodes.filter(n => !old.has(n.id)),
      changed: after.nodes.filter(n => old.has(n.id) && signature(n) !== signature(old.get(n.id))),
      removed: [...old.values()].filter(n => !now.has(n.id)) };
  }
  function ancestry(id, nodes) {
    const map = new Map(nodes.map(n => [n.id, n])), seen = new Set(), path = [];
    let node = map.get(id);
    while (node && !seen.has(node.id)) { seen.add(node.id); path.unshift(node); node = map.get(node.parent); }
    return { path, cycle: Boolean(node) };
  }
  function initialLayout(nodes, mode = 'network') {
    const ordered = [...nodes].sort((a, b) => a.id.localeCompare(b.id));
    const map = new Map(ordered.map(n => [n.id, n]));
    const hash = id => [...id].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 19) / 4294967296;
    const roots = ordered.filter(n => !n.parent || !map.has(n.parent));
    const rootIds = new Set(roots.map(n => n.id));
    const branches = ordered.filter(n => rootIds.has(n.parent));
    const levels = new Map();
    for (const n of ordered) {
      const depth = Math.max(0, ancestry(n.id, ordered).path.length - 1);
      if (!levels.has(depth)) levels.set(depth, []);
      levels.get(depth).push(n.id);
    }
    return new Map(ordered.map(n => {
      const chain = ancestry(n.id, ordered).path, depth = chain.length - 1;
      if (mode === 'hierarchy') {
        const level = levels.get(depth), i = level.indexOf(n.id);
        return [n.id, { x: 80 + depth / Math.max(1, levels.size - 1) * 790, y: 42 + (i + .5) / level.length * 490 }];
      }
      if (rootIds.has(n.id)) return [n.id, { x: 475 + (roots.length > 1 ? Math.cos(hash(n.id) * 6.28) * 95 : 0), y: 287 + (roots.length > 1 ? Math.sin(hash(n.id) * 6.28) * 75 : 0) }];
      const family = chain[1] || n, ix = branches.findIndex(b => b.id === family.id);
      const angle = ix < 0 ? hash(n.id) * Math.PI * 2 : ix / Math.max(branches.length, 1) * Math.PI * 2 - Math.PI / 2;
      const jitter = depth > 1 ? (hash(n.id) - .5) * 1.05 : 0;
      const radius = depth <= 1 ? 182 : Math.min(405, 225 + depth * 48);
      return [n.id, { x: 475 + Math.cos(angle + jitter) * radius, y: 287 + Math.sin(angle + jitter) * radius * .61 }];
    }));
  }
  function relax(points, edges, targets, steps = 1) {
    const all = [...points.values()];
    if (all.length > 350) return; // Large snapshots stay on the bounded deterministic layout.
    for (let step = 0; step < steps; step++) {
      for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
        const a = all[i], b = all[j]; let dx = a.x - b.x, dy = a.y - b.y;
        if (dx === 0 && dy === 0) dx = .1;
        const d2 = Math.max(100, dx * dx + dy * dy), d = Math.sqrt(d2);
        if (d < 135) { const f = Math.min(3.5, 300 / d2 + Math.max(0, 62 - d) * .08); a.x += dx / d * f; a.y += dy / d * f; b.x -= dx / d * f; b.y -= dy / d * f; }
        // Labels sit below nodes: separate overlapping label baselines without
        // inventing graph links or changing the meaning of a node's position.
        if (Math.abs(dx) < 135 && Math.abs(dy) < 29) {
          const push = (29 - Math.abs(dy)) * .11, sign = dy >= 0 ? 1 : -1;
          a.y += sign * push; b.y -= sign * push;
        }
      }
      for (const e of edges) {
        const a = points.get(e.source), b = points.get(e.target); if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.max(1, Math.hypot(dx, dy));
        if (d > 140) { const f = (d - 140) * .004; a.x += dx / d * f; a.y += dy / d * f; b.x -= dx / d * f; b.y -= dy / d * f; }
        // Labels sit below nodes: separate overlapping label baselines without
        // inventing graph links or changing the meaning of a node's position.
        if (Math.abs(dx) < 135 && Math.abs(dy) < 29) {
          const push = (29 - Math.abs(dy)) * .11, sign = dy >= 0 ? 1 : -1;
          a.y += sign * push; b.y -= sign * push;
        }
      }
      for (const [id, p] of points) {
        const t = targets.get(id); if (t) { p.x += (t.x - p.x) * .018; p.y += (t.y - p.y) * .018; }
        p.x = Math.max(40, Math.min(910, p.x)); p.y = Math.max(35, Math.min(545, p.y));
      }
    }
  }
  function wilson(successes, n, z = 1.959963984540054) {
    if (!Number.isSafeInteger(successes) || !Number.isSafeInteger(n) || n <= 0 || successes < 0 || successes > n || !finite(z) || z <= 0) return null;
    const p = successes / n, z2 = z * z, den = 1 + z2 / n;
    const center = (p + z2 / (2 * n)) / den;
    const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den;
    return [successes === 0 ? 0 : Math.max(0, center - half), successes === n ? 1 : Math.min(1, center + half)];
  }
  function fromCounts(row, classes) {
    const cm = row?.confusion_matrix;
    if (!cm || !Array.isArray(classes) || !classes.length || new Set(classes).size !== classes.length) return null;
    const entries = [];
    for (const [truth, preds] of Object.entries(cm)) {
      if (!classes.includes(truth) || !preds || typeof preds !== 'object' || Array.isArray(preds)) return null;
      for (const [pred, count] of Object.entries(preds)) {
        if (!Number.isSafeInteger(count) || count < 0) return null;
        entries.push([truth, pred, count]);
      }
    }
    const n = entries.reduce((s, e) => s + e[2], 0);
    if (!Number.isSafeInteger(n) || !n) return null;
    if (row.n !== undefined && row.n !== n) return null;
    const perClass = classes.map(label => {
      let tp = 0, fp = 0, fn = 0, tn = 0;
      for (const [truth, pred, count] of entries) {
        if (truth === label && pred === label) tp += count;
        else if (pred === label) fp += count;
        else if (truth === label) fn += count;
        else tn += count;
      }
      return { label, tp, fp, fn, tn, f1: ratio(2 * tp, 2 * tp + fp + fn), fpr: ratio(fp, fp + tn), fnr: ratio(fn, fn + tp),
        fpr_interval: wilson(fp, fp + tn), fnr_interval: wilson(fn, fn + tp) };
    });
    const abstained = Number.isSafeInteger(row.n_abstained) && row.n_abstained >= 0 ? row.n_abstained : null;
    return { n, per_class: perClass, accuracy: entries.reduce((s, [t, p, v]) => s + (t === p ? v : 0), 0) / n,
      macro_f1: mean(perClass.map(c => c.f1)), macro_fpr: mean(perClass.map(c => c.fpr)), macro_fnr: mean(perClass.map(c => c.fnr)),
      coverage: abstained === null ? null : ratio(n, n + abstained), source: 'recomputed_from_counts' };
  }
  return { finite, graph, difference, ancestry, initialLayout, relax, wilson, fromCounts };
});
