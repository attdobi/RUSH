(() => {
  const WIDTH = 760;
  const HEIGHT = 500;
  const COLORS = {
    root: '#4d9bff',
    positive: '#4de0a6',
    boundary: '#ffb74d',
    exception: '#ff6f91',
    negative: '#ff4d6d',
    provenance: '#4dd0e1',
    fallback: '#888'
  };

  let currentPayload = null;
  let currentVersion = '';
  let currentFocus = null;
  let availableVersions = [];
  const pendingPulseNodeIds = new Set();

  function activeDemo() {
    return typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
  }
  function isMnistDemo() {
    return activeDemo()?.id === 'mnist';
  }
  function policyGraphArea() {
    return activeDemo()?.policyGraph?.area || 'Generative_AI';
  }
  function policyGraphVersion() {
    return activeDemo()?.policyGraph?.version || currentVersion || 'v0.1';
  }
  function policyGraphRootId() {
    return activeDemo()?.policyGraph?.rootId || 'GA.root';
  }
  // Static, no-API path for demos whose policy graph lives entirely in the
  // repo (e.g. MNIST_Digits/v0.1). We read edges.json + per-node .md over
  // regular fetch() and skip /api/policy/*.
  function demoUsesLocalPolicyGraph() {
    return isMnistDemo();
  }
  function localPolicyBase() {
    const demo = activeDemo();
    if (!demo) return '';
    const path = demo.policyGraph?.path || 'policy-graph/MNIST_Digits/v0.1';
    return `../${path}`;
  }

  function qs(selector) {
    return document.querySelector(selector);
  }

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, char => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[char]));
  }

  function status(message, isError = false) {
    rushApiStatus('#policyGraphStatus', message, isError);
  }

  function setUnavailable(message = 'Local API not running — start <code>python scripts/rush_web_server.py</code> to load the policy graph.') {
    const wrap = qs('#policyGraphSvgWrap');
    if (wrap) wrap.innerHTML = `<div class="empty-state">${message}</div>`;
    status('Policy graph unavailable.', true);
  }

  function nodeColor(node) {
    const id = String(node.id || '');
    const type = String(node.node_type || '').toLowerCase();
    const polarity = String(node.polarity || '').toLowerCase();
    if (type === 'root' || id === 'GA.root' || id === 'MD.root') return COLORS.root;
    if (id.startsWith('MD.digit.') || type === 'digit_class') return COLORS.positive;
    if (id.includes('.boundary.') || type === 'boundary') return COLORS.boundary;
    if (id.includes('.exception.') || type === 'exception') return COLORS.exception;
    if (id.includes('.negative.') || polarity === 'negative') return COLORS.negative;
    if (id.includes('.provenance.') || type === 'provenance') return COLORS.provenance;
    if (
      type === 'category' ||
      polarity === 'positive' ||
      id.includes('.scene_geometry.') ||
      id === 'GA.surface_texture' ||
      id.includes('.surface_texture.') ||
      id === 'GA.visual_artifacts' ||
      id.includes('.visual_artifacts.')
    ) return COLORS.positive;
    return COLORS.fallback;
  }

  function edgeSource(edge) {
    return String(edge.source || edge.source_node_id || '');
  }

  function edgeTarget(edge) {
    return String(edge.target || edge.target_node_id || edge.to || '');
  }

  function familyOf(id) {
    const parts = String(id || '').split('.');
    if (parts.length <= 2) return id;
    return `${parts[0]}.${parts[1]}`;
  }

  function depthOf(id) {
    const value = String(id || '');
    if (value === 'GA.root' || value === 'MD.root') return 0;
    return Math.max(1, value.split('.').length - 1);
  }

  function childrenFor(id, nodes) {
    if (!id) return [];
    return nodes.filter(node => node.id !== id && (node.parent === id || String(node.id).startsWith(`${id}.`)));
  }

  function nodeIdFromPolicyFile(path) {
    return String(path || '').split('/').pop().replace(/\.md$/i, '');
  }

  function pulseNodes(nodeSelection, ids) {
    if (!ids?.size) return;
    const circles = nodeSelection.select('circle').filter(d => ids.has(d.id));
    if (circles.empty()) return;
    circles.classed('pulse-new', true);
    window.setTimeout(() => circles.classed('pulse-new', false), 2000);
  }

  function hasChildren(node, nodes) {
    return childrenFor(node.id, nodes).length > 0;
  }

  function nodeRadius(node, nodes) {
    if (node.id === 'GA.root' || node.id === 'MD.root' || String(node.node_type).toLowerCase() === 'root') return 14;
    if (hasChildren(node, nodes) || depthOf(node.id) <= 2) return 10;
    return 7;
  }

  function nodeLabel(node) {
    if (isMnistDemo()) {
      if (node.id === 'MD.root') return 'Prompt root';
      if (node.digit) return String(node.digit);
      const digit = /^MD\.digit\.(\d)$/.exec(String(node.id || ''));
      if (digit) return digit[1];
    }
    return truncate(node.id || node.title || '?', 28);
  }

  function promptVersionLabel(version = currentVersion) {
    return `Generator Prompt ${version || 'v_n'} (rendered)`;
  }

  function truncate(text, max = 26) {
    const value = String(text || '');
    return value.length > max ? `${value.slice(0, max - 1)}…` : value;
  }

  function populateVersions(versions, selected) {
    const select = qs('#policyGraphVersion');
    if (!select) return;
    const list = (Array.isArray(versions) ? versions : [])
      .map(version => version?.version || version)
      .filter(Boolean);
    availableVersions = list;
    if (!list.length) {
      select.innerHTML = rushApiOptionHtml('', 'No generator prompt versions found', true);
      updateVersionStepper('');
      return;
    }
    select.innerHTML = list.map(version => rushApiOptionHtml(version, version, version === selected)).join('');
    select.value = selected || list[list.length - 1];
    updateVersionStepper(select.value);
  }

  function setupVersionStepper() {
    const prev = qs('#policyGraphPrevVersion');
    const next = qs('#policyGraphNextVersion');
    if (prev && prev.dataset.policyStepperReady !== 'true') {
      prev.dataset.policyStepperReady = 'true';
      prev.addEventListener('click', () => stepVersion(-1));
    }
    if (next && next.dataset.policyStepperReady !== 'true') {
      next.dataset.policyStepperReady = 'true';
      next.addEventListener('click', () => stepVersion(1));
    }
  }

  function stepVersion(delta) {
    const select = qs('#policyGraphVersion');
    if (!select || !availableVersions.length) return;
    const current = select.value || currentVersion || availableVersions[availableVersions.length - 1];
    const index = Math.max(0, availableVersions.indexOf(current));
    const nextIndex = index + delta;
    if (nextIndex < 0 || nextIndex >= availableVersions.length) return;
    select.value = availableVersions[nextIndex];
    select.dispatchEvent(new Event('change', { bubbles: true }));
    updateVersionStepper(select.value);
  }

  function updateVersionStepper(selected = currentVersion) {
    setupVersionStepper();
    const prev = qs('#policyGraphPrevVersion');
    const next = qs('#policyGraphNextVersion');
    const note = qs('#policyGraphNextNote');
    const list = availableVersions.filter(Boolean);
    const current = selected || qs('#policyGraphVersion')?.value || currentVersion || list[list.length - 1] || '';
    const index = list.indexOf(current);
    const hasPrevious = index > 0;
    const hasNext = index >= 0 && index < list.length - 1;
    if (prev) {
      prev.disabled = !hasPrevious;
      prev.textContent = hasPrevious ? `Previous: ${list[index - 1]}` : 'Previous';
    }
    if (next) {
      next.disabled = !hasNext;
      next.textContent = hasNext ? `Next: ${list[index + 1]}` : 'Next version';
      next.setAttribute('aria-label', hasNext ? `Load generator prompt ${list[index + 1]}` : 'Accept a policy proposal to materialize the next generator prompt version');
    }
    if (note) {
      note.textContent = hasNext ? 'loaded version available' : 'accept a proposal to materialize the next version';
    }
  }

  async function loadPolicyVersionsForArea() {
    const fallbackVersion = policyGraphVersion();
    if (!window.RUSH_API?.available) {
      return { versions: [fallbackVersion], current: fallbackVersion };
    }
    try {
      const area = policyGraphArea();
      const payload = await rushApiGetJson(`/api/policy/versions?area=${encodeURIComponent(area)}`);
      const versions = Array.isArray(payload.versions) ? payload.versions : [];
      const current = payload.current || versions[versions.length - 1]?.version || versions[versions.length - 1] || fallbackVersion;
      return { versions: versions.length ? versions : [fallbackVersion], current };
    } catch (error) {
      return { versions: [fallbackVersion], current: fallbackVersion };
    }
  }

  function stripFrontmatter(markdown) {
    return String(markdown || '').replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '').trim();
  }

  function parseFrontmatter(markdown) {
    const match = /^---\s*\n([\s\S]*?)\n---\s*\n?/.exec(String(markdown || ''));
    if (!match) return {};
    const fields = {};
    for (const line of match[1].split(/\r?\n/)) {
      const item = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
      if (!item) continue;
      let value = item[2].trim();
      if (value === 'null') value = null;
      else value = value.replace(/^['"]|['"]$/g, '');
      fields[item[1]] = value;
    }
    return fields;
  }

  function markdownSection(markdown, heading) {
    const lines = stripFrontmatter(markdown).split(/\r?\n/);
    const start = lines.findIndex(line => line.trim().toLowerCase() === `## ${heading}`.toLowerCase());
    if (start < 0) return '';
    const end = lines.findIndex((line, index) => index > start && /^##\s+/.test(line.trim()));
    return lines.slice(start + 1, end < 0 ? undefined : end).join('\n').trim();
  }

  function cacheBustedUrl(url) {
    return typeof window.cacheBust === 'function' ? window.cacheBust(url) : url;
  }

  async function fetchNoStore(url) {
    return fetch(cacheBustedUrl(url), { cache: 'no-store' });
  }

  function inlineMarkdown(text) {
    return esc(text)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\[\[([^\]]+)\]\]/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  }

  function renderMarkdown(markdown) {
    const lines = stripFrontmatter(markdown).split(/\r?\n/);
    const html = [];
    let listType = null;
    const closeList = () => {
      if (listType) html.push(`</${listType}>`);
      listType = null;
    };

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        closeList();
        continue;
      }
      const heading = /^(#{1,4})\s+(.+)$/.exec(trimmed);
      if (heading) {
        closeList();
        const level = Math.min(5, heading[1].length + 2);
        html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      const unordered = /^[-*]\s+(.+)$/.exec(trimmed);
      const ordered = /^\d+\.\s+(.+)$/.exec(trimmed);
      if (unordered || ordered) {
        const desired = ordered ? 'ol' : 'ul';
        if (listType !== desired) {
          closeList();
          listType = desired;
          html.push(`<${desired}>`);
        }
        html.push(`<li>${inlineMarkdown((unordered || ordered)[1])}</li>`);
        continue;
      }
      closeList();
      html.push(`<p>${inlineMarkdown(trimmed)}</p>`);
    }
    closeList();
    return html.join('') || '<p class="muted">No markdown body found.</p>';
  }

  function panelShell(node, markdownHtml = '<p class="muted">Loading node markdown…</p>') {
    const color = nodeColor(node);
    const clearButton = currentFocus ? '<button id="policyGraphClear" type="button" aria-label="Clear selection and return to full graph">Clear selection</button>' : '';
    return `${clearButton}
      <div class="policy-node-kicker" style="--node-color:${color}">${esc(node.id)}</div>
      <h3>${esc(node.title || node.id)}</h3>
      <dl class="policy-node-meta">
        <div><dt>Type</dt><dd>${esc(node.node_type || 'unknown')}</dd></div>
        <div><dt>Polarity</dt><dd>${esc(node.polarity || 'mixed')}</dd></div>
        <div><dt>Parent</dt><dd>${esc(node.parent || '—')}</dd></div>
      </dl>
      <div id="policyNodeMarkdown" class="policy-node-markdown">${markdownHtml}</div>`;
  }

  async function openPanel(node) {
    const panel = qs('#policyGraphPanel');
    if (!panel || !node) return;
    const hasMarkdown = typeof node.markdown === 'string' && node.markdown.trim();
    const localMarkdownMissing = demoUsesLocalPolicyGraph() && node.markdown_loaded === false;
    const initialMarkdown = hasMarkdown
      ? renderMarkdown(node.markdown)
      : localMarkdownMissing
        ? '<p class="muted">No markdown body found.</p>'
        : '<p class="muted">Loading node markdown…</p>';
    panel.innerHTML = panelShell(node, initialMarkdown);
    qs('#policyGraphClear')?.addEventListener('click', () => applySelection(null));
    if (hasMarkdown || localMarkdownMissing) return;

    try {
      const version = currentVersion || currentPayload?.version || policyGraphVersion();
      // Local-file demos (mnist) load static Markdown from the repo; GenAI keeps
      // the /policy-graph/... HTTP path served by the local API server.
      const path = demoUsesLocalPolicyGraph()
        ? `${localPolicyBase()}/${encodeURIComponent(node.id)}.md`
        : `/policy-graph/${encodeURIComponent(policyGraphArea())}/${encodeURIComponent(version)}/${encodeURIComponent(node.id)}.md`;
      const response = await fetchNoStore(path);
      if (!response.ok) throw new Error(`Markdown not found (${response.status})`);
      const markdown = await response.text();
      node.markdown = markdown;
      const body = qs('#policyNodeMarkdown');
      if (body) body.innerHTML = renderMarkdown(markdown);
    } catch (error) {
      const body = qs('#policyNodeMarkdown');
      if (body) body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    }
  }

  function ancestorChain(id, nodes) {
    const byId = new Map(nodes.map(n => [n.id, n]));
    const chain = new Set();
    let cur = byId.get(id);
    while (cur && cur.parent && cur.parent !== cur.id) {
      if (chain.has(cur.parent)) break; // safety: avoid cycles
      chain.add(cur.parent);
      cur = byId.get(cur.parent);
    }
    return chain;
  }

  function descendantSet(id, nodes) {
    const byParent = new Map();
    nodes.forEach(n => {
      if (!n.parent) return;
      if (!byParent.has(n.parent)) byParent.set(n.parent, []);
      byParent.get(n.parent).push(n.id);
    });
    const out = new Set();
    const stack = [id];
    while (stack.length) {
      const cur = stack.pop();
      (byParent.get(cur) || []).forEach(child => {
        if (!out.has(child)) { out.add(child); stack.push(child); }
      });
    }
    // Fallback: also walk id-prefix descendants in case `parent` field is missing
    nodes.forEach(n => { if (n.id !== id && String(n.id).startsWith(`${id}.`)) out.add(n.id); });
    return out;
  }

  function applySelection(id) {
    currentFocus = id || null;
    const panel = qs('#policyGraphPanel');
    // Clear all persistent selection classes first (hover classes are untouched)
    d3.selectAll('.policy-node').classed('selected ancestor descendant dimmed', false);
    d3.selectAll('.policy-link').classed('ancestor-edge descendant-edge dimmed', false);
    if (!id) {
      if (panel) panel.innerHTML = '<h3>Policy node details</h3><p class="muted">Hover a node to trace its neighbors. Click a node to see its lineage to the root and read its Markdown.</p>';
      return;
    }
    const nodes = currentPayload?.nodes || [];
    const ancestors = ancestorChain(id, nodes);
    const descendants = descendantSet(id, nodes);
    const related = new Set([id, ...ancestors, ...descendants]);
    d3.selectAll('.policy-node')
      .classed('selected', d => d.id === id)
      .classed('ancestor', d => ancestors.has(d.id))
      .classed('descendant', d => descendants.has(d.id))
      .classed('dimmed', d => !related.has(d.id));
    d3.selectAll('.policy-link')
      .classed('ancestor-edge', d => (ancestors.has(d.sourceId) && (d.targetId === id || ancestors.has(d.targetId))) || (ancestors.has(d.targetId) && (d.sourceId === id || ancestors.has(d.sourceId))))
      .classed('descendant-edge', d => (descendants.has(d.sourceId) || d.sourceId === id) && (descendants.has(d.targetId) || d.targetId === id))
      .classed('dimmed', d => !(related.has(d.sourceId) && related.has(d.targetId)));
    const node = nodes.find(n => n.id === id);
    if (node) openPanel(node);
  }

  function renderGraph(payload, focusId = null) {
    currentPayload = payload;
    currentVersion = payload.version || currentVersion;
    currentFocus = focusId;

    const wrap = qs('#policyGraphSvgWrap');
    if (!wrap) return;
    if (!window.d3) {
      setUnavailable('D3 failed to load from the CDN; policy graph cannot render.');
      return;
    }

    // Always render the full graph. focusId is now selection state only, not a filter.
    const allNodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    const nodes = allNodes.map(node => ({ ...node }));
    const nodeSet = new Set(nodes.map(n => n.id));
    const links = (Array.isArray(payload.edges) ? payload.edges : [])
      .map(edge => ({ ...edge, source: edgeSource(edge), target: edgeTarget(edge), sourceId: edgeSource(edge), targetId: edgeTarget(edge) }))
      .filter(edge => nodeSet.has(edge.sourceId) && nodeSet.has(edge.targetId));

    const legendBase = isMnistDemo()
      ? [
          ['root', COLORS.root],
          ['digit_class', COLORS.positive],
          ['other', COLORS.fallback]
        ]
      : [
          ['root', COLORS.root],
          ['positive', COLORS.positive],
          ['boundary', COLORS.boundary],
          ['exception', COLORS.exception],
          ['negative', COLORS.negative],
          ['provenance', COLORS.provenance],
          ['other', COLORS.fallback]
        ];
    const legendItems = legendBase
      .map(([label, color]) => `<span><i style="background:${color}"></i>${esc(label)}</span>`)
      .join('');
    const edgeLegend = links.some(edge => String(edge.type || edge.edge_type || '').toLowerCase() === 'confused_with')
      ? '<span><i class="policy-legend-line confused-with"></i>confused_with</span>'
      : '';

    wrap.innerHTML = `<div class="policy-graph-render-caption">
        <span>${esc(promptVersionLabel(payload.version || currentVersion))}</span>
        <p>The rendered graph below is the active generator prompt. Accepted SME diffs create later versions; future growth slots stay explicit until executed.</p>
      </div>
      <div class="policy-graph-layout">
        <div class="policy-graph-canvas" aria-label="Interactive policy force graph">
          <svg id="policyGraphSvg" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="Generator prompt policy graph ${esc(payload.version || '')}"></svg>
        </div>
        <aside id="policyGraphPanel" class="policy-graph-panel" aria-live="polite">
          <h3>Policy node details</h3>
          <p class="muted">Hover a node to trace its neighbors. Click a node to see its lineage to the root and read its Markdown.</p>
        </aside>
      </div>
      <div class="policy-graph-legend">${legendItems}${edgeLegend}</div>`;

    qs('#policyGraphTitle').textContent = payload.title || 'Generator Prompt v_n (policy graph)';

    const svg = d3.select('#policyGraphSvg');
    const viewport = svg.append('g').attr('class', 'policy-graph-viewport');
    svg.call(d3.zoom().scaleExtent([0.55, 4]).on('zoom', event => viewport.attr('transform', event.transform)));

    viewport.append('rect')
      .attr('width', WIDTH)
      .attr('height', HEIGHT)
      .attr('rx', 16)
      .attr('fill', '#0e1219');

    const neighborMap = new Map();
    nodes.forEach(node => neighborMap.set(node.id, new Set([node.id])));
    links.forEach(link => {
      neighborMap.get(link.sourceId)?.add(link.targetId);
      neighborMap.get(link.targetId)?.add(link.sourceId);
    });

    function linkClass(edge) {
      const type = String(edge.type || edge.edge_type || '').toLowerCase();
      if (type === 'confused_with') return 'confused-with';
      const sourceFamily = familyOf(edge.sourceId);
      const targetFamily = familyOf(edge.targetId);
      if (edge.sourceId === 'GA.root' || edge.targetId === 'GA.root') return 'root-link';
      if (edge.sourceId === 'MD.root' || edge.targetId === 'MD.root') return 'root-link';
      return sourceFamily === targetFamily ? 'same-family' : 'cross-family';
    }

    const link = viewport.append('g')
      .attr('class', 'policy-links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('class', edge => `policy-link ${linkClass(edge)}`)
      .attr('stroke-width', edge => linkClass(edge) === 'same-family' ? 1.8 : 1.2);

    const node = viewport.append('g')
      .attr('class', 'policy-nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', d => `policy-node ${d.id === 'GA.root' || hasChildren(d, allNodes) ? 'parent-node' : 'leaf-node'}`)
      .attr('tabindex', 0)
      .attr('role', 'button')
      .attr('aria-label', d => `${d.id}: ${d.title || d.id}`)
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', d => nodeRadius(d, allNodes))
      .attr('fill', '#111927')
      .attr('stroke', d => nodeColor(d))
      .attr('stroke-width', 2.4);

    node.append('text')
      .attr('y', d => nodeRadius(d, allNodes) + 13)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9.5)
      .style('opacity', 1)
      .text(d => nodeLabel(d));

    node.on('mouseover', (_, d) => highlight(d.id))
      .on('mouseout', clearHighlight)
      .on('click', (event, d) => {
        event.stopPropagation();
        applySelection(d.id);
      })
      .on('keydown', (event, d) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          applySelection(d.id);
        }
      });

    const digitNodes = nodes.filter(d => String(d.id || '').startsWith('MD.digit.'));
    const digitOrder = new Map(digitNodes
      .slice()
      .sort((a, b) => Number(a.digit ?? String(a.id).split('.').pop()) - Number(b.digit ?? String(b.id).split('.').pop()))
      .map((d, index) => [d.id, index]));
    if (isMnistDemo()) {
      const radius = 178;
      nodes.forEach(d => {
        if (d.id === 'MD.root') {
          d.x = WIDTH / 2;
          d.y = HEIGHT / 2;
          return;
        }
        const index = digitOrder.get(d.id);
        if (Number.isFinite(index)) {
          const angle = (-Math.PI / 2) + (index / Math.max(1, digitOrder.size)) * Math.PI * 2;
          d.x = WIDTH / 2 + Math.cos(angle) * radius;
          d.y = HEIGHT / 2 + Math.sin(angle) * radius;
        }
      });
    }

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id)
        .distance(edge => String(edge.type || edge.edge_type || '').toLowerCase() === 'confused_with' ? 175 : 104)
        .strength(edge => String(edge.type || edge.edge_type || '').toLowerCase() === 'confused_with' ? 0.18 : 0.62))
      .force('charge', d3.forceManyBody().strength(isMnistDemo() ? -360 : -230))
      .force('center', d3.forceCenter(WIDTH / 2, HEIGHT / 2))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d, allNodes) + (isMnistDemo() ? 46 : hasChildren(d, allNodes) ? 42 : 28)));

    if (isMnistDemo()) {
      simulation
        .force('x', d3.forceX(d => {
          if (d.id === 'MD.root') return WIDTH / 2;
          const index = digitOrder.get(d.id);
          if (!Number.isFinite(index)) return WIDTH / 2;
          const angle = (-Math.PI / 2) + (index / Math.max(1, digitOrder.size)) * Math.PI * 2;
          return WIDTH / 2 + Math.cos(angle) * 178;
        }).strength(0.18))
        .force('y', d3.forceY(d => {
          if (d.id === 'MD.root') return HEIGHT / 2;
          const index = digitOrder.get(d.id);
          if (!Number.isFinite(index)) return HEIGHT / 2;
          const angle = (-Math.PI / 2) + (index / Math.max(1, digitOrder.size)) * Math.PI * 2;
          return HEIGHT / 2 + Math.sin(angle) * 178;
        }).strength(0.18));
    }

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    const visiblePulseIds = new Set(nodes.map(d => d.id).filter(id => pendingPulseNodeIds.has(id)));
    if (visiblePulseIds.size) {
      let didPulse = false;
      const triggerPulse = () => {
        if (didPulse) return;
        didPulse = true;
        pulseNodes(node, visiblePulseIds);
        visiblePulseIds.forEach(id => pendingPulseNodeIds.delete(id));
      };
      simulation.on('end.pulseNew', triggerPulse);
      window.setTimeout(triggerPulse, 900);
    }

    function highlight(id) {
      const neighbors = neighborMap.get(id) || new Set([id]);
      // Use hover-only classes so we never clobber persistent selection classes.
      node.classed('hover-dim', d => !neighbors.has(d.id))
        .classed('hover-trace', d => neighbors.has(d.id));
      link.classed('hover-dim', d => d.sourceId !== id && d.targetId !== id)
        .classed('hover-trace', d => d.sourceId === id || d.targetId === id);
    }

    function clearHighlight() {
      // Only remove hover classes; selection classes (set by applySelection) stay intact.
      node.classed('hover-dim hover-trace', false);
      link.classed('hover-dim hover-trace', false);
    }

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }

  function openPolicyNodeById(id) {
    const nodeId = String(id || '');
    if (!nodeId || !currentPayload) return false;
    const node = (currentPayload.nodes || []).find(item => item.id === nodeId);
    if (!node) return false;
    applySelection(nodeId);
    qs('#grow')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return true;
  }

  window.rushOpenPolicyNode = openPolicyNodeById;

  // Defensive backfill: ensure every node has a path to GA.root so the d3 view
  // never shows orphaned floats when a proposal-added node lands without an
  // explicit subtype_of edge in its frontmatter.
  function backfillParentEdges(payload) {
    const nodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
    const edges = Array.isArray(payload?.edges) ? payload.edges : [];
    const nodeIds = new Set(nodes.map(n => n.id));
    const existing = new Set(edges.map(e => `${edgeSource(e)}\u2192${edgeTarget(e)}`));
    let addedExplicit = 0;
    let addedPrefix = 0;
    let addedRoot = 0;
    function add(from, to, kind) {
      const key = `${from}\u2192${to}`;
      if (from === to || !nodeIds.has(from) || !nodeIds.has(to) || existing.has(key)) return false;
      edges.push({ source: from, target: to, type: kind, synthetic: true });
      existing.add(key);
      return true;
    }
    function hasAnyEdge(id) {
      for (const edge of edges) {
        if (edgeSource(edge) === id || edgeTarget(edge) === id) return true;
      }
      return false;
    }
    nodes.forEach(node => {
      if (!node || node.id === 'GA.root') return;
      // 1. Honor explicit parent frontmatter when the target exists in this payload.
      if (node.parent && nodeIds.has(node.parent)) {
        if (add(node.id, node.parent, 'subtype_of_inferred')) addedExplicit += 1;
        return;
      }
      // 2. Fall back to id-prefix parent (drop dotted segments until something matches).
      const parts = String(node.id).split('.');
      let attached = false;
      for (let i = parts.length - 1; i > 0; i -= 1) {
        const candidate = parts.slice(0, i).join('.');
        if (nodeIds.has(candidate)) {
          if (add(node.id, candidate, 'subtype_of_inferred')) addedPrefix += 1;
          attached = true;
          break;
        }
      }
      if (attached) return;
      // 3. Last resort: attach to GA.root so no node ever floats on stage.
      if (!hasAnyEdge(node.id) && nodeIds.has('GA.root')) {
        if (add(node.id, 'GA.root', 'subtype_of_inferred')) addedRoot += 1;
      }
    });
    payload.edges = edges;
    if (addedExplicit || addedPrefix || addedRoot) {
      payload._backfilled_edges = { explicit: addedExplicit, prefix: addedPrefix, root: addedRoot };
    }
    return payload;
  }

  // Build a static payload for a demo that keeps its policy graph in the
  // repo. Reads edges.json for subtype_of edges, then adds virtual
  // confused_with edges from the active demo config so the boundary pairs
  // show up as dashed cross-links.
  async function loadStaticLocalGraph() {
    const demo = activeDemo();
    const base = localPolicyBase();
    const rootId = policyGraphRootId();
    const classes = Array.isArray(demo.classes) ? demo.classes : [];
    const nodeIdFor = typeof demo.classNodeId === 'function' ? demo.classNodeId : (c => `MD.digit.${c}`);
    const version = demo.policyGraph?.version || 'v0.1';
    const hydrateNodeFromMarkdown = async fallback => {
      const url = `${base}/${encodeURIComponent(fallback.id)}.md`;
      try {
        const response = await fetchNoStore(url);
        if (!response.ok) throw new Error(`Markdown not found (${response.status})`);
        const markdown = await response.text();
        const frontmatter = parseFrontmatter(markdown);
        return {
          ...fallback,
          ...frontmatter,
          id: frontmatter.id || fallback.id,
          title: frontmatter.title || fallback.title,
          node_type: frontmatter.node_type || fallback.node_type,
          polarity: frontmatter.polarity || fallback.polarity,
          parent: Object.prototype.hasOwnProperty.call(frontmatter, 'parent') ? frontmatter.parent : fallback.parent,
          version: frontmatter.version || fallback.version || version,
          markdown,
          markdown_body: stripFrontmatter(markdown),
          positive_criteria: markdownSection(markdown, 'Positive criteria'),
          distinguishing_features: markdownSection(markdown, 'Distinguishing features'),
          confused_with: markdownSection(markdown, 'Hard negatives / confusions'),
          markdown_loaded: true
        };
      } catch (error) {
        return {
          ...fallback,
          version: fallback.version || version,
          markdown: '',
          markdown_error: error.message,
          markdown_loaded: false
        };
      }
    };
    // 1) Fetch edges.json (subtype_of edges).
    let baseEdges = [];
    try {
      const url = `${base}/edges.json`;
      const response = await fetchNoStore(url);
      if (response.ok) baseEdges = await response.json();
    } catch (error) {
      baseEdges = [];
    }
    // 2) Build nodes from their live markdown files, falling back to stubs if missing.
    const nodeStubs = [
      { id: rootId, title: 'MNIST Digits — root', node_type: 'root', polarity: 'mixed', parent: null, version },
      ...classes.map(cls => ({
        id: nodeIdFor(cls),
        title: `Digit ${cls}`,
        node_type: 'digit_class',
        polarity: 'positive',
        parent: rootId,
        digit: cls,
        version
      }))
    ];
    const nodes = await Promise.all(nodeStubs.map(hydrateNodeFromMarkdown));
    // 3) Confused_with virtual edges from demos.js confusionPairs.
    const confusedEdges = (demo.confusionPairs || []).flatMap(entry => {
      const [a, b] = entry.pair || [];
      if (!a || !b) return [];
      return [{ source_node_id: nodeIdFor(a), target_node_id: nodeIdFor(b), edge_type: 'confused_with', provenance: 'demo config', synthetic: true }];
    });
    return {
      title: demo.sectionCopy?.policyGraphTitle || 'MNIST Generator Prompt v0.1 (policy graph)',
      version,
      available_versions: [version],
      nodes,
      edges: [...baseEdges, ...confusedEdges]
    };
  }

  async function loadGraph(version = '') {
    // Static-local demos (mnist) can render straight from repo files when the
    // API is unavailable. When the API is present, still prefer the area-scoped
    // /api/policy/* endpoints so selectors never mix versions across demos.
    if (demoUsesLocalPolicyGraph() && !window.RUSH_API?.available) {
      try {
        status('Loading policy graph…');
        const payload = await loadStaticLocalGraph();
        backfillParentEdges(payload);
        currentFocus = null;
        currentVersion = payload.version || version;
        populateVersions(payload.available_versions, payload.version);
        renderGraph(payload, null);
        const confusedCount = payload.edges.filter(e => (e.edge_type || e.type) === 'confused_with').length;
        status(`Loaded ${payload.nodes?.length || 0} node(s), ${payload.edges?.length || 0} edge(s) (${confusedCount} confused_with).`);
      } catch (error) {
        const wrap = qs('#policyGraphSvgWrap');
        if (wrap) wrap.innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
        status(`Policy graph failed: ${error.message}`, true);
      }
      return;
    }
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      const params = new URLSearchParams();
      if (version) params.set('version', version);
      params.set('area', policyGraphArea());
      const query = `?${params.toString()}`;
      status('Loading policy graph…');
      const payload = await rushApiGetJson(`/api/policy/graph${query}`);
      backfillParentEdges(payload);
      currentFocus = null;
      currentVersion = payload.version || version;
      populateVersions(payload.available_versions || window.RUSH_API?.catalog?.policyVersions || [currentVersion], payload.version || currentVersion);
      renderGraph(payload, null);
      const backfilled = payload._backfilled_edges;
      const backfillNote = backfilled ? ` · backfilled ${backfilled.explicit + backfilled.prefix + backfilled.root} parent edge(s)` : '';
      status(`Loaded ${payload.nodes?.length || 0} node(s), ${payload.edges?.length || 0} edge(s)${backfillNote}.`);
    } catch (error) {
      if (demoUsesLocalPolicyGraph()) {
        try {
          const payload = await loadStaticLocalGraph();
          backfillParentEdges(payload);
          currentFocus = null;
          currentVersion = payload.version || version;
          populateVersions(payload.available_versions, payload.version);
          renderGraph(payload, null);
          status(`Loaded local fallback ${payload.nodes?.length || 0} node(s), ${payload.edges?.length || 0} edge(s).`);
          return;
        } catch (fallbackError) {
          // Surface the original API error below; fallback only improves offline/old-backend demos.
        }
      }
      const wrap = qs('#policyGraphSvgWrap');
      if (wrap) wrap.innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Policy graph failed: ${error.message}`, true);
    }
  }

  async function initPolicyGraph(api) {
    // Static-local demos (mnist) render from repo files if the API is offline.
    if (demoUsesLocalPolicyGraph() && !api.available) {
      qs('#policyGraphVersion')?.addEventListener('change', event => loadGraph(event.target.value));
      await loadGraph(policyGraphVersion());
      return;
    }
    if (!api.available) {
      setUnavailable();
      return;
    }
    const versionPayload = await loadPolicyVersionsForArea();
    populateVersions(versionPayload.versions, versionPayload.current);
    if (window.RUSH_API?.catalog) {
      window.RUSH_API.catalog.policyVersions = versionPayload.versions;
      window.RUSH_API.catalog.currentPolicyVersion = versionPayload.current;
    }
    const selected = qs('#policyGraphVersion')?.value || versionPayload.current || '';
    qs('#policyGraphVersion')?.addEventListener('change', event => loadGraph(event.target.value));
    await loadGraph(selected);
  }

  rushApiOnReady(initPolicyGraph);
  window.addEventListener('rush-api-catalog', event => {
    const versions = event.detail?.policyVersions || [];
    const latestItem = versions[versions.length - 1];
    const latest = event.detail?.currentPolicyVersion || latestItem?.version || latestItem || '';
    if (latest && latest !== currentVersion) loadGraph(latest);
  });
  window.addEventListener('rush-policy-accepted', event => {
    const files = [
      ...(Array.isArray(event.detail?.files_added) ? event.detail.files_added : []),
      ...(Array.isArray(event.detail?.files_changed) ? event.detail.files_changed : [])
    ];
    files.map(nodeIdFromPolicyFile).filter(Boolean).forEach(id => pendingPulseNodeIds.add(id));
  });
})();
