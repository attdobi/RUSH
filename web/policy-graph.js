(() => {
  const WIDTH = 720;
  const HEIGHT = 460;
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
    if (type === 'root' || id === 'GA.root') return COLORS.root;
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
    if (value === 'GA.root') return 0;
    return Math.max(1, value.split('.').length - 1);
  }

  function childrenFor(id, nodes) {
    if (!id) return [];
    return nodes.filter(node => node.id !== id && (node.parent === id || String(node.id).startsWith(`${id}.`)));
  }

  function hasChildren(node, nodes) {
    return childrenFor(node.id, nodes).length > 0;
  }

  function nodeRadius(node, nodes) {
    if (node.id === 'GA.root' || String(node.node_type).toLowerCase() === 'root') return 14;
    if (hasChildren(node, nodes) || depthOf(node.id) <= 2) return 10;
    return 7;
  }

  function truncate(text, max = 26) {
    const value = String(text || '');
    return value.length > max ? `${value.slice(0, max - 1)}…` : value;
  }

  function populateVersions(versions, selected) {
    const select = qs('#policyGraphVersion');
    if (!select) return;
    const list = Array.isArray(versions) ? versions : [];
    if (!list.length) {
      select.innerHTML = rushApiOptionHtml('', 'No policy versions found', true);
      return;
    }
    select.innerHTML = list.map(version => rushApiOptionHtml(version, version, version === selected)).join('');
    select.value = selected || list[list.length - 1];
  }

  function stripFrontmatter(markdown) {
    return String(markdown || '').replace(/^---\s*\n[\s\S]*?\n---\s*\n/, '').trim();
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
    const backButton = currentFocus ? '<button id="policyGraphBack" type="button">Back to full graph</button>' : '';
    return `${backButton}
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
    panel.innerHTML = panelShell(node);
    qs('#policyGraphBack')?.addEventListener('click', () => renderGraph(currentPayload, null));

    try {
      const version = currentVersion || currentPayload?.version || '';
      const path = `/policy-graph/Generative_AI/${encodeURIComponent(version)}/${encodeURIComponent(node.id)}.md`;
      const response = await fetch(path);
      if (!response.ok) throw new Error(`Markdown not found (${response.status})`);
      const markdown = await response.text();
      const body = qs('#policyNodeMarkdown');
      if (body) body.innerHTML = renderMarkdown(markdown);
    } catch (error) {
      const body = qs('#policyNodeMarkdown');
      if (body) body.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
    }
  }

  function graphSubset(payload, focusId) {
    const allNodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    if (!focusId) return allNodes;
    const keep = new Set([focusId]);
    allNodes.forEach(node => {
      if (node.id === focusId || node.parent === focusId || String(node.id).startsWith(`${focusId}.`)) keep.add(node.id);
    });
    return allNodes.filter(node => keep.has(node.id));
  }

  function renderGraph(payload, focusId = currentFocus) {
    currentPayload = payload;
    currentVersion = payload.version || currentVersion;
    currentFocus = focusId;

    const wrap = qs('#policyGraphSvgWrap');
    if (!wrap) return;
    if (!window.d3) {
      setUnavailable('D3 failed to load from the CDN; policy graph cannot render.');
      return;
    }

    const allNodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    const nodeSet = new Set(graphSubset(payload, focusId).map(node => node.id));
    const nodes = allNodes.filter(node => nodeSet.has(node.id)).map(node => ({ ...node }));
    const links = (Array.isArray(payload.edges) ? payload.edges : [])
      .map(edge => ({ ...edge, source: edgeSource(edge), target: edgeTarget(edge), sourceId: edgeSource(edge), targetId: edgeTarget(edge) }))
      .filter(edge => nodeSet.has(edge.sourceId) && nodeSet.has(edge.targetId));

    const legendItems = [
      ['root', COLORS.root],
      ['positive', COLORS.positive],
      ['boundary', COLORS.boundary],
      ['exception', COLORS.exception],
      ['negative', COLORS.negative],
      ['provenance', COLORS.provenance],
      ['other', COLORS.fallback]
    ].map(([label, color]) => `<span><i style="background:${color}"></i>${esc(label)}</span>`).join('');

    wrap.innerHTML = `<div class="policy-graph-layout">
        <div class="policy-graph-canvas" aria-label="Interactive policy force graph">
          <svg id="policyGraphSvg" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="Policy graph ${esc(payload.version || '')}"></svg>
        </div>
        <aside id="policyGraphPanel" class="policy-graph-panel" aria-live="polite">
          <h3>Policy node details</h3>
          <p class="muted">Hover a node to trace its neighbors. Click a node to read its Markdown and drill into its subtree when one exists.</p>
        </aside>
      </div>
      <div class="policy-graph-legend">${legendItems}</div>`;

    qs('#policyGraphTitle').textContent = payload.title || 'Cold-start GenAI policy';

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
      const sourceFamily = familyOf(edge.sourceId);
      const targetFamily = familyOf(edge.targetId);
      if (edge.sourceId === 'GA.root' || edge.targetId === 'GA.root') return 'root-link';
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
      .attr('stroke-width', d => d.id === focusId ? 4 : 2.4);

    node.append('text')
      .attr('y', d => nodeRadius(d, allNodes) + 13)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9.5)
      .style('opacity', 1)
      .text(d => d.id || d.title || '?');

    node.on('mouseover', (_, d) => highlight(d.id))
      .on('mouseout', clearHighlight)
      .on('click', (event, d) => {
        event.stopPropagation();
        if (hasChildren(d, allNodes) && d.id !== currentFocus) {
          renderGraph(payload, d.id);
          window.requestAnimationFrame(() => openPanel(d));
        } else {
          openPanel(d);
        }
      })
      .on('keydown', (event, d) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openPanel(d);
        }
      });

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(80).strength(0.58))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(WIDTH / 2, HEIGHT / 2))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d, allNodes) + (hasChildren(d, allNodes) ? 42 : 28)));

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function highlight(id) {
      const neighbors = neighborMap.get(id) || new Set([id]);
      node.classed('dimmed', d => !neighbors.has(d.id))
        .classed('highlighted', d => neighbors.has(d.id));
      link.classed('dimmed', d => d.sourceId !== id && d.targetId !== id)
        .classed('highlighted', d => d.sourceId === id || d.targetId === id);
    }

    function clearHighlight() {
      node.classed('dimmed highlighted', false);
      link.classed('dimmed highlighted', false);
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
    openPanel(node);
    qs('#policy-graph')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return true;
  }

  window.rushOpenPolicyNode = openPolicyNodeById;

  async function loadGraph(version = '') {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      const query = version ? `?version=${encodeURIComponent(version)}` : '';
      status('Loading policy graph…');
      const payload = await rushApiGetJson(`/api/policy/graph${query}`);
      currentFocus = null;
      currentVersion = payload.version || version;
      populateVersions(payload.available_versions, payload.version);
      renderGraph(payload, null);
      status(`Loaded ${payload.nodes?.length || 0} node(s), ${payload.edges?.length || 0} edge(s).`);
    } catch (error) {
      const wrap = qs('#policyGraphSvgWrap');
      if (wrap) wrap.innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Policy graph failed: ${error.message}`, true);
    }
  }

  async function initPolicyGraph(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    const selected = qs('#policyGraphVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || '';
    qs('#policyGraphVersion')?.addEventListener('change', event => loadGraph(event.target.value));
    await loadGraph(selected);
  }

  rushApiOnReady(initPolicyGraph);
  window.addEventListener('rush-api-catalog', event => {
    const versions = event.detail?.policyVersions || [];
    const latest = event.detail?.currentPolicyVersion || versions[versions.length - 1]?.version || '';
    if (latest && latest !== currentVersion) loadGraph(latest);
  });
})();
