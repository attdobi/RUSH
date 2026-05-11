(() => {
  const COLORS = {
    root: '#4d9bff',
    positive: '#4de0a6',
    boundary: '#ffb74d',
    exception: '#ff6f91',
    negative: '#ff4d6d',
    provenance: '#4dd0e1',
    fallback: '#888'
  };
  const VIEWBOX = { width: 720, height: 480, cx: 360, cy: 240, radius: 180 };

  function status(message, isError = false) {
    rushApiStatus('#policyGraphStatus', message, isError);
  }

  function setUnavailable() {
    const wrap = $('#policyGraphSvgWrap');
    if (wrap) wrap.innerHTML = '<div class="empty-state">Local API not running — start <code>python scripts/rush_web_server.py</code> to load the policy graph.</div>';
    status('Policy graph unavailable.', true);
  }

  function nodeColor(node) {
    const id = String(node.id || '');
    const type = String(node.node_type || '').toLowerCase();
    const polarity = String(node.polarity || '').toLowerCase();
    if (type === 'root' || id === 'GA.root') return COLORS.root;
    if (id.includes('.boundary.') || type === 'boundary') return COLORS.boundary;
    if (id.includes('.exception.')) return COLORS.exception;
    if (id.includes('.negative.') || polarity === 'negative') return COLORS.negative;
    if (id.includes('.provenance.')) return COLORS.provenance;
    if (
      type === 'category' ||
      polarity === 'positive' ||
      id.includes('.visual_artifacts.') ||
      id.includes('.surface_texture.') ||
      id.includes('.scene_geometry.')
    ) return COLORS.positive;
    return COLORS.fallback;
  }

  function truncate(text, max = 30) {
    const value = String(text || '');
    return value.length > max ? `${value.slice(0, max - 1)}…` : value;
  }

  function edgeSource(edge) {
    return edge.source || edge.source_node_id;
  }

  function edgeTarget(edge) {
    return edge.target || edge.target_node_id || edge.to;
  }

  function layoutNodes(nodes) {
    const root = nodes.find(node => node.id === 'GA.root') || nodes.find(node => node.node_type === 'root');
    const rest = nodes.filter(node => node !== root).sort((a, b) => String(a.id || '').localeCompare(String(b.id || '')));
    const positions = new Map();
    if (root) positions.set(root.id, { x: VIEWBOX.cx, y: VIEWBOX.cy, r: 24 });
    const count = Math.max(1, rest.length);
    rest.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / count - Math.PI / 2;
      positions.set(node.id, {
        x: VIEWBOX.cx + Math.cos(angle) * VIEWBOX.radius,
        y: VIEWBOX.cy + Math.sin(angle) * VIEWBOX.radius,
        r: 16
      });
    });
    return positions;
  }

  function populateVersions(versions, selected) {
    const select = $('#policyGraphVersion');
    if (!select) return;
    const list = Array.isArray(versions) ? versions : [];
    if (!list.length) {
      select.innerHTML = rushApiOptionHtml('', 'No policy versions found', true);
      return;
    }
    select.innerHTML = list.map(version => rushApiOptionHtml(version, version, version === selected)).join('');
    select.value = selected || list[list.length - 1];
  }

  function renderGraph(payload) {
    const wrap = $('#policyGraphSvgWrap');
    if (!wrap) return;
    const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
    const edges = Array.isArray(payload.edges) ? payload.edges : [];
    const positions = layoutNodes(nodes);
    const edgesHtml = edges.map(edge => {
      const source = positions.get(edgeSource(edge));
      const target = positions.get(edgeTarget(edge));
      if (!source || !target) return '';
      return `<line x1="${source.x.toFixed(1)}" y1="${source.y.toFixed(1)}" x2="${target.x.toFixed(1)}" y2="${target.y.toFixed(1)}" stroke="#ffffff55" stroke-width="1.2"><title>${esc(edge.edge_type || 'edge')}</title></line>`;
    }).join('');
    const nodesHtml = nodes.map(node => {
      const pos = positions.get(node.id);
      if (!pos) return '';
      const color = nodeColor(node);
      const label = truncate(node.title || node.id);
      return `<g class="policy-graph-node" transform="translate(${pos.x.toFixed(1)} ${pos.y.toFixed(1)})">
        <circle r="${pos.r}" fill="#111927" stroke="${color}" stroke-width="3"><title>${esc(node.id)} · ${esc(node.node_type)} · ${esc(node.polarity)}</title></circle>
        <text y="${pos.r + 16}" text-anchor="middle">${esc(label)}</text>
      </g>`;
    }).join('');
    const legendItems = [
      ['root', COLORS.root],
      ['positive', COLORS.positive],
      ['boundary', COLORS.boundary],
      ['exception', COLORS.exception],
      ['negative', COLORS.negative],
      ['provenance', COLORS.provenance]
    ].map(([label, color]) => `<span><i style="background:${color}"></i>${esc(label)}</span>`).join('');
    wrap.innerHTML = `<h3>${esc(payload.title || `Cold-start GenAI policy ${payload.version || ''}`)}</h3>
      <svg viewBox="0 0 ${VIEWBOX.width} ${VIEWBOX.height}" role="img" aria-label="Policy graph ${esc(payload.version || '')}">
        <rect width="${VIEWBOX.width}" height="${VIEWBOX.height}" rx="18" fill="#0e1219"></rect>
        ${edgesHtml}
        ${nodesHtml}
      </svg>
      <div class="policy-graph-legend">${legendItems}</div>`;
    $('#policyGraphTitle').textContent = payload.title || 'Cold-start GenAI policy';
  }

  async function loadGraph(version = '') {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      const query = version ? `?version=${encodeURIComponent(version)}` : '';
      status('Loading policy graph…');
      const payload = await rushApiGetJson(`/api/policy/graph${query}`);
      populateVersions(payload.available_versions, payload.version);
      renderGraph(payload);
      status(`Loaded ${payload.nodes?.length || 0} node(s), ${payload.edges?.length || 0} edge(s).`);
    } catch (error) {
      $('#policyGraphSvgWrap').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Policy graph failed: ${error.message}`, true);
    }
  }

  async function initPolicyGraph(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    const selected = $('#policyGraphVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || '';
    $('#policyGraphVersion')?.addEventListener('change', event => loadGraph(event.target.value));
    await loadGraph(selected);
  }

  rushApiOnReady(initPolicyGraph);
  window.addEventListener('rush-api-catalog', event => {
    const versions = event.detail?.policyVersions || [];
    const latest = event.detail?.currentPolicyVersion || versions[versions.length - 1]?.version || '';
    if (latest) loadGraph(latest);
  });
})();
