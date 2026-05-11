(() => {
  const PANELS = [
    ['model_disagreement', 'Model disagreement'],
    ['boundary_concentration', 'Boundary concentration'],
    ['consistent_pair_disagreement', 'Consistent pair disagreement'],
    ['majority_wrong', 'Majority wrong'],
    ['policy_clarity_hot_spots', 'Policy clarity hot spots']
  ];

  const KNOWN_LABELS = ['gen_ai', 'not_gen_ai', 'abstain'];

  function status(message, isError = false) {
    rushApiStatus('#insightsStatus', message, isError);
  }

  function setUnavailable() {
    rushApiUnavailable('#insights');
    $('#insightsPanels').innerHTML = '';
  }

  function populateRuns() {
    const select = $('#insightsRunId');
    if (!select) return;
    const selected = select.value || window.RUSH_API?.catalog?.runs?.[0]?.run_id || '';
    select.innerHTML = rushApiRunOptions(selected, false);
    if (selected) select.value = selected;
  }

  function labelBadgeClass(label) {
    const globalClass = window.labelBadgeClass || window.rushLabelBadgeClass;
    if (typeof globalClass === 'function') return globalClass(label);
    if (!label || !KNOWN_LABELS.includes(label)) return 'dev';
    return String(label).replaceAll('_', '-');
  }

  function labelBadge(label) {
    return `<span class="badge ${labelBadgeClass(label)}">${esc(label || '—')}</span>`;
  }

  function imgIdThumb(imageId) {
    const id = imageId || '';
    if (!id) return '<span class="muted">—</span>';
    const fn = (typeof window !== 'undefined' && window.thumbnailSrcForImageId)
      ? window.thumbnailSrcForImageId
      : (imgId) => `/api/thumbnail?path=${encodeURIComponent(`data/images/genai-classification/thumbnails/${imgId}.jpg`)}`;
    const src = fn(id);
    const fallbackSrc = `../data/images/genai-classification/thumbnails/${encodeURIComponent(id)}.jpg`;
    return `<img class="row-thumb thumb-loading" src="${esc(src)}" data-fallback-src="${esc(fallbackSrc)}" alt="${esc(id)}" title="${esc(id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="if(this.dataset.fallbackSrc){this.src=this.dataset.fallbackSrc;this.dataset.fallbackSrc='';}else{this.outerHTML='<span class=&quot;muted&quot;>'+this.alt+'</span>'}" />`;
  }

  function votesHtml(votes) {
    if (!Array.isArray(votes) || !votes.length) return '<span class="muted">—</span>';
    return window.rushSortEnsembleLast(votes, (a, b) =>
      String(a.model_id || a.labeler_id || '').localeCompare(String(b.model_id || b.labeler_id || ''))
    ).slice(0, 6).map(vote => {
      const id = vote.labeler_id || vote.model_id || 'unknown';
      const label = vote.label || vote.vote || '—';
      const suffix = window.rushIsEnsembleRow(vote) ? ' <small class="muted">· ensemble</small>' : '';
      return `<span class="mini-chip">${esc(id)}${suffix}: ${labelBadge(label)}</span>`;
    }).join(' ');
  }

  function labelsHtml(labels) {
    if (!Array.isArray(labels) || !labels.length) return '<span class="muted">—</span>';
    return labels.slice(0, 6).map(label => labelBadge(label)).join(' ');
  }

  function renderHotSpots(rows) {
    const filteredRows = rows.filter(row => Number(row.flip_rate || 0) >= 0.05);
    if (!filteredRows.length) return '<div class="empty-state compact-empty">No policy clarity hot spots yet — needs ≥2 scored runs of the same images with flip-rate ≥ 0.05.</div>';
    return table(['Image', 'Flip rate', 'Runs', 'Labels observed'], filteredRows.map(row => [
      imgIdThumb(row.image_id),
      isNumber(row.flip_rate) ? row.flip_rate.toFixed(2) : '—',
      esc(row.n_runs ?? '—'),
      labelsHtml(row.labels_observed)
    ]));
  }

  function renderMajorityWrong(rows) {
    return table(['Image', 'SME truth', 'Majority label', 'Votes'], rows.map(row => [
      imgIdThumb(row.image_id),
      labelBadge(row.sme_truth),
      labelBadge(row.majority_label),
      votesHtml(row.votes)
    ]));
  }

  function renderDisagreement(rows) {
    return table(['Image', 'Votes'], rows.map(row => [
      imgIdThumb(row.image_id),
      votesHtml(row.votes)
    ]));
  }

  function renderBoundary(rows) {
    return table(['L0 bucket', 'Images', 'Top L2 nodes'], rows.map(row => [
      `<strong>${esc(row.l0_bucket || '—')}</strong>`,
      esc(row.n_images ?? '—'),
      labelsHtml(row.top_l2_nodes)
    ]));
  }

  function renderPairDisagreement(rows) {
    return table(['Pair', 'Disagreements', 'Fraction'], rows.map(row => [
      labelsHtml(row.pair),
      esc(row.n_disagreements ?? '—'),
      isNumber(row.fraction) ? rushApiFormatMetric(row.fraction) : '—'
    ]));
  }

  function table(headers, rows, emptyMessage = 'No rows for this insight yet.') {
    if (!rows.length) return `<div class="empty-state compact-empty">${esc(emptyMessage)}</div>`;
    const head = headers.map(header => `<th>${esc(header)}</th>`).join('');
    const body = rows.slice(0, 10).map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('');
    return `<div class="compact-table"><table class="misalignment"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderPanel(key, title, rows) {
    const sourceRows = Array.isArray(rows) ? rows : [];
    const safeRows = sourceRows.slice(0, 10);
    let body = '';
    if (key === 'policy_clarity_hot_spots') body = renderHotSpots(sourceRows);
    else if (key === 'majority_wrong') body = renderMajorityWrong(safeRows);
    else if (key === 'model_disagreement') body = renderDisagreement(safeRows);
    else if (key === 'boundary_concentration') body = renderBoundary(safeRows);
    else if (key === 'consistent_pair_disagreement') body = renderPairDisagreement(safeRows);
    return `<article class="insight-panel"><h3>${esc(title)}</h3>${body}</article>`;
  }

  function renderInsights(payload) {
    const target = $('#insightsPanels');
    if (!target) return;
    target.innerHTML = PANELS.map(([key, title]) => renderPanel(key, title, payload?.[key])).join('');
  }

  async function loadInsights() {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    const runId = $('#insightsRunId')?.value || '';
    if (!runId) {
      $('#insightsPanels').innerHTML = '<div class="empty-state">Select a scored run to load insights.</div>';
      status('Select a run.');
      return;
    }
    try {
      status(`Loading insights for ${runId}…`);
      const payload = await rushApiGetJson(`/api/insights?run_id=${encodeURIComponent(runId)}`);
      renderInsights(payload);
      status(`Loaded insights for ${payload.run_id || runId}.`);
    } catch (error) {
      $('#insightsPanels').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Insights failed: ${error.message}`, true);
    }
  }

  async function initInsights(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    populateRuns();
    $('#insightsRunId')?.addEventListener('change', loadInsights);
    $('#refreshInsights')?.addEventListener('click', loadInsights);
    await loadInsights();
  }

  rushApiOnReady(initInsights);
  window.addEventListener('rush-api-catalog', populateRuns);
})();
