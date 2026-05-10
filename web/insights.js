(() => {
  const PANELS = [
    ['policy_clarity_hot_spots', 'Policy clarity hot spots'],
    ['majority_wrong', 'Majority wrong'],
    ['model_disagreement', 'Model disagreement'],
    ['boundary_concentration', 'Boundary concentration'],
    ['consistent_pair_disagreement', 'Consistent pair disagreement']
  ];

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

  function votesHtml(votes) {
    if (!Array.isArray(votes) || !votes.length) return '<span class="muted">—</span>';
    return votes.slice(0, 6).map(vote => {
      const id = vote.labeler_id || vote.model_id || 'unknown';
      const label = vote.label || vote.vote || '—';
      return `<span class="mini-chip">${esc(id)}: ${esc(label)}</span>`;
    }).join(' ');
  }

  function labelsHtml(labels) {
    if (!Array.isArray(labels) || !labels.length) return '<span class="muted">—</span>';
    return labels.slice(0, 6).map(label => `<span class="mini-chip">${esc(label)}</span>`).join(' ');
  }

  function renderHotSpots(rows) {
    return table(['Image', 'Flip rate', 'Runs', 'Labels observed'], rows.map(row => [
      `<strong>${esc(row.image_id || '—')}</strong>`,
      isNumber(row.flip_rate) ? row.flip_rate.toFixed(2) : '—',
      esc(row.n_runs ?? '—'),
      labelsHtml(row.labels_observed)
    ]));
  }

  function renderMajorityWrong(rows) {
    return table(['Image', 'SME truth', 'Majority label', 'Votes'], rows.map(row => [
      `<strong>${esc(row.image_id || '—')}</strong>`,
      esc(row.sme_truth || '—'),
      esc(row.majority_label || '—'),
      votesHtml(row.votes)
    ]));
  }

  function renderDisagreement(rows) {
    return table(['Image', 'Votes'], rows.map(row => [
      `<strong>${esc(row.image_id || '—')}</strong>`,
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

  function table(headers, rows) {
    if (!rows.length) return '<div class="empty-state compact-empty">No rows for this insight yet.</div>';
    const head = headers.map(header => `<th>${esc(header)}</th>`).join('');
    const body = rows.slice(0, 10).map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('');
    return `<div class="compact-table"><table class="misalignment"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderPanel(key, title, rows) {
    const safeRows = Array.isArray(rows) ? rows.slice(0, 10) : [];
    let body = '';
    if (key === 'policy_clarity_hot_spots') body = renderHotSpots(safeRows);
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
