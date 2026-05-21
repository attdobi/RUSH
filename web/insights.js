(() => {
  const PANELS = [
    ['majority_wrong', 'Majority wrong'],
    ['model_disagreement', 'Model disagreement'],
    ['boundary_concentration', 'Boundary concentration'],
    ['consistent_pair_disagreement', 'Consistent pair disagreement']
  ];
  const SECONDARY_PANELS = PANELS.slice(1);

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

  function thumbnailSrcForRepoPath(repoRelPath) {
    const path = String(repoRelPath || '').replace(/^\.\//, '').replace(/^\/+/, '');
    if (!path) return '';
    const fn = (typeof window !== 'undefined' && typeof window.thumbnailSrcForPath === 'function')
      ? window.thumbnailSrcForPath
      : (repoPath) => `/api/thumbnail?path=${encodeURIComponent(repoPath)}`;
    return fn(path);
  }

  function imgThumbCell(row) {
    const isSynthetic = row?.is_synthetic_demo_candidate === true;
    const id = row?.image_id || (isSynthetic ? row?.sample_id : '') || '';
    if (!id) return '<span class="muted">—</span>';
    const syntheticSrc = isSynthetic && typeof window.syntheticThumbDataUri === 'function'
      ? window.syntheticThumbDataUri(row)
      : '';
    const src = syntheticSrc
      || thumbnailSrcForRepoPath(row?.repo_rel_path)
      || (typeof window.thumbnailSrcForImageId === 'function' ? window.thumbnailSrcForImageId(id) : '');
    const thumb = src
      ? `<img class="row-thumb thumb-loading" src="${attr(src)}" alt="${attr(id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />`
      : '<div class="row-thumb thumb-fallback mini-thumb-fallback"><strong>no image</strong></div>';
    return `<div class="thumb-wrap insight-thumb-wrap">${thumb}<div><button type="button" class="image-id-button" data-open-justifications="${attr(id)}"><strong>${esc(id)}</strong></button></div></div>`;
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

  function isMissingScoringError(error) {
    return /not scored|missing scoring|scoring failed/i.test(String(error?.message || error || ''));
  }

  function renderScoreRunEmpty(runId, message) {
    return `<div class="empty-state">${esc(message)} <button type="button" data-score-run-id="${attr(runId)}">Score this run</button><span id="insightsScoreStatus" class="status-line" role="status"></span></div>`;
  }

  async function scoreRun(event) {
    const button = event.target.closest('[data-score-run-id]');
    if (!button) return;
    const runId = button.dataset.scoreRunId || '';
    if (!runId) return;
    const scoreStatus = $('#insightsScoreStatus');
    try {
      button.disabled = true;
      if (scoreStatus) scoreStatus.textContent = `Computing scoring for ${runId}…`;
      status(`Computing scoring for ${runId}…`);
      await rushApiPostJson(`/api/runs/${encodeURIComponent(runId)}/compute-now`, {});
      await loadInsights();
    } catch (error) {
      const message = `Score failed: ${error.message}`;
      if (scoreStatus) scoreStatus.textContent = message;
      status(message, true);
      button.disabled = false;
    }
  }

  function renderMajorityWrong(rows) {
    return table(['Image', 'SME truth', 'Majority label', 'Votes'], rows.map(row => ({
      imageId: row.image_id,
      cells: [
        imgThumbCell(row),
        labelBadge(row.sme_truth),
        labelBadge(row.majority_label),
        votesHtml(row.votes)
      ]
    })));
  }

  function renderDisagreement(rows) {
    return table(['Image', 'Votes'], rows.map(row => ({
      imageId: row.image_id,
      cells: [
        imgThumbCell(row),
        votesHtml(row.votes)
      ]
    })));
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
    const body = rows.slice(0, 10).map(row => {
      const cells = Array.isArray(row) ? row : (row?.cells || []);
      const imageId = Array.isArray(row) ? '' : (row?.imageId || '');
      const rowAttrs = imageId ? ` data-image-id="${attr(imageId)}"` : '';
      return `<tr${rowAttrs}>${cells.map(cell => `<td>${cell}</td>`).join('')}</tr>`;
    }).join('');
    return `<div class="compact-table"><table class="misalignment"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderPanel(key, title, rows) {
    const sourceRows = Array.isArray(rows) ? rows : [];
    const safeRows = sourceRows.slice(0, 10);
    let body = '';
    if (key === 'majority_wrong') body = renderMajorityWrong(safeRows);
    else if (key === 'model_disagreement') body = renderDisagreement(safeRows);
    else if (key === 'boundary_concentration') body = renderBoundary(safeRows);
    else if (key === 'consistent_pair_disagreement') body = renderPairDisagreement(safeRows);
    return `<article class="insight-panel ${key === 'majority_wrong' ? 'insight-panel-primary' : ''}"><h3>${esc(title)}</h3>${body}</article>`;
  }

  function renderInsights(payload) {
    const target = $('#insightsPanels');
    if (!target) return;
    const majority = renderPanel('majority_wrong', 'Majority wrong — review these first', payload?.majority_wrong);
    const secondary = SECONDARY_PANELS.map(([key, title]) => renderPanel(key, title, payload?.[key])).join('');
    target.innerHTML = `${majority}<details class="insights-more"><summary>More cuts <span class="muted">model disagreement, boundary concentration, pair disagreement</span></summary><div class="insights-more-grid">${secondary}</div></details>`;
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
      $('#insightsPanels').innerHTML = isMissingScoringError(error)
        ? renderScoreRunEmpty(runId, error.message)
        : `<div class="empty-state">${esc(error.message)}</div>`;
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
    $('#insightsPanels')?.addEventListener('click', scoreRun);
    await loadInsights();
  }

  rushApiOnReady(initInsights);
  window.addEventListener('rush-api-catalog', populateRuns);
})();
