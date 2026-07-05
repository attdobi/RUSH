(() => {
  const KNOWN_LABELS = ['gen_ai', 'not_gen_ai', 'abstain'];
  const MNIST_EMPTY_RUN_MESSAGE = 'No scored MNIST run yet — run labeling to populate.';
  const PANEL_TARGETS = ['#scoreInsightSme', '#scoreInsightModel', '#scoreInsightBoundary'];

  function activeDemo() {
    return typeof window.rushActiveDemo === 'function'
      ? window.rushActiveDemo()
      : { id: 'genai', classes: ['gen_ai', 'not_gen_ai'] };
  }

  function activeDemoId() {
    return activeDemo().id || 'genai';
  }

  function isMnistDemo() {
    return activeDemoId() === 'mnist';
  }

  function status(message, isError = false) {
    rushApiStatus('#scoreInsightStatus', message, isError);
  }

  function setPanels(html) {
    for (const selector of PANEL_TARGETS) {
      const target = $(selector);
      if (target) target.innerHTML = html;
    }
  }

  function setUnavailable() {
    setPanels('<div class="empty-state compact-empty">Local API offline — start the rush web server to enable folded insight cuts.</div>');
    status('Local API offline.', true);
  }

  function selectedScoreRunId() {
    return $('#runPicker')?.value || '';
  }

  function labelBadgeClass(label) {
    const value = String(label || '');
    const classes = Array.isArray(activeDemo().classes) ? activeDemo().classes.map(String) : [];
    if (isMnistDemo() && classes.includes(value)) return `digit-${value.replace(/[^a-z0-9_-]/gi, '')}`;
    const globalClass = window.labelBadgeClass || window.rushLabelBadgeClass;
    if (typeof globalClass === 'function') return globalClass(label);
    if (!value || !KNOWN_LABELS.includes(value)) return 'dev';
    return value.replaceAll('_', '-');
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

  function boundaryPairChip(record) {
    const pair = record?.is_boundary === true && Array.isArray(record?.is_boundary_between) && record.is_boundary_between.length === 2
      ? record.is_boundary_between
      : null;
    if (!pair) return '';
    return `<span class="boundary-pair-chip" title="boundary pair">${esc(pair[0])} ↔ ${esc(pair[1])}</span>`;
  }

  function votesHtml(votes) {
    if (!Array.isArray(votes) || !votes.length) return '<span class="muted">—</span>';
    return window.rushSortEnsembleLast(votes, (a, b) =>
      String(a.model_id || a.labeler_id || '').localeCompare(String(b.model_id || b.labeler_id || ''))
    ).slice(0, 6).map(vote => {
      const id = vote.labeler_id || vote.model_id || 'unknown';
      const label = vote.label || vote.vote || '—';
      const suffix = window.rushIsEnsembleRow(vote) ? ' <small class="muted">· ensemble</small>' : '';
      return `<span class="mini-chip">${esc(id)}${suffix}: ${labelBadge(label)}${boundaryPairChip(vote)}</span>`;
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
    return `<div class="empty-state compact-empty">${esc(message)} <button type="button" data-compute-target="insights" data-run-id="${attr(runId)}">Compute now</button></div>`;
  }

  function table(headers, rows, emptyMessage = 'No rows for this cut yet.') {
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

  function renderMajorityWrong(rows) {
    return table(['Image', 'SME truth', 'Majority label', 'Votes'], rows.map(row => ({
      imageId: row.image_id,
      cells: [
        imgThumbCell(row),
        labelBadge(row.sme_truth),
        labelBadge(row.majority_label),
        votesHtml(row.votes)
      ]
    })), 'No majority-vs-SME misses for this run.');
  }

  function renderDisagreement(rows) {
    return table(['Image', 'Votes'], rows.map(row => ({
      imageId: row.image_id,
      cells: [
        imgThumbCell(row),
        votesHtml(row.votes)
      ]
    })), 'No model disagreement rows for this run.');
  }

  function renderBoundary(rows) {
    return table(['Bucket', 'Images', 'Top nodes'], rows.map(row => [
      `<strong>${esc(row.l0_bucket || row.bucket || '—')}</strong>`,
      esc(row.n_images ?? '—'),
      labelsHtml(row.top_l2_nodes)
    ]), 'No boundary concentration rows for this run.');
  }

  function renderPairDisagreement(rows) {
    return table(['Pair', 'Disagreements', 'Fraction'], rows.map(row => [
      labelsHtml(row.pair),
      esc(row.n_disagreements ?? '—'),
      isNumber(row.fraction) ? rushApiFormatMetric(row.fraction) : '—'
    ]), 'No recurring pair disagreements for this run.');
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

  function renderScoreInsights(payload) {
    const smeTarget = $('#scoreInsightSme');
    const modelTarget = $('#scoreInsightModel');
    const boundaryTarget = $('#scoreInsightBoundary');
    if (!smeTarget || !modelTarget || !boundaryTarget) return;
    const hasRows = ['majority_wrong', 'model_disagreement', 'boundary_concentration', 'consistent_pair_disagreement']
      .some(key => Array.isArray(payload?.[key]) && payload[key].length > 0);
    if (isMnistDemo() && !hasRows) {
      setPanels(`<div class="empty-state">${esc(MNIST_EMPTY_RUN_MESSAGE)}</div>`);
      return;
    }
    smeTarget.innerHTML = renderPanel('majority_wrong', 'Majority wrong — review these first', payload?.majority_wrong);
    modelTarget.innerHTML = renderPanel('model_disagreement', 'Model disagreement', payload?.model_disagreement);
    boundaryTarget.innerHTML = [
      renderPanel('boundary_concentration', 'Boundary concentration', payload?.boundary_concentration),
      renderPanel('consistent_pair_disagreement', 'Recurring pair disagreement', payload?.consistent_pair_disagreement)
    ].join('');
  }

  async function loadScoreInsights() {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    const runId = selectedScoreRunId();
    if (!runId) {
      setPanels('<div class="empty-state compact-empty">Select a scored run in §3 Recent runs to load folded insight cuts.</div>');
      status('Select a scored run to load review cuts.');
      return;
    }
    try {
      status(`Loading folded insight cuts for ${runId}…`);
      const params = new URLSearchParams();
      params.set('demo', activeDemoId());
      params.set('run_id', runId);
      const payload = await rushApiGetJson(`/api/insights?${params.toString()}`);
      renderScoreInsights(payload);
      status(`Loaded folded insight cuts for ${payload.run_id || runId}.`);
    } catch (error) {
      const html = isMissingScoringError(error)
        ? renderScoreRunEmpty(runId, error.message)
        : `<div class="empty-state compact-empty">${esc(error.message)}</div>`;
      setPanels(html);
      status(`Folded insights failed: ${error.message}`, true);
    }
  }

  async function initScoreInsights(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    await loadScoreInsights();
  }

  rushApiOnReady(initScoreInsights);
  window.addEventListener('rush-score-run-selected', loadScoreInsights);
  window.addEventListener('rush-api-catalog', loadScoreInsights);
})();
