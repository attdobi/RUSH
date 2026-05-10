(() => {
  const COLORS = ['#4de0a6', '#82b5ff', '#ffd166', '#ff6f91', '#d394ff', '#7dd0ff'];

  function status(message, isError = false) {
    rushApiStatus('#decisionQualityStatus', message, isError);
  }

  function setUnavailable() {
    rushApiUnavailable('#decision-quality-overview');
    $('#decisionQualityCards').innerHTML = '';
    $('#decisionQualityChart').innerHTML = '';
  }

  function populateFilters() {
    const runSelect = $('#decisionQualityRunId');
    if (runSelect) runSelect.innerHTML = rushApiRunOptions(runSelect.value, true, 'All scored runs');
    const versionSelect = $('#decisionQualityPolicyVersion');
    if (versionSelect) versionSelect.innerHTML = rushApiPolicyVersionOptions(versionSelect.value, true, 'All policy versions');
  }

  function buildQuery() {
    const params = new URLSearchParams();
    const runId = $('#decisionQualityRunId')?.value || '';
    const policyVersion = $('#decisionQualityPolicyVersion')?.value || '';
    if (runId) params.set('run_id', runId);
    if (policyVersion) params.set('policy_version', policyVersion);
    return params.toString() ? `/api/decision-quality?${params.toString()}` : '/api/decision-quality';
  }

  function aggregateLabelers(runs) {
    const grouped = new Map();
    for (const run of runs) {
      for (const labeler of (Array.isArray(run.labelers) ? run.labelers : [])) {
        const id = labeler.labeler_id || 'unknown';
        const metrics = labeler.metrics || {};
        if (!grouped.has(id)) grouped.set(id, { labeler_id: id, labeler_type: labeler.labeler_type || 'llm', count: 0, n: 0, accuracy: 0, f1: 0, fpr: 0, fnr: 0 });
        const row = grouped.get(id);
        row.count += 1;
        row.n += Number(metrics.n || 0);
        for (const key of ['accuracy', 'f1', 'fpr', 'fnr']) {
          if (isNumber(metrics[key])) row[key] += metrics[key];
        }
      }
    }
    return Array.from(grouped.values()).map(row => ({
      ...row,
      accuracy: row.count ? row.accuracy / row.count : null,
      f1: row.count ? row.f1 / row.count : null,
      fpr: row.count ? row.fpr / row.count : null,
      fnr: row.count ? row.fnr / row.count : null
    })).sort((a, b) => (b.accuracy || 0) - (a.accuracy || 0));
  }

  function renderCards(runs) {
    const target = $('#decisionQualityCards');
    if (!target) return;
    const rows = aggregateLabelers(runs);
    if (!rows.length) {
      target.innerHTML = '<div class="empty-state">No labeler metrics found for the current filters.</div>';
      return;
    }
    target.innerHTML = rows.map(row => `<article class="quality-card">
      <span>${esc(row.labeler_type)}</span>
      <strong>${esc(row.labeler_id)}</strong>
      <p>Accuracy ${rushApiFormatMetric(row.accuracy)} · F1 ${rushApiFormatMetric(row.f1)}<br>FPR ${rushApiFormatMetric(row.fpr)} · FNR ${rushApiFormatMetric(row.fnr)}<br>${esc(row.n)} images across ${esc(row.count)} run(s)</p>
    </article>`).join('');
  }

  function collectSeries(runs) {
    const sortedRuns = runs.slice().sort((a, b) => (a.started_at || '').localeCompare(b.started_at || ''));
    const series = new Map();
    sortedRuns.forEach((run, index) => {
      for (const labeler of (Array.isArray(run.labelers) ? run.labelers : [])) {
        const accuracy = labeler.metrics?.accuracy;
        if (!isNumber(accuracy)) continue;
        const id = labeler.labeler_id || 'unknown';
        if (!series.has(id)) series.set(id, []);
        series.get(id).push({ index, run, accuracy });
      }
    });
    return { sortedRuns, series };
  }

  function renderChart(runs) {
    const target = $('#decisionQualityChart');
    if (!target) return;
    const { sortedRuns, series } = collectSeries(runs);
    if (!sortedRuns.length || !series.size) {
      target.innerHTML = '<div class="empty-state">No accuracy history available yet.</div>';
      return;
    }
    const width = 760;
    const height = 260;
    const pad = { left: 46, right: 18, top: 18, bottom: 48 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const denom = Math.max(1, sortedRuns.length - 1);
    const x = index => pad.left + (index / denom) * innerW;
    const y = value => pad.top + (1 - Math.max(0, Math.min(1, value))) * innerH;
    const grid = [0, 0.25, 0.5, 0.75, 1].map(value => {
      const yy = y(value).toFixed(1);
      return `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}" class="chart-grid" /><text x="8" y="${Number(yy) + 4}" class="chart-label">${Math.round(value * 100)}%</text>`;
    }).join('');
    const lines = Array.from(series.entries()).map(([id, points], index) => {
      const color = COLORS[index % COLORS.length];
      const pointText = points.map(point => `${x(point.index).toFixed(1)},${y(point.accuracy).toFixed(1)}`).join(' ');
      const circles = points.map(point => `<circle cx="${x(point.index).toFixed(1)}" cy="${y(point.accuracy).toFixed(1)}" r="3.5" fill="${color}"><title>${esc(id)} · ${esc(point.run.run_id || '')}: ${rushApiFormatMetric(point.accuracy)}</title></circle>`).join('');
      return `<polyline points="${pointText}" fill="none" stroke="${color}" stroke-width="2.5" />${circles}`;
    }).join('');
    const labels = sortedRuns.map((run, index) => {
      const shortId = String(run.run_id || `run-${index + 1}`).slice(0, 13);
      return `<text x="${x(index).toFixed(1)}" y="${height - 18}" class="chart-label" text-anchor="middle">${esc(shortId)}</text>`;
    }).join('');
    const legend = Array.from(series.keys()).map((id, index) => `<span><i style="background:${COLORS[index % COLORS.length]}"></i>${esc(id)}</span>`).join('');
    target.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Accuracy across runs over time">${grid}${lines}${labels}</svg><div class="chart-legend">${legend}</div></div>`;
  }

  async function loadDecisionQuality() {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      status('Loading decision quality…');
      const payload = await rushApiGetJson(buildQuery());
      const runs = Array.isArray(payload.runs) ? payload.runs : [];
      if (Array.isArray(payload.policy_versions) && payload.policy_versions.length) {
        window.RUSH_API.catalog.policyVersions = payload.policy_versions.map(version => ({ version }));
        populateFilters();
      }
      renderCards(runs);
      renderChart(runs);
      status(`Loaded ${runs.length} scored run(s).`);
    } catch (error) {
      $('#decisionQualityCards').innerHTML = '';
      $('#decisionQualityChart').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Decision quality failed: ${error.message}`, true);
    }
  }

  async function initDecisionQuality(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    populateFilters();
    $('#decisionQualityRunId')?.addEventListener('change', loadDecisionQuality);
    $('#decisionQualityPolicyVersion')?.addEventListener('change', loadDecisionQuality);
    $('#refreshDecisionQuality')?.addEventListener('click', loadDecisionQuality);
    await loadDecisionQuality();
  }

  rushApiOnReady(initDecisionQuality);
  window.addEventListener('rush-api-catalog', populateFilters);
})();
