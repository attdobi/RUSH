(() => {
  const COLORS = ['#4de0a6', '#82b5ff', '#ffd166', '#ff6f91', '#d394ff', '#7dd0ff'];
  const MNIST_EMPTY_RUN_MESSAGE = 'No scored MNIST run yet — run labeling to populate.';

  function activeDemo() {
    return typeof window.rushActiveDemo === 'function'
      ? window.rushActiveDemo()
      : { id: 'genai', classes: ['gen_ai', 'not_gen_ai'], policyGraph: { area: 'Generative_AI' } };
  }

  function activeDemoId() {
    return activeDemo().id || 'genai';
  }

  function activePolicyGraphArea() {
    return activeDemo().policyGraph?.area || 'Generative_AI';
  }

  function isMnistDemo() {
    return activeDemoId() === 'mnist';
  }

  function status(message, isError = false) {
    rushApiStatus('#decisionQualityStatus', message, isError);
  }

  function setUnavailable() {
    rushApiUnavailable('#quality');
    const summary = $('#decisionQualitySummary');
    if (summary) summary.innerHTML = '';
    const warning = $('#decisionQualityWarning');
    if (warning) {
      warning.hidden = true;
      warning.textContent = '';
    }
    const narrative = $('#decisionQualityNarrative');
    if (narrative) {
      narrative.hidden = true;
      narrative.textContent = '';
    }
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
    params.set('demo', activeDemoId());
    params.set('area', activePolicyGraphArea());
    const runId = $('#decisionQualityRunId')?.value || '';
    const policyVersion = $('#decisionQualityPolicyVersion')?.value || '';
    if (runId) params.set('run_id', runId);
    if (policyVersion) params.set('policy_version', policyVersion);
    return params.toString() ? `/api/decision-quality?${params.toString()}` : '/api/decision-quality';
  }

  function aggregateLabelers(runs) {
    const grouped = new Map();
    const metricKeys = ['accuracy', 'f1', 'precision', 'recall', 'fpr', 'fnr', 'positive_proportion', 'informedness'];
    for (const run of runs) {
      for (const labeler of (Array.isArray(run.labelers) ? run.labelers : [])) {
        const id = labeler.labeler_id || 'unknown';
        const metrics = labeler.metrics || {};
        if (!grouped.has(id)) {
          grouped.set(id, {
            labeler_id: id,
            labeler_type: labeler.labeler_type || 'llm',
            count: 0,
            n: 0,
            sums: Object.fromEntries(metricKeys.map(key => [key, 0])),
            costSum: 0,
            costCount: 0
          });
        }
        const row = grouped.get(id);
        row.count += 1;
        row.n += Number(metrics.n || 0);
        for (const key of metricKeys) {
          if (isNumber(metrics[key])) row.sums[key] += metrics[key];
        }
        if (isNumber(metrics.cost_per_1000_labels)) {
          row.costSum += metrics.cost_per_1000_labels;
          row.costCount += 1;
        }
      }
    }
    return sortLabelerRows(Array.from(grouped.values()).map(row => {
      const out = {
        labeler_id: row.labeler_id,
        labeler_type: row.labeler_type,
        count: row.count,
        n: row.n,
        cost_per_1000_labels: row.costCount ? row.costSum / row.costCount : null
      };
      for (const key of metricKeys) out[key] = row.count ? row.sums[key] / row.count : null;
      return out;
    }));
  }

  function isEnsembleRow(row) {
    return row.labeler_id === 'majority_vote' || row.labeler_type === 'ensemble';
  }

  function sortLabelerRows(rows) {
    return rows.sort((a, b) => {
      const aEnsemble = isEnsembleRow(a);
      const bEnsemble = isEnsembleRow(b);
      if (aEnsemble !== bEnsemble) return aEnsemble ? 1 : -1;
      return (b.accuracy || 0) - (a.accuracy || 0);
    });
  }

  function formatCostPerThousand(value) {
    return isNumber(value) ? `$${value.toFixed(4)}` : '—';
  }

  function formatSignedMetric(value) {
    if (!isNumber(value)) return '—';
    const sign = value > 0 ? '+' : '';
    return `${sign}${rushApiFormatMetric(value)}`;
  }

  function policyVersionCount(runs) {
    return new Set(runs.map(run => run.policy_graph_version).filter(Boolean)).size;
  }

  function totalImages(runs) {
    return runs.reduce((sum, run) => sum + (Number(run.n_images || 0) || 0), 0);
  }

  function splitBoundaryText(runs) {
    const totals = runs.reduce((acc, run) => {
      const summary = run.consensus_summary || {};
      const n = Number(run.n_images || summary.n_images_total || 0) || 0;
      const split = Number(summary.n_images_split || 0) || 0;
      const boundaryRate = isNumber(run.boundary_rate) ? run.boundary_rate : null;
      const boundary = boundaryRate === null ? Number(summary.n_images_with_boundary_flag || 0) || 0 : boundaryRate * n;
      acc.n += n;
      acc.split += split;
      acc.boundary += boundary;
      return acc;
    }, { n: 0, split: 0, boundary: 0 });
    if (!totals.n) return '—';
    return `${rushApiFormatMetric(totals.split / totals.n)} / ${rushApiFormatMetric(totals.boundary / totals.n)}`;
  }

  function renderSummary(runs, rows) {
    const target = $('#decisionQualitySummary');
    if (!target) return;
    const totalCost = runs.reduce((sum, run) => {
      const value = run.cost?.total_cost_usd;
      return isNumber(value) ? sum + value : sum;
    }, 0);
    const ensemble = rows.find(row => row.labeler_id === 'majority_vote') || rows.find(isEnsembleRow);
    const bestLabeler = rows.find(row => !isEnsembleRow(row));
    const ensembleLift = ensemble && bestLabeler && isNumber(ensemble.accuracy) && isNumber(bestLabeler.accuracy)
      ? ensemble.accuracy - bestLabeler.accuracy
      : null;
    const imageCount = totalImages(runs);
    const f1Label = isMnistDemo() ? 'Ensemble macro F1' : 'Ensemble accuracy';
    const f1Value = ensemble ? rushApiFormatMetric(isMnistDemo() ? ensemble.f1 : ensemble.accuracy) : '—';
    target.innerHTML = [
      ['Scored runs', String(runs.length)],
      ['Images scored', imageCount ? String(imageCount) : '—'],
      ['Policy versions', policyVersionCount(runs) ? String(policyVersionCount(runs)) : '—'],
      [f1Label, f1Value],
      ['Ensemble vs best model', formatSignedMetric(ensembleLift)],
      ['Split / boundary review load', splitBoundaryText(runs)],
      ['Total cost (USD)', `$${totalCost.toFixed(4)}`]
    ].map(([label, value]) => `<div class="dq-summary-card"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('');
  }

  function renderSmallNWarning(rows) {
    const target = $('#decisionQualityWarning');
    if (!target) return;
    const hasSmallN = rows.some(row => !isEnsembleRow(row) && Number(row.n || 0) < 30);
    target.hidden = !hasSmallN;
    target.textContent = hasSmallN ? 'Sample sizes are small (N < 30); metrics are indicative, not statistical.' : '';
  }

  function renderNarrative(runs, rows) {
    const target = $('#decisionQualityNarrative');
    if (!target) return;
    if (!runs.length || !rows.length) {
      target.hidden = true;
      target.textContent = '';
      return;
    }
    const ensemble = rows.find(row => row.labeler_id === 'majority_vote') || rows.find(isEnsembleRow);
    const bestLabeler = rows.find(row => !isEnsembleRow(row));
    const lift = ensemble && bestLabeler && isNumber(ensemble.accuracy) && isNumber(bestLabeler.accuracy)
      ? ensemble.accuracy - bestLabeler.accuracy
      : null;
    const liftText = isNumber(lift)
      ? `ensemble is ${formatSignedMetric(lift)} vs the best individual labeler`
      : 'ensemble comparison is waiting on scored labeler rows';
    const framing = isMnistDemo()
      ? 'watch per-digit errors, macro-F1, and confusion-pair review load for regressions by policy version.'
      : 'watch false positives/false negatives and the split/boundary review load for regressions by policy version.';
    target.innerHTML = `<strong>Demo read:</strong> treat this panel as the decision-quality gate before accepting policy growth — ${esc(liftText)}; ${esc(framing)}`;
    target.hidden = false;
  }

  function setBenchmarkState(hasRuns) {
    const target = $('#benchmarkComparison');
    if (!target || !isMnistDemo()) return;
    const strong = target.querySelector('strong');
    if (strong) strong.textContent = hasRuns ? 'RUSH MNIST result available' : MNIST_EMPTY_RUN_MESSAGE;
  }

  function renderEmptyDecisionQuality(message) {
    const summary = $('#decisionQualitySummary');
    if (summary) summary.innerHTML = '';
    const warning = $('#decisionQualityWarning');
    if (warning) {
      warning.hidden = true;
      warning.textContent = '';
    }
    const narrative = $('#decisionQualityNarrative');
    if (narrative) {
      narrative.hidden = false;
      narrative.textContent = activeDemo().sectionCopy?.qualitySub || 'Compare labels 0-9 with per-digit precision, per-digit recall, and macro-F1 once a scored run exists.';
    }
    const cards = $('#decisionQualityCards');
    if (cards) cards.innerHTML = `<div class="empty-state">${esc(message)}</div>`;
    const chart = $('#decisionQualityChart');
    if (chart) chart.innerHTML = `<div class="empty-state">${esc(message)}</div>`;
    setBenchmarkState(false);
  }

  function renderCards(runs) {
    const target = $('#decisionQualityCards');
    if (!target) return;
    const rows = aggregateLabelers(runs);
    target.classList.add('dq-table-wrap');
    renderSummary(runs, rows);
    renderSmallNWarning(rows);
    renderNarrative(runs, rows);
    if (!rows.length) {
      target.innerHTML = '<div class="empty-state">No labeler metrics found for the current filters.</div>';
      return;
    }
    const columns = isMnistDemo() ? [
      ['Labeler', row => `${esc(row.labeler_id)}${isEnsembleRow(row) ? ' <span class="dq-ensemble-pill">ensemble decision</span>' : ''}`, false],
      ['Type', row => esc(row.labeler_type), false],
      ['Accuracy', row => rushApiFormatMetric(row.accuracy), true],
      ['Macro F1', row => rushApiFormatMetric(row.f1), true],
      ['Per-digit precision', row => rushApiFormatMetric(row.precision), true],
      ['Per-digit recall', row => rushApiFormatMetric(row.recall), true],
      ['Digit labels', () => '0-9', false],
      ['N', row => esc(row.n), true],
      ['Cost / 1k labels', row => formatCostPerThousand(row.cost_per_1000_labels), true]
    ] : [
      ['Labeler', row => `${esc(row.labeler_id)}${isEnsembleRow(row) ? ' <span class="dq-ensemble-pill">ensemble decision</span>' : ''}`, false],
      ['Type', row => esc(row.labeler_type), false],
      ['Accuracy', row => rushApiFormatMetric(row.accuracy), true],
      ['F1', row => rushApiFormatMetric(row.f1), true],
      ['Precision', row => rushApiFormatMetric(row.precision), true],
      ['Recall', row => rushApiFormatMetric(row.recall), true],
      ['FPR', row => rushApiFormatMetric(row.fpr), true],
      ['FNR', row => rushApiFormatMetric(row.fnr), true],
      ['Pos. Prop.', row => rushApiFormatMetric(row.positive_proportion), true],
      ['N', row => esc(row.n), true],
      ['Informedness', row => rushApiFormatMetric(row.informedness), true],
      ['Cost / 1k labels', row => formatCostPerThousand(row.cost_per_1000_labels), true]
    ];
    const header = columns.map(([label]) => `<th scope="col">${esc(label)}</th>`).join('');
    const body = rows.map(row => `<tr class="${isEnsembleRow(row) ? 'dq-row-ensemble' : ''}">${columns.map(([, render, numeric]) => `<td class="${numeric ? 'dq-cell-num' : ''}">${render(row)}</td>`).join('')}</tr>`).join('');
    target.innerHTML = `<table class="dq-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
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
    const height = 200;
    const pad = { left: 46, right: 18, top: 14, bottom: 38 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const denom = Math.max(1, sortedRuns.length - 1);
    const x = index => pad.left + (index / denom) * innerW;
    const y = value => pad.top + (1 - Math.max(0, Math.min(1, value))) * innerH;
    const grid = [0, 0.25, 0.5, 0.75, 1].map(value => {
      const yy = y(value).toFixed(1);
      return `<line x1="${pad.left}" y1="${yy}" x2="${width - pad.right}" y2="${yy}" class="chart-grid" /><text x="8" y="${Number(yy) + 4}" class="chart-label" font-size="6.5" style="font-size:6.5px">${Math.round(value * 100)}%</text>`;
    }).join('');
    const lines = Array.from(series.entries()).map(([id, points], index) => {
      const color = COLORS[index % COLORS.length];
      const pointText = points.map(point => `${x(point.index).toFixed(1)},${y(point.accuracy).toFixed(1)}`).join(' ');
      const circles = points.map(point => `<circle cx="${x(point.index).toFixed(1)}" cy="${y(point.accuracy).toFixed(1)}" r="3.5" fill="${color}"><title>${esc(id)} · ${esc(point.run.run_id || '')}: ${rushApiFormatMetric(point.accuracy)}</title></circle>`).join('');
      return `<polyline points="${pointText}" fill="none" stroke="${color}" stroke-width="2.5" />${circles}`;
    }).join('');
    const labels = sortedRuns.map((run, index) => {
      const shortId = String(run.run_id || `run-${index + 1}`).slice(0, 13);
      return `<text x="${x(index).toFixed(1)}" y="${height - 18}" class="chart-label" font-size="6.5" style="font-size:6.5px" text-anchor="middle">${esc(shortId)}</text>`;
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
        window.RUSH_API.catalog.policyVersions = payload.policy_versions.map(version => ({ version: version?.version || version }));
        populateFilters();
      }
      if (isMnistDemo() && !runs.length) {
        renderEmptyDecisionQuality(MNIST_EMPTY_RUN_MESSAGE);
        status('No scored MNIST run yet.');
        return;
      }
      setBenchmarkState(runs.length > 0);
      renderCards(runs);
      renderChart(runs);
      status(`Loaded ${runs.length} scored run(s).`);
    } catch (error) {
      const summary = $('#decisionQualitySummary');
      if (summary) summary.innerHTML = '';
      const warning = $('#decisionQualityWarning');
      if (warning) {
        warning.hidden = true;
        warning.textContent = '';
      }
      const narrative = $('#decisionQualityNarrative');
      if (narrative) {
        narrative.hidden = true;
        narrative.textContent = '';
      }
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
