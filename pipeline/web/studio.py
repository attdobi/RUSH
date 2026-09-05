"""Read-only, bounded policy snapshots and explicit experiment lineage for the studio.

No provider calls, mutations, credentials, label rows or metric claims are exposed.
The caller supplies the evidence root; query strings cannot choose filesystem roots.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

AREAS = {"Generative_AI": "GA.root", "MNIST_Digits": "MD.root"}
VERSION = re.compile(r"^v\d+\.\d+$")
MAX_FILE = 8_000_000
MAX_VERSIONS = 500
MAX_RUNS = 200


def _read(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("Artifact is outside the evidence root or is missing")
    if resolved.stat().st_size > MAX_FILE:
        raise ValueError("Artifact exceeds the read limit")
    return resolved.read_text(encoding="utf-8")


def _versions(root: Path, area: str) -> list[str]:
    if area not in AREAS:
        raise ValueError("Unknown policy area")
    base = root / "policy-graph" / area
    if not base.resolve().is_relative_to(root.resolve()):
        raise ValueError("Policy directory is outside the evidence root")
    result = [p.name for p in base.iterdir() if p.is_dir() and VERSION.fullmatch(p.name)
              and (p / f"{AREAS[area]}.md").is_file() and (p / "edges.json").is_file()] if base.is_dir() else []
    return sorted(result, key=lambda v: tuple(map(int, v[1:].split('.'))))[-MAX_VERSIONS:]


def _markdown(text: str) -> tuple[dict[str, str], str]:
    text = re.sub(r"^(?:\s*<!--[\s\S]*?-->)+\s*", "", text)
    match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n?", text)
    if not match:
        return {}, text.strip()
    meta = {}
    for line in match[1].splitlines():
        item = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if item:
            meta[item[1]] = item[2].strip().strip('\"\'')
    return meta, text[match.end():].strip()


def snapshot(repo_root: str | Path, area: str, version: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if version not in _versions(root, area):
        raise ValueError("Unknown or incomplete policy version")
    base = root / "policy-graph" / area / version
    paths = sorted(base.glob('*.md'))
    if len(paths) > 2000:
        raise ValueError("Graph exceeds the node limit")
    nodes, edges, warnings = [], [], []
    total_bytes = 0
    for path in paths:
        raw = _read(root, path)
        total_bytes += len(raw.encode('utf-8'))
        if total_bytes > MAX_FILE:
            raise ValueError('Graph exceeds the aggregate read limit')
        meta, body = _markdown(raw)
        node_id = meta.get('id') or path.stem
        parent = meta.get('parent')
        parent = None if parent in (None, '', 'null', '~') else parent
        nodes.append({'id': node_id, 'title': meta.get('title') or node_id,
                      'node_type': meta.get('node_type') or 'guideline', 'parent': parent,
                      'polarity': meta.get('polarity') or 'mixed', 'status': meta.get('status') or 'unspecified',
                      'body': body, 'content_hash': hashlib.sha256(body.encode()).hexdigest(),
                      'source': f'policy-graph/{area}/{version}/{path.name}'})
        # An explicit frontmatter parent is evidence; an id-prefix guess is not.
        if parent:
            edges.append({'source': node_id, 'target': parent, 'type': 'subtype_of', 'provenance': 'frontmatter'})
    raw_edges = json.loads(_read(root, base / 'edges.json'))
    if not isinstance(raw_edges, list) or len(raw_edges) > 10000:
        raise ValueError('Invalid or oversized edges artifact')
    for edge in raw_edges:
        if not isinstance(edge, dict):
            warnings.append('Ignored a malformed edge')
            continue
        source = edge.get('source') or edge.get('source_node_id')
        target = edge.get('target') or edge.get('target_node_id') or edge.get('to')
        if source and target:
            edges.append({'source': str(source), 'target': str(target),
                          'type': str(edge.get('edge_type') or edge.get('type') or 'related_to'),
                          'synthetic': bool(edge.get('synthetic', False)), 'provenance': 'edges.json'})
    # Include inline non-hierarchical frontmatter edges used by RUSH node files.
    for path, node in zip(paths, nodes):
        raw = _read(root, path)
        front = re.match(r'^---\s*\n([\s\S]*?)\n---', raw)
        if not front:
            continue
        in_edges = False
        for line in front[1].splitlines():
            if line and not line[0].isspace():
                in_edges = line.strip() == 'edges:'
            elif in_edges:
                item = re.match(r'\s*-\s*\{(.*)\}\s*$', line)
                if item:
                    fields = dict((k.strip(), v.strip().strip('\"\'')) for part in item[1].split(',') for k, sep, v in [part.partition(':')] if sep)
                    target, kind = fields.get('to') or fields.get('target'), fields.get('type') or fields.get('edge_type')
                    if target and kind:
                        edges.append({'source': node['id'], 'target': target, 'type': kind, 'provenance': 'frontmatter'})
    return {'origin': 'recorded', 'area': area, 'version': version, 'nodes': nodes,
            'edges': edges, 'warnings': warnings, 'metrics_status': 'Not inferred from graph size. Historical scores require re-scoring after the F1 correction.'}


def history(repo_root: str | Path, area: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    versions = _versions(root, area)
    if not versions:
        return {'origin': 'recorded', 'area': area, 'series': [], 'warnings': ['No complete policy snapshots found.']}
    series = [{'id': 'catalog', 'title': 'Snapshot catalog · not a lineage', 'lineage': False,
               'frames': [{'version': v, 'title': v, 'status': 'snapshot', 'detail': 'Independent recorded snapshot. Adjacent catalog entries are not asserted to be parent and child.'} for v in versions]}]
    warnings = []
    if len(versions) == MAX_VERSIONS:
        warnings.append(f'Catalog is limited to the latest {MAX_VERSIONS} named versions.')
    base = root / 'data' / 'experiments'
    if not base.resolve().is_relative_to(root):
        raise ValueError('Experiment directory is outside the evidence root')
    paths = sorted(base.glob('*/experiment.json'), key=lambda p: p.parent.name, reverse=True) if base.is_dir() else []
    if len(paths) > MAX_RUNS:
        warnings.append(f'Only the latest {MAX_RUNS} experiment files were inspected.')
    for path in paths[:MAX_RUNS]:
        try:
            run = json.loads(_read(root, path))
            if not isinstance(run, dict) or run.get('dry_run') is not False or run.get('area') != area:
                continue
            initial = run.get('base_version')
            if initial not in versions:
                continue
            frames = [{'version': initial, 'title': 'Baseline', 'status': 'baseline', 'detail': 'Recorded baseline for this experiment.'}]
            current = initial
            for cycle in run.get('cycles', [])[:500]:
                if not isinstance(cycle, dict) or cycle.get('k') == 0:
                    continue
                before = str(cycle.get('generator_before') or '').removeprefix(area + '.')
                after = str(cycle.get('generator_after') or '').removeprefix(area + '.')
                if before != current:
                    warnings.append(f'{path.parent.name}: incomplete lineage; replay stops at the last verified step.')
                    break
                status = str(cycle.get('status') or 'unknown')
                if status == 'accepted':
                    if after not in versions or after == current:
                        warnings.append(f'{path.parent.name}: accepted snapshot missing; replay stops.')
                        break
                    current = after
                elif status not in ('skipped', 'rejected') or after != current:
                    continue
                frames.append({'version': current, 'title': f"Cycle {cycle.get('k', '?')} · {status}", 'status': status,
                               'detail': 'Recorded accepted policy update.' if status == 'accepted' else 'Recorded rejection/skip. The incumbent policy remains in force.'})
            if len(frames) > 1:
                series.append({'id': path.parent.name, 'title': f"Run {run.get('run_number', '?')} · {len(frames)-1} steps", 'lineage': True,
                               'frames': frames, 'source': f'data/experiments/{path.parent.name}/experiment.json'})
        except (ValueError, OSError, TypeError, AttributeError):
            warnings.append(f'{path.parent.name}: unreadable experiment skipped.')
    return {'origin': 'recorded', 'area': area, 'series': series, 'warnings': warnings,
            'metric_notice': 'Legacy gate scores are not shown as current evidence. Re-score with the corrected F1 definition.'}


def dispatch(repo_root: str | Path, url: str) -> tuple[int, dict[str, Any]]:
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    if any(len(v) != 1 for v in query.values()):
        return 400, {'error': 'Repeated query parameters are not supported'}
    area = query.get('area', ['Generative_AI'])[0]
    try:
        if parts.path == '/api/studio/history':
            return 200, history(repo_root, area)
        if parts.path == '/api/studio/snapshot':
            return 200, snapshot(repo_root, area, query.get('version', [''])[0])
        return 404, {'error': 'Unknown studio endpoint'}
    except (ValueError, OSError, TypeError):
        return 400, {'error': 'Invalid, missing or unreadable policy artifact'}
