"""Exercise the actual HTTP boundary, not only the underlying label store."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from labeling.server import create_server

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / 'web' / 'labeling' / 'fixtures.json'
TOKENS = {
    'unit-test-policy': {'id': 'demo-policy', 'name': 'Alex Rivera', 'role': 'policy'},
    'unit-test-ops': {'id': 'test-ops', 'name': 'Test Ops', 'role': 'ops'},
}


@contextmanager
def running_server():
    with tempfile.TemporaryDirectory() as tmp:
        server = create_server(host='127.0.0.1', port=0, repo_root=REPO,
                               db_path=Path(tmp) / 'labels.sqlite3',
                               seed_demo=True, tokens=TOKENS)
        worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        worker.start()
        try:
            yield f'http://127.0.0.1:{server.server_address[1]}'
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


def call(base, path, payload=None, *, token='unit-test-policy', key=None, extra=None):
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    if payload is not None:
        headers['Content-Type'] = 'application/json'
    if key:
        headers['Idempotency-Key'] = key
    headers.update(extra or {})
    request = Request(base + path, headers=headers,
                      data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urlopen(request, timeout=5) as response:
            raw = response.read()
            return response.status, json.loads(raw) if response.headers.get_content_type() == 'application/json' else raw.decode()
    except HTTPError as error:
        raw = error.read()
        try:
            result = json.loads(raw)
        except ValueError:
            result = raw.decode()
        return error.code, result


class LabelingHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_blind_boundary_static_fixture_and_authenticated_identity(self):
        blind = next(i for i in self.fixture['items']
                     if i['type'] == 'assistant' and not i.get('seed_independent_labels') and i.get('split') != 'holdout')
        with running_server() as base:
            status, _ = call(base, '/api/labeling/bootstrap', token=None)
            self.assertEqual(401, status)
            status, runtime = call(base, '/runtime.js', token=None)
            self.assertEqual(200, status)
            self.assertIn('api', runtime)
            status, _ = call(base, '/fixtures.json', token=None)
            self.assertIn(status, (403, 404))
            status, before = call(base, '/api/labeling/items/' + blind['id'])
            self.assertEqual(200, status)
            self.assertFalse(before['guidance_visible'])
            for target in before['targets']:
                self.assertIsNone(target.get('ai'))
                self.assertIn(target.get('gold'), (None, {}))
            labels = {t['id']: t['choices'][0] for t in blind['targets']}
            payload = {'base_revision': before['revision'], 'phase': 'independent', 'decision': 'save',
                       'labels': labels, 'rationale': 'Reviewed each original target independently.', 'policy_refs': []}
            status, after = call(base, f"/api/labeling/items/{blind['id']}/reviews", payload, key='first-pass-1')
            self.assertEqual(200, status, after)
            self.assertTrue(after['guidance_visible'])
            self.assertEqual(labels, after['independent_labels'])
            self.assertEqual({'input', 'output'}, set(after['independent_labels']))
            self.assertTrue(all(t.get('ai') for t in after['targets']))
            status, replay = call(base, f"/api/labeling/items/{blind['id']}/reviews", payload, key='first-pass-1')
            self.assertEqual(200, status, replay)
            self.assertEqual(after['revision'], replay['revision'])
            status, conflict = call(base, f"/api/labeling/items/{blind['id']}/reviews",
                                    {**payload, 'rationale': 'Different body'}, key='first-pass-1')
            self.assertEqual(409, status, conflict)
            status, conflict = call(base, f"/api/labeling/items/{blind['id']}/reviews", payload, key='stale-request')
            self.assertEqual(409, status, conflict)

    def test_approval_publication_and_apo_handoff_pin_an_immutable_epoch(self):
        with running_server() as base:
            status, bootstrap = call(base, '/api/labeling/bootstrap')
            self.assertEqual(200, status)
            initial_epoch = bootstrap['epoch']
            _, old_snapshot = call(base, f'/api/labeling/epochs/{initial_epoch}/export')
            _, item = call(base, '/api/labeling/items/turn-1042')
            labels = {t['id']: t['choices'][0] for t in item['targets']}
            payload = {'base_revision': item['revision'], 'phase': 'adjudication', 'decision': 'approve',
                       'labels': labels, 'rationale': 'The two targets require separate decisions under the cited policy.',
                       'policy_refs': item['targets'][0].get('ai', {}).get('policy_refs', [])}
            status, approved = call(base, '/api/labeling/items/turn-1042/reviews', payload, key='approve-1')
            self.assertEqual(200, status, approved)
            self.assertEqual('approved', approved['status'])
            status, epoch = call(base, '/api/labeling/epochs', {'expected_epoch': initial_epoch, 'note': 'Reviewed correction'}, key='publish-1')
            self.assertEqual(200, status, epoch)
            self.assertEqual(initial_epoch + 1, epoch['epoch'])
            self.assertEqual('local_published', epoch['status'])
            _, old_snapshot_again = call(base, f'/api/labeling/epochs/{initial_epoch}/export')
            self.assertEqual(old_snapshot, old_snapshot_again)
            _, snapshot = call(base, f"/api/labeling/epochs/{epoch['epoch']}/export")
            protected_ids = {i['id'] for i in self.fixture['items'] if i.get('split') == 'holdout'}
            self.assertTrue(protected_ids)
            for protected_id in protected_ids:
                self.assertNotIn(protected_id, json.dumps(snapshot))
            status, run = call(base, '/api/labeling/reruns', {'epoch': epoch['epoch'], 'note': 'Pinned handoff'}, key='rerun-1')
            self.assertEqual(200, status, run)
            self.assertEqual('awaiting_connector', run['status'])
            self.assertEqual(epoch['epoch'], run['epoch'])
            self.assertEqual(snapshot['digest'], run['digest'])

    def test_role_and_cross_origin_controls_are_server_enforced(self):
        with running_server() as base:
            _, bootstrap = call(base, '/api/labeling/bootstrap', token='unit-test-ops')
            self.assertEqual('ops', bootstrap['viewer']['role'])
            status, result = call(base, '/api/labeling/epochs', {'expected_epoch': bootstrap['epoch'], 'note': 'Not authorized'}, token='unit-test-ops', key='ops-publish')
            self.assertEqual(403, status, result)
            status, result = call(base, '/api/labeling/reruns', {'epoch': bootstrap['epoch'], 'note': 'Not authorized'}, token='unit-test-ops', key='ops-run')
            self.assertEqual(403, status, result)
            status, result = call(base, '/api/labeling/epochs', {'expected_epoch': bootstrap['epoch'], 'note': 'Cross origin'}, extra={'Origin': 'https://untrusted.example'}, key='cross-origin')
            self.assertEqual(403, status, result)


if __name__ == '__main__':
    unittest.main()
