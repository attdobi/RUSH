"""Dependency-free regression tests: python -m unittest discover -s tests -p test_studio_contracts.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


API = load('studio_api', 'pipeline/web/studio.py')
METRICS = load('studio_metrics', 'pipeline/scoring/multiclass_counts.py')


class MetricRegression(unittest.TestCase):
    def metric(self, p, t, classes=('0', '1')):
        return METRICS.compute_multiclass_metrics(p, t, classes=classes)

    def test_all_wrong_is_zero_f1(self):
        m = self.metric(['1', '0'], ['0', '1'])
        self.assertEqual(m['macro_f1'], 0)
        self.assertEqual(m['micro_f1'], 0)

    def test_never_predicted_supported_class_is_zero(self):
        m = self.metric(['0', '0'], ['0', '1'])
        self.assertEqual(m['per_class']['1']['f1'], 0)
        self.assertAlmostEqual(m['macro_f1'], 1/3, places=6)

    def test_constant_mnist_classifier_does_not_get_inflated_macro(self):
        classes = tuple(map(str, range(10)))
        m = self.metric(['0']*10, list(classes), classes)
        self.assertAlmostEqual(m['macro_f1'], 0.018182, places=6)
        self.assertEqual(m['accuracy'], 0.1)

    def test_absent_class_has_undefined_f1(self):
        m = self.metric(['0'], ['0'])
        self.assertIsNone(m['per_class']['1']['f1'])
        self.assertEqual(m['macro_f1'], 1)

    def test_false_positive_only_class_has_zero_f1(self):
        self.assertEqual(self.metric(['1'], ['0'])['per_class']['1']['f1'], 0)

    def test_abstention_is_reported(self):
        m = self.metric(['0', 'abstain'], ['0', '1'])
        self.assertEqual((m['n'], m['n_abstained'], m['accuracy']), (1, 1, 1))

    def test_all_abstain_has_no_decided_quality(self):
        m = self.metric(['abstain'], ['0'])
        self.assertIsNone(m['macro_f1'])
        self.assertEqual(m['n_abstained'], 1)

    def test_empty_data(self):
        self.assertIsNone(self.metric([], [])['micro_f1'])

    def test_unknown_matching_labels_not_correct(self):
        self.assertEqual(self.metric(['other'], ['other'])['accuracy'], 0)

    def test_input_contracts(self):
        with self.assertRaises(ValueError): self.metric(['0'], [])
        with self.assertRaises(ValueError): self.metric([], [], ())
        with self.assertRaises(ValueError): self.metric([], [], ('0', '0'))

    def test_micro_equals_accuracy_for_closed_single_label(self):
        m = self.metric(['0', '1', '0', '0'], ['0', '1', '1', '0'])
        self.assertEqual(m['micro_f1'], m['accuracy'])


class StudioApiRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for v in ['v0.1', 'v1.1']:
            self.write_policy(v)

    def tearDown(self):
        self.tmp.cleanup()

    def write_policy(self, v, body='A general policy rule.'):
        base = self.root / 'policy-graph' / 'Generative_AI' / v
        base.mkdir(parents=True, exist_ok=True)
        (base/'GA.root.md').write_text('---\nid: GA.root\nnode_type: root\nparent: null\ntitle: Policy intent\n---\n'+body)
        (base/'GA.boundary.md').write_text('---\nid: GA.boundary\nnode_type: boundary\nparent: GA.root\n---\nA boundary rule.')
        (base/'edges.json').write_text('[]')

    def write_run(self, name='exp-test', **overrides):
        run = {'area':'Generative_AI', 'dry_run':False, 'base_version':'v0.1', 'run_number':1,
               'cycles':[{'k':1,'status':'accepted','generator_before':'Generative_AI.v0.1','generator_after':'Generative_AI.v1.1'},
                         {'k':2,'status':'skipped','generator_before':'Generative_AI.v1.1','generator_after':'Generative_AI.v1.1'}]}
        run.update(overrides)
        path = self.root/'data'/'experiments'/name/'experiment.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(run))
        return path

    def test_snapshot_contains_real_body_and_hash(self):
        s = API.snapshot(self.root,'Generative_AI','v0.1')
        self.assertEqual(s['origin'],'recorded')
        self.assertTrue(any(n['body']=='A general policy rule.' for n in s['nodes']))
        self.assertTrue(all(len(n['content_hash'])==64 for n in s['nodes']))

    def test_metadata_version_does_not_change_body_hash(self):
        a = API.snapshot(self.root,'Generative_AI','v0.1')
        b = API.snapshot(self.root,'Generative_AI','v1.1')
        self.assertEqual([n['content_hash'] for n in a['nodes']],[n['content_hash'] for n in b['nodes']])

    def test_content_edit_changes_hash(self):
        old = API.snapshot(self.root,'Generative_AI','v0.1')
        self.write_policy('v0.1','A different generalized rule.')
        new = API.snapshot(self.root,'Generative_AI','v0.1')
        self.assertNotEqual(old['nodes'][1]['content_hash'],new['nodes'][1]['content_hash'])

    def test_explicit_parent_is_recorded_not_inferred(self):
        s = API.snapshot(self.root,'Generative_AI','v0.1')
        self.assertEqual(s['edges'][0]['provenance'],'frontmatter')

    def test_catalog_is_not_claimed_as_lineage(self):
        h = API.history(self.root,'Generative_AI')
        self.assertFalse(h['series'][0]['lineage'])

    def test_run_lineage_keeps_incumbent_on_skip(self):
        self.write_run()
        h = API.history(self.root,'Generative_AI')['series'][1]
        self.assertTrue(h['lineage'])
        self.assertEqual([f['version'] for f in h['frames']],['v0.1','v1.1','v1.1'])

    def test_dry_runs_and_unknown_dry_run_status_not_evidence(self):
        self.write_run(dry_run=True)
        self.write_run(name='unknown',dry_run=None)
        self.assertEqual(len(API.history(self.root,'Generative_AI')['series']),1)

    def test_lineage_break_is_not_silently_connected(self):
        self.write_run(cycles=[{'k':1,'status':'accepted','generator_before':'Generative_AI.v99.1','generator_after':'Generative_AI.v1.1'}])
        h = API.history(self.root,'Generative_AI')
        self.assertEqual(len(h['series']),1)
        self.assertTrue(h['warnings'])

    def test_missing_accepted_snapshot_stops_replay(self):
        self.write_run(cycles=[{'k':1,'status':'accepted','generator_before':'Generative_AI.v0.1','generator_after':'Generative_AI.v9.9'}])
        self.assertTrue(API.history(self.root,'Generative_AI')['warnings'])

    def test_invalid_area_and_path_traversal_fail_closed(self):
        for url in ['/api/studio/history?area=../../etc','/api/studio/snapshot?version=../../etc/passwd']:
            self.assertEqual(API.dispatch(self.root,url)[0],400)

    def test_duplicate_parameters_rejected(self):
        self.assertEqual(API.dispatch(self.root,'/api/studio/history?area=Generative_AI&area=MNIST_Digits')[0],400)

    def test_symlink_cannot_expose_external_file(self):
        with tempfile.TemporaryDirectory() as other:
            outside=Path(other)/'secret.md';outside.write_text('do not expose')
            target=self.root/'policy-graph/Generative_AI/v0.1/GA.root.md'
            target.unlink();target.symlink_to(outside)
            with self.assertRaises(ValueError):API.snapshot(self.root,'Generative_AI','v0.1')

    def test_reading_does_not_mutate_artifacts(self):
        self.write_run()
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        API.history(self.root,'Generative_AI');API.snapshot(self.root,'Generative_AI','v0.1')
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_unknown_endpoint_is_404(self):
        self.assertEqual(API.dispatch(self.root,'/api/studio/not-here')[0],404)


if __name__ == '__main__':
    unittest.main()
