"""Research evidence contracts; stdlib-only, no model or database calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

API = load('research_api', ROOT/'pipeline/web/research.py')
SHELL = load('research_shell', ROOT/'pipeline/web/research_shell.py')
ID = 'exp-20260707T004341-b9c4f0'
AREA = 'Generative_AI'

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for v in ('v0.1','v1.1'):
            p = self.root/'policy-graph'/AREA/v;p.mkdir(parents=True)
            (p/'GA.root.md').write_text('# Reference policy')
            (p/'edges.json').write_text('[]')
        self.run = {'area':AREA,'dry_run':False,'base_version':'v0.1','seed':13,'splits':{'test_ids':['held-1'],'holdout_n':50},'cycles':[
            {'k':0,'kind':'baseline','generator_before':AREA+'.v0.1','generator_after':AREA+'.v0.1'},
            {'k':1,'status':'accepted','generator_before':AREA+'.v0.1','generator_after':AREA+'.v1.1','train_ids':['train-1'],'metrics':{'test':{'system':{'n':20,'macro_fpr':.1}}}},
            {'k':2,'status':'skipped','generator_before':AREA+'.v1.1','generator_after':AREA+'.v1.1','train_ids':['train-2']}]}
        self.path = self.root/'data/experiments'/ID/'experiment.json';self.path.parent.mkdir(parents=True)
    def save(self):self.path.write_text(json.dumps(self.run))
    def read(self):self.save();return API.read_run(self.root,AREA,ID)
    def test_explicit_lineage(self):self.assertEqual([f['version'] for f in self.read()['frames']],['v0.1','v1.1','v1.1'])
    def test_rejected_step_keeps_incumbent(self):self.assertEqual(self.read()['frames'][-1]['before_version'],'v1.1')
    def test_positive_id_overlap_is_exposed(self):
        self.run['cycles'][1]['train_ids']=['held-1'];self.assertEqual(self.read()['split_audit']['train_gate_overlap'],1)
    def test_disjoint_identifiers_only(self):self.assertEqual(self.read()['split_audit']['train_gate_overlap'],0)
    def test_missing_train_identifiers_is_unknown(self):
        del self.run['cycles'][1]['train_ids'];self.assertIsNone(self.read()['split_audit']['train_gate_overlap'])
    def test_resampled_split_not_a_fixed_disjointness_pass(self):
        self.run['test_mode']='resample';self.assertIsNone(self.read()['split_audit']['train_gate_overlap'])
    def test_bad_parent_stops_lineage(self):
        self.run['cycles'][1]['generator_before']=AREA+'.v90.1';r=self.read();self.assertEqual(len(r['frames']),1);self.assertTrue(r['warnings'])
    def test_missing_accepted_artifact_stops_lineage(self):
        self.run['cycles'][1]['generator_after']=AREA+'.v90.1';self.assertEqual(len(self.read()['frames']),1)
    def test_skipped_cannot_change_version(self):
        self.run['cycles'][1]['status']='skipped';self.assertEqual(len(self.read()['frames']),1)
    def test_dry_run_rejected(self):
        self.run['dry_run']=True
        with self.assertRaises(ValueError):self.read()
    def test_missing_dry_flag_not_evidence(self):
        del self.run['dry_run']
        with self.assertRaises(ValueError):self.read()
    def test_cross_area_rejected(self):
        self.run['area']='MNIST_Digits'
        with self.assertRaises(ValueError):self.read()
    def test_baseline_pin_mismatch_rejected(self):
        self.run['cycles'][0]['generator_after']=AREA+'.v1.1'
        with self.assertRaises(ValueError):self.read()
    def test_unknown_metrics_not_zero(self):self.assertNotIn('metrics',self.read()['cycles'][0])
    def test_nan_cleaned(self):
        self.run['cycles'][1]['metrics']['test']['system']['macro_fpr']=float('nan')
        self.assertIsNone(self.read()['cycles'][1]['metrics']['test']['system']['macro_fpr'])
    def test_no_model_credentials_exposed(self):
        self.run['api_key']='secret';self.run['cycles'][1]['api_key']='secret';self.assertNotIn('secret',json.dumps(self.read()))
    def test_read_does_not_mutate_evidence(self):
        self.save();before=self.path.read_bytes();API.read_run(self.root,AREA,ID);self.assertEqual(before,self.path.read_bytes())
    def test_duplicate_cycle_stops_lineage(self):
        self.run['cycles'].insert(2,self.run['cycles'][1].copy());r=self.read();self.assertEqual(len(r['frames']),2);self.assertTrue(r['warnings'])
    def test_unknown_transition_stops_lineage(self):
        self.run['cycles'][1]['status']='mysterious';self.assertEqual(len(self.read()['frames']),1)
    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'outside.json';target.write_text(json.dumps(self.run));self.path.symlink_to(target)
            with self.assertRaises(ValueError):API.read_run(self.root,AREA,ID)
    def test_incomplete_policy_not_claimed(self):
        (self.root/'policy-graph'/AREA/'v1.1'/'edges.json').unlink();self.assertEqual(len(self.read()['frames']),1)
    def test_duplicate_query_rejected(self):self.assertEqual(API.dispatch(self.root,f'/api/studio/research-run?area={AREA}&id={ID}&id={ID}')[0],400)
    def test_query_root_cannot_be_chosen(self):self.assertEqual(API.dispatch(self.root,f'/api/studio/research-run?area={AREA}&id={ID}&root=/etc')[0],400)
    def test_traversal_rejected(self):self.assertEqual(API.dispatch(self.root,f'/api/studio/research-run?area={AREA}&id=../../etc/passwd')[0],400)
    def test_unknown_endpoint_404(self):self.assertEqual(API.dispatch(self.root,'/api/studio/nope')[0],404)
    def test_shell_keeps_original_controls(self):
        source='<html><head><title>Old</title></head><body><section id="experiment"><input id="experimentTestN" value="100"/></section></body></html>'
        result=SHELL.enhance_lab_html(source);self.assertIn('id="experimentTestN" value="100"',result);self.assertIn('research.js',result)
    def test_shell_idempotence(self):
        source='<head></head><section id="experiment"></section>';once=SHELL.enhance_lab_html(source);self.assertEqual(once,SHELL.enhance_lab_html(once))
    def test_shadow_methods_isolated(self):self.assertIn('src="studio-about.js',SHELL.enhance_lab_html('<div id="studioView"></div><script src="about.js?v=1"></script>'))

if __name__=='__main__':unittest.main()
