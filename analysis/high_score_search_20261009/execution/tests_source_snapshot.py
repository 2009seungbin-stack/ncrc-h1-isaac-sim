"""CPU regression for official rsl_rl layout recognition and snapshot retries."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('snapshot_helper', Path(__file__).with_name('source_snapshot.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='snapshot_test_', dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root/'rsl_rl'
        self.train = self.root/'train'
        self.train.mkdir()
        for name in m.PROTECTED:
            path = self.train/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# provided source\n', encoding='utf-8')
        self.reference = self.root/'reference.json'
        self.reference.write_text(json.dumps({name: hashlib.sha256((self.train/name).read_bytes()).hexdigest()
                                              for name in m.PROTECTED}), encoding='utf-8')
        for name in ('__init__.py', 'algorithms/ppo.py', 'runners/on_policy_runner.py'):
            self.write_source(name)

    def write_source(self, name):
        path = self.package/name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Would raise if an implementation module were imported during preflight.
        path.write_text('raise RuntimeError("must never import during source preflight")\n', encoding='utf-8')

    def main(self):
        args = ['source_snapshot.py', '--train-dir', str(self.train), '--reference', str(self.reference)]
        with mock.patch.object(m, '__file__', str(self.root/'source_snapshot.py')), \
             mock.patch('sys.argv', args), \
             mock.patch.object(m.importlib.util, 'find_spec', return_value=SimpleNamespace(origin=str(self.package/'__init__.py'))), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m.main()

    def test_legacy_layout_hashes_all_package_python(self):
        self.write_source('modules/actor_critic.py')
        self.write_source('extensions/helper.py')
        result = m.snapshot_runtime_sources(self.package)
        self.assertEqual(result['policy_layout'], 'legacy_actor_critic')
        self.assertIn(str((self.package/'extensions/helper.py').resolve()), result['files'])

    def test_new_official_modular_layout_hashes_all_package_python(self):
        self.write_source('models/mlp_model.py')
        self.write_source('modules/mlp.py')
        self.write_source('modules/distribution.py')
        result = m.snapshot_runtime_sources(self.package)
        self.assertEqual(result['policy_layout'], 'modular_mlp_model')
        self.assertIn(str((self.package/'modules/distribution.py').resolve()), result['files'])

    def test_incomplete_layout_is_rejected(self):
        self.write_source('models/mlp_model.py')
        with self.assertRaisesRegex(ValueError, 'recognized actor'):
            m.snapshot_runtime_sources(self.package)

    def test_failed_report_preserved_then_retry_new_layout_succeeds(self):
        with self.assertRaises(SystemExit):
            self.main()
        prior_bytes = (self.root/'source_snapshot_report.json').read_bytes()
        self.assertFalse((self.root/'source_hashes_current.json').exists())
        self.write_source('models/mlp_model.py')
        self.write_source('modules/mlp.py')
        self.main()
        prior = list(self.root.glob('source_snapshot_report.previous_*.json'))
        self.assertEqual(len(prior), 1)
        self.assertEqual(prior[0].read_bytes(), prior_bytes)
        self.assertTrue((self.root/'source_hashes_current.json').is_file())
        self.assertEqual(json.loads((self.root/'protected_runtime_sources.json').read_text(encoding='utf-8'))['policy_layout'],
                         'modular_mlp_model')

    def test_successful_snapshot_cannot_be_overwritten(self):
        self.write_source('modules/actor_critic.py')
        self.main()
        manifest = (self.root/'source_hashes_current.json').read_bytes()
        with self.assertRaises(SystemExit):
            self.main()
        self.assertEqual((self.root/'source_hashes_current.json').read_bytes(), manifest)

    def test_changed_provided_source_cannot_be_accepted(self):
        self.write_source('modules/actor_critic.py')
        (self.train/'train.py').write_text('# changed provided training source\n', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'SOURCE MISMATCH'):
            self.main()
        self.assertFalse((self.root/'source_hashes_current.json').exists())
        self.assertFalse((self.root/'protected_runtime_sources.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
