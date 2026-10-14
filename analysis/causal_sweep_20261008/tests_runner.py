"""CPU-only contract tests; run with python -B tests_runner.py."""
import ast
import importlib.util
import json
from pathlib import Path
import signal
import tarfile
import tempfile
import time
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('causal_runner', Path(__file__).with_name('runner.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class PureContracts(unittest.TestCase):
    def test_utf8_patch_preserves_every_byte_outside_dictionary(self):
        prefix = '# 한글 설명\r\n인사 = "안녕"; REWARD_WEIGHTS = '.encode()
        suffix = '  # 원문 유지\r\ndef untouched():\r\n    return "그대로"\r\n'.encode()
        original = prefix + b"{'old': 5}" + suffix
        for arm in r.ARMS:
            result = r.patch_rewards(original, arm)
            self.assertEqual(result, prefix + repr(r.arm_weights(arm)).encode() + suffix)
            ast.parse(result.decode())

    def test_patch_rejects_missing_duplicate_and_non_dictionary_assignments(self):
        for original in (b'x = {}', b'REWARD_WEIGHTS = dict(a=1)',
                         b'REWARD_WEIGHTS = {}\nREWARD_WEIGHTS = {}'):
            with self.subTest(original=original), self.assertRaises(ValueError):
                r.patch_rewards(original, 'control')

    def test_each_intervention_changes_exactly_one_effective_weight(self):
        base = r.expected_weights('control')
        for arm, changed in [('torque', 'dof_torques_l2'), ('posture', 'flat_orientation_l2')]:
            candidate = r.expected_weights(arm)
            self.assertEqual(set(candidate), set(base))
            self.assertEqual([key for key in base if base[key] != candidate[key]], [changed])

    def test_deadline_reserves_45_minutes_at_ten_hour_boundary(self):
        self.assertEqual(r.validate_deadlines(1000, 34300, now=1001), 37000)
        with self.assertRaises(ValueError):
            r.validate_deadlines(1000, 34300.001, now=1001)

    def test_deadline_rejects_nonfinite_future_start_and_expired(self):
        for start, end, now in [(float('nan'), 2000, 1000), (1000, float('inf'), 1001),
                                (1007, 2000, 1000), (1000, 2000, 2000)]:
            with self.subTest(start=start, end=end, now=now), self.assertRaises(ValueError):
                r.validate_deadlines(start, end, now=now)

    def test_yaml_parser_is_scoped_to_rewards_and_does_not_construct_objects(self):
        text = 'weight: 999\nrewards:\n  term:\n    func: !!python/name:bad\n    weight: -0.3\nother:\n  term:\n    weight: 99\n'
        self.assertEqual(r.yaml_reward_weights(text), {'term': -0.3})


class RunnerContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='causal_runner_test_', dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.train = self.base/'analysis', self.base/'train'
        self.root.mkdir()
        self.train.mkdir()
        for name in r.INPUTS:
            (self.root/name).write_text('# unchanged analysis source\n', encoding='utf-8')
        for name in r.PROTECTED:
            path = self.train/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# provided source\n', encoding='utf-8')
        self.original = '# 원문\nREWARD_WEIGHTS = {"old": 1}\n'.encode()
        (self.train/'quadruped_rewards.py').write_bytes(self.original)
        self.manifest = self.base/'sources.json'
        self.manifest.write_text(json.dumps({name: r.sha(self.train/name) for name in r.PROTECTED}))
        self.start = time.time()-30
        self.deadline = self.start+33300
        self.runner = self.make_runner()

    def make_runner(self, resume=False):
        return r.Runner(self.root, self.train, 'unused-isaac', self.manifest,
                        self.start, self.deadline, resume=resume)

    def test_guard_rejects_training_source_edit(self):
        self.runner.initialize()
        (self.train/'train.py').write_text('# changed')
        with self.assertRaisesRegex(RuntimeError, 'Protected training'):
            self.runner.guard()

    def test_guard_rejects_evaluator_edit(self):
        self.runner.initialize()
        (self.root/'batch_eval.py').write_text('# changed')
        with self.assertRaisesRegex(RuntimeError, 'Analysis evaluator'):
            self.runner.guard()

    def test_unknown_external_reward_edit_is_never_overwritten(self):
        self.runner.initialize()
        external = b'REWARD_WEIGHTS = {"external": 4}\n'
        (self.train/'quadruped_rewards.py').write_bytes(external)
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized external'):
            self.runner.restore_rewards()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), external)

    def test_resume_restores_known_applied_rewards(self):
        self.runner.initialize()
        self.runner.apply('torque')
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), self.original)

    def test_completed_arm_resume_never_launches_training(self):
        self.runner.initialize()
        checkpoint = self.root/'complete.pt'
        checkpoint.write_bytes(b'checkpoint evidence')
        self.runner.data['arms']['control'] = dict(status='complete', checkpoint='complete.pt', sha256=r.sha(checkpoint))
        self.runner.save()
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        with mock.patch.object(r.subprocess, 'Popen') as launch:
            resumed.train_arm('control')
        launch.assert_not_called()

    def test_interrupted_arm_resume_never_retrains(self):
        self.runner.initialize()
        self.runner.data['arms']['control'] = dict(status='started')
        self.runner.save()
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        with mock.patch.object(r.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'never retrain'):
                resumed.train_arm('control')
        launch.assert_not_called()

    def test_completed_checkpoint_tamper_blocks_resume_arm(self):
        self.runner.initialize()
        checkpoint = self.root/'complete.pt'
        checkpoint.write_bytes(b'original')
        self.runner.data['arms']['control'] = dict(status='complete', checkpoint='complete.pt', sha256=r.sha(checkpoint))
        checkpoint.write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint hash mismatch'):
            self.runner.train_arm('control')

    def test_expired_resume_restores_rewards_before_rejecting_compute(self):
        self.runner.initialize()
        self.runner.apply('posture')
        resumed = self.make_runner(resume=True)
        with mock.patch.object(resumed, 'now', return_value=self.deadline+1):
            with self.assertRaisesRegex(ValueError, 'passed'):
                resumed.initialize()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), self.original)

    def test_insufficient_training_budget_does_not_apply_or_launch(self):
        self.runner.initialize()
        with mock.patch.object(self.runner, 'now', return_value=self.deadline-100), \
             mock.patch.object(self.runner, 'apply') as apply, \
             mock.patch.object(r.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'training estimate'):
                self.runner.train_arm('control')
        apply.assert_not_called()
        launch.assert_not_called()

    def test_sigterm_path_restores_rewards_and_records_recovery(self):
        def interrupted_training(arm):
            self.runner.apply(arm)
            self.runner.data['arms'][arm] = {'status': 'started'}
            r.signal_stop(signal.SIGTERM, None)
        with mock.patch.object(self.runner, 'train_arm', side_effect=interrupted_training), \
             mock.patch.object(self.runner, 'bundle') as bundle, \
             mock.patch.object(r, 'stop_process') as stop:
            with self.assertRaises(r.StopRequested):
                self.runner.execute()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), self.original)
        state = json.loads((self.root/'state.json').read_text())
        self.assertEqual(state['stage'], 'STOPPED_REQUIRES_REVIEW')
        self.assertEqual(state['cleanup_errors'], [])
        self.assertEqual(state['arms']['control']['status'], 'started')
        stop.assert_called_once_with(None)
        bundle.assert_called_once()

    def test_preserve_checkpoints_and_parameters_survive_source_deletion(self):
        run = self.train/'logs/rsl_rl/quadruped'/('stamp_'+self.runner.run_name('control'))
        (run/'params').mkdir(parents=True)
        files = {'model_0.pt': b'first checkpoint', 'model_1999.pt': b'final checkpoint',
                 'params/env.yaml': b'rewards: {}', 'params/agent.yaml': b'seed: 42',
                 'events.out.tfevents.test': b'event evidence'}
        for name, content in files.items():
            (run/name).write_bytes(content)
        self.runner.preserve('control')
        destination = self.root/'preserved/control'/run.name
        for name, content in files.items():
            self.assertEqual((destination/name).read_bytes(), content)
            (run/name).unlink()
        self.runner.preserve('control')
        for name, content in files.items():
            self.assertEqual((destination/name).read_bytes(), content)

    def test_bundle_regenerates_archive_and_manifest_matches_file_hashes(self):
        evidence = self.root/'evidence.json'
        archive = self.root.parent/(self.root.name+'_results.tar.gz')
        evidence.write_bytes(b'{"version":1}')
        self.runner.bundle()
        first_hash = r.sha(archive)
        evidence.write_bytes(b'{"version":2}')
        self.runner.bundle()
        self.assertNotEqual(r.sha(archive), first_hash)
        manifest = json.loads((self.root/'manifest.json').read_text())
        for name, entry in manifest['files'].items():
            self.assertEqual(entry['sha256'], r.sha(self.root/name))
            self.assertEqual(entry['bytes'], (self.root/name).stat().st_size)
        sidecar = json.loads(archive.with_name(archive.name+'.sha256.json').read_text())
        self.assertEqual(sidecar['sha256'], r.sha(archive))
        with tarfile.open(archive) as tar:
            self.assertEqual(tar.extractfile(self.root.name+'/evidence.json').read(), b'{"version":2}')
            archived_manifest = json.loads(tar.extractfile(self.root.name+'/manifest.json').read())
        self.assertEqual(archived_manifest, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
