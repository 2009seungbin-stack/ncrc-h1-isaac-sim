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
    def test_exited_wrapper_does_not_leave_live_training_descendants(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = 0
        live = {'group': True}
        signals = []
        def killpg(pid, sig):
            self.assertEqual(pid, 12345)
            if not live['group']:
                raise ProcessLookupError()
            if sig:
                signals.append(sig)
                if sig == signal.SIGTERM:
                    live['group'] = False
        with mock.patch.object(r.os, 'killpg', side_effect=killpg, create=True), \
             mock.patch.object(r.signal, 'SIGKILL', 9, create=True), \
             mock.patch.object(r.time, 'monotonic', side_effect=range(0, 500, 31)), \
             mock.patch.object(r.time, 'sleep'):
            r.stop_process(process)
        self.assertEqual(signals, [signal.SIGINT, signal.SIGTERM])

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
                r.patch_rewards(original, 'S1_tracking_pair')

    def test_tracking_strengths_keep_ratio_and_other_weights(self):
        for arm, linear, yaw in [('S1_tracking_pair',3.0,2.25), ('R1_tracking_double',4.0,3.0)]:
            candidate = r.expected_weights(arm)
            self.assertEqual(candidate['track_lin_vel_xy_exp'], linear)
            self.assertEqual(candidate['track_ang_vel_z_exp'], yaw)
            self.assertEqual(linear/yaw, 4/3)
            self.assertEqual({k:v for k,v in candidate.items() if not k.startswith('track_')},
                             {k:v for k,v in r.BASE.items() if not k.startswith('track_')})

    def test_deadline_reserves_45_minutes_at_six_hour_boundary(self):
        self.assertEqual(r.validate_deadlines(1000, 19900, now=1001), 22600)
        with self.assertRaises(ValueError):
            r.validate_deadlines(1000, 19900.001, now=1001)

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
            (self.root/name).parent.mkdir(parents=True, exist_ok=True)
            (self.root/name).write_text('# unchanged analysis source\n', encoding='utf-8')
        for name in r.PROTECTED:
            path = self.train/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# provided source\n', encoding='utf-8')
        self.original = ('# 원문\nREWARD_WEIGHTS = '+repr(r.BASE)+'\n').encode()
        (self.train/'quadruped_rewards.py').write_bytes(self.original)
        self.manifest = self.base/'sources.json'
        self.manifest.write_text(json.dumps({name: r.sha(self.train/name) for name in r.PROTECTED}))
        refs = []
        for name in ('C0_5500', 'F10_3000'):
            path = self.root/('references/'+name+'.pt')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'fixed checkpoint '+name.encode())
            refs.append(dict(name=name, checkpoint='references/'+name+'.pt', sha256=r.sha(path)))
        (self.root/'reference_models.json').write_text(json.dumps(refs), encoding='utf-8')
        runtime_file = self.train/'runtime_ppo.py'
        runtime_file.write_text('# protected installed PPO source\n', encoding='utf-8')
        (self.root/'protected_runtime_sources.json').write_text(json.dumps(dict(
            algorithm_package='rsl_rl', files={str(runtime_file.resolve()): r.sha(runtime_file)})), encoding='utf-8')
        self.start = time.time()-30
        self.deadline = self.start+18900
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

    def test_guard_rejects_runtime_ppo_edit(self):
        self.runner.initialize()
        (self.train/'runtime_ppo.py').write_text('# unauthorized PPO edit', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Protected PPO/runtime'):
            self.runner.guard()

    def test_reference_checkpoint_tamper_blocks_initialize(self):
        (self.root/'references/F10_3000.pt').write_bytes(b'tampered fixed model')
        with self.assertRaisesRegex(RuntimeError, 'Fixed reference checkpoint'):
            self.runner.initialize()
        self.assertFalse((self.root/'fresh_rewards_original.py').exists())

    def test_unknown_fresh_reward_term_blocks_initialize(self):
        (self.train/'quadruped_rewards.py').write_bytes(b'REWARD_WEIGHTS = {"unknown": 1}\n')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            self.runner.initialize()
        self.assertFalse((self.root/'fresh_rewards_original.py').exists())

    def test_observed_slow_first_arm_blocks_second_without_reward_edit(self):
        self.runner.initialize()
        self.runner.data['arms']['S1_tracking_pair'] = dict(status='complete',
            training_time_profile=dict(recent_seconds_per_iteration=6.5))
        estimate = self.runner.training_estimate()
        self.assertGreater(estimate, 10000)
        with mock.patch.object(self.runner, 'now', return_value=self.deadline-9000), \
             mock.patch.object(self.runner, 'apply') as apply, \
             mock.patch.object(r.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'training estimate'):
                self.runner.train_arm('R1_tracking_double')
        apply.assert_not_called()
        launch.assert_not_called()

    def test_timing_profile_does_not_count_started_unfinished_iteration(self):
        log = self.root/'training.log'
        log.write_text('Learning iteration 0/1500\nIteration time: 5.0s\n'
                       'Learning iteration 1/1500\n', encoding='utf-8')
        profile = r.training_profile(log)
        self.assertEqual(profile['completed_iterations'], 1)
        self.assertEqual(profile['recent_seconds_per_iteration'], 5.0)

    def test_unknown_external_reward_edit_is_never_overwritten(self):
        self.runner.initialize()
        external = b'REWARD_WEIGHTS = {"external": 4}\n'
        (self.train/'quadruped_rewards.py').write_bytes(external)
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized external'):
            self.runner.restore_rewards()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), external)

    def test_resume_restores_known_applied_rewards(self):
        self.runner.initialize()
        self.runner.apply('R1_tracking_double')
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        self.assertEqual((self.train/'quadruped_rewards.py').read_bytes(), self.original)

    def test_completed_arm_resume_never_launches_training(self):
        self.runner.initialize()
        checkpoint = self.root/'complete.pt'
        checkpoint.write_bytes(b'checkpoint evidence')
        self.runner.data['arms']['S1_tracking_pair'] = dict(status='complete', checkpoint='complete.pt', sha256=r.sha(checkpoint))
        self.runner.save()
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        with mock.patch.object(r.subprocess, 'Popen') as launch:
            resumed.train_arm('S1_tracking_pair')
        launch.assert_not_called()

    def test_interrupted_arm_resume_never_retrains(self):
        self.runner.initialize()
        self.runner.data['arms']['S1_tracking_pair'] = dict(status='started')
        self.runner.save()
        resumed = self.make_runner(resume=True)
        resumed.initialize()
        with mock.patch.object(r.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError, 'never retrain'):
                resumed.train_arm('S1_tracking_pair')
        launch.assert_not_called()

    def test_completed_checkpoint_tamper_blocks_resume_arm(self):
        self.runner.initialize()
        checkpoint = self.root/'complete.pt'
        checkpoint.write_bytes(b'original')
        self.runner.data['arms']['S1_tracking_pair'] = dict(status='complete', checkpoint='complete.pt', sha256=r.sha(checkpoint))
        checkpoint.write_bytes(b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint hash mismatch'):
            self.runner.train_arm('S1_tracking_pair')

    def test_expired_resume_restores_rewards_before_rejecting_compute(self):
        self.runner.initialize()
        self.runner.apply('R1_tracking_double')
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
                self.runner.train_arm('S1_tracking_pair')
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
        self.assertEqual(state['arms']['S1_tracking_pair']['status'], 'started')
        stop.assert_called_once_with(None)
        bundle.assert_called_once()

    def test_preserve_checkpoints_and_parameters_survive_source_deletion(self):
        run = self.train/'logs/rsl_rl/quadruped'/('stamp_'+self.runner.run_name('S1_tracking_pair'))
        (run/'params').mkdir(parents=True)
        files = {'model_0.pt': b'first checkpoint', 'model_1499.pt': b'final checkpoint',
                 'params/env.yaml': b'rewards: {}', 'params/agent.yaml': b'seed: 42',
                 'events.out.tfevents.test': b'event evidence'}
        for name, content in files.items():
            (run/name).write_bytes(content)
        self.runner.preserve('S1_tracking_pair')
        destination = self.root/'preserved/S1_tracking_pair'/run.name
        for name, content in files.items():
            self.assertEqual((destination/name).read_bytes(), content)
            (run/name).unlink()
        self.runner.preserve('S1_tracking_pair')
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
