"""CPU-only preflight tests: mocked processes, GPU observations, and model loads."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_preflight as p

c = p.streaming
LAUNCHER = c.Identity(14000, 111, 1, c.LAUNCHER_ARGV, 'S')
STREAMER = c.Identity(14001, 112, 14000, c.STREAMER_ARGV, 'S')
CPU = c.Identity(14002, 113, 1, ('python', '/other/tool.py'), 'S')


def reader_for(items):
    return {item.pid: item for item in items}.get


class ClassificationTests(unittest.TestCase):
    def classify(self, gpu=None, items=None, reader=None):
        items = [LAUNCHER, STREAMER] if items is None else items
        return p.classify_processes({STREAMER.pid} if gpu is None else gpu,
                                    items, reader or reader_for(items))

    def test_exact_official_gpu_pair_is_allowed_with_identity_evidence(self):
        result = self.classify()
        self.assertEqual(result['classification'], 'only_exact_official_streamer')
        self.assertEqual(result['launcher']['start_ticks'], 111)
        self.assertTrue(result['official_streamer_present'])

    def test_no_gpu_jobs_is_classified_separately_from_controller_readiness(self):
        self.assertEqual(self.classify(set(), [CPU])['classification'], 'no_gpu_jobs')
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            p.verify_controller_launcher([CPU], reader_for([CPU]))

    def test_empty_gpu_with_launcher_no_child_can_support_later_handshake(self):
        self.assertEqual(p.verify_controller_launcher([LAUNCHER], reader_for([LAUNCHER])),
                         LAUNCHER.record())

    def test_missing_unknown_multiple_and_pid1_gpu_rejected(self):
        for gpu, items in (({999}, [LAUNCHER, STREAMER]), ({CPU.pid}, [CPU]),
                           ({STREAMER.pid, CPU.pid}, [LAUNCHER, STREAMER, CPU]),
                           ({1}, [replace(STREAMER, pid=1)])):
            with self.subTest(gpu=gpu), self.assertRaises(RuntimeError):
                self.classify(gpu, items)

    def test_wrong_parent_or_nonexact_argv_rejected(self):
        variants = [([replace(LAUNCHER, ppid=2), STREAMER]),
                    ([LAUNCHER, replace(STREAMER, ppid=1)]),
                    ([replace(LAUNCHER, argv=LAUNCHER.argv + ('--extra',)), STREAMER]),
                    ([LAUNCHER, replace(STREAMER, argv=('python3', '/streaming/livestream_webrtc.py'))]),
                    ([LAUNCHER, replace(STREAMER, argv=STREAMER.argv + ('--extra',))]),
                    ([replace(LAUNCHER, start_ticks=0), STREAMER])]
        for items in variants:
            with self.subTest(items=items), self.assertRaises(RuntimeError):
                self.classify(items=items)

    def test_paused_dead_and_duplicate_identities_rejected(self):
        for items in ([LAUNCHER, replace(STREAMER, state='Z')],
                      [replace(LAUNCHER, state='T'), STREAMER],
                      [LAUNCHER, replace(STREAMER, state='t')],
                      [LAUNCHER, STREAMER, STREAMER],
                      [LAUNCHER, STREAMER, replace(LAUNCHER, pid=15000)]):
            with self.subTest(items=items), self.assertRaises(RuntimeError):
                self.classify(items=items)

    def test_recycled_reparented_disappeared_and_newly_dead_process_rejected(self):
        for changed in (None, replace(STREAMER, start_ticks=999),
                        replace(STREAMER, ppid=2), replace(STREAMER, state='Z'),
                        replace(STREAMER, state='T')):
            reader = lambda pid, changed=changed: LAUNCHER if pid == LAUNCHER.pid else changed
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                self.classify(reader=reader)

    def test_competing_jobs_block_even_before_gpu_allocation(self):
        for argv in (('python', '/x/train.py'), ('python', '/x/train.py', '--task', 'Go2'),
                     ('python', '/x/evaluation.py'), ('python', '/x/t_eval.py'),
                     ('python', '/x/play.py'), ('python', '/x/benchmark.py'),
                     ('python', '/x/practice_sim.py'), ('python', '/x/ch_01.py'),
                     ('python', '/x/runner.py', '--authorized-execution')):
            item = replace(CPU, argv=argv)
            with self.subTest(argv=argv), self.assertRaisesRegex(RuntimeError, 'Competing'):
                self.classify(set(), [LAUNCHER, item])

    def test_dead_competing_job_is_not_running(self):
        item = replace(CPU, argv=('python', '/x/train.py'), state='Z')
        self.assertEqual(self.classify(set(), [LAUNCHER, item])['classification'], 'no_gpu_jobs')

    def test_controller_unknown_child_and_ambiguous_children_rejected(self):
        variants = ([LAUNCHER, replace(CPU, ppid=LAUNCHER.pid)],
                    [LAUNCHER, STREAMER, replace(STREAMER, pid=15000)],
                    [replace(LAUNCHER, state='T')])
        for items in variants:
            with self.subTest(items=items), self.assertRaises(RuntimeError):
                p.verify_controller_launcher(items, reader_for(items))

    def test_controller_recycled_launcher_rejected(self):
        with self.assertRaises(RuntimeError):
            p.verify_controller_launcher([LAUNCHER], lambda pid: replace(LAUNCHER, start_ticks=999))

    def test_observation_uses_only_nvidia_smi_xml_read(self):
        xml = '<nvidia_smi_log><gpu><product_name>GPU</product_name><processes>' \
              '<process_info><pid>14001</pid><type>G</type></process_info>' \
              '</processes></gpu></nvidia_smi_log>'
        with patch.object(c, 'identities', return_value=[LAUNCHER, STREAMER]), \
             patch.object(c, 'read_identity', side_effect=reader_for([LAUNCHER, STREAMER])), \
             patch.object(p.subprocess, 'run', return_value=SimpleNamespace(stdout=xml)) as ran, \
             patch.object(c, 'pidfd_backend') as backend:
            result = p.observe_processes()
        self.assertEqual(ran.call_args.args[0], ['nvidia-smi', '-q', '-x'])
        self.assertTrue(ran.call_args.kwargs['check'])
        self.assertEqual(result['controller_launcher']['pid'], LAUNCHER.pid)
        backend.assert_not_called()

    def test_unknown_gpu_xml_is_not_idle(self):
        for xml in ('<wrong/>', '<nvidia_smi_log><gpu/></nvidia_smi_log>',
                    '<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>'):
            with self.subTest(xml=xml), self.assertRaises(RuntimeError):
                c.gpu_pids_from_xml(xml)


class FileTests(unittest.TestCase):
    def test_runtime_versions_require_both_exact_builds(self):
        self.assertTrue(p.verify_versions(dict(p.REQUIRED_VERSIONS)))
        for versions in ({}, {'rsl_rl_lib': '5.0.1', 'torch': '2.7.0'},
                         {'rsl_rl_lib': '5.0.0', 'torch': '2.7.0+cu128'}):
            with self.subTest(versions=versions), self.assertRaises(RuntimeError):
                p.verify_versions(versions)

    def package(self, root):
        names = ['live_preflight.py', 'streaming_control_libc.py', 'evaluation.py', 't_eval.py',
                 'reference_models.json', 'previous_source_hashes_20261002.json',
                 'fresh_rewards_original.py', 'recovery_probe.zip']
        files = {}
        for name in names:
            path = root / name
            path.write_bytes(name.encode())
            files[name] = {'bytes': path.stat().st_size, 'sha256': p.sha(path)}
        return {'files': files}

    def test_complete_package_and_missing_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.package(root)
            self.assertEqual(p.verify_package(root, manifest), manifest['files'])
            del manifest['files']['t_eval.py']
            with self.assertRaisesRegex(RuntimeError, 'Required'):
                p.verify_package(root, manifest)

    def test_package_modified_bytes_or_size_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.package(root)
            manifest['files']['evaluation.py']['sha256'] = '0' * 64
            with self.assertRaises(RuntimeError):
                p.verify_package(root, manifest)
            manifest = self.package(root)
            manifest['files']['evaluation.py']['bytes'] += 1
            with self.assertRaises(RuntimeError):
                p.verify_package(root, manifest)

    def test_package_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('../escape', '/absolute', 'a\\b', '.', None):
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    p.local_member(root, name)

    def runtime(self, root, legacy=False):
        names = ['__init__.py', 'algorithms/ppo.py', 'runners/on_policy_runner.py']
        names += ['modules/actor_critic.py'] if legacy else ['models/mlp_model.py', 'modules/mlp.py']
        files = {}
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# source\n')
            files[str(path.resolve())] = p.sha(path)
        return {'algorithm_package': 'rsl_rl', 'package_root': str(root.resolve()), 'files': files}

    def test_runtime_modular_and_legacy_layouts_verified(self):
        for legacy in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                snapshot = self.runtime(root, legacy)
                self.assertEqual(p.verify_runtime_sources(snapshot, root)['file_count'], len(snapshot['files']))

    def test_runtime_modified_source_and_new_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self.runtime(root)
            (root / '__init__.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                p.verify_runtime_sources(snapshot, root)
            snapshot = self.runtime(root)
            (root / 'unexpected.py').write_text('new')
            with self.assertRaisesRegex(RuntimeError, 'inventory'):
                p.verify_runtime_sources(snapshot, root)

    def test_runtime_missing_layout_or_wrong_root_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = self.runtime(root)
            snapshot['package_root'] = str(root / 'elsewhere')
            with self.assertRaises(RuntimeError):
                p.verify_runtime_sources(snapshot, root)
            snapshot['package_root'] = str(root)
            path = root / 'modules/mlp.py'
            del snapshot['files'][str(path.resolve())]
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'layout'):
                p.verify_runtime_sources(snapshot, root)

    def test_provided_sources_and_original_reward_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in (*p.PROTECTED, 'quadruped_rewards.py'):
                path = root / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text(name)
            expected = {name: p.sha(root / name) for name in p.PROTECTED}
            digest = p.sha(root / 'quadruped_rewards.py')
            self.assertEqual(p.verify_provided_sources(root, expected, digest)['provided'], expected)
            with self.assertRaises(RuntimeError):
                p.verify_provided_sources(root, expected, '0' * 64)
            (root / 'train.py').write_text('changed')
            with self.assertRaises(RuntimeError):
                p.verify_provided_sources(root, expected, digest)

    def test_failed_receipt_is_preserved_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = p.run(root, root)
            self.assertEqual(result['status'], 'FAILED')
            receipt = root / p.RECEIPT
            initial = receipt.read_bytes()
            self.assertFalse(json.loads(initial)['service_signals_sent'])
            with self.assertRaises(FileExistsError):
                p.run(root, root)
            self.assertEqual(receipt.read_bytes(), initial)


class LoaderTests(unittest.TestCase):
    def fixture(self, root):
        items, checked = [], {}
        for name in ('C0_5500', 'F10_3000'):
            path = root / (name + '.pt')
            path.write_bytes(b'own frozen test placeholder')
            digest = p.sha(path)
            items.append(dict(name=name, checkpoint=str(path), sha256=digest,
                              actor_observation_normalization=False, policy_clip_actions=None))
            checked[path.name] = dict(sha256=digest, bytes=path.stat().st_size)
        return items, checked

    def load(self, root, checked, items):
        evaluator = SimpleNamespace(read_manifest=Mock(return_value=items),
            load_actor=Mock(side_effect=lambda item, legacy, device: (object(), {'device': device})))
        with patch.dict(sys.modules, {'evaluation': evaluator, 't_eval': SimpleNamespace()}):
            result = p.load_reference_models_cpu(root, checked)
        return result, evaluator

    def test_all_reference_loads_are_cpu_and_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, checked = self.fixture(root)
            result, evaluator = self.load(root, checked, items)
            self.assertEqual(set(result), {'C0_5500', 'F10_3000'})
            for call in evaluator.load_actor.call_args_list:
                self.assertEqual(call.args[2], 'cpu')
                self.assertIsNone(call.args[0]['policy_clip_actions'])

    def test_reference_cannot_silently_override_clipping_or_normalization(self):
        for field, value in (('policy_clip_actions', 1.0), ('actor_observation_normalization', True)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                items, checked = self.fixture(root)
                items[0][field] = value
                with self.assertRaisesRegex(RuntimeError, 'contract'):
                    self.load(root, checked, items)

    def test_unlisted_or_wrong_hash_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, checked = self.fixture(root)
            checked['C0_5500.pt']['sha256'] = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'manifest'):
                self.load(root, checked, items)
            del checked['C0_5500.pt']
            with self.assertRaisesRegex(RuntimeError, 'manifest'):
                self.load(root, checked, items)

    def test_reference_must_be_inside_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, checked = self.fixture(root)
            items[0]['checkpoint'] = str(root.parent / 'outside.pt')
            with self.assertRaisesRegex(RuntimeError, 'outside'):
                self.load(root, checked, items)

    def test_duplicate_names_or_missing_control_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, checked = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'Both'):
                self.load(root, checked, items[:1])
            items[1]['name'] = 'C0_5500'
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                self.load(root, checked, items)

    def test_optional_manifest_must_itself_be_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items, checked = self.fixture(root)
            (root / 'existing_b1_models.json').write_text(json.dumps({'models': [items[0]]}))
            with self.assertRaisesRegex(RuntimeError, 'Optional'):
                self.load(root, checked, items)


if __name__ == '__main__':
    unittest.main(verbosity=2)
