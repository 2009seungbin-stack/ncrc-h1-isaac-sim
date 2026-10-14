"""Meaningful CPU contract checks; never opens a server or imports Isaac/PPO."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('b1_runner',Path(__file__).with_name('runner.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
BASE=Path(__file__).resolve().parents[1]


class RunnerContracts(unittest.TestCase):
    def test_frozen_candidate_exact_dictionary_only(self):
        original=BASE.parent/'short_sweep_20261002/recovered_20261003/short_sweep_20261002/fresh_rewards_original.py'
        candidate=BASE/'candidates/B1_selective_cost_rewards.py'
        design=json.loads((BASE/'design.json').read_text(encoding='utf-8'))
        self.assertEqual(r.sha(original),r.ORIGINAL_SHA)
        self.assertEqual(r.validate_candidate(original.read_bytes(),candidate.read_bytes(),design['candidates'][r.ARM]['weights']),design['candidates'][r.ARM]['weights'])

    def test_mutation_outside_dictionary_refused(self):
        original=b'# original\nREWARD_WEIGHTS={"track_x":1}\n'
        candidate=b'# altered\nREWARD_WEIGHTS={"track_x":1}\n'
        with self.assertRaises(ValueError):
            r.validate_candidate(original,candidate,{'track_x':1})

    def test_unicode_ast_offsets_are_bytes(self):
        data='# 한국어\nREWARD_WEIGHTS={"feet_air_time":0}\n'.encode()
        start,end,value=r.dictionary_span(data)
        self.assertEqual(data[start:end],b'{"feet_air_time":0}')
        self.assertEqual(value,{'feet_air_time':0})

    def test_backward_wall_clock_cannot_extend_deadline(self):
        values={'wall':1000,'mono':50}
        clock=r.BudgetClock(900,lambda:values['wall'],lambda:values['mono'])
        values.update(wall=500,mono=200)
        self.assertEqual(clock.now(),1150)
        self.assertEqual(clock.deadlines['shutdown_confirmed_latest'],15300)

    def test_all_deadlines_anchored_to_allocation(self):
        clock=r.BudgetClock(100,lambda:200,lambda:1000)
        self.assertEqual(clock.deadlines,{k:100+v for k,v in r.OFFSETS.items()})
        self.assertEqual(clock.deadlines['compute_end'],11800)
        with self.assertRaises(ValueError):
            r.BudgetClock(300,lambda:200,lambda:1000)

    def test_expired_setup_rejected(self):
        clock=r.BudgetClock(100,lambda:1001,lambda:0)
        with self.assertRaises(r.PhaseTimeout):
            clock.require('setup_end')

    def test_four_hour_training_admission_margin(self):
        rate=4.818008522014856
        self.assertLess(r.training_estimate(1500,rate)+5700,14400)
        self.assertGreater(r.training_estimate(1800,rate)+5700,14400)
        self.assertGreater(r.training_estimate(1500,5.3)+5700,14400)

    def test_profile_uses_logged_iteration_without_resume(self):
        text='Learning iteration 1499/1500\nIteration time: 4.9s\n'
        result=r.training_profile(text)
        self.assertEqual(result['completed_iterations'],1500)
        self.assertEqual(result['recent_seconds_per_iteration'],4.9)
        with self.assertRaises(RuntimeError):
            r.training_profile('Learning iteration 7/1600\n')

    def test_effective_config_weights_env_seed_length_resume(self):
        weights=json.loads((BASE/'design.json').read_text(encoding='utf-8'))['candidates'][r.ARM]['weights']
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'params').mkdir()
            (run/'params/env.yaml').write_text('decimation: 4\nsim:\n  dt: 0.005\nscene:\n  num_envs: 6144\nrewards:\n'+''.join('  '+k+':\n    weight: '+str(v)+'\n' for k,v in weights.items()),encoding='utf-8')
            configs=''.join(role+':\n  class_name: MLPModel\n  hidden_dims:\n  - 512\n  - 256\n  - 128\n  activation: elu\n  obs_normalization: false\n' for role in ('actor','critic'))
            agent=run/'params/agent.yaml';agent.write_text('seed: 42\nmax_iterations: 1500\nresume: false\nclip_actions: null\nempirical_normalization: {}\n'+configs)
            self.assertEqual(r.verify_config(run,weights)['effective_weights'],weights)
            good=agent.read_text()
            for changed in (good.replace('obs_normalization: false','obs_normalization: true'),
                            good.replace('activation: elu','activation: tanh'),
                            good.replace('clip_actions: null','clip_actions: 1.0'),
                            good.replace('empirical_normalization: {}','empirical_normalization: true')):
                agent.write_text(changed)
                with self.assertRaises(RuntimeError):r.verify_config(run,weights)
            agent.write_text('seed: 42\nmax_iterations: 1500\nresume: true\n')
            with self.assertRaises(RuntimeError):r.verify_config(run,weights)

    def test_provided_and_fresh_runtime_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected={}
            for name in r.PROTECTED:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());expected[name]=r.sha(path)
            runtime=root/'ppo.py';runtime.write_bytes(b'original PPO')
            manifest=dict(algorithm_package='rsl_rl',files={str(runtime.resolve()):r.sha(runtime)})
            self.assertTrue(r.source_guard(root,expected,manifest)['ok'])
            runtime.write_bytes(b'changed PPO')
            with self.assertRaises(RuntimeError):r.source_guard(root,expected,manifest)

    def test_shell_leader_exit_does_not_skip_group_stop(self):
        process=Mock(pid=12345);process.poll.return_value=0
        with patch.object(r,'group_alive',side_effect=[True,False]),patch.object(r.os,'killpg',create=True) as send,patch.object(r.signal,'SIGKILL',9,create=True):
            r.stop_process(process)
        send.assert_called_once_with(12345,r.signal.SIGINT)

    def selection_model(self, terminal=None, count=1000):
        row=dict(episodes=64,survived=64-(terminal is not None),valid_samples=63*1000+count,
                 completed_control_steps=1000,nonfinite_count=0,per_episode_valid_samples=[count]+[1000]*63,
                 first_terminal_steps=[terminal]+[None]*63,first_timeout_steps=[None]*64,
                 sum_squared_error={a:.04*(63*1000+count) for a in ('vx','vy','yaw')})
        return dict(diagnostics_complete=True,seeds={'2029':{'scenarios':{'G1':row}}})

    def test_selection_tie_prefers_later_checkpoint(self):
        model=self.selection_model()
        self.assertLess(r.subset_rank(model,['G1'],2029,1499),r.subset_rank(model,['G1'],2029,1000))

    def test_selection_fall_precedes_quality_and_bad_mask_rejected(self):
        fall=self.selection_model(500,500)
        good=self.selection_model()
        self.assertGreater(r.subset_rank(fall,['G1'],2029,1499),r.subset_rank(good,['G1'],2029,1000))
        with self.assertRaises(ValueError):r.subset_rank(self.selection_model(None,999),['G1'],2029,1499)

    def test_critical_archive_preserves_exact_selected_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();(root/'preserved').mkdir()
            ck=root/'preserved/model_1499.pt';ck.write_bytes(b'exact frozen selected')
            (root/'state.json').write_text('{}');(root/'unselected.pt').write_bytes(b'not critical')
            obj=r.Runner.__new__(r.Runner);obj.root=root;obj.clock=r.BudgetClock(100,lambda:101,lambda:0)
            obj.data={'selected':{'checkpoint':'preserved/model_1499.pt','sha256':r.sha(ck)}}
            receipt=obj.archive(True)
            self.assertEqual(r.sha(receipt['path']),receipt['sha256'])
            with tarfile.open(receipt['path']) as tf:
                names=tf.getnames()
                self.assertIn('run/preserved/model_1499.pt',names)
                self.assertNotIn('run/unselected.pt',names)
                self.assertEqual(hashlib.sha256(tf.extractfile('run/preserved/model_1499.pt').read()).hexdigest(),r.sha(ck))

    def test_deadline_reader_stops_long_archive(self):
        from io import BytesIO
        clock=r.BudgetClock(100,lambda:13901,lambda:0)
        with self.assertRaises(r.PhaseTimeout):
            r.DeadlineReader(BytesIO(b'data'),clock,'dashboard_shutdown_request_latest').read(4)

    def test_unknown_reward_edit_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.train=Path(tmp);obj.root=Path(tmp);obj.original=b'original';obj.candidate=b'candidate'
            target=obj.train/'quadruped_rewards.py';target.write_bytes(b'external edit')
            with self.assertRaises(RuntimeError):obj.restore()
            self.assertEqual(target.read_bytes(),b'external edit')

    def test_initialization_failure_is_not_suppressed_or_rolled_back(self):
        obj=r.Runner.__new__(r.Runner);obj.data=None;obj.owns_mutation=False
        obj.initialize=Mock(side_effect=ValueError('preflight failed'))
        obj.restore=Mock()
        with self.assertRaisesRegex(ValueError,'preflight failed'):
            obj.execute()
        obj.restore.assert_not_called()

    def test_signal_stops_without_training_restart(self):
        with self.assertRaises(r.StopRequested):r.signal_stop(r.signal.SIGTERM,None)

    def test_evaluation_audit_snapshot_is_immutable_and_not_periodic_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp);obj.train=Path(tmp);obj.isaac='isaaclab.sh'
            obj.design=json.loads((BASE/'design.json').read_text(encoding='utf-8'))
            obj.guard=Mock(return_value={'ok':True,'provided':{},'runtime':{},'checked_epoch':123})
            command=obj.evaluator_command(obj.root/'manifest.json',obj.root/'results/selection/flat','selection',['G1','G6'])
            path=Path(command[command.index('--source-audit')+1]);digest=r.sha(path)
            r.atomic_json(obj.root/'current_source_audit.json',{'checked_epoch':456})
            self.assertNotEqual(path,obj.root/'current_source_audit.json')
            self.assertEqual(r.sha(path),digest)
            with self.assertRaises(RuntimeError):obj.evaluator_command(obj.root/'manifest.json',obj.root/'results/selection/flat','selection',['G1','G6'])


if __name__=='__main__':unittest.main(verbosity=2)
