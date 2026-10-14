"""Meaningful CPU contract checks; never opens a server or imports Isaac/PPO."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('fresh_runner',Path(__file__).with_name('runner.py'))
r=importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
BASE=Path(__file__).resolve().parents[1]
OLD=BASE.parent/'baseline_redesign_20261009_4h'
TEST_ARM='TEST_CANDIDATE'
TEST_WEIGHTS=dict(track_lin_vel_xy_exp=2.,track_ang_vel_z_exp=1.5,feet_air_time=0.,
    lin_vel_z_l2=-2.,ang_vel_xy_l2=-.05,dof_torques_l2=-.0001,dof_acc_l2=-2.5e-7,
    action_rate_l2=-.01,flat_orientation_l2=0.,dof_pos_limits=0.)


class RunnerContracts(unittest.TestCase):
    def identity(self,pid=100,ticks=10,pgid=100,session=100):
        return dict(pid=pid,start_ticks=ticks,pgid=pgid,session=session,ppid=1,state='S',cmdline='owned child')

    def owned(self):
        process=Mock(pid=100)
        process.runner_identity=dict(leader=self.identity(),command=['isaaclab.sh','-p','train.py'],seen={100:10,101:11})
        return process

    def fixture_runner(self,root,iterations=1500,stored_iteration=1499,capture_final=True):
        import torch
        obj=r.Runner.__new__(r.Runner)
        obj.root=root/'experiment';obj.root.mkdir()
        obj.train=root/'train';obj.train.mkdir()
        obj.arm=TEST_ARM;obj.run_name=TEST_ARM+'_4h_100';obj.isaac='isaaclab.sh'
        obj.original=b'original';obj.candidate=b'candidate';obj.active_reward=obj.original
        (obj.train/'quadruped_rewards.py').write_bytes(obj.original)
        obj.clock=r.BudgetClock(100,lambda:200,lambda:0)
        obj.design={'budget':{'planning_iteration_seconds':4.7,'planning_multiplier':1.05,'training_boot_seconds':120,
                    'selection_seconds':900,'validation_seconds':1650,'computation_margin_seconds':300}}
        obj.expected_weights=TEST_WEIGHTS;obj.process=None;obj.capture=None
        obj.data={'stage':'SCREEN_FINISHED','training':{'status':'not_started'},'evaluations':{}}
        obj.guard=Mock();obj.save=Mock();obj.restore=Mock();obj.request_streaming=Mock();obj.job=Mock()
        run=obj.train/'logs/rsl_rl/quadruped'/('date_'+obj.run_name);run.mkdir(parents=True)
        (run/'params').mkdir();(run/'params/env.yaml').write_text('env');(run/'params/agent.yaml').write_text('agent')
        (obj.root/'logs').mkdir()
        (obj.root/'logs'/('train_'+TEST_ARM+'.log')).write_text(''.join(
            'Learning iteration '+str(i)+'/1500\nIteration time: 4.7s\n' for i in range(iterations)))
        if capture_final:
            source=run/'model_1499.pt'
            torch.save({'iter':stored_iteration,'actor_state_dict':{'weight':torch.ones(2)}},source)
            capture=r.CheckpointCapture(run.parent,obj.run_name);capture.link(run,source)
            source.unlink()
        return obj,run

    def test_actual_postcleanup_final_path_absent_preserved_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp))
            with patch.object(r.CheckpointCapture,'start'),patch.object(r.CheckpointCapture,'stop'),patch.object(r,'verify_config',return_value={'effective_weights':TEST_WEIGHTS}):
                obj.train_once()
            self.assertFalse((run/'model_1499.pt').exists())
            self.assertTrue(obj.data['training']['iterations_complete'])
            self.assertEqual(obj.data['training']['status'],'complete_verified_preserved')
            self.assertEqual(obj.data['training']['verified_final_checkpoint']['iteration'],1499)
            self.assertFalse(obj.data['training']['original_final_checkpoint_present_after_provided_cleanup'])
            self.assertEqual(obj.job.call_args.args[1],'train_'+TEST_ARM)

    def test_complete_iterations_missing1499_distinct_no_best_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp),capture_final=False)
            (run/'model_best.pt').write_bytes(b'not final')
            with patch.object(r.CheckpointCapture,'start'),patch.object(r.CheckpointCapture,'stop'),patch.object(r,'verify_config',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'1500 iterations complete; exact preserved1499 unavailable'):
                    obj.train_once()
            self.assertTrue(obj.data['training']['iterations_complete'])
            self.assertEqual(obj.data['training']['status'],'iterations_complete_exact_checkpoint_unavailable')

    def test_partial_iterations_distinct_from_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp),iterations=1400,capture_final=False)
            with patch.object(r.CheckpointCapture,'start'),patch.object(r.CheckpointCapture,'stop'),patch.object(r,'verify_config',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'exact1500 scratch iterations'):
                    obj.train_once()
            self.assertEqual(obj.data['training']['status'],'incomplete_iterations_no_retry')

    def test_wrong_preserved_iteration_is_fatal_not_timeout_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp),stored_iteration=1000)
            with patch.object(r.CheckpointCapture,'start'),patch.object(r.CheckpointCapture,'stop'),patch.object(r,'verify_config',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'wrong exact iteration'):
                    obj.train_once()
            self.assertEqual(obj.data['training']['status'],'iterations_complete_exact_checkpoint_invalid')

    def test_checkpoint_nonfinite_and_hash_change_refused(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model_1499.pt'
            torch.save({'iter':1499,'actor_state_dict':{'weight':torch.tensor([float('nan')])}},path)
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):r.verify_checkpoint(path,1499)
            with self.assertRaisesRegex(RuntimeError,'frozen SHA256'):r.verify_checkpoint(path,1499,'0'*64)

    def test_owned_group_reused_leader_pid_refused(self):
        with patch.object(r,'group_snapshot',return_value=[self.identity(ticks=99)]):
            with self.assertRaisesRegex(RuntimeError,'reused'):r.verified_owned_group(self.owned())

    def test_owned_group_parent_exit_keeps_current_descendant_identity(self):
        process=self.owned()
        with patch.object(r,'group_snapshot',return_value=[self.identity(101,11)]):
            self.assertEqual(r.verified_owned_group(process)[0]['pid'],101)

    def test_owned_group_all_old_members_gone_cannot_signal_new_group(self):
        with patch.object(r,'group_snapshot',return_value=[self.identity(102,1000)]):
            with self.assertRaisesRegex(RuntimeError,'No current member'):r.verified_owned_group(self.owned())

    def test_owned_group_session_mismatch_refused_before_signal(self):
        with patch.object(r,'group_snapshot',return_value=[self.identity(101,11,session=99)]):
            with self.assertRaisesRegex(RuntimeError,'session identity mismatch'):r.verified_owned_group(self.owned())

    def test_stop_requires_current_identity_before_every_signal(self):
        process=self.owned();process.poll.return_value=None
        with patch.object(r,'group_alive',return_value=True),patch.object(r,'verified_owned_group',side_effect=RuntimeError('identity mismatch')),patch.object(r.os,'killpg',create=True) as send,patch.object(r.signal,'SIGKILL',9,create=True):
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):r.stop_process(process)
            send.assert_not_called()

    def test_profile_rejects_duplicate_skipped_or_resumed_headers(self):
        for text in ('Learning iteration 1/1500\n','Learning iteration 0/1500\nLearning iteration 0/1500\n',
                     'Learning iteration 0/1500\nLearning iteration 2/1500\n'):
            with self.assertRaisesRegex(RuntimeError,'contiguous'):r.training_profile(text)

    def test_new_deadlines_protect25minute_dashboard_margin(self):
        self.assertEqual(r.OFFSETS['training_begin_latest'],1800)
        self.assertEqual(r.OFFSETS['dashboard_shutdown_request_latest'],12900)
        self.assertEqual(r.OFFSETS['shutdown_confirmed_latest']-r.OFFSETS['dashboard_shutdown_request_latest'],1500)
        self.assertLess(1800+r.training_estimate(1500,4.818008522014856),9600-r.PROCESS_CLEANUP_RESERVE_SECONDS)

    def test_screen_timeout_stops_without_repeat_and_keeps_latest_train_begin(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp)
            obj.clock=r.BudgetClock(100,lambda:1200,lambda:0)
            obj.data={'stage':'PREPARED','screen':{'status':'not_started'},'evaluations':{}}
            obj.references=[];obj.existing_b1=[];obj.process=None
            obj.screen_enabled=True
            obj.save=Mock();obj.request_streaming=Mock();obj.evaluate_phase=Mock(side_effect=r.PhaseTimeout('cutoff'))
            obj.screen_existing()
            self.assertEqual(obj.clock.deadlines['screen_end'],1900)
            self.assertEqual(obj.data['screen']['status'],'incomplete_timeout_no_retry')
            self.assertEqual(obj.evaluate_phase.call_count,1)
            with self.assertRaisesRegex(RuntimeError,'already attempted'):obj.screen_existing()

    def test_disabled_screen_never_calls_evaluator_or_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp);obj.screen_enabled=False
            obj.data={'screen':{'status':'not_started'}}
            obj.save=Mock();obj.request_streaming=Mock();obj.evaluate_phase=Mock()
            obj.screen_existing()
            self.assertEqual(obj.data['screen']['status'],'disabled_by_frozen_design')
            obj.request_streaming.assert_not_called();obj.evaluate_phase.assert_not_called()

    def test_full_compute_admission_stricter_than30minute_start_backstop(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp))
            obj.clock=r.BudgetClock(100,lambda:1750,lambda:0)
            obj.design['budget']['planning_iteration_seconds']=4.818008522014856
            with self.assertRaisesRegex(r.PhaseTimeout,'bounded selection/validation and300s'):
                obj.train_once()
            obj.job.assert_not_called()
            self.assertEqual((obj.train/'quadruped_rewards.py').read_bytes(),obj.original)

    def test_late_training_cannot_begin_even_if_estimate_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj,run=self.fixture_runner(Path(tmp))
            obj.clock=r.BudgetClock(100,lambda:1901,lambda:0)
            with self.assertRaisesRegex(r.PhaseTimeout,'training_begin_latest'):obj.train_once()
            obj.job.assert_not_called()
            self.assertEqual((obj.train/'quadruped_rewards.py').read_bytes(),obj.original)

    def test_release_ack_preserves_final_compute_stage_for_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp);obj.streaming_counter=0
            obj.clock=r.BudgetClock(100,lambda:200,lambda:0)
            obj.data={'stage':'COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION'}
            def saved(stage=None):
                if stage:obj.data['stage']=stage
                request=obj.data['streaming_request']
                r.atomic_json(obj.root/'streaming_status.json',dict(token=request['token'],mode='release',status='released',verified_epoch=200))
            obj.save=Mock(side_effect=saved)
            obj.request_streaming('release','cleanup','dashboard_shutdown_request_latest')
            self.assertEqual(obj.data['stage'],'COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION')
            self.assertEqual(len(obj.data['streaming_handshakes']),1)

    def test_terminal_complete_published_only_after_cleanup_release_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp)
            obj.clock=r.BudgetClock(100,lambda:200,lambda:0)
            obj.data={'stage':'PREPARED','training':{'status':'complete_verified_preserved'},'evaluations':{}}
            obj.owns_mutation=True;obj.process=None;obj.capture=None
            obj.initialize=Mock();obj.screen_existing=Mock();obj.train_once=Mock();obj.select=Mock()
            def save(stage=None):
                if stage:obj.data['stage']=stage
            obj.save=Mock(side_effect=save)
            def validate():
                obj.data['compute_completed']=True;obj.save('COMPUTE_FINISHED_CLEANUP_PENDING')
            obj.validate=Mock(side_effect=validate)
            obj.preserve=Mock();obj.restore=Mock();obj.guard=Mock();obj.archive=Mock(return_value={})
            stages=[]
            obj.request_streaming=Mock(side_effect=lambda *args:stages.append(obj.data['stage']))
            obj.execute()
            self.assertEqual(stages,['COMPUTE_FINISHED_CLEANUP_PENDING'])
            self.assertEqual(obj.data['stage'],'COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION')
            self.assertTrue(obj.data['cleanup']['complete'])

    def test_terminal_error_published_only_after_cleanup_release_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp)
            obj.clock=r.BudgetClock(100,lambda:200,lambda:0)
            obj.data={'stage':'PREPARED','training':{'status':'complete_verified_preserved'},'evaluations':{}}
            obj.owns_mutation=True;obj.process=None;obj.capture=None
            obj.initialize=Mock();obj.screen_existing=Mock();obj.train_once=Mock();obj.select=Mock()
            def save(stage=None):
                if stage:obj.data['stage']=stage
            obj.save=Mock(side_effect=save)
            obj.validate=Mock(side_effect=RuntimeError('evaluation failure'))
            obj.preserve=Mock();obj.restore=Mock();obj.guard=Mock();obj.archive=Mock(return_value={})
            stages=[];obj.request_streaming=Mock(side_effect=lambda *args:stages.append(obj.data['stage']))
            with self.assertRaisesRegex(RuntimeError,'evaluation failure'):obj.execute()
            self.assertEqual(stages,['ERROR_CLEANUP_PENDING'])
            self.assertEqual(obj.data['stage'],'STOPPED_PARTIAL_RETRIEVE_AND_STOP_ALLOCATION')

    def test_current_streaming_error_ack_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp);obj.streaming_counter=0
            obj.clock=r.BudgetClock(100,lambda:200,lambda:0);obj.data={'stage':'PREPARED'}
            def saved(stage=None):
                request=obj.data['streaming_request']
                r.atomic_json(obj.root/'streaming_status.json',dict(token=request['token'],mode='hold',status='error',verified_epoch=200))
            obj.save=Mock(side_effect=saved)
            with self.assertRaisesRegex(RuntimeError,'invalid/current-error ACK'):
                obj.request_streaming('hold','screen','training_begin_latest')

    def test_placeholder_candidate_refuses_execution_constructor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,value in [('design.json',{}),('sources.json',{}),('protected_runtime_sources.json',{}),('preflight.json',{})]:
                r.atomic_json(root/name,value)
            (root/'fresh_rewards_original.py').write_bytes(b'original')
            with self.assertRaisesRegex(ValueError,'parent-confirmed'):
                r.Runner(root,root,'isaaclab.sh',root/'sources.json',r.time.time(),root/'preflight.json')

    def test_frozen_candidate_exact_dictionary_only(self):
        original=(OLD/'execution/package/fresh_rewards_original.py').read_bytes()
        start,end,_=r.dictionary_span(original)
        candidate=original[:start]+repr(TEST_WEIGHTS).encode()+original[end:]
        self.assertEqual(r.sha(OLD/'execution/package/fresh_rewards_original.py'),r.ORIGINAL_SHA)
        self.assertEqual(r.validate_candidate(original,candidate,TEST_WEIGHTS),TEST_WEIGHTS)

    def test_actual_final_parent_candidate_and_budget_match(self):
        design=json.loads((BASE/'design.json').read_text(encoding='utf-8'))
        name=design['candidate_name']
        original=(OLD/'execution/package/fresh_rewards_original.py').read_bytes()
        candidate=(BASE/'execution'/design['candidate_reward_file']).read_bytes()
        self.assertEqual(name,'S5_torque_half')
        self.assertEqual(r.validate_candidate(original,candidate,design['candidates'][name]['weights']),design['candidates'][name]['weights'])
        self.assertEqual(design['budget']['deadlines_relative_to_actual_allocation_start'],r.OFFSETS)
        self.assertIs(design['evaluation']['screen']['enabled'],False)
        self.assertEqual(design['budget']['computation_margin_seconds'],300)

    def test_corrupt1000_selection_does_not_become_timeout_fallback(self):
        obj=r.Runner.__new__(r.Runner)
        obj.request_streaming=Mock();obj.checkpoint=Mock(side_effect=[{'iteration':1499},RuntimeError('Nonfinite tensor')])
        obj.evaluate_phase=Mock();obj.references=[]
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):obj.select()
        obj.evaluate_phase.assert_not_called()

    def test_missing1000_falls_back_to_exact1499_without_optimality(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=r.Runner.__new__(r.Runner);obj.root=Path(tmp);obj.process=None
            obj.clock=r.BudgetClock(100,lambda:200,lambda:0)
            obj.data={'evaluations':{}};obj.references=[]
            final=obj.root/'final.pt';final.write_bytes(b'exact final')
            row=dict(name=TEST_ARM+'_1499',checkpoint='final.pt',iteration=1499,sha256=r.sha(final))
            obj.request_streaming=Mock();obj.checkpoint=Mock(side_effect=[row,r.CheckpointMissing('1000 missing')])
            obj.evaluate_phase=Mock();obj.save=Mock()
            obj.select()
            self.assertFalse(obj.data['selected']['checkpoint_optimality_claim'])
            self.assertEqual(obj.data['selected']['iteration'],1499)
            obj.evaluate_phase.assert_not_called()

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
        clock=r.BudgetClock(100,lambda:1301,lambda:0)
        with self.assertRaises(r.PhaseTimeout):
            clock.require('setup_end')

    def test_setup_target_is_not_extended_compute(self):
        clock=r.BudgetClock(100,lambda:1060,lambda:0)
        clock.require('setup_end')
        self.assertEqual(r.OFFSETS['setup_end'],1200)
        self.assertEqual(r.OFFSETS['compute_end'],11700)
        self.assertEqual(r.OFFSETS['dashboard_shutdown_request_latest'],12900)
        self.assertEqual(r.OFFSETS['shutdown_confirmed_latest'],14400)

    def test_four_hour_training_admission_margin(self):
        rate=4.818008522014856
        self.assertLess(r.training_estimate(1500,rate)+900+900+1650+300,11700)
        self.assertGreater(r.training_estimate(1800,rate)+900+900+1650+300,11700)
        self.assertGreater(r.training_estimate(1500,5.3)+900+900+1650+300,11700)

    def test_profile_uses_logged_iteration_without_resume(self):
        text=''.join('Learning iteration '+str(i)+'/1500\nIteration time: 4.9s\n' for i in range(1500))
        result=r.training_profile(text)
        self.assertEqual(result['completed_iterations'],1500)
        self.assertAlmostEqual(result['recent_seconds_per_iteration'],4.9)
        with self.assertRaises(RuntimeError):
            r.training_profile('Learning iteration 7/1600\n')

    def test_effective_config_weights_env_seed_length_resume(self):
        weights=TEST_WEIGHTS
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
        with patch.object(r,'group_alive',side_effect=[True,False]),patch.object(r,'verified_owned_group',return_value=[{'pid':12345}]),patch.object(r.os,'killpg',create=True) as send,patch.object(r.signal,'SIGKILL',9,create=True):
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
            obj=r.Runner.__new__(r.Runner);obj.root=root;obj.arm=TEST_ARM;obj.clock=r.BudgetClock(100,lambda:101,lambda:0)
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
        clock=r.BudgetClock(100,lambda:13001,lambda:0)
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
            obj.design=json.loads((OLD/'design.json').read_text(encoding='utf-8'))
            obj.guard=Mock(return_value={'ok':True,'provided':{},'runtime':{},'checked_epoch':123})
            command=obj.evaluator_command(obj.root/'manifest.json',obj.root/'results/selection/flat','selection',['G1','G6'])
            path=Path(command[command.index('--source-audit')+1]);digest=r.sha(path)
            r.atomic_json(obj.root/'current_source_audit.json',{'checked_epoch':456})
            self.assertNotEqual(path,obj.root/'current_source_audit.json')
            self.assertEqual(r.sha(path),digest)
            with self.assertRaises(RuntimeError):obj.evaluator_command(obj.root/'manifest.json',obj.root/'results/selection/flat','selection',['G1','G6'])




if __name__=='__main__':unittest.main(verbosity=2)
