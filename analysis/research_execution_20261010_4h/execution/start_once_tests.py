"""Local CPU mocks only: no server, service signal, actual child or GPU work."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import start_once as launcher


class Child:
    def __init__(self,pid):
        self.pid=pid
        self.returncode=None

    def poll(self):
        return self.returncode


class LauncherContracts(unittest.TestCase):
    def setUp(self):
        # Windows tests supply only the POSIX constants; no native signal runs.
        sigkill=patch.object(launcher.signal,'SIGKILL',9,create=True)
        sigkill.start();self.addCleanup(sigkill.stop)

    def fixture(self,root):
        for name in ('runner.py','streaming_control_libc.py'):
            (root/name).write_text('# CPU dummy source\n')
        preflight=root/'preflight_receipt.json'
        preflight.write_text(json.dumps(dict(user_authorized_execution=True,actual_allocation_start_epoch=100)))
        hashes=root/'source_hashes_current.json';hashes.write_text('{}')
        return argparse.Namespace(root=root,train_dir=Path('/workspace/training/quadruped'),isaac='isaaclab.sh',
            source_hashes=hashes,preflight=preflight,server_start_epoch=100,authorized_execution=True)

    def identity(self,pid,command):
        return dict(pid=pid,start_ticks=pid*10,ppid=os.getpid(),pgid=pid,session=pid,argv=command,cmdline=' '.join(command))

    def successful_popen(self,root,calls,training_status='not_started',wrong_ticks=False,helper_fails=False):
        def create(command,**kwargs):
            calls.append((command,kwargs))
            if len(calls)==2 and helper_fails:
                raise FileNotFoundError('mock helper launch failed')
            child=Child(1000+len(calls))
            if len(calls)==1:
                record=self.identity(child.pid,command)
                if wrong_ticks:record['start_ticks']+=1
                (root/'state.json').write_text(json.dumps(dict(runner_identity=record,training={'status':training_status})))
            return child
        return create

    def run_launch(self,args,popen,identities,cleanup):
        with patch.object(launcher.runner,'proc_identity',side_effect=identities), \
                patch.object(launcher.runner,'register_owned_process'), \
                patch.object(launcher,'graceful_stop_runner',side_effect=lambda child,*_:cleanup(child)), \
                contextlib.redirect_stdout(io.StringIO()):
            return launcher.launch(args,popen=popen,wall=lambda:200,sleep=lambda _:None)

    def test_exact_fresh_runner_then_one_helper_with_proven_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);calls=[]
            cleanup=lambda _:self.fail('successful launch must not signal')
            identities=lambda pid:self.identity(pid,calls[0 if pid==1001 else 1][0])
            created,helper,clock=self.run_launch(args,self.successful_popen(root,calls),identities,cleanup)
            self.assertEqual(len(calls),2)
            self.assertEqual((created.pid,helper.pid),(1001,1002))
            helper_argv=calls[1][0]
            self.assertEqual(helper_argv[helper_argv.index('--runner-pid')+1],'1001')
            self.assertEqual(helper_argv[helper_argv.index('--runner-start-ticks')+1],'10010')
            self.assertEqual(helper_argv[helper_argv.index('--deadline-epoch')+1],'11800')
            self.assertTrue(all(k['start_new_session'] for _,k in calls))
            receipt=json.loads((root/'launch_receipt.json').read_text())
            self.assertFalse(receipt['GPU_training_started_by_launcher'])
            self.assertTrue(receipt['controller_release_ACK_still_required'])
            self.assertEqual(receipt['initial_training_status'],'not_started')
            self.assertEqual(clock.deadlines['shutdown_confirmed_latest'],14500)

    def test_missing_helper_source_refuses_any_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);(root/'streaming_control_libc.py').unlink()
            with self.assertRaisesRegex(RuntimeError,'source missing'):
                launcher.launch(args,popen=lambda *a,**k:self.fail('No child authorized'),wall=lambda:200)
            self.assertFalse((root/'launch_once.json').exists())

    def test_helper_launch_failure_cleans_only_runner_and_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);calls=[];cleaned=[]
            identities=lambda pid:self.identity(pid,calls[0][0])
            with self.assertRaises(FileNotFoundError):
                self.run_launch(args,self.successful_popen(root,calls,helper_fails=True),identities,lambda child:cleaned.append(child.pid))
            self.assertEqual(len(calls),2)
            self.assertEqual(cleaned,[1001])
            failed=json.loads((root/'launch_failed.json').read_text())
            self.assertFalse(failed['automatic_retry_authorized'])
            self.assertTrue((root/'launch_once.json').exists())
            self.assertFalse((root/'launch_receipt.json').exists())

    def test_training_prior_controller_ack_refuses_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);calls=[];cleaned=[]
            identities=lambda pid:self.identity(pid,calls[0][0])
            with self.assertRaisesRegex(RuntimeError,'preceded the first controller ACK'):
                self.run_launch(args,self.successful_popen(root,calls,training_status='started'),identities,lambda child:cleaned.append(child.pid))
            self.assertEqual(len(calls),1);self.assertEqual(cleaned,[1001])

    def test_state_start_ticks_mismatch_refuses_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);calls=[];cleaned=[]
            identities=lambda pid:self.identity(pid,calls[0][0])
            with self.assertRaisesRegex(RuntimeError,'state identity'):
                self.run_launch(args,self.successful_popen(root,calls,wrong_ticks=True),identities,lambda child:cleaned.append(child.pid))
            self.assertEqual(len(calls),1);self.assertEqual(cleaned,[1001])

    def test_no_authorization_or_wrong_allocation_refuses_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            args.authorized_execution=False
            with self.assertRaisesRegex(ValueError,'user-authorized'):
                launcher.launch(args,popen=lambda *a,**k:self.fail('No child'),wall=lambda:200)
            self.assertFalse((root/'launch_once.json').exists())
            args.authorized_execution=True;args.server_start_epoch=99
            with self.assertRaisesRegex(ValueError,'clock mismatch'):
                launcher.launch(args,popen=lambda *a,**k:self.fail('No child'),wall=lambda:200)

    def test_one_shot_claim_prohibits_second_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root);(root/'launch_once.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                launcher.launch(args,popen=lambda *a,**k:self.fail('No child'),wall=lambda:200)

    def test_prior_state_or_helper_claim_prohibits_launch(self):
        for name in ('state.json','preserved','streaming_controller_once.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.fixture(root);(root/name).write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'no retry or resume'):
                    launcher.launch(args,popen=lambda *a,**k:self.fail('No child'),wall=lambda:200)

    def test_current_argv_and_parent_proof_are_mandatory(self):
        for field,value in (('argv',['wrong']),('ppid',-1),('start_ticks',0),('pgid',1),('session',1)):
            with self.subTest(field=field):
                child=Child(1001);identity=self.identity(1001,['exact']);identity[field]=value
                times=iter((1,1,3))
                with patch.object(launcher.runner,'proc_identity',return_value=identity):
                    with self.assertRaisesRegex(RuntimeError,'proof timed out'):
                        launcher.wait_identity(child,['exact'],2,wall=lambda:next(times),sleep=lambda _:None)

    def test_exited_runner_cannot_receive_controller(self):
        child=Child(1001);child.returncode=1
        with self.assertRaisesRegex(RuntimeError,'exited'):
            launcher.wait_identity(child,['exact'],2,wall=lambda:1)

    def test_missing_state_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            times=iter((1,3))
            with self.assertRaisesRegex(RuntimeError,'10s initialization bound'):
                launcher.wait_runner_state(Path(tmp),Child(1001),{'pid':1001,'start_ticks':1},2,wall=lambda:next(times),sleep=lambda _:None)

    def test_expired_setup_refuses_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            with self.assertRaises(launcher.runner.PhaseTimeout):
                launcher.launch(args,popen=lambda *a,**k:self.fail('No child'),wall=lambda:1301)

    def audit(self,root,helper,hold=False,release=True):
        entries=[dict(writer_pid=helper.pid,event='controller_started',epoch=1)]
        if hold:entries.append(dict(writer_pid=helper.pid,event='stage_acknowledged',status='held_gpu_released',epoch=2))
        if release:entries.append(dict(writer_pid=helper.pid,event='stage_released',resumer_reaped=True,epoch=3))
        entries.append(dict(writer_pid=helper.pid,event='controller_finished',epoch=4))
        (root/'streaming_control_audit.jsonl').write_text('\n'.join(json.dumps(item) for item in entries))

    def test_helper_early_loss_stops_runner_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);created=Child(1001);helper=Child(1002);helper.returncode=1
            (root/'state.json').write_text(json.dumps({'stage':'RUNNING_TRAIN'}));stopped=[]
            def stop(child):stopped.append(child.pid);child.returncode=1
            with patch.object(launcher,'graceful_stop_runner',side_effect=lambda child,*_:stop(child)):
                with self.assertRaisesRegex(RuntimeError,'controller exited'):
                    launcher.monitor(root,created,helper,launcher.runner.BudgetClock(100,lambda:200,lambda:0),sleep=lambda _:None)
            self.assertEqual(stopped,[1001])
            self.assertFalse(json.loads((root/'launch_finished.json').read_text())['controller_recovery_evidence']['verified'])

    def test_wrapper_interrupt_cleans_runner_and_checks_helper_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);created=Child(1001);helper=Child(1002);self.audit(root,helper,hold=True)
            stopped=[]
            def stop(child):stopped.append(child.pid);child.returncode=1;helper.returncode=0
            def interrupt(_):raise launcher.runner.StopRequested('mock SIGTERM')
            with patch.object(launcher,'graceful_stop_runner',side_effect=lambda child,*_:stop(child)):
                with self.assertRaises(launcher.runner.StopRequested):
                    launcher.monitor(root,created,helper,launcher.runner.BudgetClock(100,lambda:200,lambda:0),sleep=interrupt)
            self.assertEqual(stopped,[1001])
            result=json.loads((root/'launch_finished.json').read_text())
            self.assertTrue(result['controller_recovery_evidence']['verified'])
            self.assertFalse(result['allocation_off_confirmed'])

    def test_helper_normal_exit_after_terminal_does_not_stop_archiving_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);created=Child(1001);helper=Child(1002);helper.returncode=0;self.audit(root,helper,hold=True)
            (root/'state.json').write_text(json.dumps({'stage':'COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION'}))
            with patch.object(launcher,'graceful_stop_runner',side_effect=lambda *_:self.fail('Do not interrupt owned archives')):
                launcher.monitor(root,created,helper,launcher.runner.BudgetClock(100,lambda:200,lambda:0),sleep=lambda _:setattr(created,'returncode',0))

    def test_controller_hold_without_recovery_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);created=Child(1001);helper=Child(1002);created.returncode=0;helper.returncode=0
            self.audit(root,helper,hold=True,release=False)
            with self.assertRaisesRegex(RuntimeError,'recovery evidence incomplete'):
                launcher.monitor(root,created,helper,launcher.runner.BudgetClock(100,lambda:200,lambda:0))

    def test_helper_wait_cannot_extend_compute_recovery_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);created=Child(1001);helper=Child(1002);created.returncode=1
            clock=launcher.runner.BudgetClock(100,lambda:11831,lambda:0)
            with self.assertRaisesRegex(RuntimeError,'recovery evidence incomplete'):
                launcher.monitor(root,created,helper,clock,sleep=lambda _:self.fail('No wait after compute+30'))

    def test_wrapper_signal_requests_normal_cleanup(self):
        with self.assertRaises(launcher.runner.StopRequested):launcher.wrapper_signal(15,None)

    def test_runner_grace_does_not_escalate_during55s_child_cleanup(self):
        now={'value':200.};process=Child(1001);command=['python','runner.py']
        process.runner_identity={'command':command};signals=[]
        identity=self.identity(1001,command)
        clock=launcher.runner.BudgetClock(100,lambda:now['value'],lambda:0)
        def sleep(_):now['value']+=1
        with patch.object(launcher.runner,'group_alive',side_effect=lambda _:now['value']<260), \
                patch.object(launcher.runner,'verified_owned_group',return_value=[identity]), \
                patch.object(launcher.os,'killpg',side_effect=lambda pid,sig:signals.append((now['value'],sig)),create=True):
            launcher.graceful_stop_runner(process,clock,sleep)
        self.assertEqual(signals,[(200.,launcher.signal.SIGINT)])
        self.assertEqual(now['value'],260.)

    def test_wrapper_every_escalation_refreshes_current_identity(self):
        now={'value':200.};process=Child(1001);command=['python','runner.py']
        process.runner_identity={'command':command};signals=[];checks=[]
        identity=self.identity(1001,command)
        clock=launcher.runner.BudgetClock(100,lambda:now['value'],lambda:0)
        def sleep(_):now['value']+=1
        def verify(_):checks.append(now['value']);return [identity]
        with patch.object(launcher.runner,'group_alive',side_effect=lambda _:len(signals)<3), \
                patch.object(launcher.runner,'verified_owned_group',side_effect=verify), \
                patch.object(launcher.os,'killpg',side_effect=lambda pid,sig:signals.append((now['value'],sig)),create=True):
            launcher.graceful_stop_runner(process,clock,sleep)
        self.assertEqual(checks,[200.,320.,335.])
        self.assertEqual([item[1] for item in signals],[launcher.signal.SIGINT,launcher.signal.SIGTERM,launcher.signal.SIGKILL])

    def test_wrapper_refuses_changed_current_runner_argv_before_signal(self):
        process=Child(1001);process.runner_identity={'command':['expected']}
        with patch.object(launcher.runner,'group_alive',return_value=True), \
                patch.object(launcher.runner,'verified_owned_group',return_value=[self.identity(1001,['changed'])]), \
                patch.object(launcher.os,'killpg',side_effect=lambda *_:self.fail('No mismatched identity signal'),create=True):
            with self.assertRaisesRegex(RuntimeError,'current argv changed'):
                launcher.graceful_stop_runner(process,launcher.runner.BudgetClock(100,lambda:200,lambda:0))

    def test_wrapper_cleanup_cannot_wait_past_OFFminus60(self):
        now={'value':14435.};process=Child(1001);command=['python','runner.py']
        process.runner_identity={'command':command};signals=[]
        clock=launcher.runner.BudgetClock(100,lambda:now['value'],lambda:0)
        def sleep(_):now['value']+=1
        with patch.object(launcher.runner,'group_alive',side_effect=lambda _:len(signals)<3), \
                patch.object(launcher.runner,'verified_owned_group',return_value=[self.identity(1001,command)]), \
                patch.object(launcher.os,'killpg',side_effect=lambda pid,sig:signals.append(sig),create=True):
            launcher.graceful_stop_runner(process,clock,sleep)
        self.assertEqual(now['value'],14440.)


if __name__=='__main__':
    unittest.main(verbosity=2)
