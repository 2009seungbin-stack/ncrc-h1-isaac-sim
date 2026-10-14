"""CPU-only safety tests for the service controller; never signals real processes."""
import importlib.util
import ctypes
import errno
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('streaming_control_libc', Path(__file__).with_name('streaming_control_libc.py'))
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)

ROOT = Path('/workspace/student_workspace/b1_baseline_20261009_4h')
START = 1791547031
LAUNCHER = c.Identity(14000, 111, 1, c.LAUNCHER_ARGV, 'S')
STREAMER = c.Identity(14001, 112, 14000, c.STREAMER_ARGV, 'S')
RUNNER = c.Identity(16000, 100, 1,
    ('/isaac-sim/kit/python/bin/python3', str(ROOT/'runner.py'), '--root', str(ROOT),
     '--server-start-epoch', str(START), '--authorized-execution'), 'S')


class SafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows lacks these Linux constants. All signal operations remain mocks.
        for name, value in (('SIGSTOP',19),('SIGCONT',18),('SIGKILL',9),('SIGHUP',1)):
            item=patch.object(signal,name,value,create=True)
            item.start()
            cls.addClassCleanup(item.stop)
        item=patch.object(c.signal,'pidfd_send_signal',Mock(),create=True)
        item.start()
        cls.addClassCleanup(item.stop)

    def setUp(self):
        c._PIDFD_BACKEND = None

    def test_proc_stat_starttime_with_parenthesized_name(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'14000'
            folder.mkdir()
            fields = ['S', '1'] + ['0']*17 + ['111'] + ['0']*3
            (folder/'stat').write_text('14000 (bash (worker)) '+' '.join(fields))
            (folder/'cmdline').write_bytes(b'\0'.join(x.encode() for x in c.LAUNCHER_ARGV)+b'\0')
            item = c.read_identity(14000, Path(temp))
            self.assertEqual(item.start_ticks, 111)
            self.assertTrue(c.is_launcher(item))

    def test_pid_reuse_is_not_same_process(self):
        replaced = c.Identity(LAUNCHER.pid, 999, 1, LAUNCHER.argv, 'S')
        self.assertFalse(LAUNCHER.same_process(replaced))

    def test_changed_parent_or_argv_is_not_same_process(self):
        self.assertFalse(LAUNCHER.same_process(c.Identity(14000,111,2,LAUNCHER.argv,'S')))
        self.assertFalse(LAUNCHER.same_process(c.Identity(14000,111,1,('bash',),'S')))

    def test_only_exact_pid1_child_launcher_allowed(self):
        self.assertTrue(c.is_launcher(LAUNCHER))
        self.assertFalse(c.is_launcher(c.Identity(1,111,0,c.LAUNCHER_ARGV,'S')))
        self.assertFalse(c.is_launcher(c.Identity(14000,111,60,c.LAUNCHER_ARGV,'S')))
        self.assertFalse(c.is_launcher(c.Identity(14000,111,1,('/bin/bash','other.py'),'S')))
        self.assertFalse(c.is_launcher(c.Identity(14000,111,1,c.LAUNCHER_ARGV,'Z')))

    def test_streamer_must_be_exact_child(self):
        self.assertTrue(c.is_streamer(STREAMER, LAUNCHER))
        self.assertFalse(c.is_streamer(c.Identity(14001,112,1,c.STREAMER_ARGV,'S'), LAUNCHER))
        self.assertFalse(c.is_streamer(c.Identity(14001,112,14000,('python','other.py'),'S'), LAUNCHER))

    def test_real_runner_binding(self):
        self.assertTrue(c.runner_identity_ok(RUNNER, ROOT, START))
        self.assertFalse(c.runner_identity_ok(RUNNER, ROOT, START+1))
        self.assertFalse(c.runner_identity_ok(STREAMER, ROOT, START))
        altered = c.Identity(16000,100,1,RUNNER.argv+('--root',str(ROOT)),'S')
        self.assertFalse(c.runner_identity_ok(altered,ROOT,START))

    def test_live_training_blocks_service_control(self):
        actor = c.Identity(17000,113,16000,
            ('/isaac-sim/kit/python/bin/python3','/workspace/training/quadruped/train.py',
             '--_run_isaaclab_child','--task','Quadruped-v0'),'S')
        state = dict(stage='RUNNING_selection_flat', training=dict(status='complete',process_group_exited=True))
        self.assertFalse(c.should_arm(state,[actor,LAUNCHER,STREAMER]))
        self.assertTrue(c.should_arm(state,[LAUNCHER,STREAMER]))

    def test_practice_and_edu_match_supervisor_activity(self):
        for script in ('practice_sim.py','ch_01.py'):
            item=c.Identity(17000,113,1,('python','/workspace/'+script),'S')
            self.assertTrue(c.has_training([item]))

    def test_training_exit_proof_required(self):
        self.assertFalse(c.should_arm(dict(training=dict(status='started',process_group_exited=True)),[]))
        self.assertFalse(c.should_arm(dict(training=dict(status='complete',process_group_exited=False)),[]))
        self.assertFalse(c.should_arm(dict(stage='COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION',
                     training=dict(status='complete',process_group_exited=True)),[]))

    def test_terminal_recovery_requires_all_owned_groups_exit(self):
        state=dict(training=dict(pid=17000,process_group_exited=True),
                   evaluations={'a':dict(pid=18000,process_group_exited=False)})
        self.assertFalse(c.groups_exited(state))
        state['evaluations']['a']['process_group_exited']=True
        self.assertTrue(c.groups_exited(state))

    def test_gpu_xml_includes_graphics_and_compute_processes(self):
        xml='<nvidia_smi_log><gpu><processes>'+''.join(
            '<process_info><pid>'+str(pid)+'</pid><type>'+kind+'</type></process_info>'
            for pid,kind in [(14001,'G'),(17000,'C')])+'</processes></gpu></nvidia_smi_log>'
        self.assertEqual(c.gpu_pids_from_xml(xml),{14001,17000})
        with self.assertRaises(RuntimeError):
            c.gpu_pids_from_xml('<wrong><gpu/></wrong>')
        with self.assertRaises(RuntimeError):
            c.gpu_pids_from_xml('<nvidia_smi_log><gpu/></nvidia_smi_log>')
        with self.assertRaises(RuntimeError):
            c.gpu_pids_from_xml('<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>')
        with self.assertRaises(RuntimeError):
            c.gpu_pids_from_xml('<nvidia_smi_log><gpu><processes><process_info><pid>N/A</pid>'
                                 '</process_info></processes></gpu></nvidia_smi_log>')

    def test_handle_refuses_pid1_without_opening_pidfd(self):
        with patch.object(c.os,'pidfd_open',create=True) as opened:
            with self.assertRaises(RuntimeError):
                c.Handle(c.Identity(1,111,0,c.LAUNCHER_ARGV,'S'),Mock())
            opened.assert_not_called()

    def test_pidfd_binding_detects_pid_reuse_and_closes(self):
        replacement=c.Identity(14000,222,1,c.LAUNCHER_ARGV,'S')
        with patch.object(c.os,'pidfd_open',return_value=123,create=True), \
             patch.object(c.os,'close') as closed, patch.object(c,'read_identity',return_value=replacement):
            with self.assertRaises(RuntimeError):
                c.Handle(LAUNCHER,Mock())
            closed.assert_called_once_with(123)

    def test_only_allowed_signal_roles(self):
        with patch.object(c.os,'pidfd_open',return_value=123,create=True), \
             patch.object(c,'read_identity',return_value=LAUNCHER), \
             patch.object(c.signal,'pidfd_send_signal',create=True) as sent:
            handle=c.Handle(LAUNCHER,Mock())
            with self.assertRaises(RuntimeError): handle.send(signal.SIGTERM)
            sent.assert_not_called()
            handle.send(signal.SIGSTOP)
            self.assertEqual(sent.call_args.args[0:2],(123,signal.SIGSTOP))
        with patch.object(c.os,'pidfd_open',return_value=124,create=True), \
             patch.object(c,'read_identity',return_value=STREAMER), \
             patch.object(c.signal,'pidfd_send_signal',create=True) as sent:
            handle=c.Handle(STREAMER,Mock())
            with self.assertRaises(RuntimeError): handle.send(signal.SIGSTOP)
            handle.send(signal.SIGTERM)
            self.assertEqual(sent.call_args.args[0:2],(124,signal.SIGTERM))

    def test_signal_skips_gone_or_reused_process(self):
        with patch.object(c.os,'pidfd_open',return_value=123,create=True), \
             patch.object(c,'read_identity',side_effect=[LAUNCHER,None]), \
             patch.object(c.signal,'pidfd_send_signal',create=True) as sent:
            handle=c.Handle(LAUNCHER,Mock())
            self.assertFalse(handle.send(signal.SIGSTOP))
            sent.assert_not_called()

    def test_resume_cancels_pending_stop_even_when_state_not_T(self):
        with patch.object(c.os,'pidfd_open',return_value=123,create=True), \
             patch.object(c,'read_identity',return_value=LAUNCHER), \
             patch.object(c.signal,'pidfd_send_signal',create=True) as sent:
            handle=c.Handle(LAUNCHER,Mock())
            handle.resume('pending_stop_recovery')
            self.assertEqual(sent.call_args.args[0:2],(123,signal.SIGCONT))

    def test_gpu_remaining_must_be_real_owned_evaluator(self):
        argv=('python',str(ROOT/'evaluation.py'),'--_run_isaaclab_child','--phase','selection',
              '--task','Quadruped-v0','--manifest',str(ROOT/'selection_models.json'))
        item=c.Identity(18001,444,18000,argv,'S',18000)
        state=dict(evaluations={'selection_flat':dict(status='started',process_group_id=18000)})
        self.assertTrue(c.owned_evaluator(item,ROOT,state))
        self.assertFalse(c.owned_evaluator(STREAMER,ROOT,state))
        altered=c.Identity(18001,444,18000,argv,'S',19000)
        self.assertFalse(c.owned_evaluator(altered,ROOT,state))
        other_root=c.Identity(18001,444,18000,argv[:-1]+('/elsewhere/model.json',),'S',18000)
        self.assertFalse(c.owned_evaluator(other_root,ROOT,state))

    def test_resumer_recovers_after_controller_pipe_EOF(self):
        launcher=Mock()
        launcher.current.return_value=LAUNCHER
        with patch.object(c.os,'pipe',side_effect=[(10,11),(12,13)]), \
             patch.object(c.os,'fork',return_value=0,create=True), \
             patch.object(c.os,'close'), patch.object(c.os,'write'), patch.object(c.os,'read',return_value=b''), \
             patch.object(c.os,'_exit',side_effect=SystemExit), \
             patch.object(c,'read_identity',return_value=RUNNER), \
             patch.object(c.signal,'signal'), \
             patch.object(c.select,'select',return_value=([10],[],[])):
            with self.assertRaises(SystemExit):
                c.start_resumer(launcher,RUNNER,9999999999,Mock())
        launcher.resume.assert_called_once_with('controller_pipe_closed_or_killed')

    def test_resumer_does_not_exit_before_pending_parent_STOP_at_deadline(self):
        launcher=Mock()
        launcher.current.return_value=LAUNCHER
        clock=Mock()
        clock.now.return_value=100
        with patch.object(c.os,'pipe',side_effect=[(10,11),(12,13)]), \
             patch.object(c.os,'fork',return_value=0,create=True), \
             patch.object(c.os,'close'), patch.object(c.os,'write'), patch.object(c.os,'read',return_value=b'R'), \
             patch.object(c.os,'_exit',side_effect=SystemExit), \
             patch.object(c,'read_identity',return_value=RUNNER), \
             patch.object(c,'Clock',return_value=clock), patch.object(c.signal,'signal'), \
             patch.object(c.select,'select',side_effect=[([],[],[]),([10],[],[])]):
            with self.assertRaises(SystemExit):
                c.start_resumer(launcher,RUNNER,99,Mock())
        # Repeated CONT protects a STOP racing with the first recovery. Pipe EOF/R
        # proves the parent has finished its final recovery before watcher exits.
        self.assertGreaterEqual(launcher.resume.call_count,3)

    def test_parent_requires_ready_ack_before_returning_pause_permission(self):
        launcher=Mock()
        with patch.object(c.os,'pipe',side_effect=[(10,11),(12,13)]), \
             patch.object(c.os,'fork',return_value=123,create=True), \
             patch.object(c.os,'close'), patch.object(c.os,'read',return_value=b'A'), \
             patch.object(c.select,'select',return_value=([12],[],[])):
            self.assertEqual(c.start_resumer(launcher,RUNNER,9999999999,Mock()),(123,11))
        launcher.send.assert_not_called()

    def test_resumer_has_hard_recovery_grace_if_controller_stalls(self):
        launcher=Mock()
        launcher.current.return_value=LAUNCHER
        clock=Mock()
        clock.now.side_effect=[100,100,100,200,200]
        with patch.object(c.os,'pipe',side_effect=[(10,11),(12,13)]), \
             patch.object(c.os,'fork',return_value=0,create=True), \
             patch.object(c.os,'close'), patch.object(c.os,'write'), \
             patch.object(c.os,'_exit',side_effect=SystemExit), \
             patch.object(c,'read_identity',return_value=RUNNER), \
             patch.object(c,'Clock',return_value=clock), patch.object(c.signal,'signal'), \
             patch.object(c.select,'select',return_value=([],[],[])):
            with self.assertRaises(SystemExit):
                c.start_resumer(launcher,RUNNER,99,Mock())
        self.assertEqual(launcher.resume.call_args.args[0],
                         'compute_deadline_bounded_recovery_grace_ended')

    def test_missing_ready_ack_closes_liveness_pipe_and_reaps_without_STOP(self):
        launcher=Mock()
        with patch.object(c.os,'pipe',side_effect=[(10,11),(12,13)]), \
             patch.object(c.os,'fork',return_value=123,create=True), \
             patch.object(c.os,'close') as closed, patch.object(c.os,'waitpid',create=True) as reaped, \
             patch.object(c.select,'select',return_value=([],[],[])):
            with self.assertRaises(RuntimeError):
                c.start_resumer(launcher,RUNNER,9999999999,Mock())
        self.assertIn(unittest.mock.call(11),closed.call_args_list)
        reaped.assert_called_once_with(123,0)
        launcher.send.assert_not_called()

    def test_libc_adapter_sets_exact_signatures_and_uses_real_exported_symbols(self):
        library=Mock()
        library.pidfd_open.return_value=123
        library.pidfd_send_signal.return_value=0
        with patch.object(c.ctypes,'CDLL',return_value=library) as loaded:
            backend=c.PidfdBackend(prefer_native=False)
            self.assertEqual(backend.open(14000),123)
            backend.send(123,signal.SIGCONT)
        loaded.assert_called_once_with(None,use_errno=True)
        self.assertEqual(library.pidfd_open.argtypes,(ctypes.c_int,ctypes.c_uint))
        self.assertEqual(library.pidfd_open.restype,ctypes.c_int)
        self.assertEqual(library.pidfd_send_signal.argtypes,
                         (ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint))
        self.assertEqual(library.pidfd_send_signal.restype,ctypes.c_int)
        library.pidfd_send_signal.assert_called_once_with(123,int(signal.SIGCONT),None,0)

    def test_libc_missing_either_symbol_has_no_fallback(self):
        library=Mock(spec=['pidfd_open'])
        with patch.object(c.ctypes,'CDLL',return_value=library), \
             patch.object(c.os,'kill') as unsafe:
            with self.assertRaisesRegex(RuntimeError,'Both real libc'):
                c.PidfdBackend(prefer_native=False)
            unsafe.assert_not_called()

    def test_libc_errno_propagates_process_gone_without_numeric_kill(self):
        library=Mock()
        library.pidfd_open.return_value=-1
        library.pidfd_send_signal.return_value=-1
        with patch.object(c.ctypes,'CDLL',return_value=library), \
             patch.object(c.ctypes,'get_errno',return_value=errno.ESRCH), \
             patch.object(c.os,'kill') as unsafe:
            backend=c.PidfdBackend(prefer_native=False)
            with self.assertRaises(OSError) as opened: backend.open(14000)
            self.assertEqual(opened.exception.errno,errno.ESRCH)
            with self.assertRaises(OSError) as sent: backend.send(123,signal.SIGCONT)
            self.assertEqual(sent.exception.errno,errno.ESRCH)
            unsafe.assert_not_called()

    def test_runtime_without_python_wrappers_resolves_libc(self):
        library=Mock()
        with patch.object(c.os,'pidfd_open',None,create=True), \
             patch.object(c.signal,'pidfd_send_signal',None), \
             patch.object(c.ctypes,'CDLL',return_value=library):
            self.assertEqual(c.PidfdBackend().mode,'libc_pidfd')

    def test_probe_only_signals_zero_to_its_own_pidfd_and_closes(self):
        backend=Mock()
        backend.mode='libc_pidfd'
        backend.open.return_value=123
        with patch.object(c.os,'name','posix'), patch.object(c.os,'getpid',return_value=20000), \
             patch.object(c,'pidfd_backend',return_value=backend), patch.object(c.os,'close') as closed:
            evidence=c.probe_pidfd()
        backend.open.assert_called_once_with(20000,0)
        backend.send.assert_called_once_with(123,0)
        closed.assert_called_once_with(123)
        self.assertFalse(evidence['service_signals_sent'])

    def test_probe_failure_closes_descriptor_without_service_action(self):
        backend=Mock()
        backend.open.return_value=123
        backend.send.side_effect=OSError(errno.EPERM,'not permitted')
        with patch.object(c.os,'name','posix'), patch.object(c,'pidfd_backend',return_value=backend), \
             patch.object(c.os,'close') as closed:
            with self.assertRaises(OSError): c.probe_pidfd()
        closed.assert_called_once_with(123)
        backend.send.assert_called_once_with(123,0)

    def test_unknown_launcher_child_is_never_terminated(self):
        stopped=c.Identity(14000,111,1,c.LAUNCHER_ARGV,'T')
        launcher=Mock()
        launcher.current.return_value=stopped
        unknown=c.Identity(15000,444,14000,('python','other.py'),'S')
        with patch.object(c,'Handle') as handles:
            with self.assertRaises(RuntimeError):
                c.stop_streamers(launcher,[unknown],9999999999,Mock())
            handles.assert_not_called()

    def test_gpu_launcher_is_not_paused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'state.json').write_text(json.dumps(dict(stage='RUNNING_selection_flat',
                training=dict(status='complete',process_group_exited=True))))
            with patch.object(c,'read_identity',return_value=RUNNER), \
                 patch.object(c,'identities',return_value=[LAUNCHER,STREAMER]), \
                 patch.object(c,'gpu_pids',return_value={LAUNCHER.pid}), \
                 patch.object(c,'Handle') as handles, \
                 patch.object(c.signal,'signal'):
                with self.assertRaisesRegex(RuntimeError,'owns GPU'):
                    c.control(root,RUNNER,9999999999,Mock())
                handles.assert_not_called()

    def test_runner_termination_finishes_without_service_action(self):
        with patch.object(c,'read_identity',return_value=None), \
             patch.object(c,'Handle') as handles, patch.object(c.signal,'signal'):
            c.control(ROOT,RUNNER,9999999999,Mock())
            handles.assert_not_called()


if __name__=='__main__':
    unittest.main()
