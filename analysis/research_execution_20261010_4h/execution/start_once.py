"""One CPU launcher for runner -> exact identity proof -> one streaming helper.

Use IsaacLab's Python environment. No allocation API, retry, resume or GPU work
exists in this wrapper; the runner release ACK is the gate for its first train.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

import runner

TERMINAL_STAGES=frozenset(('COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION',
    'STOPPED_PARTIAL_RETRIEVE_AND_STOP_ALLOCATION','RECOVERY_FAILED'))
RUNNER_GRACE_SECONDS=120


def wrapper_signal(signum,_frame):
    raise runner.StopRequested('Launcher signal '+str(signum))


def graceful_stop_runner(process,clock,sleep=time.sleep):
    """Do not kill the runner while its own55s child cleanup is still working."""
    cutoff=clock.deadlines['shutdown_confirmed_latest']-60
    for sig,seconds in ((signal.SIGINT,RUNNER_GRACE_SECONDS),(signal.SIGTERM,15),(signal.SIGKILL,5)):
        process.poll()
        if not runner.group_alive(process.pid):return
        members=runner.verified_owned_group(process)
        if not members:
            if runner.group_alive(process.pid):raise RuntimeError('No current owned runner identity; refuse signal')
            return
        leader=next((item for item in members if item['pid']==process.pid),None)
        if leader is not None and leader.get('argv')!=process.runner_identity['command']:
            raise RuntimeError('Owned runner current argv changed; refuse wrapper signal')
        # The original runner command and current member start ticks/session
        # are rechecked immediately before every signal, including escalation.
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:return
        until=min(clock.now()+seconds,cutoff)
        while clock.now()<until:
            process.poll()
            if not runner.group_alive(process.pid):return
            sleep(.1)
    if runner.group_alive(process.pid):
        raise RuntimeError('Owned runner did not finish bounded graceful cleanup')


def wait_identity(process,command,deadline,wall=time.time,sleep=time.sleep):
    while wall()<deadline:
        if process.poll() is not None:
            raise RuntimeError('Created child exited before current identity proof')
        identity=runner.proc_identity(process.pid)
        if (identity and identity.get('argv')==command and identity['pid']==process.pid and
                identity['start_ticks']>0 and identity['ppid']==os.getpid() and
                identity['pgid']==process.pid and identity['session']==process.pid):
            return identity
        sleep(.02)
    raise RuntimeError('Current created-child PID/starttime/argv proof timed out')


def wait_runner_state(root,process,identity,deadline,wall=time.time,sleep=time.sleep):
    while wall()<deadline:
        if process.poll() is not None:
            raise RuntimeError('Runner exited during preflight initialization')
        path=root/'state.json'
        if path.exists():
            state=json.loads(path.read_text(encoding='utf-8'))
            recorded=state.get('runner_identity',{})
            if (recorded.get('pid'),recorded.get('start_ticks'))!=(identity['pid'],identity['start_ticks']):
                raise RuntimeError('Runner state identity does not match current created process')
            if state.get('training',{}).get('status')!='not_started':
                raise RuntimeError('Training preceded the first controller ACK; refuse helper dispatch')
            return state
        sleep(.02)
    raise RuntimeError('Fresh runner state unavailable within10s initialization bound')


def launch(args,popen=subprocess.Popen,wall=time.time,sleep=time.sleep):
    root=args.root.resolve()
    clock=runner.BudgetClock(args.server_start_epoch,wall=wall)
    clock.require('setup_end')
    receipt=json.loads(args.preflight.read_text(encoding='utf-8'))
    if not args.authorized_execution or receipt.get('user_authorized_execution') is not True:
        raise ValueError('Current user-authorized preflight is required')
    if receipt.get('actual_allocation_start_epoch')!=args.server_start_epoch:
        raise ValueError('Preflight allocation clock mismatch')
    if not all((root/name).is_file() for name in ('runner.py','streaming_control_libc.py')):
        raise RuntimeError('Required runner/helper source missing; no child launch')
    if any((root/name).exists() for name in ('state.json','preserved','streaming_controller_once.json')):
        raise RuntimeError('Prior experiment/helper evidence exists; no retry or resume')
    claim=root/'launch_once.json'
    with claim.open('x',encoding='utf-8') as stream:
        json.dump(dict(one_shot=True,allocation_start_epoch=args.server_start_epoch,
            launcher_pid=os.getpid(),claimed_epoch=clock.now()),stream)
        stream.flush();os.fsync(stream.fileno())
    command=[sys.executable,str(root/'runner.py'),'--root',str(root),
        '--train-dir',str(args.train_dir),'--isaac',args.isaac,
        '--source-hashes',str(args.source_hashes.resolve()),'--preflight',str(args.preflight.resolve()),
        '--server-start-epoch',str(args.server_start_epoch),'--authorized-execution']
    created=None
    try:
        with (root/'runner_console.log').open('xb') as console:
            created=popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=console,
                stderr=subprocess.STDOUT,start_new_session=True)
        # Parent-created leader anchors group cleanup before any adaptive wait.
        runner.register_owned_process(created,command)
        deadline=min(clock.deadlines['setup_end'],clock.now()+10)
        current=wait_identity(created,command,deadline,clock.now,sleep)
        state=wait_runner_state(root,created,current,deadline,clock.now,sleep)
        helper_command=[sys.executable,str(root/'streaming_control_libc.py'),
            '--root',str(root),'--runner-pid',str(current['pid']),
            '--runner-start-ticks',str(current['start_ticks']),
            '--allocation-start-epoch',str(int(args.server_start_epoch)),
            '--deadline-epoch',str(int(clock.deadlines['compute_end'])),
            '--audit',str(root/'streaming_control_audit.jsonl')]
        with (root/'streaming_control_console.log').open('xb') as console:
            helper=popen(helper_command,cwd=root,stdin=subprocess.DEVNULL,stdout=console,
                stderr=subprocess.STDOUT,start_new_session=True)
        helper_identity=wait_identity(helper,helper_command,min(clock.deadlines['setup_end'],clock.now()+2),clock.now,sleep)
        launch_receipt=dict(launched_epoch=clock.now(),allocation_start_epoch=args.server_start_epoch,
            runner=current,helper=helper_identity,runner_command=command,helper_command=helper_command,
            initial_training_status=state['training']['status'],controller_launched_once=True,
            GPU_training_started_by_launcher=False,controller_release_ACK_still_required=True)
        runner.atomic_json(root/'launch_receipt.json',launch_receipt)
        print(json.dumps(launch_receipt),flush=True)
        return created,helper,clock
    except BaseException as exc:
        cleanup_error=None
        if created is not None:
            try:
                # The runner receives a normal signal and owns its child/reward
                # cleanup. Current /proc identity is rechecked before signals.
                graceful_stop_runner(created,clock,sleep)
            except BaseException as error:
                cleanup_error=repr(error)
        runner.atomic_json(root/'launch_failed.json',dict(error=repr(exc),
            runner_cleanup_error=cleanup_error,automatic_retry_authorized=False,checked_epoch=clock.now()))
        raise


def controller_recovery_evidence(root,helper):
    """Report actual helper audit/ACK evidence; never infer allocation OFF."""
    path=root/'streaming_control_audit.jsonl'
    entries=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []
    own=[item for item in entries if item.get('writer_pid')==helper.pid]
    started=any(item.get('event')=='controller_started' for item in own)
    finished=any(item.get('event')=='controller_finished' for item in own)
    holds=[item for item in own if item.get('event')=='stage_acknowledged' and item.get('status')=='held_gpu_released']
    releases=[item for item in own if item.get('event')=='stage_released' and item.get('resumer_reaped') is True]
    acknowledgements=[item for item in own if item.get('event')=='stage_acknowledged' and item.get('status')=='released']
    recovered=all(any(item.get('epoch',0)>=hold.get('epoch',0) for item in releases+acknowledgements) for hold in holds)
    errors=[item for item in entries if item.get('event') in ('controller_error','resumer_error','resumer_reap_deadline')]
    verified=(helper.poll()==0 and started and finished and recovered and not errors)
    return dict(verified=verified,controller_returncode=helper.poll(),started_audit=started,
        finished_audit=finished,hold_ack_count=len(holds),released_after_each_hold=recovered,
        resumer_reaped_release_count=len(releases),errors=errors,allocation_off_confirmed=False)


def monitor(root,created,helper,clock,sleep=time.sleep):
    """Abort on wrapper signal/helper loss; always wait only for bounded recovery."""
    failure=None;cleanup_errors=[]
    try:
        while created.poll() is None:
            if helper.poll() is not None:
                state=json.loads((root/'state.json').read_text(encoding='utf-8'))
                if helper.poll()!=0 or state.get('stage') not in TERMINAL_STAGES:
                    raise RuntimeError('Streaming controller exited before terminal runner cleanup')
            if clock.now()>=clock.deadlines['dashboard_shutdown_request_latest']:
                raise runner.PhaseTimeout('Launcher reached portal shutdown request deadline')
            sleep(.5)
    except BaseException as exc:
        failure=repr(exc)
        raise
    finally:
        # Current exact owned group identities are checked before every signal;
        # helper is never numerically signaled or rearmed here.
        if created.poll() is None:
            try:graceful_stop_runner(created,clock,sleep)
            except BaseException as exc:cleanup_errors.append('runner_stop: '+repr(exc))
        cutoff=min(clock.now()+30,clock.deadlines['compute_end']+30,
                   clock.deadlines['dashboard_shutdown_request_latest'])
        while helper.poll() is None and clock.now()<cutoff:
            sleep(.1)
        try:recovery=controller_recovery_evidence(root,helper)
        except BaseException as exc:
            recovery=dict(verified=False,error=repr(exc),allocation_off_confirmed=False)
        if not recovery.get('verified'):cleanup_errors.append('Controller recovery evidence incomplete')
        runner.atomic_json(root/'launch_finished.json',dict(runner_returncode=created.poll(),
            helper_returncode=helper.poll(),finished_epoch=clock.now(),error=failure,
            cleanup_errors=cleanup_errors,controller_recovery_evidence=recovery,allocation_off_confirmed=False))
        if failure is None and cleanup_errors:
            raise RuntimeError('; '.join(cleanup_errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-dir',type=Path,default=Path('/workspace/training/quadruped'))
    parser.add_argument('--isaac',default='/workspace/IsaacLab/isaaclab.sh')
    parser.add_argument('--source-hashes',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--server-start-epoch',type=int,required=True)
    parser.add_argument('--authorized-execution',action='store_true')
    args=parser.parse_args()
    if os.name!='posix' or not args.authorized_execution:
        parser.error('Require current authorized Linux execution; Windows supports mocked CPU tests only')
    old={sig:signal.signal(sig,wrapper_signal) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        created,helper,clock=launch(args)
        monitor(args.root.resolve(),created,helper,clock)
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)


if __name__=='__main__':
    main()
