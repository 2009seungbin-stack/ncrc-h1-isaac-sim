"""Bounded control of the platform's default streaming child only.

No Isaac, actor, training, or PPO imports. Does not change the frozen runner,
sources, supervisor, or process groups. Linux pidfds prevent PID-reuse signals.
"""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


LAUNCHER_ARGV = ('/bin/bash', '/isaac-sim/python.sh', '/streaming/livestream_webrtc.py')
STREAMER_ARGV = ('/isaac-sim/kit/python/bin/python3', '/streaming/livestream_webrtc.py')
COMPUTE_OFFSET = 11700
RECOVERY_GRACE_SECONDS = 30
TERMINAL_STAGES = frozenset(('COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION',
                            'STOPPED_PARTIAL_RETRIEVE_AND_STOP_ALLOCATION', 'RECOVERY_FAILED'))


@dataclass(frozen=True)
class Identity:
    pid: int
    start_ticks: int
    ppid: int
    argv: tuple
    state: str
    pgrp: int = 0

    def same_process(self, other):
        return (other is not None and self.pid == other.pid and
                self.start_ticks == other.start_ticks and self.ppid == other.ppid and
                self.argv == other.argv)

    def record(self):
        return dict(pid=self.pid, start_ticks=self.start_ticks, ppid=self.ppid,
                    argv=list(self.argv), state=self.state, pgrp=self.pgrp)


def read_identity(pid, proc_root=Path('/proc')):
    try:
        folder = Path(proc_root) / str(pid)
        raw = (folder / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        argv = tuple(x.decode('utf-8', 'surrogateescape') for x in
                     (folder / 'cmdline').read_bytes().split(b'\0') if x)
        return Identity(int(pid), int(fields[19]), int(fields[1]), argv, fields[0], int(fields[2]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def identities(proc_root=Path('/proc')):
    result = []
    for folder in Path(proc_root).iterdir():
        if folder.name.isdecimal():
            item = read_identity(int(folder.name), proc_root)
            if item is not None:
                result.append(item)
    return result


def alive(item):
    return item is not None and item.state not in ('Z', 'X', 'x')


def is_launcher(item):
    return alive(item) and item.pid > 1 and item.ppid == 1 and item.argv == LAUNCHER_ARGV


def is_streamer(item, launcher):
    return (alive(item) and item.pid > 1 and item.ppid == launcher.pid and
            item.argv == STREAMER_ARGV)


def has_training(items):
    for item in items:
        if not alive(item):
            continue
        # The supervisor's actual detection contract; no synthetic marker process.
        if any(Path(arg).name == 'train.py' for arg in item.argv) and '--task' in item.argv:
            return True
        if any(Path(arg).name == 'practice_sim.py' for arg in item.argv):
            return True
        if any(re.fullmatch(r'ch_[0-9]+\.py', Path(arg).name) for arg in item.argv):
            return True
    return False


def runner_identity_ok(item, root, allocation_start):
    if not alive(item) or item.pid <= 1 or len(item.argv) < 2:
        return False
    args = item.argv
    if args[1] != str(Path(root) / 'runner.py') or '--authorized-execution' not in args:
        return False
    try:
        return (args.count('--root') == 1 and args[args.index('--root') + 1] == str(root) and
                args.count('--server-start-epoch') == 1 and
                float(args[args.index('--server-start-epoch') + 1]) == allocation_start)
    except (ValueError, IndexError):
        return False


def gpu_pids_from_xml(text):
    tree = ET.fromstring(text)
    if tree.tag != 'nvidia_smi_log' or tree.find('gpu') is None:
        raise RuntimeError('Unrecognized nvidia-smi XML; no GPU signal permission')
    result = set()
    for gpu in tree.findall('gpu'):
        listing = gpu.find('processes')
        if listing is None or (listing.text or '').strip() not in ('', 'No running processes found'):
            raise RuntimeError('GPU process inventory unavailable; no signal permission')
    for process in tree.findall('.//gpu/processes/process_info'):
        value = process.findtext('pid', '')
        if not value.isdecimal():
            raise RuntimeError('Unrecognized GPU process PID; fail closed')
        result.add(int(value))
    return result


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True,
                            timeout=8, check=True)
    return gpu_pids_from_xml(result.stdout)


def groups_exited(state):
    records = [state.get('training', {})] + list(state.get('evaluations', {}).values())
    return all(r.get('process_group_exited') is True for r in records if r.get('pid'))


def should_arm(state, items):
    training = state.get('training', {})
    return (training.get('status') == 'complete' and
            training.get('process_group_exited') is True and
            state.get('stage') not in TERMINAL_STAGES and not has_training(items))


def owned_evaluator(item, root, state):
    """Only the real runner-owned evaluation group may remain on the GPU."""
    if (not alive(item) or len(item.argv) < 2 or item.argv[1] != str(root/'evaluation.py') or
            '--task' not in item.argv):
        return False
    args = item.argv
    try:
        if (args.count('--phase') != 1 or args[args.index('--phase')+1] not in ('selection','validation') or
                args.count('--manifest') != 1 or
                not Path(args[args.index('--manifest')+1]).is_relative_to(root)):
            return False
    except (IndexError, ValueError):
        return False
    return any(record.get('status') == 'started' and
               record.get('process_group_id') == item.pgrp and item.pgrp > 1
               for record in state.get('evaluations', {}).values())


class Audit:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)

    def write(self, event, **fields):
        entry = dict(epoch=time.time(), writer_pid=os.getpid(), event=event, **fields)
        raw = (json.dumps(entry, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
        if len(raw) > 4096:
            raise RuntimeError('Audit record exceeds atomic append bound')
        if os.write(self.fd, raw) != len(raw):
            raise RuntimeError('Short audit write')
        os.fsync(self.fd)


class Handle:
    """An identity-checked pidfd; never falls back to os.kill or group signals."""
    def __init__(self, item, audit):
        if item.pid <= 1 or item.pid == os.getpid():
            raise RuntimeError('Refuse init/self process handle')
        self.item, self.audit = item, audit
        self.fd = os.pidfd_open(item.pid, 0)
        if not item.same_process(read_identity(item.pid)):
            os.close(self.fd)
            raise RuntimeError('Process identity changed while opening pidfd')

    def current(self):
        item = read_identity(self.item.pid)
        return item if self.item.same_process(item) else None

    def send(self, sig):
        current = self.current()
        if not alive(current):
            self.audit.write('signal_skipped_process_gone', identity=self.item.record(), signal=int(sig))
            return False
        # Only this exact service launcher may STOP/CONT, only its exact child TERM/KILL.
        if sig in (signal.SIGSTOP, signal.SIGCONT):
            allowed = is_launcher(current)
        elif sig in (signal.SIGTERM, signal.SIGKILL):
            allowed = current.argv == STREAMER_ARGV and current.ppid == self.item.ppid and current.pid > 1
        else:
            allowed = False
        if not allowed:
            raise RuntimeError('Signal target not exact authorized streaming service')
        signal.pidfd_send_signal(self.fd, sig, None, 0)
        self.audit.write('signal_sent', identity=current.record(), signal=int(sig))
        return True

    def resume(self, reason):
        current = self.current()
        # CONT also cancels a pending STOP that is not visible in /proc yet.
        if alive(current):
            self.send(signal.SIGCONT)
            self.audit.write('launcher_resumed', reason=reason, identity=current.record())
        else:
            self.audit.write('resume_unneeded', reason=reason, identity=self.item.record())

    def close(self):
        os.close(self.fd)


class Clock:
    def __init__(self):
        self.wall, self.mono = time.time(), time.monotonic()

    def now(self):
        return max(time.time(), self.wall + time.monotonic() - self.mono)


def start_resumer(launcher, runner, deadline, audit):
    """Fork before STOP. EOF guarantees recovery even after main-controller SIGKILL."""
    read_fd, write_fd = os.pipe()
    ready_read, ready_write = os.pipe()
    child = os.fork()
    if child:
        os.close(read_fd)
        os.close(ready_write)
        try:
            # Never STOP until the child has installed recovery handlers.
            timeout = max(0, min(3, deadline - Clock().now()))
            ready, _, _ = select.select([ready_read], [], [], timeout)
            if not ready or os.read(ready_read, 1) != b'A':
                raise RuntimeError('Bounded resumer readiness acknowledgement missing')
            audit.write('resumer_readiness_verified', resumer_pid=child)
        except BaseException:
            os.close(write_fd)  # EOF causes last-chance recovery and child exit.
            os.waitpid(child, 0)
            raise
        finally:
            os.close(ready_read)
        return child, write_fd
    os.close(write_fd)
    os.close(ready_read)
    clock = Clock()
    # Independent child must handle its own termination by restoring the launcher.
    def interrupted(signum, frame):
        raise InterruptedError('Resumer signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    reason = 'resumer_exception'
    recovering = False
    recovery_cutoff = None
    try:
        os.write(ready_write, b'A')
        os.close(ready_write)
        ready_write = None
        audit.write('resumer_ready', launcher=launcher.item.record())
        while True:
            current_runner = read_identity(runner.pid)
            if not runner.same_process(current_runner) or not alive(current_runner):
                reason = 'runner_terminated'
                recovering = True
            if clock.now() >= deadline:
                reason = 'compute_deadline'
                recovering = True
            if launcher.current() is None:
                reason = 'launcher_disappeared'
                break
            if recovering:
                if recovery_cutoff is None:
                    recovery_cutoff = min(deadline + RECOVERY_GRACE_SECONDS,
                                          clock.now() + RECOVERY_GRACE_SECONDS)
                if clock.now() >= recovery_cutoff:
                    reason += '_bounded_recovery_grace_ended'
                    break
                # Remain until parent's pipe closes; do not exit before a pending
                # parent STOP within the bounded recovery grace. Parent has
                # bounded I/O and observes the same deadline on its next loop.
                launcher.resume(reason)
            ready, _, _ = select.select([read_fd], [], [], 0.2 if recovering else 0.5)
            if ready:
                value = os.read(read_fd, 1)
                reason = 'controller_released' if value else 'controller_pipe_closed_or_killed'
                break
    except BaseException as exc:
        audit.write('resumer_error', error=repr(exc))
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            launcher.resume(reason)
            audit.write('resumer_finished', reason=reason)
        finally:
            os.close(read_fd)
            if ready_write is not None:
                os.close(ready_write)
            os._exit(0)


def stop_streamers(launcher, items, deadline, audit, root=None):
    current = launcher.current()
    if current is None or current.state not in ('T', 't'):
        raise RuntimeError('Launcher must be held before child termination')
    children = [item for item in items if alive(item) and item.ppid == current.pid]
    if any(not is_streamer(item, current) for item in children):
        raise RuntimeError('Unknown launcher child; do not terminate it')
    handles = []
    clock = Clock()
    try:
        for child in children:
            handle = Handle(child, audit)
            handles.append(handle)
            handle.send(signal.SIGTERM)
        soft_end = min(deadline, clock.now() + 10)
        while any(alive(h.current()) for h in handles) and clock.now() < soft_end:
            time.sleep(0.2)
        for handle in handles:
            if alive(handle.current()):
                handle.send(signal.SIGKILL)
        release_end = min(deadline, clock.now() + 10)
        while clock.now() < release_end:
            occupied = gpu_pids()
            if current.pid in occupied:
                raise RuntimeError('Launcher unexpectedly owns GPU; release control')
            still_gpu = [h.item.pid for h in handles if h.item.pid in occupied]
            if not still_gpu:
                if root is None:
                    raise RuntimeError('Frozen experiment root required for remaining GPU classification')
                state = json.loads((root/'state.json').read_text())
                classified = []
                for pid in sorted(occupied):
                    item = read_identity(pid)
                    if not owned_evaluator(item, root, state):
                        audit.write('unexpected_remaining_gpu_process', pid=pid,
                                    identity=item.record() if item else None)
                        raise RuntimeError('Remaining GPU process is not a runner-owned evaluator')
                    classified.append(item.record())
                audit.write('streamer_gpu_released', launcher=current.record(),
                            terminated_streamer_pids=[h.item.pid for h in handles],
                            remaining_owned_evaluators=classified)
                return
            time.sleep(0.5)
        raise RuntimeError('Streamer GPU release not verified within bound')
    finally:
        for handle in handles:
            handle.close()


def control(root, runner, deadline, audit):
    clock = Clock()
    launcher, resumer_pid, pipe_fd = None, None, None
    reason = 'controller_exception'
    try:
        while clock.now() < deadline:
            current_runner = read_identity(runner.pid)
            if not runner.same_process(current_runner) or not alive(current_runner):
                reason = 'runner_terminated'
                break
            state = json.loads((root / 'state.json').read_text())
            if state.get('stage') in TERMINAL_STAGES and groups_exited(state):
                reason = 'compute_finished_or_stopped_groups_exited'
                break
            items = identities()
            if launcher is None and should_arm(state, items):
                candidates = [item for item in items if is_launcher(item)]
                if len(candidates) > 1:
                    raise RuntimeError('Multiple exact official launchers; fail closed')
                if candidates:
                    item = candidates[0]
                    if item.state in ('T', 't'):
                        raise RuntimeError('Launcher already paused by another owner')
                    occupied = gpu_pids()
                    if item.pid in occupied:
                        raise RuntimeError('Official launcher itself owns GPU; do not pause')
                    launcher = Handle(item, audit)
                    audit.write('launcher_identified_non_gpu', identity=item.record(),
                                observed_gpu_pids=sorted(occupied))
                    resumer_pid, pipe_fd = start_resumer(launcher, runner, deadline, audit)
                    if clock.now() >= deadline or not alive(read_identity(runner.pid)):
                        reason = 'deadline_or_runner_exit_before_pause'
                        break
                    launcher.send(signal.SIGSTOP)
                    if (not alive(read_identity(resumer_pid)) or clock.now() >= deadline or
                            not runner.same_process(read_identity(runner.pid))):
                        raise RuntimeError('Resumer/runner lifetime changed during pause; recover now')
                    stopped_end = min(deadline, clock.now() + 3)
                    while clock.now() < stopped_end and (launcher.current() is not None and
                            launcher.current().state not in ('T', 't')):
                        time.sleep(0.05)
                    if launcher.current() is None or launcher.current().state not in ('T', 't'):
                        raise RuntimeError('Launcher stopped state not verified')
                    # Fresh child identities after STOP, never a cached PID list.
                    stop_streamers(launcher, identities(), deadline, audit, root)
            elif launcher is not None:
                if not alive(read_identity(resumer_pid)):
                    raise RuntimeError('Recovery resumer exited unexpectedly; release launcher now')
                current = launcher.current()
                if not alive(current) or current.state not in ('T', 't'):
                    raise RuntimeError('Held launcher exited or resumed externally; do not re-pause/rearm')
                if has_training(items):
                    raise RuntimeError('Unexpected training/practice appeared; release service control')
                exact_streamers = [i for i in items if is_streamer(i, current)]
                if exact_streamers:
                    raise RuntimeError('Streamer restarted under held launcher; release and report')
            time.sleep(0.5)
        else:
            reason = 'compute_deadline'
        audit.write('controller_finished', reason=reason)
    finally:
        # Keep recovery reliable even when a second Ctrl+C arrives during cleanup.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            if launcher is not None:
                launcher.resume(reason)
        finally:
            if pipe_fd is not None:
                try:
                    os.write(pipe_fd, b'R')
                except BrokenPipeError:
                    pass
                os.close(pipe_fd)
            if resumer_pid is not None:
                # Resumer is non-GPU, bounded by deadline, and never signalled as a group.
                os.waitpid(resumer_pid, 0)
            if launcher is not None:
                launcher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runner-pid', type=int, required=True)
    parser.add_argument('--runner-start-ticks', type=int, required=True)
    parser.add_argument('--allocation-start-epoch', type=int, required=True)
    parser.add_argument('--deadline-epoch', type=int, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    if (os.name != 'posix' or not hasattr(os, 'pidfd_open') or
            not hasattr(signal, 'pidfd_send_signal') or not hasattr(os, 'fork')):
        parser.error('Require Linux pidfds and fork; no unsafe signal fallback')
    root = args.root.resolve()
    if args.deadline_epoch != args.allocation_start_epoch + COMPUTE_OFFSET:
        parser.error('Compute deadline must equal the frozen allocation clock +11700s')
    if Clock().now() >= args.deadline_epoch:
        parser.error('Compute deadline already passed; no service control')
    runner = read_identity(args.runner_pid)
    if (not runner_identity_ok(runner, root, args.allocation_start_epoch) or
            runner.start_ticks != args.runner_start_ticks):
        parser.error('Exact active runner PID/starttime/argv proof is required')
    audit = Audit(args.audit)
    audit.write('controller_started', runner=runner.record(), root=str(root),
                deadline_epoch=args.deadline_epoch, pid1_signal_allowed=False,
                process_group_signals_allowed=False, source_modifications=False)
    def interrupted(signum, frame):
        raise InterruptedError('Controller signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        control(root, runner, args.deadline_epoch, audit)
    except BaseException as exc:
        audit.write('controller_error', error=repr(exc))
        raise
    finally:
        os.close(audit.fd)


if __name__ == '__main__':
    main()
