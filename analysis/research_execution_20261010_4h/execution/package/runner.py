"""One named scratch run and existing-B1 screen, bounded by four-hour allocation.

Linux server only. No allocation creation/stop API, submission, retry or resume.
The operator owns portal shutdown and must verify OFF by the recorded deadline.
Every GPU child owns a new process group; phase transitions require group exit.
"""
import argparse
import ast
import datetime as dt
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import tarfile
import threading
import time

PROTECTED = ('train.py', 'play.py', 'go2_task/agent_cfg.py', 'go2_task/env_cfg.py')
ORIGINAL_SHA = '09630e6f548b16a2fa0a81eef73aa393cf5f12eadccd0bad8a144b924e8c1be7'
OFFSETS = dict(setup_end=1200, training_begin_latest=1800,
               training_end_latest=9600, selection_end_latest=10500,
               compute_end=11700, dashboard_shutdown_request_latest=12900,
               shutdown_confirmed_latest=14400)
WEIGHTS = dict(zip(('G1','G2','G3','G4','G5','G6','G7'), (.15,.15,.20,.15,.15,.10,.10)))
CANDIDATE_NAME_PLACEHOLDER = 'CANDIDATE_NAME_PLACEHOLDER'
SCREEN_MAX_SECONDS = 900
PROCESS_CLEANUP_RESERVE_SECONDS = 55


class StopRequested(BaseException):
    pass


class PhaseTimeout(RuntimeError):
    pass


class CheckpointMissing(RuntimeError):
    pass


def signal_stop(signum, _frame):
    raise StopRequested('Signal '+str(signum))


def sha(path, clock=None, deadline_key='dashboard_shutdown_request_latest'):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            if clock is not None:
                clock.require(deadline_key)
            digest.update(block)
    return digest.hexdigest()


class DeadlineReader:
    def __init__(self, stream, clock, deadline_key):
        self.stream,self.clock,self.deadline_key = stream,clock,deadline_key

    def read(self, size=-1):
        self.clock.require(self.deadline_key)
        return self.stream.read(size)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))


def dictionary_span(data):
    tree = ast.parse(data.decode('utf-8'))
    nodes = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in n.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Dict):
        raise ValueError('Require one literal REWARD_WEIGHTS dictionary')
    node = nodes[0]
    lines = data.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    return start, end, ast.literal_eval(node)


def validate_candidate(original, candidate, expected):
    a, b, _ = dictionary_span(original)
    c, d, actual = dictionary_span(candidate)
    if original[:a] != candidate[:c] or original[b:] != candidate[d:]:
        raise ValueError('Candidate changes bytes outside the reward dictionary')
    if actual != expected or len(actual) != 10:
        raise ValueError('Candidate reward values differ from frozen ten-weight design')
    for key, value in actual.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid reward value: '+key)
        if (key.startswith('track_') and value <= 0) or (key.endswith('_l2') and value > 0):
            raise ValueError('Invalid reward sign: '+key)
    return actual


class BudgetClock:
    def __init__(self, allocation_start, wall=time.time, monotonic=time.monotonic):
        self.wall, self.monotonic = wall, monotonic
        self.boot_wall, self.boot_monotonic = wall(), monotonic()
        if not math.isfinite(allocation_start) or allocation_start > self.boot_wall+5:
            raise ValueError('Invalid actual allocation start')
        self.start = allocation_start
        self.deadlines = {k: allocation_start+v for k, v in OFFSETS.items()}

    def now(self):
        return max(self.wall(), self.boot_wall+self.monotonic()-self.boot_monotonic)

    def require(self, key):
        if self.now() >= self.deadlines[key]:
            raise PhaseTimeout('Allocation deadline passed: '+key)


def training_estimate(iterations, rate, multiplier=1.05, boot=120):
    if iterations < 0 or not math.isfinite(rate) or rate <= 0 or not math.isfinite(multiplier) or multiplier < 1 or boot < 0:
        raise ValueError('Invalid training estimate inputs')
    return iterations*rate*multiplier+boot


def training_profile(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    iterations = re.findall(r'Learning iteration (\d+)/(\d+)', text)
    durations = [float(x) for x in re.findall(r'Iteration time:\s+([0-9.]+)s', text)]
    indexes = [int(index) for index, _ in iterations]
    if iterations and any(int(total) != 1500 for _, total in iterations):
        raise RuntimeError('Training log reports an unexpected maximum iteration count')
    if indexes != list(range(len(indexes))) or len(indexes) > 1500:
        raise RuntimeError('Training iteration headers are not unique contiguous scratch iterations')
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise RuntimeError('Training log has invalid iteration durations')
    return dict(samples=len(durations), completed_iterations=(int(iterations[-1][0])+1 if iterations else 0),
                exact_iteration_coverage=indexes == list(range(1500)),
                recent_seconds_per_iteration=(sum(durations[-50:])/len(durations[-50:]) if durations else None))


def yaml_reward_weights(text):
    result, active, term = {}, False, None
    for line in text.splitlines():
        if line == 'rewards:':
            active = True
            continue
        if active and line and not line[0].isspace() and not line.startswith('#'):
            break
        if not active:
            continue
        match = re.fullmatch(r'  ([A-Za-z_][A-Za-z_0-9]*):(?: null)?', line)
        if match:
            term = match.group(1)
        match = re.fullmatch(r'    weight: (\S+)\s*', line)
        if match and term:
            result[term] = float(match.group(1))
    return result


def verify_config(run, expected):
    env = (run/'params/env.yaml').read_text(encoding='utf-8')
    agent = (run/'params/agent.yaml').read_text(encoding='utf-8')
    actual = yaml_reward_weights(env)
    if any(actual.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Effective reward settings mismatch: '+str(actual))
    for key, value in [('seed','42'), ('max_iterations','1500'), ('resume','false')]:
        if not re.search(r'^'+key+': '+value+r'\s*$', agent, re.M):
            raise RuntimeError('Effective agent config mismatch: '+key)
    if not re.search(r'^  num_envs: 6144\s*$', env, re.M):
        raise RuntimeError('Effective environment count mismatch')
    if not re.search(r'^clip_actions: null\s*$',agent,re.M):
        raise RuntimeError('Actual action clipping must be explicit null')
    if not re.search(r'^empirical_normalization: (?:\{\}|false)\s*$',agent,re.M):
        raise RuntimeError('Actual empirical normalization must be disabled or empty identity')
    actor_configs={}
    for role in ('actor','critic'):
        match=re.search(r'^'+role+r':\n((?:[ \t].*\n|\n)+)',agent,re.M)
        if not match:
            raise RuntimeError('Unrecognized effective actor/critic configuration: '+role)
        block=match.group(1)
        for key,value in [('class_name','MLPModel'),('activation','elu'),('obs_normalization','false')]:
            if not re.search(r'^  '+key+': '+value+r'\s*$',block,re.M):
                raise RuntimeError('Actual '+role+' '+key+' differs from frozen evaluator contract')
        dims=re.search(r'^  hidden_dims:\n((?:  - \d+\n)+)',block,re.M)
        if not dims or [int(v) for v in re.findall(r'\d+',dims.group(1))] !=[512,256,128]:
            raise RuntimeError('Actual '+role+' hidden dimensions differ from512/256/128')
        actor_configs[role]=dict(class_name='MLPModel',activation='elu',obs_normalization=False,hidden_dims=[512,256,128])
    decimation=re.search(r'^decimation: (\d+)\s*$',env,re.M)
    sim_block=re.search(r'^sim:\n((?:[ \t].*\n|\n)+)',env,re.M)
    physics_dt=re.search(r'^  dt: ([0-9.]+)\s*$',sim_block.group(1),re.M) if sim_block else None
    if not decimation or not physics_dt or float(physics_dt.group(1))*int(decimation.group(1)) !=.02:
        raise RuntimeError('Actual physics timing does not give0.02s control steps')
    return dict(effective_weights=actual, env_sha256=sha(run/'params/env.yaml'),
                agent_sha256=sha(run/'params/agent.yaml'),actor_observation_normalization=False,
                policy_clip_actions=None,actor_configs=actor_configs,control_step_seconds=.02)


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def proc_identity(pid, proc_root=Path('/proc')):
    """Identity anchored to kernel start ticks, never a previously stored GPU PID."""
    folder = proc_root/str(pid)
    try:
        text = (folder/'stat').read_text()
        tail = text[text.rfind(')')+2:].split()
        argv=[part.decode('utf-8',errors='replace') for part in (folder/'cmdline').read_bytes().split(b'\0') if part]
        command=' '.join(argv)
    except FileNotFoundError:
        return None
    return dict(pid=int(pid),state=tail[0],ppid=int(tail[1]),pgid=int(tail[2]),
                session=int(tail[3]),start_ticks=int(tail[19]),cmdline=command,argv=argv)


def group_snapshot(pgid, proc_root=Path('/proc')):
    members=[]
    for folder in proc_root.iterdir():
        if folder.name.isdigit():
            value=proc_identity(int(folder.name),proc_root)
            if value and value['pgid']==pgid:
                members.append(value)
    return members


def register_owned_process(process, command):
    leader=proc_identity(process.pid)
    if leader is None or leader['pgid']!=process.pid or leader['session']!=process.pid:
        raise RuntimeError('Cannot confirm newly created owned process session')
    process.runner_identity=dict(leader=leader,command=list(command),
        seen={leader['pid']:leader['start_ticks']})
    return dict(leader=leader,expected_command=list(command))


def verified_owned_group(process):
    owned=getattr(process,'runner_identity',None)
    if owned is None:
        raise RuntimeError('Owned group identity missing; refuse signal')
    members=group_snapshot(process.pid)
    if not members:
        return []
    leader=owned['leader']
    if any(m['session']!=process.pid or m['start_ticks']<leader['start_ticks'] for m in members):
        raise RuntimeError('Current group/session identity mismatch; refuse signal')
    current_leader=next((m for m in members if m['pid']==process.pid),None)
    if current_leader and current_leader['start_ticks']!=leader['start_ticks']:
        raise RuntimeError('Owned leader PID was reused; refuse signal')
    known=[m for m in members if owned['seen'].get(m['pid'])==m['start_ticks']]
    if not known:
        raise RuntimeError('No current member of the observed owned group remains; refuse signal')
    for member in members:
        previous=owned['seen'].get(member['pid'])
        if previous is not None and previous!=member['start_ticks']:
            raise RuntimeError('Owned member PID was reused; refuse signal')
        owned['seen'][member['pid']]=member['start_ticks']
    return members


def stop_process(process):
    if process is None:
        return
    # The shell leader may exit while its GPU Python descendants are alive.
    for sig, seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        process.poll()
        if not group_alive(process.pid):
            return
        # Refresh exact identities immediately before each owned group signal.
        if not verified_owned_group(process):
            if group_alive(process.pid):
                raise RuntimeError('Group exists without readable owned identity; refuse signal')
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_alive(process.pid):
                return
            time.sleep(.1)
    process.poll()
    if group_alive(process.pid):
        raise RuntimeError('Owned process group remains alive after SIGKILL')


class CheckpointCapture:
    """Inotify preserves checkpoint inodes before the provided post-train cleanup.

    Hardlinks remain inside the unique owned run (same filesystem), with .capture
    suffixes. Completed bytes are copied to preserved/ only after stable checks.
    This never changes provided training code or catches a deleted file by guess.
    """
    SPOOL = '_runner_checkpoint_capture'

    def __init__(self, parent, run_name):
        self.parent,self.run_name=Path(parent),run_name
        self.done=threading.Event()
        self.error=None
        self.fd=None
        self.watches={}
        self.thread=None

    def link(self, run, source):
        if source.is_symlink() or not source.is_file():
            return
        if not re.fullmatch(r'model_\d+\.pt',source.name):
            return
        spool=run/self.SPOOL
        spool.mkdir(exist_ok=True)
        target=spool/(source.name+'.capture')
        try:
            os.link(source,target)
        except FileExistsError:
            if source.stat().st_ino!=target.stat().st_ino or source.stat().st_dev!=target.stat().st_dev:
                raise RuntimeError('Checkpoint name overwritten with a different inode')
        except FileNotFoundError:
            # Deletion raced before capture. The final exact-checkpoint guard
            # will report unavailable evidence; no model_best substitution.
            return

    def discover(self):
        runs=list(self.parent.glob('*_'+self.run_name))
        if len(runs)>1:
            raise RuntimeError('Duplicate owned training folders during capture')
        for run in runs:
            if run.is_symlink() or not run.is_dir():
                raise RuntimeError('Invalid owned run folder')
            if run not in self.watches.values():
                wd=self.libc.inotify_add_watch(self.fd,os.fsencode(run),0x00000100|0x00000008|0x00000080)
                if wd<0:
                    raise OSError(ctypes.get_errno(),'inotify owned run watch failed')
                self.watches[wd]=run
            for source in run.glob('model_*.pt'):
                self.link(run,source)

    def start(self):
        if os.name!='posix':
            raise RuntimeError('Checkpoint capture requires Linux inotify')
        self.parent.mkdir(parents=True,exist_ok=True)
        self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.inotify_init1.argtypes=[ctypes.c_int]
        self.libc.inotify_init1.restype=ctypes.c_int
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        self.libc.inotify_add_watch.restype=ctypes.c_int
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:
            raise OSError(ctypes.get_errno(),'inotify init failed')
        wd=self.libc.inotify_add_watch(self.fd,os.fsencode(self.parent),0x00000100|0x00000080)
        if wd<0:
            os.close(self.fd);self.fd=None
            raise OSError(ctypes.get_errno(),'inotify parent watch failed')
        self.thread=threading.Thread(target=self._watch,name='owned-checkpoint-capture',daemon=True)
        self.thread.start()

    def _watch(self):
        try:
            self.discover()
            while not self.done.is_set():
                try:
                    data=os.read(self.fd,65536)
                except BlockingIOError:
                    self.done.wait(.01)
                    continue
                offset=0
                while offset<len(data):
                    wd,mask,cookie,length=struct.unpack_from('iIII',data,offset)
                    name=os.fsdecode(data[offset+16:offset+16+length].split(b'\0',1)[0])
                    offset+=16+length
                    if mask&0x00004000:
                        raise RuntimeError('Checkpoint inotify queue overflow; capture evidence incomplete')
                    run=self.watches.get(wd)
                    if run is not None:
                        self.link(run,run/name)
                    else:
                        self.discover()
        except BaseException as exc:
            self.error=exc

    def healthy(self):
        if self.error:
            raise RuntimeError('Checkpoint capture failed: '+repr(self.error))

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(2)
            if self.thread.is_alive():
                raise RuntimeError('Checkpoint capture thread did not terminate')
        if self.fd is not None:
            os.close(self.fd);self.fd=None
        self.healthy()


def verify_checkpoint(path, iteration, expected_sha=None):
    """CPU, weights_only: exact saved iteration and finite floating tensors."""
    import torch
    path=Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size<=0:
        raise RuntimeError('Exact preserved checkpoint missing: '+str(path))
    before=sha(path)
    if expected_sha and before!=expected_sha:
        raise RuntimeError('Checkpoint differs from frozen SHA256')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or value.get('iter')!=iteration:
        raise RuntimeError('Preserved checkpoint stores wrong exact iteration')
    if not any(key in value for key in ('actor_state_dict','model_state_dict')):
        raise RuntimeError('Preserved checkpoint has no supported actor state')
    count=0
    def inspect(item):
        nonlocal count
        if isinstance(item,torch.Tensor):
            count+=1
            if (item.is_floating_point() or item.is_complex()) and not torch.isfinite(item).all().item():
                raise RuntimeError('Nonfinite tensor in preserved checkpoint')
        elif isinstance(item,dict):
            for child in item.values(): inspect(child)
        elif isinstance(item,(tuple,list)):
            for child in item: inspect(child)
    inspect(value)
    if count==0 or sha(path)!=before:
        raise RuntimeError('Checkpoint tensor coverage/immutability verification failed')
    return dict(sha256=before,bytes=path.stat().st_size,iteration=iteration,
                exact_stored_iteration_verified=True,finite_tensor_count=count)


def source_guard(train, expected, runtime):
    if set(expected) != set(PROTECTED):
        raise ValueError('Require exactly four provided source hashes')
    actual = {k: sha(train/k) for k in expected}
    if actual != expected:
        raise RuntimeError('Protected provided source mismatch')
    if runtime.get('algorithm_package') != 'rsl_rl' or not runtime.get('files'):
        raise ValueError('Require installed rsl_rl source snapshot')
    for path, digest in runtime['files'].items():
        if not Path(path).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid PPO runtime source manifest')
        if sha(path) != digest:
            raise RuntimeError('Protected PPO/runtime source mismatch: '+path)
    if runtime.get('package_root'):
        inventory = {str(p.resolve()) for p in Path(runtime['package_root']).rglob('*.py') if p.is_file()}
        if inventory != set(runtime['files']):
            raise RuntimeError('Protected PPO source inventory changed')
    return dict(ok=True, provided=actual, runtime=runtime['files'], checked_epoch=time.time())


def subset_rank(model, scenarios, seed, iteration):
    """Selection rank only; missing rows or sample audits invalidate ranking."""
    rows = model['seeds'][str(seed)]['scenarios']
    if set(rows) != set(scenarios) or model.get('diagnostics_complete') is not True:
        raise ValueError('Incomplete selection scenario/diagnostic coverage')
    quality, xy2, falls = 0., 0., 0
    for sid in scenarios:
        row = rows[sid]
        n, ep, survived = row['valid_samples'], row['episodes'], row['survived']
        if n <= 0 or ep != 64 or row['completed_control_steps'] != 1000 or row['nonfinite_count']:
            raise ValueError('Incomplete selection sample coverage')
        if len(row['per_episode_valid_samples']) != 64 or sum(row['per_episode_valid_samples']) != n:
            raise ValueError('Selection sample reconciliation failed')
        for count, terminal, timeout in zip(row['per_episode_valid_samples'], row['first_terminal_steps'], row['first_timeout_steps']):
            if timeout is not None and timeout != 999:
                raise ValueError('Selection early timeout')
            events = [e for e in (terminal, timeout) if e is not None]
            if count != (min(events) if events else 1000):
                raise ValueError('Selection horizon mask mismatch')
        mse = [row['sum_squared_error'][a]/n for a in ('vx','vy','yaw')]
        if any(not math.isfinite(x) or x < 0 for x in mse):
            raise ValueError('Selection nonfinite error sum')
        quality += WEIGHTS[sid]*(survived/ep)*math.exp(-math.sqrt(sum(mse)/3))
        xy2 += WEIGHTS[sid]*(mse[0]+mse[1])/2
        falls += ep-survived
    normalization = sum(WEIGHTS[s] for s in scenarios)
    return (falls, -quality/normalization, math.sqrt(xy2/normalization), -iteration)


class Runner:
    def __init__(self, root, train, isaac, source_hashes, allocation_start, preflight):
        self.root, self.train = Path(root).resolve(), Path(train).resolve()
        self.isaac = str(isaac)
        self.clock = BudgetClock(allocation_start)
        self.design = json.loads((self.root/'design.json').read_text(encoding='utf-8'))
        self.expected = json.loads(Path(source_hashes).read_text(encoding='utf-8'))
        self.runtime = json.loads((self.root/'protected_runtime_sources.json').read_text(encoding='utf-8'))
        self.preflight = json.loads(Path(preflight).read_text(encoding='utf-8'))
        self.original = (self.root/'fresh_rewards_original.py').read_bytes()
        self.arm=self.design.get('candidate_name',CANDIDATE_NAME_PLACEHOLDER)
        if self.arm==CANDIDATE_NAME_PLACEHOLDER or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{1,63}',self.arm):
            raise ValueError('Require parent-confirmed final candidate_name before execution')
        reward_path=Path(self.design['candidate_reward_file'])
        if reward_path.is_absolute() or '..' in reward_path.parts:
            raise ValueError('Candidate reward path must stay within frozen root')
        self.candidate = (self.root/reward_path).read_bytes()
        self.process, self.data, self.inputs = None, None, None
        self.owns_mutation = False
        self.active_reward = self.original
        self.capture=None
        self.streaming_counter=0
        self.expected_weights = self.design['candidates'][self.arm]['weights']
        validate_candidate(self.original, self.candidate, self.expected_weights)
        if sha(self.root/'fresh_rewards_original.py') != ORIGINAL_SHA:
            raise ValueError('Packaged reward original hash mismatch')
        self.run_name = self.arm+'_4h_'+str(int(allocation_start))
        self.references = json.loads((self.root/'reference_models.json').read_text(encoding='utf-8'))
        self.screen_enabled=self.design['evaluation'].get('screen',{}).get('enabled') is True
        self.existing_b1 = json.loads((self.root/'existing_b1_models.json').read_text(encoding='utf-8')) if self.screen_enabled else []

    def now(self):
        return self.clock.now()

    def save(self, stage=None):
        if not self.data:
            return
        if stage:
            self.data['stage'] = stage
        self.data['updated_epoch'] = self.now()
        self.data['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(self.root/'state.json', self.data)
        print(json.dumps({'stage': self.data['stage'], 'epoch': self.data['updated_epoch']}), flush=True)

    def guard(self):
        audit = source_guard(self.train, self.expected, self.runtime)
        if (self.train/'quadruped_rewards.py').read_bytes() != self.active_reward:
            raise RuntimeError('Current reward bytes changed outside owned phase transition')
        if self.inputs and any(sha(self.root/k) != v for k, v in self.inputs.items()):
            raise RuntimeError('Frozen experiment input changed')
        atomic_json(self.root/'current_source_audit.json', audit)
        return audit

    def initialize(self):
        self.clock.require('setup_end')
        if (self.root/'state.json').exists() or self.runs() or (self.root/'preserved').exists():
            raise RuntimeError('Prior run evidence exists; retry/resume is prohibited')
        training = self.design['training']
        if (training['iterations'], training['num_envs'], training['seed'], training['resume'], training['policies']) != (1500,6144,42,False,1):
            raise ValueError('Frozen one-arm scratch training contract mismatch')
        if self.design['user_budget_seconds'] != 14400 or self.design['budget']['deadlines_relative_to_actual_allocation_start'] != OFFSETS:
            raise ValueError('Frozen four-hour deadline contract mismatch')
        for phase in ('screen','selection','validation'):
            cfg=self.design['evaluation'][phase]
            if phase=='screen' and not self.screen_enabled:
                continue
            expected=(['G1','G3','G5','G6'],64,[2029],404,10037) if phase!='validation' else (
                ['G1','G2','G3','G4','G5','G6','G7'],128,[2039,2063],409,11003)
            if tuple(cfg[k] for k in ('scenarios','num_envs','reset_seeds','startup_seed','terrain_seed'))!=expected:
                raise ValueError('Frozen '+phase+' protocol mismatch')
        receipt = self.preflight
        if receipt.get('actual_allocation_start_epoch') != self.clock.start:
            raise ValueError('Preflight allocation clock differs')
        for key in ('user_authorized_execution','retrieval_smoke_verified','no_competing_gpu_jobs_verified','runtime_verified','reference_loader_verified','checkpoint_capture_probe_verified'):
            if receipt.get(key) is not True:
                raise ValueError('Required preflight evidence missing: '+key)
        if receipt.get('verified_epoch', 0) < self.clock.start or receipt.get('verified_epoch', 0) > self.now()+5:
            raise ValueError('Preflight evidence timestamp is not current allocation')
        if not re.fullmatch(r'[0-9a-f]{64}', receipt.get('probe_sha256','')):
            raise ValueError('Preflight retrieval probe hash missing')
        if receipt.get('probe_local_sha256') != receipt.get('probe_sha256'):
            raise ValueError('Preflight local/remote probe hash mismatch')
        capture_probe=receipt.get('checkpoint_capture_probe',{})
        if any(capture_probe.get(key) is not True for key in ('ok','cpu_only','inotify_new_run_discovery',
                'hardlink_inode_verified','original_unlink_preservation_verified')):
            raise ValueError('Require actual Linux checkpoint capture compatibility evidence')
        if not self.clock.start<=capture_probe.get('verified_epoch',0)<=self.now()+5:
            raise ValueError('Checkpoint capture probe is not from the current allocation')
        actual_versions = receipt.get('runtime_versions',{})
        if any(not isinstance(actual_versions.get(k),str) or not actual_versions[k] for k in ('rsl_rl_lib','torch')):
            raise ValueError('Require actual current runtime version evidence')
        self.guard()
        if [r['name'] for r in self.references] != ['C0_5500','F10_3000']:
            raise ValueError('Require both exact reference policies')
        for model in self.references:
            path = Path(model['checkpoint'])
            if path.is_absolute() or '..' in path.parts or sha(self.root/path) != model['sha256']:
                raise ValueError('Reference policy path/hash mismatch')
        if (self.train/'quadruped_rewards.py').read_bytes() != self.original:
            raise RuntimeError('Active reward file differs from known original; refuse overwrite')
        if self.screen_enabled and ([m.get('iteration') for m in self.existing_b1]!=[600,1000,1499] or len({m['name'] for m in self.existing_b1})!=3):
            raise ValueError('Require exact existing B1 checkpoints600/1000/1499')
        if {m['name'] for m in self.existing_b1}&{m['name'] for m in self.references}:
            raise ValueError('Existing policy/reference names collide')
        for model in self.existing_b1:
            path=Path(model['checkpoint'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Existing B1 checkpoint path escapes frozen root')
            verify_checkpoint(self.root/path,model['iteration'],model['sha256'])
        if not (self.design['budget'].get('planning_multiplier')==1.05 and
                self.design['budget'].get('training_boot_seconds')==120 and
                self.design['budget'].get('selection_seconds')==900 and
                self.design['budget'].get('validation_seconds')==1650 and
                self.design['budget'].get('computation_margin_seconds')==300):
            raise ValueError('Frozen screen/training admission budget mismatch')
        if self.screen_enabled and self.design['budget'].get('screen_max_seconds')!=SCREEN_MAX_SECONDS:
            raise ValueError('Frozen optional screen duration mismatch')
        files = [p for p in self.root.rglob('*') if p.is_file() and (p.suffix == '.py' or p.name in ('design.json','reference_models.json','existing_b1_models.json','protected_runtime_sources.json')) and '__pycache__' not in p.parts]
        self.inputs = {p.relative_to(self.root).as_posix():sha(p) for p in files}
        self.data = dict(schema_version=1, stage='PREPARED', allocation_start_epoch=self.clock.start,
                         deadlines_epoch=self.clock.deadlines, training=dict(status='not_started',run_name=self.run_name),
                         evaluations={},screen=dict(status='not_started'),selected=None,candidate_name=self.arm,
                         input_hashes=self.inputs, source_hashes=self.expected,
                         preflight=self.preflight, original_reward_sha256=ORIGINAL_SHA,
                         runner_identity=proc_identity(os.getpid()),
                         official_score_prediction=None, allocation_off_confirmed=False,
                         warnings=['One training seed; references have unequal training length',
                                   'Internal comparison only; no official prediction or automatic follow-up'])
        (self.root/'logs').mkdir(exist_ok=True)
        self.save()
        self.owns_mutation = True

    def runs(self):
        return list((self.train/'logs/rsl_rl/quadruped').glob('*_'+self.run_name))

    def restore(self):
        target = self.train/'quadruped_rewards.py'
        current = target.read_bytes()
        if current not in (self.original,self.candidate):
            raise RuntimeError('Unknown external reward modification; refusing rollback overwrite')
        atomic_bytes(target,self.original)
        self.active_reward = self.original
        if sha(target) != ORIGINAL_SHA:
            raise RuntimeError('Reward rollback hash mismatch')
        atomic_json(self.root/'reward_restored.json',dict(restored=True,sha256=sha(target),verified_epoch=self.now()))

    def preserve(self):
        runs = self.runs()
        if len(runs) > 1:
            raise RuntimeError('Ambiguous duplicate named training output')
        for run in runs:
            for source in run.rglob('*'):
                if not source.is_file() or '__pycache__' in source.parts or CheckpointCapture.SPOOL in source.parts:
                    continue
                target = self.root/'preserved'/self.arm/run.name/source.relative_to(run)
                target.parent.mkdir(parents=True,exist_ok=True)
                try:
                    before = (source.stat().st_size,source.stat().st_mtime_ns)
                except FileNotFoundError:
                    continue
                if target.exists() and (target.stat().st_size,target.stat().st_mtime_ns) == before:
                    continue
                tmp = target.with_name(target.name+'.copying')
                try:
                    shutil.copy2(source,tmp)
                    if before == (source.stat().st_size,source.stat().st_mtime_ns):
                        tmp.replace(target)
                except FileNotFoundError:
                    # The provided train may remove original .pt paths after
                    # learn(). Consume linked capture bytes below instead.
                    if tmp.exists(): tmp.unlink()
            for source in (run/CheckpointCapture.SPOOL).glob('model_*.pt.capture'):
                target=self.root/'preserved'/self.arm/run.name/source.name.removesuffix('.capture')
                target.parent.mkdir(parents=True,exist_ok=True)
                before=(source.stat().st_size,source.stat().st_mtime_ns)
                if target.exists() and (target.stat().st_size,target.stat().st_mtime_ns)==before:
                    continue
                tmp=target.with_name(target.name+'.copying')
                shutil.copy2(source,tmp)
                if before==(source.stat().st_size,source.stat().st_mtime_ns): tmp.replace(target)
        return runs

    def request_streaming(self,mode,label,deadline_key):
        if mode not in ('hold','release'):
            raise ValueError('Unknown streaming mode')
        self.clock.require(deadline_key)
        self.streaming_counter+=1
        requested=self.now()
        token=label+'_'+str(self.streaming_counter)+'_'+str(time.time_ns())
        self.data['streaming_request']=dict(mode=mode,token=token,requested_epoch=requested)
        previous_stage=self.data['stage']
        self.save('STREAMING_HANDSHAKE_'+label)
        limit=min(requested+30,self.clock.deadlines[deadline_key]-PROCESS_CLEANUP_RESERVE_SECONDS)
        expected='held_gpu_released' if mode=='hold' else 'released'
        while self.now()<limit:
            path=self.root/'streaming_status.json'
            if path.exists():
                status=json.loads(path.read_text(encoding='utf-8'))
                if status.get('token')==token:
                    if status.get('mode')!=mode or status.get('status')!=expected or status.get('verified_epoch',0)<requested:
                        raise RuntimeError('Streaming controller returned invalid/current-error ACK')
                    self.data.setdefault('streaming_handshakes',[]).append(status)
                    self.save(previous_stage)
                    return
            time.sleep(.2)
        raise PhaseTimeout('Current streaming '+mode+' ACK unavailable before phase cutoff')

    def screen_existing(self):
        if self.data['screen']['status']!='not_started':
            raise RuntimeError('Existing-policy screen already attempted; no retry')
        if not self.screen_enabled:
            self.data['screen']=dict(status='disabled_by_frozen_design',executed=False,
                reason='Train-first allocation preserves conservative compute/retrieval/shutdown budget')
            self.save('SCREEN_DISABLED_TRAIN_FIRST')
            return
        began=self.now()
        self.clock.deadlines['screen_end']=min(began+SCREEN_MAX_SECONDS,
                                             self.clock.deadlines['training_begin_latest'])
        self.data['deadlines_epoch']=dict(self.clock.deadlines)
        self.data['screen'].update(status='started',started_epoch=began,deadline_epoch=self.clock.deadlines['screen_end'])
        self.save('SCREEN_PREPARED')
        try:
            self.request_streaming('hold','screen','screen_end')
            result=self.evaluate_phase('screen',self.references+self.existing_b1,'screen_end')
            cfg=self.design['evaluation']['screen']
            ranks=[dict(model=m,rank=list(subset_rank(result[m['name']],cfg['scenarios'],
                                cfg['reset_seeds'][0],m.get('iteration',0)))) for m in self.existing_b1]
            self.data['screen'].update(status='complete',ranking=ranks,ended_epoch=self.now(),
                affects_new_training_candidate=False,official_score_prediction=None)
            atomic_json(self.root/'existing_b1_screen.json',self.data['screen'])
        except PhaseTimeout as exc:
            if self.process is not None and group_alive(self.process.pid):
                raise RuntimeError('Screen group remains alive; refuse new training')
            self.process=None
            self.data['screen'].update(status='incomplete_timeout_no_retry',reason=str(exc),ended_epoch=self.now())
            for label,record in self.data['evaluations'].items():
                if label.startswith('screen_') and record.get('status')=='started':
                    record['status']='incomplete_timeout_no_retry'
            atomic_json(self.root/'existing_b1_screen.json',self.data['screen'])
        self.save('SCREEN_FINISHED')

    def inspect_log(self, log):
        text = log.read_text(encoding='utf-8',errors='replace')
        if re.search(r'(?:PhysX|Pxg|GPU).{0,180}(?:overflow|capacity exceeded)',text,re.I) or re.search(r'(?:overflow|capacity exceeded).{0,180}(?:PhysX|Pxg)',text,re.I):
            raise RuntimeError('Physics overflow in child log')
        if re.search(r'(?:loss|reward|gradient|std|action|observation|tensor|mean).{0,100}\b(?:nan|[+-]?inf)\b',text,re.I):
            raise RuntimeError('Nonfinite numerical value in child log')
        return text

    def job(self, command, label, deadline_key, training=False):
        self.guard()
        self.clock.require(deadline_key)
        process_end = self.clock.deadlines[deadline_key]-PROCESS_CLEANUP_RESERVE_SECONDS
        if self.now() >= process_end:
            raise PhaseTimeout('No owned-process cleanup margin remains for '+label)
        began = self.now()
        checked, last_profile_checkpoint, last_inspect = False, 0, 0
        log = self.root/'logs'/(label+'.log')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.train)+os.pathsep+env.get('PYTHONPATH','')
        record = self.data['training'] if training else self.data['evaluations'].setdefault(label,{})
        with log.open('xb') as stream:
            self.process = subprocess.Popen(command,cwd=self.train,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(status='started',pid=self.process.pid,process_group_id=self.process.pid,started_epoch=began,command=command)
            try:
                record['owned_process_identity']=register_owned_process(self.process,command)
                self.save('RUNNING_'+label)
                while self.process.poll() is None:
                    verified_owned_group(self.process)
                    if training and self.capture is not None: self.capture.healthy()
                    if self.now() >= process_end:
                        raise PhaseTimeout('Child computation cutoff protects group cleanup: '+label)
                    if self.now()-last_inspect >= 15:
                        text = self.inspect_log(log)
                        self.guard()
                        if training:
                            runs = self.preserve()
                            if runs and (runs[0]/'params/env.yaml').exists() and (runs[0]/'params/agent.yaml').exists() and (runs[0]/'model_0.pt').exists():
                                if not checked:
                                    record['configuration'] = verify_config(runs[0],self.expected_weights)
                                    record['initial_checkpoint_sha256'] = sha(runs[0]/'model_0.pt')
                                    checked = True
                                    self.save()
                            if not checked and self.now()-began > 120:
                                raise RuntimeError('Training boot exceeds120s without effective config and model_0')
                            profile = training_profile(text)
                            completed = profile['completed_iterations']
                            if completed >=100 and completed//100 > last_profile_checkpoint:
                                rate = max(self.design['budget']['planning_iteration_seconds'],profile['recent_seconds_per_iteration'])
                                projected = training_estimate(max(0,1500-completed),rate,
                                    multiplier=self.design['budget']['planning_multiplier'],boot=0)
                                record['training_time_profile'] = profile
                                if self.now()+projected > process_end:
                                    raise PhaseTimeout('Observed training rate cannot preserve downstream reservations')
                                downstream=sum(self.design['budget'][key] for key in
                                    ('selection_seconds','validation_seconds','computation_margin_seconds'))
                                if self.now()+projected+downstream>self.clock.deadlines['compute_end']:
                                    raise PhaseTimeout('Observed training ETA cannot preserve bounded evaluation plus300s compute margin')
                                last_profile_checkpoint = completed//100
                                self.save()
                        last_inspect = self.now()
                    time.sleep(.5)
            finally:
                stop_process(self.process)
                record['process_group_exited'] = not group_alive(self.process.pid)
                record['returncode'] = self.process.poll()
                record['ended_epoch'] = self.now()
                if training:
                    self.preserve()
                self.save()
        code = self.process.returncode
        self.process = None
        self.inspect_log(log)
        self.guard()
        if code:
            raise RuntimeError(label+' exited with code '+str(code))
        record['status'] = 'complete'
        record['seconds'] = self.now()-began
        self.save()

    def train_once(self):
        if self.data['training']['status'] != 'not_started':
            raise RuntimeError('Training was already attempted; no retry or resume')
        self.clock.require('training_begin_latest')
        self.request_streaming('release','train','training_begin_latest')
        self.clock.require('training_begin_latest')
        estimate = training_estimate(1500,self.design['budget']['planning_iteration_seconds'],
            self.design['budget']['planning_multiplier'],self.design['budget']['training_boot_seconds'])
        downstream=sum(self.design['budget'][key] for key in
            ('selection_seconds','validation_seconds','computation_margin_seconds'))
        if self.now()+estimate+downstream>self.clock.deadlines['compute_end']:
            raise PhaseTimeout('Full1500 ETA plus bounded selection/validation and300s margin exceeds compute deadline')
        if self.now()+estimate > self.clock.deadlines['training_end_latest']-PROCESS_CLEANUP_RESERVE_SECONDS:
            raise PhaseTimeout('Insufficient time for1500 with protected evaluation/retrieval reserve')
        self.guard()
        atomic_bytes(self.train/'quadruped_rewards.py',self.candidate)
        self.active_reward = self.candidate
        if (self.train/'quadruped_rewards.py').read_bytes() != self.candidate:
            raise RuntimeError('Applied named candidate reward bytes mismatch')
        self.data['training']['expected_weights'] = self.expected_weights
        command = [self.isaac,'-p','train.py','--task','Quadruped-v0','--num_envs','6144','--seed','42',
                   '--max_iterations','1500','--run_name',self.run_name,'--headless']
        self.capture=CheckpointCapture(self.train/'logs/rsl_rl/quadruped',self.run_name)
        self.capture.start()
        try:
            self.job(command,'train_'+self.arm,'training_end_latest',training=True)
        finally:
            self.capture.stop()
        runs = self.preserve()
        if len(runs) !=1:
            raise RuntimeError('Missing unique training run')
        self.data['training']['configuration'] = verify_config(runs[0],self.expected_weights)
        profile = training_profile((self.root/'logs'/('train_'+self.arm+'.log')).read_text(encoding='utf-8',errors='replace'))
        self.data['training']['training_time_profile'] = profile
        self.data['training']['original_final_checkpoint_present_after_provided_cleanup']=(runs[0]/'model_1499.pt').is_file()
        self.data['training']['iterations_complete']=profile['completed_iterations']==1500 and profile['exact_iteration_coverage'] and profile['samples']==1500
        if not self.data['training']['iterations_complete']:
            self.data['training']['status']='incomplete_iterations_no_retry'
            raise RuntimeError('Training log did not confirm exact1500 scratch iterations')
        try:
            final=self.checkpoint(1499)
        except CheckpointMissing as exc:
            self.data['training']['status']='iterations_complete_exact_checkpoint_unavailable'
            self.data['training']['checkpoint_error']=str(exc)
            raise RuntimeError('1500 iterations complete; exact preserved1499 unavailable: '+str(exc)) from exc
        except RuntimeError as exc:
            self.data['training']['status']='iterations_complete_exact_checkpoint_invalid'
            self.data['training']['checkpoint_error']=str(exc)
            raise
        self.data['training']['verified_final_checkpoint']=final
        self.data['training']['status']='complete_verified_preserved'
        self.restore()
        self.guard()
        self.save('TRAIN_COMPLETE')

    def checkpoint(self, iteration):
        paths = list((self.root/'preserved'/self.arm).glob('*/model_'+str(iteration)+'.pt'))
        if len(paths) !=1 or paths[0].stat().st_size <=0:
            raise CheckpointMissing('Require exact unique checkpoint model_'+str(iteration)+'.pt')
        receipt=verify_checkpoint(paths[0],iteration)
        return dict(name=self.arm+'_'+str(iteration),checkpoint=paths[0].relative_to(self.root).as_posix(),**receipt)

    def evaluator_command(self, manifest, out, phase, scenarios):
        cfg = self.design['evaluation'][phase]
        audit=self.guard()
        audit_path=self.root/'audit_records'/(phase+'_'+out.name+'_source_audit.json')
        if audit_path.exists():
            raise RuntimeError('Immutable per-job source audit already exists; no evaluation retry')
        atomic_json(audit_path,audit)
        return [self.isaac,'-p',str(self.root/'evaluation.py'),'--manifest',str(manifest),'--out',str(out),
                '--phase',phase,'--scenarios',','.join(scenarios),'--num_envs',str(cfg['num_envs']),
                '--seeds',','.join(map(str,cfg['reset_seeds'])),'--startup_seed',str(cfg['startup_seed']),
                '--terrain_seed',str(cfg['terrain_seed']),'--train-dir',str(self.train),'--task','Quadruped-v0',
                '--task-module','go2_task','--source-audit',str(audit_path),'--headless']

    def evaluate_phase(self, phase, models, deadline_key):
        self.restore()
        self.guard()
        manifest = self.root/(phase+'_manifest.json')
        # Existing screen policies precede training config; the frozen loader
        # contract and per-model/preflight receipts apply, never invented config.
        models = [dict(m,actor_observation_normalization=m.get('actor_observation_normalization',False),
                       policy_clip_actions=m.get('policy_clip_actions')) for m in models]
        atomic_json(manifest,models)
        phase_dir = self.root/'results'/phase
        cfg = self.design['evaluation'][phase]
        groups = [('flat',('G1','G2','G6')),('rough',('G3',)),('slope',('G4',)),('stairs',('G5',)),('dr',('G7',))]
        dirs = []
        for terrain, ids in groups:
            ids = [sid for sid in ids if sid in cfg['scenarios']]
            if not ids:
                continue
            out = phase_dir/terrain
            self.job(self.evaluator_command(manifest,out,phase,ids),phase+'_'+terrain,deadline_key)
            dirs.append(out)
        command = [self.isaac,'-p',str(self.root/'evaluation.py'),'--merge',*[str(p) for p in dirs],
                   '--out',str(phase_dir),'--phase',phase]
        self.job(command,phase+'_merge',deadline_key)
        complete = json.loads((phase_dir/'evaluation_complete.json').read_text(encoding='utf-8'))
        if complete.get('complete') is not True:
            raise RuntimeError('Evaluation merge reports incomplete '+phase)
        result = json.loads((phase_dir/'models.json').read_text(encoding='utf-8'))
        if set(result) != {m['name'] for m in models}:
            raise RuntimeError('Evaluation policy coverage mismatch')
        for model in models:
            if result[model['name']]['checkpoint_sha256'] != model['sha256']:
                raise RuntimeError('Evaluated checkpoint hash mismatch')
        return result

    def select(self):
        self.request_streaming('hold','selection','selection_end_latest')
        final = self.checkpoint(1499)
        chosen, fallback = final, None
        try:
            try:
                candidates = [self.checkpoint(1000),final]
            except CheckpointMissing as exc:
                raise PhaseTimeout('Selection checkpoint1000 absent; fallback1499: '+str(exc))
            result = self.evaluate_phase('selection',self.references+candidates,'selection_end_latest')
            cfg = self.design['evaluation']['selection']
            ranking = [(subset_rank(result[m['name']],cfg['scenarios'],cfg['reset_seeds'][0],m['iteration']),m) for m in candidates]
            chosen = min(ranking,key=lambda p:p[0])[1]
            self.data['selection_ranking'] = [{'rank':list(rank),'model':m} for rank,m in ranking]
        except PhaseTimeout as exc:
            fallback = str(exc)
            # job() has already stopped and verified its owned group.
            if self.process is not None and group_alive(self.process.pid):
                raise RuntimeError('Selection group remains alive; refusing validation')
            self.process = None
            for label,record in self.data['evaluations'].items():
                if label.startswith('selection_') and record.get('status')=='started':
                    record['status']='incomplete_timeout_no_retry'
        if sha(self.root/chosen['checkpoint']) !=chosen['sha256']:
            raise RuntimeError('Selected checkpoint changed before freezing')
        self.data['selected'] = dict(chosen,selection_fallback=fallback,frozen_epoch=self.now(),
                                     checkpoint_optimality_claim=fallback is None)
        atomic_json(self.root/'selected_checkpoint.json',self.data['selected'])
        self.save('CHECKPOINT_FROZEN')

    def validate(self):
        selected = self.data['selected']
        if sha(self.root/selected['checkpoint']) !=selected['sha256']:
            raise RuntimeError('Frozen selected policy hash changed')
        candidate = dict(selected,name='candidate')
        models = self.evaluate_phase('validation',self.references+[candidate],'compute_end')
        spec = importlib.util.spec_from_file_location('frozen_decision_rules',self.root/'decision_rules.py')
        rules = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rules)
        decision = rules.decide(models)
        atomic_json(self.root/'decision.json',decision)
        atomic_json(self.root/'summary.json',dict(models=models,decision=decision,selected=selected,
                    official_score_prediction=None,training_seeds=[42],complete_validation=True))
        self.data['compute_completed']=True
        # Keep the one-shot controller alive until its cleanup release ACK.
        # Publishing a terminal stage here would race its automatic shutdown.
        self.save('COMPUTE_FINISHED_CLEANUP_PENDING')

    def archive(self, critical=False):
        """Critical bundle precedes full data; atomic replacement and exact hashes."""
        exclusions = ('.tmp','.copying','.lock','.tar.gz')
        files = [p for p in sorted(self.root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                 and not p.name.endswith(exclusions) and p.name not in ('manifest.json','critical_manifest.json')]
        if critical:
            selected = self.data.get('selected') if self.data else None
            checkpoint = self.root/selected['checkpoint'] if selected else None
            if checkpoint is None:
                paths = list((self.root/'preserved'/self.arm).glob('*/model_*.pt'))
                checkpoint = max(paths,key=lambda p:int(p.stem.split('_')[-1])) if paths else None
            files = [p for p in files if (p.suffix=='.json' and 'results' not in p.parts) or p.name in ('env.yaml','agent.yaml')]
            if checkpoint and checkpoint not in files:
                files.append(checkpoint)
        manifest_name = 'critical_manifest.json' if critical else 'manifest.json'
        manifest = self.root/manifest_name
        validation_complete=bool(self.data and self.data.get('stage')=='COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION'
                                 and (self.root/'summary.json').is_file())
        for p in files:
            self.clock.require('dashboard_shutdown_request_latest')
        atomic_json(manifest,dict(created_epoch=self.now(),kind='critical_partial' if critical else 'full',
                     contains_complete_validation=validation_complete,local_retrieval_verified=False,
                     selected=self.data.get('selected') if self.data else None,
                     files={p.relative_to(self.root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p,self.clock)) for p in files}))
        suffix = '_critical.tar.gz' if critical else '_results.tar.gz'
        output = self.root.parent/(self.root.name+suffix)
        tmp = output.with_name(output.name+'.tmp')
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tf:
            for p in files+[manifest]:
                self.clock.require('dashboard_shutdown_request_latest')
                info = tf.gettarinfo(str(p),arcname=self.root.name+'/'+p.relative_to(self.root).as_posix())
                with p.open('rb') as stream:
                    tf.addfile(info,DeadlineReader(stream,self.clock,'dashboard_shutdown_request_latest'))
        tmp.replace(output)
        receipt = dict(path=str(output),bytes=output.stat().st_size,sha256=sha(output,self.clock),
                       kind='critical_partial' if critical else 'full',created_epoch=self.now(),
                       contains_complete_validation=validation_complete,local_retrieval_verified=False)
        atomic_json(output.with_name(output.name+'.sha256.json'),receipt)
        print('RETRIEVE '+json.dumps(receipt),flush=True)
        return receipt

    def execute(self):
        failed = None
        try:
            self.initialize()
            self.screen_existing()
            self.train_once()
            self.select()
            self.validate()
        except BaseException as exc:
            failed = exc
            if self.data:
                for record in [self.data['training']]+list(self.data['evaluations'].values()):
                    if record.get('status') == 'started':
                        record['status'] = 'interrupted_no_retry'
                self.data['error'] = repr(exc)
                self.save('ERROR_CLEANUP_PENDING')
            raise
        finally:
            old = {sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
            errors=[]
            try:
                if not self.owns_mutation:
                    if failed is not None:
                        raise failed
                    raise RuntimeError('No owned experiment state was initialized')
                for label,operation in [('stop',lambda:stop_process(self.process)),
                    ('checkpoint_capture_stop',lambda:self.capture.stop() if self.capture else None),
                    ('preserve',self.preserve),('restore',self.restore),('guard',self.guard),
                    ('streaming_release',lambda:self.request_streaming('release','cleanup','dashboard_shutdown_request_latest'))]:
                    try:
                        operation()
                    except Exception as exc:
                        errors.append(label+': '+repr(exc))
                if self.data:
                    self.data['cleanup'] = dict(complete=not errors,process_and_restore_complete=not errors,errors=list(errors),checked_epoch=self.now())
                    final_stage=('RECOVERY_FAILED' if errors else
                        ('COMPUTE_COMPLETE_RETRIEVE_AND_STOP_ALLOCATION' if failed is None and self.data.get('compute_completed') else
                         'STOPPED_PARTIAL_RETRIEVE_AND_STOP_ALLOCATION'))
                    self.save(final_stage)
                    for critical in (True,False):
                        try:
                            # Leave portal shutdown margin even if compression stalls.
                            if self.now() >= self.clock.deadlines['dashboard_shutdown_request_latest']:
                                raise PhaseTimeout('Archive skipped at portal shutdown request deadline')
                            receipt = self.archive(critical)
                            self.data['critical_archive' if critical else 'full_archive']=receipt
                        except Exception as exc:
                            errors.append(('critical archive' if critical else 'full archive') + ': '+repr(exc))
                    self.data['cleanup']['archive_errors']=errors
                    self.data['cleanup']['complete']=not errors
                    self.save('RECOVERY_FAILED' if errors else None)
            finally:
                for sig,handler in old.items():
                    signal.signal(sig,handler)
            if errors and failed is None:
                raise RuntimeError('; '.join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--train-dir',type=Path,default=Path('/workspace/training/quadruped'))
    parser.add_argument('--isaac',default='/workspace/IsaacLab/isaaclab.sh')
    parser.add_argument('--source-hashes',type=Path,required=True)
    parser.add_argument('--server-start-epoch',type=float,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--authorized-execution',action='store_true')
    args = parser.parse_args()
    if os.name != 'posix' or not args.authorized_execution:
        parser.error('Require authorized Linux execution; local Windows supports CPU tests only')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal_stop)
        Runner(args.root,args.train_dir,args.isaac,args.source_hashes,args.server_start_epoch,args.preflight).execute()


if __name__ == '__main__':
    main()
