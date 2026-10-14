"""Bounded Go2 tracking-strength screen. Analysis only; never submit or shut down a PC.

Run on Linux with the server's *actual* start time, not the runner start time.
The operator must retrieve the archive and stop the server before hard_deadline.
SIGINT/SIGTERM are handled; SIGKILL/power loss cannot run Python cleanup. On the
next invocation, a recognized applied reward file is restored before resuming.
An interrupted training arm is NEVER retrained, even when --resume is used.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import time
import traceback

ARMS = ('S1_tracking_pair', 'R1_tracking_double')
ITERATIONS = 1500
ALLOCATION_SECONDS = 6 * 3600
RETRIEVAL_RESERVE = 45 * 60
PROTECTED = ('train.py', 'play.py', 'go2_task/agent_cfg.py', 'go2_task/env_cfg.py')
INPUTS = ('runner.py', 'batch_eval.py', 't_eval.py', 't_eval_go2_core.py',
          'source_snapshot.py', 'reference_models.json', 'references/C0_5500.pt',
          'references/F10_3000.pt', 'protected_runtime_sources.json')
BASE = dict(track_lin_vel_xy_exp=2.0, track_ang_vel_z_exp=1.5, feet_air_time=.01,
            lin_vel_z_l2=-3.0, ang_vel_xy_l2=-.08, action_rate_l2=-.01,
            dof_acc_l2=-2.5e-7, dof_torques_l2=-.0002,
            flat_orientation_l2=0.0, dof_pos_limits=0.0)
DELTAS = {'reference': {},
          'S1_tracking_pair': {'track_lin_vel_xy_exp': 3.0, 'track_ang_vel_z_exp': 2.25},
          'R1_tracking_double': {'track_lin_vel_xy_exp': 4.0, 'track_ang_vel_z_exp': 3.0}}
GROUPS = (('flat', 'G1,G2,G6', 1800), ('rough', 'G3', 900),
          ('slope', 'G4', 900), ('stairs', 'G5', 900), ('dr', 'G7', 900))
# Prior measured scratch training estimate is 4.818 s/iteration.
# Per-process emergency caps are separate from the total reserved evaluation time.
EVAL_RESERVE = 2400
HISTORICAL_SECONDS_PER_ITERATION = 4.818008522014856
TRAIN_ESTIMATE = ITERATIONS * HISTORICAL_SECONDS_PER_ITERATION
TRAIN_CAP = 9000
TRAIN_BOOT_ALLOWANCE = 120
WEIGHTS = dict(zip(('G1','G2','G3','G4','G5','G6','G7'), (.15,.15,.2,.15,.15,.1,.1)))


class StopRequested(BaseException):
    pass


def signal_stop(signum, _frame):
    raise StopRequested('Signal ' + str(signum))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))


def arm_weights(arm):
    return dict(BASE, **DELTAS[arm])


def expected_weights(arm):
    return arm_weights(arm)


def patch_rewards(original, arm):
    """Replace exactly the assignment value, preserving all other source bytes."""
    tree = ast.parse(original.decode('utf-8'))
    nodes = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in n.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Dict):
        raise ValueError('Require exactly one REWARD_WEIGHTS dictionary assignment')
    node = nodes[0]
    lines = original.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    result = original[:start] + repr(arm_weights(arm)).encode('utf-8') + original[end:]
    ast.parse(result.decode('utf-8'))
    return result


def validate_original_rewards(original):
    tree = ast.parse(original.decode('utf-8'))
    nodes = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'REWARD_WEIGHTS' for t in n.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Dict):
        raise ValueError('Require one literal REWARD_WEIGHTS dictionary')
    value = ast.literal_eval(nodes[0])
    if not value or not set(value) <= set(BASE):
        raise ValueError('Fresh reward dictionary has unknown or no allowed terms')
    for key, weight in value.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError('Fresh reward weights must be finite numbers: '+key)
        if (key.startswith('track_') and weight <= 0) or (key.endswith('_l2') and weight > 0):
            raise ValueError('Unexpected reward sign in fresh source: '+key)
    return value


def reference_models(root):
    root = Path(root)
    models = json.loads((root/'reference_models.json').read_text(encoding='utf-8'))
    if [m['name'] for m in models] != ['C0_5500', 'F10_3000']:
        raise ValueError('Require exact fixed C0_5500 and F10_3000 comparison models')
    for model in models:
        path = Path(model['checkpoint'])
        if path.is_absolute() or '..' in path.parts or sha(root/path) != model['sha256']:
            raise RuntimeError('Fixed reference checkpoint path or hash mismatch: '+model['name'])
    return models


def training_profile(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace') if Path(path).exists() else ''
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    iterations = re.findall(r'Learning iteration (\d+)/(\d+)', text)
    durations = [float(x) for x in re.findall(r'Iteration time:\s+([0-9.]+)s', text)]
    recent = durations[-50:]
    return dict(samples=len(durations),
                completed_iterations=min(len(durations), int(iterations[-1][0])+1) if iterations else 0,
                recent_seconds_per_iteration=sum(recent)/len(recent) if recent else HISTORICAL_SECONDS_PER_ITERATION)


def validate_deadlines(server_start, compute_deadline, now=None):
    now = time.time() if now is None else now
    if not all(math.isfinite(x) for x in (server_start, compute_deadline, now)):
        raise ValueError('Deadlines must be finite Unix timestamps')
    hard = server_start + ALLOCATION_SECONDS
    if server_start > now + 5:
        raise ValueError('Server start cannot be in the future')
    if compute_deadline > hard - RETRIEVAL_RESERVE:
        raise ValueError('Reserve at least 45 minutes before the 6-hour allocation limit')
    if compute_deadline <= now:
        raise ValueError('Compute deadline has passed')
    return hard


def yaml_reward_weights(text):
    """Read scalar weights without constructing Python objects from Isaac YAML."""
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


def verify_config(run, arm):
    env = (run / 'params/env.yaml').read_text(encoding='utf-8')
    agent = (run / 'params/agent.yaml').read_text(encoding='utf-8')
    actual = yaml_reward_weights(env)
    for key, expected in expected_weights(arm).items():
        if actual.get(key) != expected:
            raise RuntimeError(f'{arm}: effective reward {key}: {actual.get(key)} != {expected}')
    for key, value in [('seed','42'), ('max_iterations',str(ITERATIONS)), ('resume','false')]:
        if not re.search(r'^' + key + ': ' + value + r'\s*$', agent, re.M):
            raise RuntimeError(f'{arm}: agent config did not confirm {key}={value}')
    if not re.search(r'^  num_envs: 6144\s*$', env, re.M):
        raise RuntimeError(f'{arm}: environment config did not confirm 6144 environments')
    return {'effective_weights': actual, 'env_sha256': sha(run/'params/env.yaml'),
            'agent_sha256': sha(run/'params/agent.yaml')}


def stop_process(process):
    if process is None:
        return
    def group_alive():
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False
    # The isaaclab.sh leader can exit before its Python/GPU descendants. Group
    # liveness, not the wrapper's return code, governs cleanup and escalation.
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        process.poll()  # Reap an exited leader without assuming its group died.
        if not group_alive():
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            process.poll()
            if not group_alive():
                return
            time.sleep(.1)
    process.poll()
    if not group_alive():
        return
    raise RuntimeError('Child process group did not exit after SIGKILL')


class Runner:
    def __init__(self, root, train, isaac, source_hashes, server_start, compute_deadline, resume=False):
        self.root, self.train = Path(root).resolve(), Path(train).resolve()
        self.isaac = str(isaac)
        self.expected = json.loads(Path(source_hashes).read_text(encoding='utf-8'))
        if not all(k in self.expected for k in PROTECTED):
            raise ValueError('Source hash file must include all four protected sources')
        for key, value in self.expected.items():
            if Path(key).is_absolute() or '..' in Path(key).parts or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Invalid protected source manifest entry')
        self.runtime_manifest = json.loads((self.root/'protected_runtime_sources.json').read_text(encoding='utf-8'))
        self.expected_runtime = self.runtime_manifest['files']
        if not self.expected_runtime or self.runtime_manifest.get('algorithm_package') != 'rsl_rl':
            raise ValueError('Require read-only rsl_rl source snapshot before launching')
        for key, value in self.expected_runtime.items():
            if not Path(key).is_absolute() or not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Invalid protected runtime source manifest entry')
        self.server_start, self.deadline = server_start, compute_deadline
        self.hard_deadline = server_start + ALLOCATION_SECONDS
        self.resume = resume
        self.data = None
        self.original = None
        self.process = None
        self.started_monotonic = time.monotonic()
        self.started_wall = time.time()

    def now(self):
        # A backward wall-clock correction must not extend the compute budget.
        return max(time.time(), self.started_wall + time.monotonic() - self.started_monotonic)

    def save(self, stage=None):
        if stage:
            self.data['stage'] = stage
        self.data['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(self.root/'state.json', self.data)
        print(json.dumps({'stage': self.data['stage'], 'utc': self.data['updated_utc']}), flush=True)

    def guard(self):
        current = {k: sha(self.train/k) for k in self.expected}
        if current != self.expected:
            raise RuntimeError('Protected training source hash mismatch; execution refused')
        if {k: sha(k) for k in self.expected_runtime} != self.expected_runtime:
            raise RuntimeError('Protected PPO/runtime source hash mismatch; execution refused')
        if self.data and {k: sha(self.root/k) for k in INPUTS} != self.data['input_hashes']:
            raise RuntimeError('Analysis evaluator/runner source changed during this experiment')

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard()
        state_path, backup = self.root/'state.json', self.root/'fresh_rewards_original.py'
        contract = dict(server_start_epoch=self.server_start, compute_deadline=self.deadline,
                        hard_deadline=self.hard_deadline, arms=list(ARMS), iterations=ITERATIONS,
                        train_envs=6144, train_seed=42, eval_envs=128, reset_seeds=[809,1201],
                        terrain_seed=9049, startup_seed=303, source_hashes=self.expected,
                        runtime_source_hashes=self.expected_runtime)
        if state_path.exists():
            if not self.resume:
                raise RuntimeError('Existing state requires --resume; no training will be repeated')
            self.data = json.loads(state_path.read_text(encoding='utf-8'))
            if self.data['contract'] != contract:
                raise RuntimeError('Resume contract differs from the original fixed experiment')
            for entry in list(self.data['arms'].values()) + list(self.data['evaluations'].values()):
                if entry.get('status') == 'started' and entry.get('pid'):
                    try:
                        os.killpg(entry['pid'], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError('Prior process group is still active; stop it before recovery/resume')
            self.original = backup.read_bytes()
            if sha(backup) != self.data['original_reward_sha256']:
                raise RuntimeError('Original reward backup hash mismatch')
            self.guard()
            self.restore_rewards()
        else:
            validate_deadlines(self.server_start, self.deadline, self.now())
            if self.resume:
                raise RuntimeError('--resume requires existing experiment state')
            if backup.exists():
                raise RuntimeError('Orphan reward backup requires review before a new experiment')
            for arm in ARMS:
                if self.runs(arm) or (self.root/'preserved'/arm).exists():
                    raise RuntimeError('Existing arm output without state; refuse duplicate training')
            self.original = (self.train/'quadruped_rewards.py').read_bytes()
            validate_original_rewards(self.original)
            reference_models(self.root)
            for arm in ARMS:
                patch_rewards(self.original, arm)
            atomic_bytes(backup, self.original)
            self.data = dict(version=1, contract=contract, stage='PREPARED',
                             original_reward_sha256=sha(backup),
                             input_hashes={k: sha(self.root/k) for k in INPUTS},
                             arms={}, evaluations={}, warnings=[
                                 'Internal comparison only; no official score or submission',
                                 'One training seed; reset seeds are not independent training runs',
                                 'Unchanged batch_eval has known one-step push-observation timing issue',
                                 'Prior reset seeds informed proposal; this is not unseen held-out evaluation'])
            self.save()
        # Recovery above is allowed even after the compute deadline has expired.
        validate_deadlines(self.server_start, self.deadline, self.now())
        (self.root/'logs').mkdir(exist_ok=True)

    def restore_rewards(self):
        current = (self.train/'quadruped_rewards.py').read_bytes()
        known = [self.original] + [patch_rewards(self.original, arm) for arm in DELTAS]
        if current not in known:
            raise RuntimeError('Unrecognized external reward edit; refuse to overwrite it')
        atomic_bytes(self.train/'quadruped_rewards.py', self.original)
        atomic_json(self.root/'reward_restored.json', {'restored': True, 'sha256': sha(self.train/'quadruped_rewards.py')})

    def apply(self, arm):
        self.guard()
        self.restore_rewards()
        applied = patch_rewards(self.original, arm)
        atomic_bytes(self.train/'quadruped_rewards.py', applied)
        atomic_bytes(self.root/(arm+'_applied_rewards.py'), applied)

    def run_name(self, arm):
        return 'tracking_20261009_' + arm

    def runs(self, arm):
        return list((self.train/'logs/rsl_rl/quadruped').glob('*_' + self.run_name(arm)))

    def preserve(self, arm):
        runs = self.runs(arm)
        if len(runs) > 1:
            raise RuntimeError('Multiple training runs for the same arm; refuse ambiguous evidence')
        for run in runs:
            dest = self.root/'preserved'/arm/run.name
            dest.mkdir(parents=True, exist_ok=True)
            for source in run.glob('model_*.pt'):
                target = dest/source.name
                try:
                    before = (source.stat().st_size, source.stat().st_mtime_ns)
                    if target.exists() and (target.stat().st_size, target.stat().st_mtime_ns) == before:
                        continue
                    tmp = target.with_name(target.name+'.copying')
                    shutil.copy2(source, tmp)
                    if before == (source.stat().st_size, source.stat().st_mtime_ns):
                        tmp.replace(target)
                except FileNotFoundError:
                    continue
            for source in list(run.glob('events.out.tfevents.*')) + list((run/'params').glob('*')):
                if source.is_file():
                    relative = source.relative_to(run)
                    target = dest/relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name+'.copying')
                    try:
                        shutil.copy2(source, tmp)
                        tmp.replace(target)
                    except FileNotFoundError:
                        pass
        return runs

    def job(self, command, label, cap, end, arm=None):
        self.guard()
        end = min(end, self.deadline)
        if self.now() >= end:
            raise RuntimeError('No compute time remains for '+label)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.train) + os.pathsep + env.get('PYTHONPATH', '')
        began = self.now()
        checked = False
        profile_checked = 0
        with (self.root/'logs'/(label+'.log')).open('ab') as stream:
            self.process = subprocess.Popen(command, cwd=self.train, env=env, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            if arm:
                self.data['arms'][arm]['pid'] = self.process.pid
            else:
                self.data['evaluations'][label]['pid'] = self.process.pid
            self.save('RUNNING_'+label)
            try:
                while self.process.poll() is None:
                    if self.now() >= end or self.now()-began >= cap:
                        raise RuntimeError('Compute deadline or per-job cap reached: '+label)
                    if arm:
                        runs = self.preserve(arm)
                        if runs and (runs[0]/'params/agent.yaml').exists() and (runs[0]/'params/env.yaml').exists():
                            if not checked:
                                self.data['arms'][arm]['configuration'] = verify_config(runs[0], arm)
                                self.save()
                                checked = True
                        if runs and list(runs[0].glob('model_*.pt')) and not checked:
                            raise RuntimeError('Checkpoint exists before effective configuration was verified')
                    if arm and self.now()-profile_checked >= 30:
                        profile = training_profile(self.root/'logs'/(label+'.log'))
                        if profile['samples'] >= 50:
                            self.data['arms'][arm]['training_time_profile'] = profile
                            # Admission never changes the approved iteration count. If the
                            # current job cannot finish while protecting evaluation, stop.
                            projected = (ITERATIONS-profile['completed_iterations']) * profile['recent_seconds_per_iteration'] * 1.1
                            if self.now()+projected > end:
                                raise RuntimeError('Observed training speed cannot finish before evaluation reserve: '+arm)
                            self.save()
                        profile_checked = self.now()
                    time.sleep(1)
            finally:
                stop_process(self.process)
                if arm:
                    self.preserve(arm)
        code = self.process.returncode
        self.process = None
        if code:
            raise RuntimeError(label+' exited with code '+str(code))
        self.guard()
        return self.now()-began

    def training_estimate(self):
        completed = [arm for arm in self.data['arms'].values() if arm.get('status') == 'complete']
        if not completed:
            return TRAIN_ESTIMATE
        # Include measured startup/teardown and a 10% throughput margin for the
        # next arm; no simultaneous training/evaluation processes are launched.
        observed = max(arm['training_time_profile']['recent_seconds_per_iteration'] for arm in completed)
        return max(TRAIN_ESTIMATE, ITERATIONS*observed*1.1 + TRAIN_BOOT_ALLOWANCE)

    def train_arm(self, arm):
        old = self.data['arms'].get(arm)
        if old:
            if old['status'] == 'complete':
                path = self.root/old['checkpoint']
                if sha(path) != old['sha256']:
                    raise RuntimeError('Completed checkpoint hash mismatch: '+arm)
                return
            raise RuntimeError('Training already attempted; never retrain on resume: '+arm)
        remaining = sum(candidate not in self.data['arms'] for candidate in ARMS)
        if self.now() + remaining*self.training_estimate() + EVAL_RESERVE > self.deadline:
            raise RuntimeError('Remaining time is below the measured training estimate plus evaluation reserve')
        self.apply(arm)
        self.data['arms'][arm] = dict(status='started', run_name=self.run_name(arm),
                                      expected_weights=expected_weights(arm), started_epoch=self.now())
        self.save('TRAIN_'+arm)
        command = [self.isaac, '-p', 'train.py', '--task', 'Quadruped-v0', '--num_envs', '6144',
                   '--seed', '42', '--max_iterations', str(ITERATIONS), '--run_name', self.run_name(arm), '--headless']
        seconds = self.job(command, 'train_'+arm, TRAIN_CAP, self.deadline-EVAL_RESERVE, arm)
        runs = self.preserve(arm)
        if len(runs) != 1:
            raise RuntimeError('Missing unique training output: '+arm)
        configuration = verify_config(runs[0], arm)
        files = list((self.root/'preserved'/arm).glob('*/model_1499.pt'))
        if len(files) != 1:
            raise RuntimeError('Require exact final model_1499.pt: '+arm)
        checkpoint = files[0]
        self.data['arms'][arm].update(status='complete', checkpoint=checkpoint.relative_to(self.root).as_posix(),
                                     sha256=sha(checkpoint), seconds=seconds, configuration=configuration,
                                     training_time_profile=training_profile(self.root/'logs'/('train_'+arm+'.log')))
        self.restore_rewards()
        self.save('TRAIN_COMPLETE_'+arm)
        self.bundle()

    def eval_complete(self, label, scenarios, models):
        for model in models:
            path = self.root/'results'/label/(model['name']+'.json')
            if not path.exists():
                return False
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['checkpoint_sha256'] != model['sha256'] or set(result['scenarios']) != set(scenarios.split(',')):
                return False
            if (result['num_envs'], result['startup_seed'], result['terrain_seed']) != (128,303,9049):
                return False
            if any([row['seed'] for row in rows] != [809,1201] for rows in result['scenarios'].values()):
                return False
        return True

    def evaluate(self):
        models = reference_models(self.root) + [dict(name=arm, checkpoint=self.data['arms'][arm]['checkpoint'],
                       sha256=self.data['arms'][arm]['sha256']) for arm in ARMS]
        atomic_json(self.root/'models.json', models)
        self.apply('reference')
        for terrain, scenarios, cap in GROUPS:
            label = 'common_'+terrain
            old = self.data['evaluations'].get(label)
            if old and self.eval_complete(label, scenarios, models):
                self.data['evaluations'][label]['status'] = 'complete'
                self.save()
                continue
            if old:
                raise RuntimeError('Incomplete evaluation already attempted; retain evidence and review: '+label)
            self.data['evaluations'][label] = dict(status='started', scenarios=scenarios, started_epoch=self.now())
            self.save('EVAL_'+label)
            command = [self.isaac, '-p', str(self.root/'batch_eval.py'), '--task', 'Quadruped-v0',
                       '--task_module', 'go2_task', '--train_script', str(self.train/'train.py'),
                       '--models_json', str(self.root/'models.json'), '--output_dir', str(self.root/'results'/label),
                       '--num_envs', '128', '--seeds', '809,1201', '--scenarios', scenarios,
                       '--startup_seed', '303', '--terrain_seed', '9049', '--transient_s', '0.5', '--headless']
            seconds = self.job(command, label, cap, self.deadline)
            if not self.eval_complete(label, scenarios, models):
                raise RuntimeError('Incomplete common evaluation: '+label)
            self.data['evaluations'][label].update(status='complete', seconds=seconds)
            self.save('EVAL_COMPLETE_'+label)
            self.bundle()

    def summarize(self):
        result = {'models': {}, 'warning': 'Internal full-window comparison; not official points',
                  'known_evaluator_issue': 'Unchanged one-step push-observation timing',
                  'training_seeds': [42],
                  'comparison_limit': 'Fixed references have unequal training lengths; no causal claim or official-point prediction'}
        for arm in [model['name'] for model in reference_models(self.root)] + list(ARMS):
            scenarios = {}
            for terrain, ids, _ in GROUPS:
                raw = json.loads((self.root/'results'/('common_'+terrain)/(arm+'.json')).read_text(encoding='utf-8'))
                for sid, rows in raw['scenarios'].items():
                    count = sum(row['full']['count'] for row in rows)
                    mse = [sum(row['full']['sum_squared'][axis] for row in rows)/count for axis in range(3)]
                    episodes, survived = (sum(row[key] for row in rows) for key in ('episodes','survived'))
                    tracking = math.exp(-math.sqrt(sum(mse)/3))
                    scenarios[sid] = dict(episodes=episodes, survived=survived, survival=survived/episodes,
                                          rmse=dict(zip(('vx','vy','yaw'), map(math.sqrt,mse))),
                                          tracking=tracking, score=survived/episodes*tracking)
            if set(scenarios) != set(WEIGHTS):
                raise RuntimeError('Require all G1-G7 for every model')
            result['models'][arm] = dict(scenarios=scenarios,
                weighted_quality=sum(WEIGHTS[sid]*row['score'] for sid,row in scenarios.items()),
                falls=sum(row['episodes']-row['survived'] for row in scenarios.values()))
        control = result['models']['C0_5500']['scenarios']
        for arm in ARMS:
            result['models'][arm]['survival_non_regression_vs_C0_5500'] = all(
                row['survival'] >= control[sid]['survival'] for sid,row in result['models'][arm]['scenarios'].items())
        atomic_json(self.root/'summary.json', result)

    def bundle(self):
        """Atomic replacement leaves the previous complete archive on failure."""
        files = [p for p in sorted(self.root.rglob('*')) if p.is_file()
                 and '__pycache__' not in p.parts and p.name != 'manifest.json'
                 and not p.name.endswith(('.tmp','.copying','.lock'))]
        atomic_json(self.root/'manifest.json', {'created_epoch': time.time(), 'files': {
            p.relative_to(self.root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}})
        archive = self.root.parent/(self.root.name+'_results.tar.gz')
        tmp = archive.with_name(archive.name+'.tmp')
        with tarfile.open(tmp, 'w:gz', compresslevel=1) as tf:
            for p in files + [self.root/'manifest.json']:
                tf.add(p, arcname=self.root.name+'/'+p.relative_to(self.root).as_posix())
        tmp.replace(archive)
        atomic_json(archive.with_name(archive.name+'.sha256.json'), dict(path=str(archive),
                    bytes=archive.stat().st_size, sha256=sha(archive)))
        print('RETRIEVE', archive, flush=True)

    def execute(self):
        failure = None
        try:
            self.initialize()
            for arm in ARMS:
                self.train_arm(arm)
            self.evaluate()
            self.summarize()
            self.save('COMPLETE_RETRIEVE_AND_STOP_SERVER')
        except BaseException as exc:
            failure = exc
            if self.data:
                self.data['error'] = str(exc)
                self.save('STOPPED_REQUIRES_REVIEW')
            raise
        finally:
            # Repeated termination signals must not interrupt reward restoration.
            old = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
            errors = []
            try:
                try:
                    stop_process(self.process)
                except Exception as exc:
                    errors.append('stop: '+str(exc))
                for arm in ARMS:
                    try:
                        self.preserve(arm)
                    except Exception as exc:
                        errors.append('preserve '+arm+': '+str(exc))
                if self.original is not None:
                    try:
                        self.restore_rewards()
                        self.guard()
                    except Exception as exc:
                        errors.append('restore/guard: '+str(exc))
                if self.data:
                    self.data['cleanup_errors'] = errors
                    self.save('RECOVERY_FAILED' if errors else None)
                    try:
                        self.bundle()
                    except Exception as exc:
                        errors.append('archive: '+str(exc))
                        self.data['cleanup_errors'] = errors
                        self.save('RECOVERY_FAILED')
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
            if errors and failure is None:
                raise RuntimeError('; '.join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--train-dir', type=Path, default=Path('/workspace/training/quadruped'))
    parser.add_argument('--isaac', default='/workspace/IsaacLab/isaaclab.sh')
    parser.add_argument('--source-hashes', required=True, type=Path)
    parser.add_argument('--server-start-epoch', type=float, required=True)
    parser.add_argument('--compute-deadline', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Execution requires the Linux server; local Windows supports unit tests only')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal_stop)
        runner = Runner(args.root, args.train_dir, args.isaac, args.source_hashes,
                        args.server_start_epoch, args.compute_deadline, args.resume)
        runner.execute()


if __name__ == '__main__':
    main()
