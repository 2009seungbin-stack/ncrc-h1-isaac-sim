"""Build the two-arm execution package without editing the earlier approved run."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parents[1]
PREVIOUS = ANALYSIS / 'causal_sweep_20261008'
MODEL_ROOT = ANALYSIS / 'short_sweep_20261002/inputs/models'

for name in ('runner.py', 'tests_runner.py', 'batch_eval.py', 't_eval.py',
             't_eval_go2_core.py', 'source_snapshot.py', 'previous_source_hashes_20261002.json'):
    shutil.copy2(PREVIOUS / name, ROOT / name)
for name in ('C0_5500', 'F10_3000'):
    (ROOT / 'references').mkdir(exist_ok=True)
    shutil.copy2(MODEL_ROOT / (name + '.pt'), ROOT / 'references' / (name + '.pt'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

refs = [dict(name=name, checkpoint='references/' + name + '.pt',
             sha256=sha(ROOT / 'references' / (name + '.pt')),
             training_iterations=5500 if name == 'C0_5500' else 3000,
             training_seed=42, role='fixed_comparison_only')
        for name in ('C0_5500', 'F10_3000')]
(ROOT / 'reference_models.json').write_text(json.dumps(refs, indent=2) + '\n', encoding='utf-8')

text = (ROOT / 'runner.py').read_text(encoding='utf-8')
text = text.replace('Bounded Go2 causal comparison.', 'Bounded Go2 tracking-strength screen.')
text = text.replace("ARMS = ('control', 'torque', 'posture')", "ARMS = ('S1_tracking_pair', 'R1_tracking_double')\nITERATIONS = 1500\nALLOCATION_SECONDS = 6 * 3600\nRETRIEVAL_RESERVE = 45 * 60")
text = text.replace("INPUTS = ('runner.py', 'batch_eval.py', 't_eval.py', 't_eval_go2_core.py')", "INPUTS = ('runner.py', 'batch_eval.py', 't_eval.py', 't_eval_go2_core.py',\n          'reference_models.json', 'references/C0_5500.pt', 'references/F10_3000.pt')")
start = text.index('BASE = dict(')
end = text.index('GROUPS = ', start)
text = text[:start] + """BASE = dict(track_lin_vel_xy_exp=2.0, track_ang_vel_z_exp=1.5, feet_air_time=.01,
            lin_vel_z_l2=-3.0, ang_vel_xy_l2=-.08, action_rate_l2=-.01,
            dof_acc_l2=-2.5e-7, dof_torques_l2=-.0002,
            flat_orientation_l2=0.0, dof_pos_limits=0.0)
DELTAS = {'reference': {},
          'S1_tracking_pair': {'track_lin_vel_xy_exp': 3.0, 'track_ang_vel_z_exp': 2.25},
          'R1_tracking_double': {'track_lin_vel_xy_exp': 4.0, 'track_ang_vel_z_exp': 3.0}}
""" + text[end:]
text = text.replace('Prior measured 2000-iteration', 'Prior measured scratch')
text = text.replace('EVAL_RESERVE = 1800\nTRAIN_ESTIMATE = 9636\nTRAIN_CAP = 12000',
                    'EVAL_RESERVE = 2400\nHISTORICAL_SECONDS_PER_ITERATION = 4.818008522014856\nTRAIN_ESTIMATE = ITERATIONS * HISTORICAL_SECONDS_PER_ITERATION\nTRAIN_CAP = 9000\nTRAIN_BOOT_ALLOWANCE = 120')
text = text.replace('return dict(BASE, **dict(IMPLICIT, **DELTAS[arm]))', 'return arm_weights(arm)')
text = text.replace('hard = server_start + 10 * 3600', 'hard = server_start + ALLOCATION_SECONDS')
text = text.replace('if compute_deadline > hard - 2700:', 'if compute_deadline > hard - RETRIEVAL_RESERVE:')
text = text.replace('45 minutes before the 10-hour server limit', '45 minutes before the 6-hour allocation limit')
text = text.replace("('max_iterations','2000')", "('max_iterations',str(ITERATIONS))")
text = text.replace('self.hard_deadline = server_start + 36000', 'self.hard_deadline = server_start + ALLOCATION_SECONDS')
text = text.replace('arms=list(ARMS), iterations=2000,', 'arms=list(ARMS), iterations=ITERATIONS,')
text = text.replace('self.original = (self.train/\'quadruped_rewards.py\').read_bytes()\n            for arm',
                    "self.original = (self.train/'quadruped_rewards.py').read_bytes()\n            validate_original_rewards(self.original)\n            reference_models(self.root)\n            for arm")
text = text.replace("[patch_rewards(self.original, arm) for arm in ARMS]", "[patch_rewards(self.original, arm) for arm in DELTAS]")
text = text.replace("return 'causal_20261008_' + arm", "return 'tracking_20261009_' + arm")
text = text.replace("--max_iterations', '2000'", "--max_iterations', str(ITERATIONS)")
text = text.replace('model_1999.pt', 'model_1499.pt')
text = text.replace("models = [dict(name=arm, checkpoint=self.data['arms'][arm]['checkpoint'],", "models = reference_models(self.root) + [dict(name=arm, checkpoint=self.data['arms'][arm]['checkpoint'],")
text = text.replace("self.apply('control')", "self.apply('reference')")
text = text.replace("for arm in ARMS:\n            scenarios = {}", "for arm in [model['name'] for model in reference_models(self.root)] + list(ARMS):\n            scenarios = {}")
text = text.replace("control = result['models']['control']['scenarios']", "control = result['models']['C0_5500']['scenarios']")
text = text.replace("'survival_non_regression_vs_control'", "'survival_non_regression_vs_C0_5500'")
text = text.replace("'training_seeds': [42]}", "'training_seeds': [42],\n                  'comparison_limit': 'Fixed references have unequal training lengths; no causal claim or official-point prediction'}")
text = text.replace('remaining*TRAIN_ESTIMATE', 'remaining*self.training_estimate()')
text = text.replace("seconds=seconds, configuration=configuration)", "seconds=seconds, configuration=configuration,\n                                     training_time_profile=training_profile(self.root/'logs'/('train_'+arm+'.log')))" )
text = text.replace("checked = False\n        with", "checked = False\n        profile_checked = 0\n        with")
text = text.replace("                    time.sleep(1)", """                    if arm and self.now()-profile_checked >= 30:
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
                    time.sleep(1)""")
marker = '\ndef validate_deadlines('
index = text.index(marker)
text = text[:index] + '''
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
    text = re.sub(r'\\x1b\\[[0-9;]*m', '', text)
    iterations = re.findall(r'Learning iteration (\\d+)/(\\d+)', text)
    durations = [float(x) for x in re.findall(r'Iteration time:\\s+([0-9.]+)s', text)]
    recent = durations[-50:]
    return dict(samples=len(durations),
                completed_iterations=int(iterations[-1][0])+1 if iterations else 0,
                recent_seconds_per_iteration=sum(recent)/len(recent) if recent else HISTORICAL_SECONDS_PER_ITERATION)

''' + text[index:]
marker = '    def train_arm(self, arm):'
index = text.index(marker)
text = text[:index] + '''    def training_estimate(self):
        completed = [arm for arm in self.data['arms'].values() if arm.get('status') == 'complete']
        if not completed:
            return TRAIN_ESTIMATE
        # Include measured startup/teardown and a 10% throughput margin for the
        # next arm; no simultaneous training/evaluation processes are launched.
        observed = max(arm['training_time_profile']['recent_seconds_per_iteration'] for arm in completed)
        return max(TRAIN_ESTIMATE, ITERATIONS*observed*1.1 + TRAIN_BOOT_ALLOWANCE)

''' + text[index:]
(ROOT / 'runner.py').write_text(text, encoding='utf-8', newline='\n')

# Adapt the pre-existing contract tests, then append additional allocation,
# fixed-reference and observed-throughput checks separately.
tests = (ROOT / 'tests_runner.py').read_text(encoding='utf-8')
tests = tests.replace("'control'", "'S1_tracking_pair'").replace("'torque'", "'R1_tracking_double'").replace("'posture'", "'R1_tracking_double'")
start = tests.index('    def test_each_intervention_changes_exactly_one_effective_weight')
end = tests.index('    def test_deadline_rejects_nonfinite', start)
tests = tests[:start] + '''    def test_tracking_strengths_keep_ratio_and_other_weights(self):
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

''' + tests[end:]
tests = tests.replace("self.original = '# 원문\\nREWARD_WEIGHTS = {\"old\": 1}\\n'.encode()", "self.original = ('# 원문\\nREWARD_WEIGHTS = '+repr(r.BASE)+'\\n').encode()")
tests = tests.replace('self.deadline = self.start+33300', 'self.deadline = self.start+18900')
tests = tests.replace("        self.start = time.time()-30", """        refs = []
        for name in ('C0_5500', 'F10_3000'):
            path = self.root/('references/'+name+'.pt')
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'fixed checkpoint '+name.encode())
            refs.append(dict(name=name, checkpoint='references/'+name+'.pt', sha256=r.sha(path)))
        (self.root/'reference_models.json').write_text(json.dumps(refs), encoding='utf-8')
        self.start = time.time()-30""")
tests = tests.replace("            (self.root/name).write_text", "            (self.root/name).parent.mkdir(parents=True, exist_ok=True)\n            (self.root/name).write_text")
tests = tests.replace("preserved/control", "preserved/S1_tracking_pair")
tests = tests.replace('model_1999.pt', 'model_1499.pt')
(ROOT / 'tests_runner.py').write_text(tests, encoding='utf-8', newline='\n')
print('Prepared runner, references, unchanged evaluators and adapted tests:', ROOT)
