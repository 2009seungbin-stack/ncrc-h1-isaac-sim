"""Read-only historical log analysis; writes only historical* analysis artifacts.

No policy load, simulation, network, training or evaluation execution.
"""
from pathlib import Path
import hashlib
import difflib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'analysis/baseline_redesign_20261009_4h'
REC = BASE / 'recovered_full_20261009_1427/b1_baseline_20261009_4h'
ROUND = ROOT / 'analysis/round20_execution_20261001/results/extracted/round20_20261001'
SOURCES = {
    'B1': (REC / 'logs/train_B1.log', REC / 'preserved/B1_selective_cost/2026-10-09_21-12-00_B1_4h_1791547031/params/env.yaml'),
    'F10_historical_training': (ROUND / 'logs/train_F10.log', ROUND / 'preserved/F10/2026-10-01_18-55-04_round20_F10/params/env.yaml'),
    'F00_C0_reward_historical_training': (ROUND / 'logs/train_F00.log', ROUND / 'preserved/F00/2026-10-01_13-44-14_round20_F00/params/env.yaml'),
}
METRICS = [
    'Episode_Reward/track_lin_vel_xy_exp', 'Episode_Reward/track_ang_vel_z_exp',
    'Metrics/base_velocity/error_vel_xy', 'Metrics/base_velocity/error_vel_yaw',
    'Episode_Termination/base_contact', 'Episode_Termination/time_out',
    'Mean episode length', 'Mean action std', 'Curriculum/terrain_levels',
    'Episode_Reward/dof_torques_l2', 'Episode_Reward/dof_acc_l2',
    'Episode_Reward/action_rate_l2', 'Episode_Reward/lin_vel_z_l2',
]
WINDOWS = [(500, 599), (700, 799), (900, 999), (1000, 1099), (1200, 1299), (1400, 1499), (1490, 1499)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_weights(path):
    active = False
    term = None
    values = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line == 'rewards:':
            active = True
            continue
        if active and line and not line[0].isspace():
            break
        if not active:
            continue
        m = re.fullmatch(r'  ([\w]+):(?: null)?', line)
        if m:
            term = m[1]
        m = re.fullmatch(r'    weight: (\S+)', line)
        if m and term:
            values[term] = float(m[1])
    return values

def parse_log(path):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', path.read_text(encoding='utf-8', errors='replace'))
    records = {}
    current = None
    for line in text.splitlines():
        m = re.search(r'Learning iteration\s+(\d+)/(\d+)', line)
        if m:
            current = int(m[1])
            if current in records:
                raise ValueError(f'Duplicate iteration {current}: {path}')
            records[current] = {}
            continue
        if current is not None:
            m = re.fullmatch(r'\s*([^:]+):\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*', line, re.I)
            if m and m[1].strip() in METRICS:
                records[current][m[1].strip()] = float(m[2])
    if sorted(records) != list(range(len(records))):
        raise ValueError(f'Noncontiguous iteration records: {path}')
    return records

def summarize(records, weights, start, end):
    rows = [records[i] for i in range(start, end + 1)]
    result = {}
    for metric in METRICS:
        vals = [r[metric] for r in rows]
        result[metric] = {'mean': statistics.mean(vals), 'median': statistics.median(vals), 'first': vals[0], 'last': vals[-1], 'count': len(vals)}
    for axis, term in [('xy', 'track_lin_vel_xy_exp'), ('yaw', 'track_ang_vel_z_exp')]:
        result['weight_normalized_episode_reward_' + axis] = {
            'mean': result['Episode_Reward/' + term]['mean'] / weights[term],
            'median': result['Episode_Reward/' + term]['median'] / weights[term],
            'definition': 'logged Episode_Reward divided by effective reward weight; NOT heldout tracking quality or RMSE',
        }
    return result

analysis = {
    'status': 'LOCAL_HISTORICAL_TRAINING_DIAGNOSTIC_NOT_POLICY_EVALUATION',
    'no_policy_loaded': True, 'no_simulation_or_server': True,
    'iteration_windows_inclusive': WINDOWS, 'models': {},
    'comparability_limits': [
        'F00 is a separate historical scratch run with C0 reward settings, not the official C0_5500 training trajectory.',
        'Same iteration windows and seed42 do not equate endogenous terrain/command/termination distributions.',
        'No independent training seeds; successive logged iterations are not independent experiments.',
        'Historical runtime equality with B1 is not established by log-window matching.',
        'Weight-normalized reward retains episode-duration/reset/aggregation effects and exploration; it is not deterministic-policy tracking.',
        'Portal S1/R1 reports expose final terrain/std and rounded last10 training fall rate, but no normalized tracking trace.',
        'B1 has zero heldout scenario evaluations; existing official calibration cannot accept its training metrics as replacement inputs.',
    ],
    'B1_policy_internal_Q3': None, 'B1_official_score_prediction': None,
}
for name, (log, env) in SOURCES.items():
    records = parse_log(log)
    weights = parse_weights(env)
    analysis['models'][name] = {
        'log_path': str(log), 'log_sha256': sha(log),
        'env_path': str(env), 'env_sha256': sha(env), 'effective_weights': weights,
        'iterations': len(records),
        'windows': {f'{start}-{end}': summarize(records, weights, start, end) for start, end in WINDOWS},
    }

analysis['config_comparison'] = {
    'all_three_effective_configs': {'seed': 42, 'num_envs': 6144, 'num_steps_per_env': 24, 'resume': False},
    'historical_agent_vs_B1': 'Only max_iterations (4000 versus 1500) and run_name differ in saved agent.yaml.',
    'historical_env_vs_B1': 'Only reward weights and log_dir differ in saved env.yaml.',
    'runtime_equivalence': 'Not established by matching saved configuration; exact historical dependency versions were not proved.',
    'diffs': {},
}
for name, (log, env) in SOURCES.items():
    if name == 'B1':
        continue
    analysis['config_comparison']['diffs'][name] = {}
    for kind in ('env', 'agent'):
        ref = env if kind == 'env' else env.with_name('agent.yaml')
        b1 = SOURCES['B1'][1] if kind == 'env' else SOURCES['B1'][1].with_name('agent.yaml')
        analysis['config_comparison']['diffs'][name][kind] = {
            'reference_sha256': sha(ref), 'B1_sha256': sha(b1),
            'unified_diff': '\n'.join(difflib.unified_diff(ref.read_text(encoding='utf-8-sig').splitlines(), b1.read_text(encoding='utf-8-sig').splitlines(), fromfile=name, tofile='B1', n=1)),
        }

last = '1400-1499'
comparison = {}
for name in SOURCES:
    if name == 'B1':
        continue
    comparison[name] = {}
    for metric in METRICS + ['weight_normalized_episode_reward_xy', 'weight_normalized_episode_reward_yaw']:
        b = analysis['models']['B1']['windows'][last][metric]['mean']
        r = analysis['models'][name]['windows'][last][metric]['mean']
        comparison[name][metric] = {'B1': b, 'reference': r, 'relative_change_percent': (b/r - 1)*100 if r else None}
analysis['last100_comparison'] = comparison

pred_path = ROOT / 'analysis/short_sweep_20261002/predicted_scores.json'
pred = json.loads(pred_path.read_text(encoding='utf-8-sig'))
anchor_path = ROOT / 'analysis/score_reverse_20261001/analysis.json'
anchor = json.loads(anchor_path.read_text(encoding='utf-8-sig'))['anchors']['Go2_R4']
heldout_path = ROOT / 'analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002/heldout_ranking.json'
analysis['calibration_audit'] = {
    'anchor_policy_count': 1, 'official_anchor_simulation': anchor['official_sim'],
    'official_anchor_total_assuming_report_design30': anchor['official_sim']+30,
    'legacy_prediction_status': pred['status'],
    'legacy_primary_prediction': pred['primary_prediction'],
    'legacy_mse_floor_ceiling': pred['legacy_floor_zero_local_error_ceiling_assuming_30'],
    'F10_existing_heldout_conditional_outputs_not_B1': {
        k: pred['full_predictions']['F10_3000'][k] for k in
        ['primary_total_assuming_30','global_scale_total_assuming_30','raw_self_total_assuming_30']},
    'B1_applicability': False,
    'reason': 'Requires full-window scenario tracking/survival from policy evaluation; B1 training logs provide neither.',
    'sources': {str(p): sha(p) for p in [pred_path, anchor_path, heldout_path]},
}
analysis['S1_R1_portal_training_context'] = {
    'S1_tracking_pair': {'last10_base_contact_rate_rounded': .073, 'final_terrain_rounded':5.41, 'final_std_rounded':.865, 'portal_checkpoint_iter':400, 'evaluated_checkpoint_iter':1499},
    'R1_tracking_double': {'last10_base_contact_rate_rounded': .080, 'final_terrain_rounded':5.73, 'final_std_rounded':.934, 'portal_checkpoint_iter':1499, 'evaluated_checkpoint_iter':1499},
    'tracking_trace_available':False,
    'report_scripts_or_embedded_tracking_series_found':False,
}
for name in ('S1_tracking_pair','R1_tracking_double'):
    p = ROOT / 'analysis/high_score_search_20261009/portal_recovery_20261009' / name / 'report_visible.txt'
    analysis['S1_R1_portal_training_context'][name]['report_path'] = str(p)
    analysis['S1_R1_portal_training_context'][name]['report_sha256'] = sha(p)

OUT.mkdir(parents=True, exist_ok=True)
(OUT/'historical_comparison.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'last100':comparison,'B1_last10':analysis['models']['B1']['windows']['1490-1499']},ensure_ascii=False,indent=2))
