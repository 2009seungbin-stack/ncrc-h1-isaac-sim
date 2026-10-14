"""Local descriptive training comparison. No rollout, regression, CI, or network.

It reads immutable recovered logs and effective YAML, checks raw row coverage,
and compares both common iterations and coarse terrain support. Endogenous
curriculum and policy-dependent visitation preclude a causal interpretation.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import yaml

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OLD = BASE / 'round20_execution_20261001/results/extracted/round20_20261001'
NEW = BASE / 'baseline_redesign_20261009_4h/recovered_full_20261009_1427/b1_baseline_20261009_4h'
RUNS = {'B1': (NEW / 'logs/train_B1.log', NEW / 'preserved/B1_selective_cost'),
        'F10': (OLD / 'logs/train_F10.log', OLD / 'preserved/F10'),
        'F00': (OLD / 'logs/train_F00.log', OLD / 'preserved/F00')}
RAW_KEYS = ['Mean reward', 'Mean episode length', 'Mean action std',
            'Curriculum/terrain_levels', 'Metrics/base_velocity/error_vel_xy',
            'Metrics/base_velocity/error_vel_yaw', 'Episode_Termination/base_contact',
            'Episode_Termination/time_out']
COMPONENTS = {'tracking_xy_weight_removed_proxy': 'track_lin_vel_xy_exp',
              'tracking_yaw_weight_removed_proxy': 'track_ang_vel_z_exp',
              'vertical_cost_weight_removed_proxy': 'lin_vel_z_l2',
              'roll_pitch_cost_weight_removed_proxy': 'ang_vel_xy_l2',
              'torque_cost_weight_removed_proxy': 'dof_torques_l2',
              'acceleration_cost_weight_removed_proxy': 'dof_acc_l2',
              'action_change_cost_weight_removed_proxy': 'action_rate_l2'}
KEYS = RAW_KEYS + list(COMPONENTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(name, log_path, preserved):
    runs = [p for p in preserved.iterdir() if p.is_dir()]
    assert len(runs) == 1, (name, runs)
    env_path, agent_path = runs[0] / 'params/env.yaml', runs[0] / 'params/agent.yaml'
    # BaseLoader only returns mappings/lists/strings, never constructs Python tags.
    env = yaml.load(env_path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    agent = yaml.load(agent_path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    weights = {k: float(v['weight']) for k, v in env['rewards'].items() if isinstance(v, dict)}
    log = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text(encoding='utf-8'))
    matches = list(re.finditer(r'Learning iteration (\d+)/(\d+)', log))
    rows = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else log.find('Training time:', match.end())
        if end < 0:
            end = len(log)
        block = log[match.end():end]
        row = {'iteration': int(match[1])}
        for key, number in re.findall(r'^\s*([^\n:]+):\s+([+-]?(?:\d+(?:\.\d+)?|nan|inf)(?:[eE][+-]?\d+)?)(?:s)?\s*$', block, re.M | re.I):
            row[key.strip()] = float(number)
        for proxy, term in COMPONENTS.items():
            row[proxy] = row['Episode_Reward/' + term] / weights[term]
        rows.append(row)
    totals = {int(match[2]) for match in matches}
    assert len(totals) == 1
    expected = totals.pop()
    iterations = [row['iteration'] for row in rows]
    missing_metrics = {key: [row['iteration'] for row in rows if key not in row] for key in KEYS}
    nonfinite = [(row['iteration'], key) for row in rows for key in KEYS if key in row and not math.isfinite(row[key])]
    assert iterations == list(range(expected)), (name, 'iteration coverage incomplete')
    assert not any(missing_metrics.values()) and not nonfinite, (name, missing_metrics, nonfinite)
    facts = {'name': name, 'log': str(log_path), 'log_sha256': digest(log_path),
             'env_yaml': str(env_path), 'env_sha256': digest(env_path), 'agent_sha256': digest(agent_path),
             'weights': weights, 'coverage': {'rows': len(rows), 'expected_iterations': expected,
             'contiguous_0_to_last': True, 'duplicates': 0, 'missing_metrics': [], 'nonfinite_metrics': []},
             'actor': agent.get('actor'), 'algorithm': agent.get('algorithm'),
             'seed': agent.get('seed'), 'num_steps_per_env': agent.get('num_steps_per_env'),
             'num_envs': env['scene']['num_envs'], 'episode_length_s': env['episode_length_s'],
             'control_dt': float(env['sim']['dt']) * int(env['decimation']),
             'command': {key: env['commands']['base_velocity'][key] for key in
                         ['ranges', 'resampling_time_range', 'heading_command', 'rel_standing_envs', 'rel_heading_envs']}}
    return rows, facts


def summarize(rows):
    assert rows
    return {'n': len(rows), 'iteration_min': min(row['iteration'] for row in rows),
            'iteration_max': max(row['iteration'] for row in rows),
            'mean': {key: statistics.fmean(row[key] for row in rows) for key in KEYS},
            'descriptive_std': {key: statistics.pstdev(row[key] for row in rows) for key in KEYS}}


def coarse_match(all_rows, iteration_max, width=0.5, minimum=30):
    by_model = {}
    for name, rows in all_rows.items():
        bins = {}
        for row in rows:
            if row['iteration'] < 100 or (iteration_max is not None and row['iteration'] > iteration_max):
                continue
            bucket = math.floor(row['Curriculum/terrain_levels'] / width)
            bins.setdefault(bucket, []).append(row)
        by_model[name] = bins
    shared = sorted(set.intersection(*(set(bucket for bucket, rows in bins.items() if len(rows) >= minimum)
                                       for bins in by_model.values())))
    evidence = [{'bin': [bucket * width, (bucket + 1) * width],
                 'models': {name: summarize(bins[bucket]) for name, bins in by_model.items()}}
                for bucket in shared]
    equal_bin = {name: {key: statistics.fmean(bin_['models'][name]['mean'][key] for bin_ in evidence)
                            for key in KEYS} for name in by_model} if evidence else {}
    return {'iteration_lower': 100, 'iteration_upper': iteration_max, 'width': width,
            'minimum_rows_per_model_bin': minimum, 'common_bins': evidence,
            'equal_bin_weighted_descriptive_mean': equal_bin,
            'not_a_causal_adjustment': True, 'inferential_statistics': None}


def main():
    all_rows, sources = {}, {}
    for name, (log, preserved) in RUNS.items():
        all_rows[name], sources[name] = parse(name, log, preserved)
    windows = [(100, 499), (500, 999), (1000, 1499), (300, 599), (1250, 1499)]
    same_iteration = [{'range': [start, end], 'models': {name: summarize([row for row in rows
                      if start <= row['iteration'] <= end]) for name, rows in all_rows.items()}}
                      for start, end in windows]
    result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'sources': sources, 'same_iteration': same_iteration,
              'terrain_match_first1500': coarse_match(all_rows, 1499),
              'terrain_match_all_available_sensitivity': coarse_match(all_rows, None),
              'bin_width_sensitivity': {str(width): coarse_match(all_rows, 1499, width)
                                       for width in [0.25, 1.0]},
              'limitations': ['One run per bundle, all training seed42; no independent training replicates.',
                              'Iterations are serial, dependent samples; no CI, p-value or significance claims.',
                              'Terrain level is an endogenous aggregate curriculum outcome, not a fixed terrain distribution.',
                              'Similar mean terrain level does not match terrain types, command visitation, episode weights or skill history.',
                              'Actor standard deviation differs by model/phase; no regression was used to remove it.',
                              'Configuration ranges agree, but actual commands and episode sample counts per log row are absent.',
                              'B1 runtime known5.0.1/torch2.7.0+cu128; exact F00/F10 runtime equality has not been proven by recovered version receipts.',
                              'Reward divided by its configured weight is an accumulated training-log proxy, not a physical moment/torque/velocity quantity.',
                              'Training error_vel metrics are not heldout scenario RMSE/MAE or official score.',
                              'A staged reward sequence was never trained; these data do not identify an effective switching time.'],
              'official_score_prediction': None}
    OUT.mkdir(exist_ok=True)
    (OUT / 'phase_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'phase_rows.json').write_text(json.dumps(all_rows, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'coverage': {name: sources[name]['coverage'] for name in sources},
                      'same_iteration': [{'range': w['range'], 'models': {name: {key: round(value, 6)
                      for key, value in m['mean'].items() if key in RAW_KEYS[:3]+RAW_KEYS[3:8]}
                      for name,m in w['models'].items()}} for w in same_iteration],
                      'terrain_common_bins': [{'bin': b['bin'], 'n': {n:m['n'] for n,m in b['models'].items()}}
                                              for b in result['terrain_match_first1500']['common_bins']],
                      'terrain_equal_bin': result['terrain_match_first1500']['equal_bin_weighted_descriptive_mean']}, indent=2))

    grouped = {}
    shared = result['terrain_match_first1500']['common_bins']
    for label, lower, upper in [('early', 1.5, 2.5), ('middle', 2.5, 4.5), ('late', 4.5, 5.5)]:
        chosen = [b for b in shared if lower <= b['bin'][0] < upper]
        grouped[label] = {'terrain_interval': [lower, upper], 'equal_bin_mean': {
            name: {key: statistics.fmean(b['models'][name]['mean'][key] for b in chosen) for key in KEYS}
            for name in all_rows}, 'bins': [b['bin'] for b in chosen]}
    result['terrain_groups'] = grouped
    contrasts = {}
    for label, group in grouped.items():
        b1, f10 = group['equal_bin_mean']['B1'], group['equal_bin_mean']['F10']
        compare_keys = ['Metrics/base_velocity/error_vel_xy', 'Metrics/base_velocity/error_vel_yaw'] + [k for k in COMPONENTS if 'cost' in k]
        contrasts[label] = {key: 100 * (b1[key] / f10[key] - 1) for key in compare_keys}
        contrasts[label]['base_contact_change_percentage_points'] = 100 * (
            b1['Episode_Termination/base_contact'] - f10['Episode_Termination/base_contact'])
    result['B1_vs_F10_terrain_group_descriptive_contrasts'] = contrasts
    result['curriculum_first_crossing_after_iter100'] = {
        name: {str(threshold): next((r['iteration'] for r in rows if r['iteration'] >= 100
                and r['Curriculum/terrain_levels'] >= threshold), None) for threshold in [4.5, 5.0, 5.5]}
        for name, rows in all_rows.items()}
    result['conclusions'] = [
        'The early same-iteration XY advantage over F10 is not a general advantage after terrain matching.',
        'B1 retains a limited early/middle training base-contact advantage at some shared terrain bins, with worse yaw tracking.',
        'At terrain4.5-5.5 B1 has worse XY/yaw and base-contact than F10 despite slightly lower aggregate terrain level.',
        'Against F00 B1 has less base-contact but worse XY in shared support; F00 trades stability for linear tracking.',
        'B1 torque relaxation does not deliver lower weight-removed torque proxy relative to F10; raw weighted penalty is misleading.',
        'B1 late vertical and roll/pitch proxies exceed F10; hardening those costs is a plausible research hypothesis, not an isolated effect.',
        'B1 early then F10 late is untested. Curves from separate fixed-bundle runs cannot be spliced into an effective training schedule.'
    ]
    (OUT / 'phase_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    report(result)
    plot(all_rows, result)


def report(result):
    lines = ['Local phase comparison: B1/F10/F00', '',
             'All raw logs verified: B1 1500 rows; F10/F00 4000 each, complete contiguous iteration coverage and finite metrics.',
             'Primary comparisons use100..1499. No model rank from total weighted reward.', '',
             'Shared terrain groups: 0.5-width bins, equal bin weighting, at least30 rows/model/bin.',
             'This is descriptive overlap, not matching of trajectories or causal adjustment.',
             'group | model | XY logger | yaw logger | base_contact% | actor_std | terrain_mean']
    for label, group in result['terrain_groups'].items():
        for name, m in group['equal_bin_mean'].items():
            lines.append(f"{label} | {name} | {m['Metrics/base_velocity/error_vel_xy']:.5f} | "
                         f"{m['Metrics/base_velocity/error_vel_yaw']:.5f} | {100*m['Episode_Termination/base_contact']:.4f} | "
                         f"{m['Mean action std']:.5f} | {m['Curriculum/terrain_levels']:.5f}")
    lines += ['', 'Weight-removed cost proxies (not physical units):',
              'group | model | vertical | roll_pitch | torque | acceleration | action_change']
    for label, group in result['terrain_groups'].items():
        for name, m in group['equal_bin_mean'].items():
            values = [m[key] for key in COMPONENTS if 'cost' in key]
            lines.append(label + ' | ' + name + ' | ' + ' | '.join(f'{v:.6g}' for v in values))
    lines += ['', 'Bin-width sensitivity and changing support:']
    for width, matched in [('0.5', result['terrain_match_first1500'])] + list(result['bin_width_sensitivity'].items()):
        means = matched['equal_bin_weighted_descriptive_mean']
        base, comp = means['B1'], means['F10']
        lines.append(f"width{width}: {len(matched['common_bins'])} shared bins; B1/F10 XY relative "
                     f"{100*(base['Metrics/base_velocity/error_vel_xy']/comp['Metrics/base_velocity/error_vel_xy']-1):+.3f}%, "
                     f"yaw relative {100*(base['Metrics/base_velocity/error_vel_yaw']/comp['Metrics/base_velocity/error_vel_yaw']-1):+.3f}%, "
                     f"base_contact {100*(base['Episode_Termination/base_contact']-comp['Episode_Termination/base_contact']):+.4f} percentage points")
        lines.append('support: ' + str([b['bin'] for b in matched['common_bins']]))
    lines += ['', 'Interpretation:'] + result['conclusions'] + ['', 'Limitations:'] + result['limitations']
    lines += ['', 'Fixed-versus-staged conclusion: fixed F10 remains a more credible mature-phase reference than fixed B1 on these training summaries.',
              'B1 early/middle advantages justify investigating a phase-dependent hypothesis only. No observed switching policy exists.',
              'Do not recommend a production staged reward or precise switch iteration from these data alone.',
              'No official score prediction, CI, p-value, significance, causal claim or automatic new experiment.']
    lines += ['', 'B1 versus F10, equal-bin descriptive contrasts (percent; contact uses percentage points):']
    for label, values in result['B1_vs_F10_terrain_group_descriptive_contrasts'].items():
        lines.append(label + ': ' + json.dumps({key: round(value, 4) for key, value in values.items()}))
    (OUT / 'phase_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot(all_rows, result):
    import sys
    isolated = BASE / 'baseline_redesign_20261009_4h/offline_eval/_plotdeps'
    if isolated.is_dir():
        sys.path.insert(0, str(isolated))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'B1': '#2368b8', 'F10': '#d57920', 'F00': '#69727d'}
    metrics = [('Metrics/base_velocity/error_vel_xy', 'XY training logger error', 1),
               ('Metrics/base_velocity/error_vel_yaw', 'Yaw training logger error', 1),
               ('Episode_Termination/base_contact', 'Base-contact termination (%)', 100)]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    for column, (key, label, scale) in enumerate(metrics):
        ax = axes[0, column]
        for name, rows in all_rows.items():
            selected = [r for r in rows if 100 <= r['iteration'] <= 1499]
            smooth = [statistics.fmean(r[key] for r in selected[max(0, i-99):i+1])*scale
                      for i in range(len(selected))]
            ax.plot([r['iteration'] for r in selected], smooth, label=name, color=colors[name], linewidth=2)
        ax.set_title(label); ax.set_xlabel('Iteration (trailing100 mean)'); ax.grid(alpha=.22)
        ax.legend(frameon=False)
        ax = axes[1, column]
        bins = result['terrain_match_first1500']['common_bins']
        centers = [sum(b['bin'])/2 for b in bins]
        for name in all_rows:
            ax.plot(centers, [b['models'][name]['mean'][key]*scale for b in bins],
                    'o-', label=name, color=colors[name], linewidth=2, markersize=5)
        ax.axvspan(4.5, 5.5, color='#c55555', alpha=.075)
        ax.set_xlabel('Mean-terrain bins (0.5 width, shared support)'); ax.set_title(label)
        ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.suptitle('Training phase comparison: common iterations versus shared terrain support\n'
                 'One seed/run; endogenous curriculum; no trajectory matching or inferential intervals', fontsize=14)
    fig.savefig(OUT/'phase_curves.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
