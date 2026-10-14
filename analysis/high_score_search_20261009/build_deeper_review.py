"""Recalculate existing Go2 evidence on CPU; no training or simulator imports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parent
RECOVERED = ANALYSIS / 'short_sweep_20261002/recovered_20261003/short_sweep_20261002'
OLD_EVAL = ANALYSIS / 'round20_execution_20261001/results/extracted/round20_20261001/eval'
WEIGHTS = dict(zip((f'G{i}' for i in range(1, 8)), (.15, .15, .2, .15, .15, .1, .1)))
AXES = ('vx', 'vy', 'yaw')
INPUTS = {}


def read(path):
    data = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data.decode('utf-8-sig'))


def main():
    old = read(ANALYSIS / 'score_breakthrough_20261002/third/analysis.json')
    logs = old['training_terms']['F10']
    mid, late = logs[0]['medians'], logs[1]['medians']
    drive = sum(mid['Episode_Reward/'+name] for name in ('track_lin_vel_xy_exp', 'track_ang_vel_z_exp'))
    negative = mid['negative_term_magnitude_sum_of_medians']
    air = abs(mid['Episode_Reward/feet_air_time'])
    z = abs(mid['Episode_Reward/lin_vel_z_l2'])
    torque = abs(mid['Episode_Reward/dof_torques_l2'])
    reweight = {
        'fixed_trajectory_tracking_sum_of_medians': drive,
        'negative_sum_of_medians': negative,
        'S1_tracking_gain': drive*.5,
        'R1_tracking_gain': drive,
        'S4_extra_over_S1': air+z*.5,
        'R2_extra_over_S1': torque*.5,
        'air_fraction_of_negative_median_sum': air/negative,
        'S4_extra_fraction_of_S1_tracking_gain': (air+z*.5)/(drive*.5),
        'warning': 'Fixed old trajectories and sums of marginal medians only. Not a policy gradient, learning prediction or official score.'}

    checkpoints = {}
    for checkpoint in ('3000', '3999'):
        rows = {}
        for terrain in ('flat', 'rough', 'slope', 'stairs'):
            path = OLD_EVAL / f'F10_model_{checkpoint}_A' / f'{terrain}.json'
            data = read(path)
            assert data['run_id'] == f'F10_model_{checkpoint}_A'
            for scenario, row in data['scenarios'].items():
                assert scenario not in rows
                rows[scenario] = row
        assert set(rows) == set(WEIGHTS)-{'G7'}
        checkpoints[checkpoint] = rows
    comparison = []
    for scenario in checkpoints['3000']:
        first, last = (checkpoints[key][scenario] for key in ('3000', '3999'))
        assert first['episodes'] == last['episodes'] == 512
        for key in ('segment_cmds', 'seg_s', 'push_mag', 'push_period_s'):
            assert first.get(key) == last.get(key), (scenario, key)
        for axis in AXES:
            a, b = first['rmse'][axis], last['rmse'][axis]
            comparison.append({'scenario': scenario, 'axis': axis, 'rmse_3000': a,
                               'rmse_3999': b, 'change_percent': (b/a-1)*100})
    axis_mse = {}
    for axis in AXES:
        a, b = (sum(WEIGHTS[s]*r['rmse'][axis]**2 for s, r in checkpoints[key].items())
                for key in ('3000', '3999'))
        axis_mse[axis] = {'weighted_partial_mse_3000': a, 'weighted_partial_mse_3999': b,
                          'change_percent': (b/a-1)*100}

    phases = {}
    phase_details = {}
    for path in sorted((RECOVERED/'results').glob('heldout_*/*.json')):
        data = read(path)
        model = data['model']
        for scenario, rows in data['scenarios'].items():
            key = (model, scenario)
            assert key not in phase_details
            count = sum(row['full']['count'] for row in rows)
            steady_count = sum(row['steady']['count'] for row in rows)
            full = [sum(row['full']['sum_squared'][i] for row in rows) for i in range(3)]
            steady = [sum(row['steady']['sum_squared'][i] for row in rows) for i in range(3)]
            assert count > steady_count > 0
            phase_details[key] = {'full_count': count, 'first_window_count': count-steady_count,
                                  'full_sse': full, 'first_window_sse': [a-b for a,b in zip(full, steady)]}
    for model in ('C0_5500', 'F10_3000'):
        assert {s for m,s in phase_details if m == model} == set(WEIGHTS)
        phase = {}
        for i, axis in enumerate(AXES):
            full_mse = sum(w*phase_details[(model,s)]['full_sse'][i]/phase_details[(model,s)]['full_count']
                           for s,w in WEIGHTS.items())
            window_mse = sum(w*phase_details[(model,s)]['first_window_sse'][i]/phase_details[(model,s)]['full_count']
                             for s,w in WEIGHTS.items())
            phase[axis] = {'weighted_full_mse': full_mse,
                           'weighted_first_window_mse_contribution': window_mse,
                           'first_window_fraction_of_weighted_mse': window_mse/full_mse}
        phases[model] = phase

    audit = read(ROOT/'audit.json')
    assert audit['selected_official_goal']['official_total_target'] == 93
    assert audit['budget']['planned_total_cap_hours'] == 19
    assert audit['budget']['exploration']['candidate_ids'] == ['S1_tracking_pair', 'R1_tracking_double']
    assert math.isclose(audit['budget']['protected_uncommitted_hours_after_all_caps']*60, 845)
    assert sum(row['change_percent'] > 0 for row in comparison) == 17
    out = {
        'status': 'OFFLINE_RESEARCH_ONLY_NO_NEW_POLICY_MEASUREMENTS',
        'official_total_goal': 93,
        'input_sha256': INPUTS,
        'reweighting': reweight,
        'matched_checkpoint_comparison': {
            'same_policy_training_run': True, 'set': 'A', 'scenarios': list(checkpoints['3000']),
            'coverage_weight': .9, 'G7_unmeasured_at_3999': True,
            'independent_training_seed_repeat': False,
            'axis_comparisons': comparison, 'worse_rmse_count': 17, 'total_axis_comparisons': 18,
            'weighted_partial_axis_mse': axis_mse,
            'warning': 'Historical internal comparison; earlier evaluator conditions differ from later heldout protocol. Not an official score or evidence that every new run worsens after 3000.'},
        'heldout_first_half_second': {
            'definition': 'First 0.5 seconds of each 5-second command segment, including initial settling and repeated-command segments; not exclusively changed-command response or push recovery.',
            'fraction_of_samples': .1, 'models': phases,
            'warning': 'Shares of measured squared error, not shares of official lost points or fully recoverable error.'},
        'candidate_revision': {
            'previous': ['S1_tracking_pair', 'S4_tracking_short_step'],
            'current': ['S1_tracking_pair', 'R1_tracking_double'],
            'why': 'Test 1.5x versus 2x task priority with the same tracking ratio and fixed auxiliary coefficients. Retain S4/R2 as untrained reserves.',
            'exploration_cap_hours_unchanged': 6, 'total_cap_hours_unchanged': 19,
            'score_prediction': None, 'server_started': False},
        'all_inputs_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                                    for p,digest in INPUTS.items())}
    assert out['all_inputs_unchanged']
    (ROOT/'deeper_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    # Optional local chart dependencies were installed for the host Python 3.10.
    if (ROOT/'.plot_packages').exists():
        sys.path.insert(0, str(ROOT/'.plot_packages'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    labels = ['Forward', 'Lateral', 'Yaw']
    axes[0].bar(labels, [axis_mse[a]['change_percent'] for a in AXES], color='#b74343')
    axes[0].set_title('F10: 3000 -> 3999 iterations')
    axes[0].set_ylabel('Weighted MSE change (%)')
    axes[0].text(.02, .98, '6 scenarios; 17/18 RMSE worsened', transform=axes[0].transAxes, va='top', fontsize=9)
    for i, model in enumerate(('C0_5500', 'F10_3000')):
        axes[1].bar([j + (i-.5)*.35 for j in range(3)],
                    [100*phases[model][a]['first_window_fraction_of_weighted_mse'] for a in AXES],
                    width=.35, label=model)
    axes[1].set_xticks(range(3), labels)
    axes[1].set_title('First 0.5s of each 5s segment')
    axes[1].set_ylabel('Share of weighted squared error (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Existing internal measurements: not official score predictions', fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT/'deeper_evidence.png', dpi=170)
    print(json.dumps({'rmse_worse': 17, 'rmse_compared': 18,
                      'mse_changes_percent': {a: axis_mse[a]['change_percent'] for a in AXES},
                      'F10_first_window_mse_shares': {a: phases['F10_3000'][a]['first_window_fraction_of_weighted_mse'] for a in AXES},
                      'S4_extra_fraction_of_S1_tracking_gain': reweight['S4_extra_fraction_of_S1_tracking_gain'],
                      'all_inputs_unchanged': True, 'no_training': True}, indent=2))


if __name__ == '__main__':
    main()
