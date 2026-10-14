"""Sensitivity and score headroom calculations from existing local summaries."""
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ANALYSIS = BASE.parent
DATA = ANALYSIS / 'short_sweep_20261002' / 'recovered_20261003' / 'short_sweep_20261002'
heldout = json.loads((DATA / 'heldout_ranking.json').read_text(encoding='utf-8'))['models']
anchor = json.loads((ANALYSIS / 'score_reverse_20261001' / 'analysis.json').read_text(encoding='utf-8'))['anchors']['Go2_R4']
weights = {r['sid']: r['w'] for r in anchor['rows']}
axes = ['vx', 'vy', 'yaw']


def quality(model, aw):
    return sum(weights[sid] * row['survival'] * math.exp(-math.sqrt(
        sum(aw[i] * row['rmse'][a] ** 2 for i, a in enumerate(axes))))
        for sid, row in heldout[model]['scenarios'].items())


metric_cases = {}
for name, aw in {'three_axes_equal': [1/3, 1/3, 1/3], 'linear_xy_equal': [.5, .5, 0],
                 'vx_only': [1, 0, 0], 'vy_only': [0, 1, 0], 'yaw_only': [0, 0, 1]}.items():
    qs = {m: quality(m, aw) for m in heldout}
    metric_cases[name] = {'axis_weights': aw, 'internal_quality': qs,
                          'relative_change_F10_percent': (qs['F10_3000'] / qs['C0_5500'] - 1) * 100}
grid = []
for x in range(21):
    for y in range(21 - x):
        aw = [x / 20, y / 20, (20 - x - y) / 20]
        q = {m: quality(m, aw) for m in heldout}
        grid.append({'axis_weights': aw, 'F10_minus_C0': q['F10_3000'] - q['C0_5500']})
execution = json.loads((ANALYSIS / 'short_sweep_20261002' / 'execution.json').read_text(encoding='utf-8'))
secs = [execution['last_heartbeat'][name + '_duration_seconds'] / 800
        for name in ['control', 'action', 'acc', 'linear']]
sec_per_iter = sum(secs) / len(secs)
prediction = json.loads((ANALYSIS / 'short_sweep_20261002' / 'predicted_scores.json').read_text(encoding='utf-8'))
headroom = [{'scenario': r['sid'], 'official_recorded': r['official_point'],
             'maximum': 70 * r['w'], 'headroom': 70 * r['w'] - r['official_point']}
            for r in anchor['rows']]
out = {
    'scope': 'Offline comparisons; alternate axis metrics are sensitivity hypotheses, not official formulas.',
    'metric_cases': metric_cases,
    'simplex_grid': {'step': .05, 'points': len(grid),
                     'F10_better_points': sum(r['F10_minus_C0'] > 0 for r in grid),
                     'warning': 'Fraction of this arbitrary grid is NOT a probability of official improvement.',
                     'rows': grid},
    'official_anchor_headroom': headroom,
    'C0_total_if_only_G6_becomes_perfect': 30 + anchor['official_sim'] + next(r['headroom'] for r in headroom if r['scenario'] == 'G6'),
    'MSE_floor_model_ceiling_assuming_30': 30 + 70 * math.exp(-math.sqrt(prediction['current_evaluator_floor_c'])),
    'budget_measured_seconds_per_iteration': sec_per_iter,
    'training_only_projection_hours': {str(n): n * sec_per_iter / 3600 for n in [800, 1500, 2000, 3000]},
    'training_only_three_2000_runs_hours': 3 * 2000 * sec_per_iter / 3600,
    'training_only_four_1500_runs_hours': 4 * 1500 * sec_per_iter / 3600,
    'training_only_three_2000_runs_two_training_seeds_hours': 6 * 2000 * sec_per_iter / 3600,
    'budget_warning': 'Linear projection from four completed 800-iteration runs; startup/evaluation/recovery and later slowdown are extra.'}
(BASE / 'objective_and_budget_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'simplex_grid'}, ensure_ascii=False, indent=2))
