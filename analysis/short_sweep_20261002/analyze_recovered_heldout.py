"""Offline analysis of already completed held-out evaluations; no simulator runs."""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'recovered_20261003' / 'short_sweep_20261002'
MODELS = ['C0_5500', 'F10_3000']
SCENARIOS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
AXES = ['vx', 'vy', 'yaw']
WEIGHTS = dict(zip(SCENARIOS, [.15, .15, .2, .15, .15, .1, .1]))
raw = {model: {} for model in MODELS}
for path in sorted((ROOT / 'results').glob('heldout_*/*.json')):
    data = json.loads(path.read_text())
    for scenario, rows in data['scenarios'].items():
        entry = {'seeds': {}}
        for phase in ['full', 'steady']:
            count = sum(row[phase]['count'] for row in rows)
            entry[phase] = {metric: {} for metric in ['rmse', 'mae', 'bias']}
            for i, axis in enumerate(AXES):
                entry[phase]['rmse'][axis] = math.sqrt(sum(row[phase]['sum_squared'][i] for row in rows) / count)
                entry[phase]['mae'][axis] = sum(row[phase]['sum_absolute'][i] for row in rows) / count
                entry[phase]['bias'][axis] = sum(row[phase]['sum_error'][i] for row in rows) / count
        for row in rows:
            entry['seeds'][str(row['seed'])] = math.exp(-math.sqrt(sum(v*v for v in row['full']['rmse'].values()) / 3))
        entry['segment_commands'] = rows[0]['legacy_summary']['segment_cmds']
        entry['segment_mean_actual_rounded'] = np.mean([row['legacy_summary']['segment_mean_actual'] for row in rows], axis=0).tolist()
        entry['transient_sse_fraction'] = {axis: 1 - sum(row['steady']['sum_squared'][i] for row in rows) / sum(row['full']['sum_squared'][i] for row in rows) for i, axis in enumerate(AXES)}
        raw[data['model']][scenario] = entry

ranking = json.loads((ROOT / 'heldout_ranking.json').read_text())['models']
changes = {}
for scenario in SCENARIOS:
    changes[scenario] = {
        phase: {metric: {axis: (raw['F10_3000'][scenario][phase][metric][axis] / raw['C0_5500'][scenario][phase][metric][axis] - 1) * 100 for axis in AXES}
                for metric in ['rmse', 'mae']}
        for phase in ['full', 'steady']}
    changes[scenario]['weighted_quality_contribution'] = WEIGHTS[scenario] * (ranking['F10_3000']['scenarios'][scenario]['score'] - ranking['C0_5500']['scenarios'][scenario]['score'])

aggregate = {}
for model in MODELS:
    aggregate[model] = {'weighted_axis_rmse': {axis: sum(WEIGHTS[s] * raw[model][s]['full']['rmse'][axis] for s in SCENARIOS) for axis in AXES},
                        'internal_quality': ranking[model]['partial_weighted_quality'],
                        'equal_scenario_internal_quality': sum(ranking[model]['scenarios'][s]['score'] for s in SCENARIOS) / 7,
                        'seed_internal_quality': {seed: sum(WEIGHTS[s] * raw[model][s]['seeds'][seed] for s in SCENARIOS) for seed in ['809', '1201']}}

# G1 speed response from all 256 measured episodes, not only the eight detailed trace robots.
g1 = {}
for model in MODELS:
    blocks = []
    for seed in [809, 1201]:
        with np.load(ROOT / 'results' / 'heldout_flat' / 'traces' / f'{model}_G1_{seed}.npz', allow_pickle=False) as z:
            assert z['valid'].all()
            blocks.append(z['error'][:, :, 0].astype(float))
            dt = float(z['dt'])
    errors = np.concatenate(blocks, axis=1)
    cmd = np.repeat(np.array([.3, .6, 1., -.5]), 250)
    actual = errors + cmd[:, None]
    segments = []
    for i in range(4):
        block = actual[i*250+25:(i+1)*250]
        segments.append({'command': float(cmd[i*250]), 'actual_steady_mean': float(block.mean()),
                         'vx_rmse_steady': float(np.sqrt(np.mean((block-cmd[i*250])**2)))})
    g1[model] = {'time': (np.arange(1000)*dt).tolist(), 'mean_vx': actual.mean(axis=1).tolist(),
                 'command': cmd.tolist(), 'segments_excluding_first_0_5s': segments}

out = {'scope': 'Offline analysis of recovered data only. Internal metrics, not official scores.',
       'independent_training_seed_repeats': False,
       'weights': WEIGHTS, 'raw_aggregates': raw, 'changes_percent': changes,
       'aggregate': aggregate, 'g1_time_series': g1}
(BASE / 'heldout_detailed_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
matrix = np.array([[changes[s]['full']['rmse'][a] for a in AXES] for s in SCENARIOS])
im = ax[0].imshow(matrix, vmin=-25, vmax=25, cmap='RdBu_r', aspect='auto')
ax[0].set_xticks(range(3), ['Forward vx', 'Lateral vy', 'Yaw'])
ax[0].set_yticks(range(7), ['G1 speed changes', 'G2 omni-directional', 'G3 rough', 'G4 slope', 'G5 stairs', 'G6 pushes', 'G7 randomized'])
ax[0].set_title('F10 vs C0: RMSE change (%)\nNegative = lower error')
for i in range(7):
    for j in range(3):
        ax[0].text(j, i, f'{matrix[i,j]:+.1f}%', ha='center', va='center', color='white' if abs(matrix[i,j])>16 else '#17212b')
fig.colorbar(im, ax=ax[0], shrink=.75)
for model, color in [('C0_5500','#255c99'), ('F10_3000','#d65a31')]:
    d = g1[model]
    ax[1].plot(d['time'], d['mean_vx'], label=model, color=color, lw=1.4)
ax[1].plot(g1['C0_5500']['time'], g1['C0_5500']['command'], '--', color='#333333', label='Command', lw=1)
ax[1].set(title='G1 forward speed response\nMean across 256 measured episodes', xlabel='Time (s)', ylabel='Forward velocity (m/s)')
ax[1].grid(alpha=.2); ax[1].legend(loc='lower left')
fig.suptitle('Held-out internal evaluation: 7 scenarios, 0 falls / 1792 episodes per model', fontsize=13)
fig.savefig(BASE/'heldout_error_comparison.png',dpi=160)

compact={'aggregate':aggregate,'changes_percent':changes,'g1_segments':{m:g1[m]['segments_excluding_first_0_5s'] for m in MODELS},
         'g1_transient_sse_fraction':{m:raw[m]['G1']['transient_sse_fraction'] for m in MODELS}}
print(json.dumps(compact,indent=2))
