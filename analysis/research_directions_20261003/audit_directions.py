"""Offline audit of existing Go2 traces; never starts a simulator or loads policies."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent / 'short_sweep_20261002' / 'recovered_20261003' / 'short_sweep_20261002'
MODELS = ['C0_5500', 'F10_3000']
SCENARIOS = [f'G{i}' for i in range(1, 8)]
EXPECTED = 'command3,body_lin3,body_ang3,action12,joint_vel12,joint_acc12,torque12,position3'
EFFORT = 23.5
VELOCITY = 30.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def motor_summary(t):
    """Samples are time x robot x channels; 50 Hz, last physics substep torque."""
    action, qvel, tau = t[:, :, 9:21], t[:, :, 21:33], t[:, :, 45:57]
    static_near = np.abs(tau) >= .95 * EFFORT
    qvel_clip = np.clip(qvel, -2 * VELOCITY, 2 * VELOCITY)
    upper = np.minimum(EFFORT, EFFORT * (1 - qvel_clip / VELOCITY))
    lower = np.maximum(-EFFORT, EFFORT * (-1 - qvel_clip / VELOCITY))
    # Only a diagnostic: saved torque and qvel are not synchronous actuator inputs.
    distance = np.minimum(np.abs(tau - upper), np.abs(tau - lower))
    approx_near = distance <= .05 * EFFORT
    outside = (tau > upper + .05 * EFFORT) | (tau < lower - .05 * EFFORT)
    return {
        'torque_rms_Nm': rms(tau),
        'abs_torque_p99_Nm': float(np.quantile(np.abs(tau), .99)),
        'static_95pct_cap_joint_samples_pct': float(static_near.mean() * 100),
        'static_95pct_cap_any_joint_robot_steps_pct': float(static_near.any(axis=-1).mean() * 100),
        'approx_torque_speed_boundary_joint_samples_pct': float(approx_near.mean() * 100),
        'outside_reconstructed_boundary_gt_5pct_pct': float(outside.mean() * 100),
        'joint_speed_rms_rad_s': rms(qvel),
        'abs_joint_speed_p99_rad_s': float(np.quantile(np.abs(qvel), .99)),
        'joint_speed_ge_95pct_nominal_limit_pct': float((np.abs(qvel) >= .95 * VELOCITY).mean() * 100),
        'sum_abs_tau_qvel_mean_W_diagnostic': float(np.mean(np.sum(np.abs(tau * qvel), axis=-1))),
        'action_change_rms_per_20ms': rms(np.diff(action, axis=0)),
        'body_vz_rms_m_s': rms(t[:, :, 5]),
        'body_roll_rate_rms_rad_s': rms(t[:, :, 6]),
        'body_pitch_rate_rms_rad_s': rms(t[:, :, 7]),
    }


def settling_times(error, dt=.02):
    # First run of 10 consecutive samples within +/-0.1m/s, observed over first 1.5s.
    ok = np.abs(error[:75]) <= .1
    times = []
    for i in range(ok.shape[1]):
        starts = [j for j in range(ok.shape[0] - 9) if ok[j:j+10, i].all()]
        times.append((starts[0] + 1) * dt if starts else None)
    uncensored = [x for x in times if x is not None]
    return {
        'definition': 'First of 10 consecutive 20ms samples within +/-0.1m/s; search first 1.5s',
        'fraction_without_qualifying_run': sum(x is None for x in times) / len(times),
        'median_s_among_qualifying': float(np.median(uncensored)) if uncensored else None,
        'p90_s_among_qualifying': float(np.quantile(uncensored, .9)) if uncensored else None,
    }


def error_summary(e):
    return {'rmse_xyz': np.sqrt(np.mean(e * e, axis=(0, 1))).tolist(),
            'mae_xyz': np.mean(np.abs(e), axis=(0, 1)).tolist()}


out = {
    'scope': 'Existing heldout data only; no simulator, new training, evaluation or official submission.',
    'data_contract': {'detailed_episodes_per_model_scenario': 16,
                      'error_episodes_per_model_scenario': 256, 'dt_s': .02,
                      'torque_limit_Nm': EFFORT, 'nominal_velocity_limit_rad_s': VELOCITY,
                      'dynamic_actuator': 'DCMotor v2.3.2; effort=saturation_effort=23.5, velocity_limit=30',
                      'limitations': ['Detailed motor data cover the first 8 environments per reset seed only.',
                                     '50Hz torque at last substep and post-step joint velocity cannot prove exact saturation.',
                                     'No joint positions, projected gravity, foot contacts or foot clearances were saved.',
                                     'Absolute tau*qvel is a mechanical-power diagnostic, not battery energy.',
                                     'Reset seeds are not independent training seeds.']},
    'sources': [], 'models': {}, 'g1_response': {}, 'g6_push_response': {}}

for model in MODELS:
    out['models'][model] = {}
    for sid in SCENARIOS:
        traces, errors = [], []
        for seed in [809, 1201]:
            paths = list((ROOT / 'results').glob(f'heldout_*/traces/{model}_{sid}_{seed}.npz'))
            assert len(paths) == 1, paths
            p = paths[0]
            with np.load(p, allow_pickle=False) as z:
                assert str(z['trace_columns']) == EXPECTED
                assert z['trace_first_envs'].shape == (1000, 8, 60)
                assert z['error'].shape == (1000, 128, 3)
                assert z['valid'].all()
                assert np.isclose(float(z['dt']), .02)
                traces.append(z['trace_first_envs'].astype(np.float64))
                errors.append(z['error'].astype(np.float64))
            out['sources'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        t = np.concatenate(traces, axis=1)
        e = np.concatenate(errors, axis=1)
        assert np.isfinite(t).all() and np.isfinite(e).all()
        out['models'][model][sid] = motor_summary(t)
        if sid == 'G1':
            transitions = []
            for i in range(4):
                start = 250 * i
                block = e[start:start + 250]
                transitions.append({'start_s': i * 5,
                                    'command': t[start, 0, :3].tolist(),
                                    'first_0_5s': error_summary(block[:25]),
                                    'remaining_4_5s': error_summary(block[25:]),
                                    'vx_settling': settling_times(block[:, :, 0]),
                                    'motor_first_0_5s': motor_summary(t[start:start + 25])})
            out['g1_response'][model] = transitions
        if sid == 'G6':
            # t_eval applies pushes AFTER recording step 125,250,...875.
            # First recorded response is 126,251,...876.
            windows = np.concatenate([e[k + 1:k + 1 + 75] for k in range(125, 1000, 125)], axis=1)
            out['g6_push_response'][model] = {
                'first_recorded_indices': list(range(126, 1000, 125)),
                'pooled_push_events_not_independent_episodes': 7 * 256,
                'first_0_5s': error_summary(windows[:25]),
                'first_0_5s_excluding_first_sample': error_summary(windows[1:25]),
                'following_1s': error_summary(windows[25:75]),
                'time_after_push_s': ((np.arange(75) + 1) * .02).tolist(),
                'vx_rmse_by_time': np.sqrt(np.mean(windows[:, :, 0] ** 2, axis=1)).tolist(),
                'by_reset_seed': {str(seed): {
                    'first_0_5s': error_summary(np.concatenate([er[k + 1:k + 26] for k in range(125, 1000, 125)], axis=1)),
                    'following_1s': error_summary(np.concatenate([er[k + 26:k + 76] for k in range(125, 1000, 125)], axis=1))}
                    for seed, er in zip([809, 1201], errors)},
                'protocol_caveat': 'Push is applied after obs was computed; SyncPolicy refreshes commands only. '
                                   'Next action therefore receives pre-push velocity observation for one step by code order. '
                                   'Excluding the first sample does not undo its downstream dynamics.',
            }

out['changes_percent_F10_vs_C0'] = {
    sid: {key: (out['models']['F10_3000'][sid][key] / value - 1) * 100 if value != 0 else None
          for key, value in out['models']['C0_5500'][sid].items()}
    for sid in SCENARIOS}
BASE.mkdir(parents=True, exist_ok=True)
(BASE / 'direction_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
colors = {'C0_5500': '#245b95', 'F10_3000': '#d45732'}
for i, model in enumerate(MODELS):
    response = out['g6_push_response'][model]
    axes[0].plot(response['time_after_push_s'], response['vx_rmse_by_time'],
                 color=colors[model], label=model, lw=2)
    vals = [out['models'][model][sid]['static_95pct_cap_any_joint_robot_steps_pct'] for sid in SCENARIOS]
    axes[1].bar(np.arange(7) + (i - .5) * .35, vals, .35, color=colors[model], label=model)
axes[0].axvline(.5, color='#666666', ls='--', lw=1)
axes[0].set(title='G6: forward-speed error after a push\n256 episodes / model, 7 pushes each',
            xlabel='Time after push (s)', ylabel='Forward RMSE (m/s)')
axes[1].set(title='At least one joint near the static torque cap\n16 detailed episodes / scenario / model',
            ylabel='Recorded robot steps (%)', xticks=np.arange(7), xticklabels=SCENARIOS)
for ax in axes:
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend()
fig.suptitle('Existing Go2 traces: response weakness is concentrated near disturbances', fontsize=13)
fig.savefig(BASE / 'response_and_effort.png', dpi=170)
print(json.dumps({'g6_response': {m: {k: v for k, v in out['g6_push_response'][m].items()
                                     if k not in ['time_after_push_s', 'vx_rmse_by_time']}
                                for m in MODELS}}, ensure_ascii=False, indent=2))
