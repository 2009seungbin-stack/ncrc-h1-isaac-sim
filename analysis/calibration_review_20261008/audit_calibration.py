"""Offline calibration stress test. Hypothetical errors are NOT policy evaluations."""
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'short_sweep_20261002'
data_path = PRIOR / 'recovered_20261003' / 'short_sweep_20261002' / 'heldout_ranking.json'
pred_path = PRIOR / 'predicted_scores.json'
models = json.loads(data_path.read_text(encoding='utf-8'))['models']
pred = json.loads(pred_path.read_text(encoding='utf-8'))
weights = pred['weights']
c = pred['current_evaluator_floor_c']
anchor_sim = pred['anchor_sim']


def q(model, rmse_scale=1., offset=0., gain=1., remove_axis=None):
    total = 0.
    for sid, row in model['scenarios'].items():
        mse = sum((0. if a == remove_axis else row['rmse'][a] ** 2)
                  for a in ['vx', 'vy', 'yaw']) / 3
        total += weights[sid] * row['survival'] * math.exp(-gain * math.sqrt(mse * rmse_scale**2 + offset))
    return total


control = models['C0_5500']
ratio = anchor_sim / (70 * q(control))
lo, hi = 0., 100.
for _ in range(100):
    mid = (lo + hi) / 2
    if 70 * q(control, gain=mid) > anchor_sim:
        lo = mid
    else:
        hi = mid
gain = (lo + hi) / 2


def totals(model, scale=1., remove_axis=None):
    return {
        'legacy_additive_mse_floor': 30 + 70 * q(model, scale, offset=c, remove_axis=remove_axis),
        'legacy_quality_ratio': 30 + 70 * ratio * q(model, scale, remove_axis=remove_axis),
        'illustrative_error_gain': 30 + 70 * q(model, scale, gain=gain, remove_axis=remove_axis),
    }


factual = {m: totals(d) for m, d in models.items()}
assert all(math.isclose(v, 88.14, abs_tol=1e-9) for v in factual['C0_5500'].values())
for m, d in models.items():
    assert math.isclose(q(d), d['partial_weighted_quality'], abs_tol=1e-10)
assert math.isclose(factual['F10_3000']['legacy_additive_mse_floor'], 88.3645532571, abs_tol=1e-9)
counterfactual = {m: [{'rmse_fraction_remaining': f, 'outputs': totals(d, f)}
                      for f in [1., .75, .5, .25, 0.]] for m, d in models.items()}
out = {
    'review_date_local': '2026-10-08',
    'status': 'CALIBRATION_SENSITIVITY_ONLY_NOT_VALIDATED_PREDICTION',
    'official_candidate_score': None,
    'anchor_policy_count': 1,
    'fixed_qualitative_points_assumption': 30,
    'parameters': {'legacy_mse_floor': c, 'legacy_quality_ratio': ratio, 'illustrative_error_gain': gain},
    'real_measurement_inputs_hypothetical_official_outputs': factual,
    'hypothetical_error_reductions_NOT_measured_policies': counterfactual,
    'F10_single_axis_removal_NOT_measured_policies': {a: totals(models['F10_3000'], remove_axis=a) for a in ['vx', 'vy', 'yaw']},
    'internal_quality_relative_change_percent': (q(models['F10_3000']) / q(control) - 1) * 100,
    'limitations': [
        'The error-gain mapping is an arbitrary competing assumption, not a new preferred prediction.',
        'All mappings match C0 by construction; none was validated on a second official policy outcome.',
        'Their spread is not a confidence interval or probability of reaching 90.',
        'Zero local error does not establish zero official error; synthetic error scaling preserves survival only as an assumption.',
        'No training, simulator evaluation, server access or official submission was performed.'
    ],
    'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [data_path, pred_path]],
}
BASE.mkdir(exist_ok=True)
(BASE / 'calibration_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'real_data_outputs': factual, 'zero_local_error_outputs': counterfactual['F10_3000'][-1],
                  'checks': 'C0 anchor matched by all formulas; legacy F10 output and raw quality reproduced.'}, indent=2))
