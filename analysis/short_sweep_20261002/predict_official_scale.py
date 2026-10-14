"""Unvalidated calibration outputs, retained for sensitivity analysis only.

2026-10-08: withdraw the 'primary prediction' interpretation. Does not run policies.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parents[1]
DATA = BASE / 'recovered_20261003' / 'short_sweep_20261002'
ANCHOR_DATA = WORKSPACE / 'analysis' / 'score_reverse_20261001' / 'analysis.json'
historical = json.loads(ANCHOR_DATA.read_text(encoding='utf-8'))['anchors']['Go2_R4']
WEIGHTS = {row['sid']: row['w'] for row in historical['rows']}
POINTS = {row['sid']: row['official_point'] for row in historical['rows']}
SIM = historical['official_sim']
QUALITATIVE = 30.0
assert math.isclose(sum(POINTS.values()), SIM)
assert math.isclose(sum(WEIGHTS.values()), 1.0)

def load(name):
    return json.loads((DATA / (name + '_ranking.json')).read_text())['models']

def mse(scenario):
    return sum(scenario['rmse'][a]**2 for a in ['vx', 'vy', 'yaw']) / 3

def floor_sim(model, c):
    return 70 * sum(WEIGHTS[s] * row['survival'] * math.exp(-math.sqrt(mse(row) + c))
                    for s, row in model['scenarios'].items())

heldout = load('heldout')
assert set(heldout) == {'C0_5500', 'F10_3000'}
assert all(set(m['scenarios']) == set(WEIGHTS) for m in heldout.values())
baseline = heldout['C0_5500']
left, right = 0., .25
for _ in range(100):
    middle = (left + right) / 2
    if floor_sim(baseline, middle) > SIM:
        left = middle
    else:
        right = middle
c = (left + right) / 2
assert math.isclose(floor_sim(baseline, c), SIM, abs_tol=1e-10)
raw_baseline_sim = 70 * baseline['partial_weighted_quality']

def scenario_calibration(anchor):
    result = {}
    for sid, row in anchor['scenarios'].items():
        official_q = POINTS[sid] / (70 * WEIGHTS[sid])
        ci = (-math.log(official_q / row['survival']))**2 - mse(row)
        assert ci >= 0
        scale = official_q / row['score']
        result[sid] = {'mse_floor': ci, 'scale': scale}
    return result

def scenario_prediction(model, calibration):
    floor, scale = {}, {}
    for sid, row in model['scenarios'].items():
        assert sid in calibration
        coeff = calibration[sid]
        floor[sid] = 70 * WEIGHTS[sid] * row['survival'] * math.exp(-math.sqrt(mse(row) + coeff['mse_floor']))
        scale[sid] = 70 * WEIGHTS[sid] * row['score'] * coeff['scale']
    return {'mse_floor_points': floor, 'scale_points': scale,
            'mse_floor_sum': sum(floor.values()), 'scale_sum': sum(scale.values())}

scenario_coefficients = scenario_calibration(baseline)
predictions = {}
for name, model in heldout.items():
    primary_sim = floor_sim(model, c)
    scale_sim = SIM * 70 * model['partial_weighted_quality'] / raw_baseline_sim
    local = scenario_prediction(model, scenario_coefficients)
    totals = [QUALITATIVE + primary_sim, QUALITATIVE + scale_sim,
              QUALITATIVE + local['mse_floor_sum'], QUALITATIVE + local['scale_sum']]
    predictions[name] = {'primary_sim': primary_sim, 'primary_total_assuming_30': totals[0],
                         'global_scale_total_assuming_30': totals[1],
                         'scenario_floor_total_assuming_30': totals[2],
                         'scenario_scale_total_assuming_30': totals[3],
                         'model_sensitivity_min': min(totals), 'model_sensitivity_max': max(totals),
                         'raw_self_sim': 70 * model['partial_weighted_quality'],
                         'raw_self_total_assuming_30': QUALITATIVE + 70 * model['partial_weighted_quality'],
                         'scenario_predictions': local,
                         'officially_evaluated_candidate': name == 'C0_5500'}

# These models were only measured on four scenarios. Never extrapolate the missing 45%.
saved = load('saved')
screen_coefficients = scenario_calibration(saved['C0_5500'])
partial = {}
for stage, models in [('saved', saved), ('short', load('short'))]:
    partial[stage] = {}
    for name, model in models.items():
        p = scenario_prediction(model, screen_coefficients)
        partial[stage][name] = {'covered_max_points': 70 * sum(WEIGHTS[s] for s in model['scenarios']),
                               'predicted_covered_points_mse_floor': p['mse_floor_sum'],
                               'predicted_covered_points_scale': p['scale_sum'],
                               'unmeasured_scenarios': sorted(set(WEIGHTS) - set(model['scenarios'])),
                               'total_prediction': None, 'falls': model['falls']}

out = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'UNVALIDATED_CALIBRATION_OUTPUTS_NOT_PRIMARY_PREDICTION',
    'interpretation_review_date': '2026-10-08',
    'primary_prediction': None,
    'prediction_validated': False,
    'legacy_field_note': 'Numeric fields named primary are retained for historical compatibility, not endorsed as primary predictions.',
    'primary_method': 'Legacy C0 anchored 3-axis MSE floor. Retained for sensitivity analysis only.',
    'legacy_floor_zero_local_error_ceiling_assuming_30': QUALITATIVE + 70 * math.exp(-math.sqrt(c)),
    'formula': '30 + 70*sum(w*S_local*exp(-sqrt((vx_rmse^2+vy_rmse^2+yaw_rmse^2)/3+c)))',
    'current_evaluator_floor_c': c, 'anchor_sim': SIM, 'anchor_total': SIM + QUALITATIVE,
    'qualitative_points_assumed': QUALITATIVE, 'weights': WEIGHTS,
    'anchor_source': str(ANCHOR_DATA), 'anchor_source_sha256': hashlib.sha256(ANCHOR_DATA.read_bytes()).hexdigest(),
    'measurement_source': str(DATA / 'heldout_ranking.json'),
    'measurement_source_sha256': hashlib.sha256((DATA/'heldout_ranking.json').read_bytes()).hexdigest(),
    'full_predictions': predictions, 'partial_predictions_not_totals': partial,
    'limitations': [
        'C0 matching 88.14 is calibration by construction, not out-of-sample validation.',
        'The relation between local tracking/survival and official scoring may change for F10.',
        'Model sensitivity min/max is not a confidence or prediction interval; actual score can lie outside.',
        'Design/report 30 points are assumed, not yet awarded to a new submission.',
        'No independent training seed repeats; no new simulator evaluations or submissions.',
        'The MSE-floor model ceiling is a mathematical artifact, not an official competition ceiling.',
        'Legacy calibration was based on another evaluation convention. Reanchoring is exploratory and unvalidated.'
    ]
}
(BASE / 'predicted_scores.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
lines = [
    'Go2 조건부 환산표 — 2026-10-03 자료 / 2026-10-08 해석 정정', '',
    'F10_3000의 검증된 공식 예측점수는 없다. 기존 88.36을 주 예측으로 제시한 해석을 철회한다.',
    '88.36(MSE 보정)과 88.63(비율 보정)은 C0 공식88.14 하나에 맞춘 미검증 식의 출력이다.',
    f'MSE 보정식은 내부 오차가 모두0이어도 {QUALITATIVE + 70 * math.exp(-math.sqrt(c)):.5f}점이 상한이다.',
    '이는 후보의 성능 한계가 아니라 고정 잔여 감점 가정의 결과다. 90점 가능성 판정에 사용하지 않는다.',
    '아래 기존 계산값은 식의 민감도와 계산 이력을 확인하기 위해 그대로 보존했다.', '',
    '7개 시나리오 전체 평가를 마친 두 모델',
    '모델             무보정 자체총점    MSE보정 총점    비율보정 총점',
]
for name in ['C0_5500', 'F10_3000']:
    p = predictions[name]
    lines.append(f"{name:<16} {p['raw_self_total_assuming_30']:.2f}             {p['primary_total_assuming_30']:.2f}          {p['global_scale_total_assuming_30']:.2f}")
lines += [
    '', '모든 환산 총점은 설계20+리포트10=30점을 유지한다는 조건이다. F10의 MSE 보정식 출력은58.36/70이다.',
    '0.9167×100을 공식91.67점으로 부르지 않는다. 무보정 자체총점은30+70×0.9167106=94.17이다.',
    '무보정 식은 이미 공식88.14점인C0를93.63점으로 계산한다. 따라서94.17을 공식 예측으로 사용하면 과대평가한다.', '',
    '보정 방법',
    '이번 평가의 각 시나리오3축 MSE=(RMSE_vx²+RMSE_vy²+RMSE_yaw²)/3.',
    f'기존 MSE 보정식: 환산총점=30+70×Σ[w×자체생존율×exp(-sqrt(MSE+c))], c={c:.12f}.',
    'C0의 이번 측정치를 넣었을 때 시뮬58.14가 되도록c를 맞춘 후F10에 그대로 적용했다.',
    '보조 식은C0와F10의 내부 품질 비율을 공식시뮬58.14에 적용한다. 이것도 경험적 가정이다.',
    '시나리오별로 각각C0에 보정하면MSE식88.36, 비율식88.63으로 거의 같다.',
    '88.36~88.63은 선택한 보정식에 따른 차이이며 통계적 예측구간이 아니다. 실제 점수는 범위 밖일 수 있다.',
    'C0=88.14의 재현은 맞추어 놓은 결과이므로 예측 성공 사례로 계산하지 않는다.',
    '기존 10월 1일 c≈0.02685는 과거 측정 방식 기반이다. 이번 평가는 명령 관측 동기화, 종료 처리, 전체 시간 오차 집계가 달라 현재 C0로 다시 맞췄다.', '',
    '4개 시나리오만 측정한 800회 모델: 전체 예측점수를 만들지 않는다',
    '아래는 G1/G2/G5/G6 합계만의 미검증 조건부 환산이며 배점은 38.5점이다. G3/G4/G7의 31.5점은 미측정이다.',
    '모델          MSE보정 부분점수   비율보정 부분점수   낙상/256',
]
for name in ['control','action','acc','linear']:
    p=partial['short'][name]
    lines.append(f"{name:<12} {p['predicted_covered_points_mse_floor']:.2f}/38.5          {p['predicted_covered_points_scale']:.2f}/38.5         {p['falls']}")
lines += [
    '', '해석과 한계',
    'F10의 내부 지표는 일부 개선됐지만, 공식90까지 남은 격차를 이 식으로 정량화할 수 없다.',
    'MSE 보정식은 오차가 0이어도 90보다 낮은 상한을 갖는다. 이를 대회의 상한이나 90 달성 불가능 근거로 사용하지 않는다.',
    '공식 추종 함수, 공식 낙상, 실제 명령/seed와의 관계는 미식별이며 새 후보 공식 결과로 검증된 식이 아니다.',
    '독립 학습 seed 반복은 없다. 이번에는 기존 결과만 계산했으며 서버 시작·새 학습·공식 제출을 하지 않았다.', '',
    '2026-10-08 해석 정정',
    '88.36을 주 예측으로 부르거나, 그 수치로90까지1.64점 부족하다고 정책의 격차를 단정하지 않는다.',
    '추가 감사 자료: analysis/calibration_review_20261008/calibration_audit.json.',
    '공식 결과를 높여 말하려는 수정이 아니다. 새로운 공식 결과나 학습 결과는 없다.', '',
    '이전 설명 정정',
    '이전 답변의 배점 미확인 주장은 오래된 t_eval_go2_core.py 주석을 근거로 한 오류였다.',
    '10월 1일 및 2일 검증 기록에서는 Go2 배점 [0.15,0.15,0.20,0.15,0.15,0.10,0.10]과 70+20+10 구조를 확인했다.',
    '배점 확인과 세부 추종점수 예측식 검증은 별개다. 예측 불확실성은 여전히 남는다.', '',
    '재현: python predict_official_scale.py',
    '전체계산값: predicted_scores.json',
]
(BASE/'predicted_scores.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'full_predictions':predictions,'short_partial':partial['short']},indent=2))
