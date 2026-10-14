"""Analyze preserved Go2 measurements; no training or simulator execution."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/extracted/round20_20261001'
sys.path.insert(0, str(ROOT))
import worker

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

labels = ['original_A', 'F00_model_3000_A', 'F00_model_3999_A', 'F10_model_3000_A']
data = {label: read(ROOT / 'eval' / label / 'merged.json') for label in labels}
scores = {label: dict(**worker.proxy(value), falls=sum(s['falls'] for s in value['scenarios'].values()))
          for label, value in data.items()}
original = data['original_A']['scenarios']
selected = data['F10_model_3000_A']['scenarios']
late = {}
for path in (ROOT / 'eval/F10_model_3999_A').glob('*.json'):
    late.update(read(path)['scenarios'])

def contribution(scenario, weight):
    mse = sum(scenario['rmse'][a] ** 2 for a in ('vx', 'vy', 'yaw')) / 3
    return 70 * weight * scenario['survival'] * math.exp(-math.sqrt(mse + .02684614570005177))

rows = []
for sid, weight in worker.WEIGHTS.items():
    rows.append(dict(scenario=sid, description=selected[sid]['desc'],
                     original=original[sid]['rmse'], selected=selected[sid]['rmse'],
                     selected_floor_delta=contribution(selected[sid], weight) - contribution(original[sid], weight),
                     late=late.get(sid),
                     late_floor_delta_vs_selected=contribution(late[sid], weight) - contribution(selected[sid], weight)
                     if sid in late else None))

known = sum(contribution(s, worker.WEIGHTS[sid]) for sid, s in late.items())
# Valid bound within this proxy: survival <= 1, every squared error >= 0.
late_upper = known + sum(70 * w * math.exp(-math.sqrt(.02684614570005177))
                         for sid, w in worker.WEIGHTS.items() if sid not in late)
axes = {}
for axis in ('vx', 'vy', 'yaw'):
    b = sum(w * original[sid]['rmse'][axis] for sid, w in worker.WEIGHTS.items())
    c = sum(w * selected[sid]['rmse'][axis] for sid, w in worker.WEIGHTS.items())
    axes[axis] = dict(original_weighted_mean_rmse=b, selected_weighted_mean_rmse=c,
                      reduction_percent=100 * (1 - c / b))

trends = {}
for name in ('F00', 'F10'):
    blocks = re.split(r'Learning iteration\s+(\d+)/(\d+)',
                      (ROOT / f'logs/train_{name}.log').read_text(encoding='utf-8', errors='replace'))
    values = []
    for j in range(1, len(blocks), 3):
        iteration = int(blocks[j])
        if iteration not in (1000, 1200, 2000, 2500, 3000, 3500, 3999):
            continue
        row = dict(iteration=iteration)
        for label, key in [('Mean action std', 'action_std'), ('Mean reward', 'reward'),
                           ('Mean value loss', 'value_loss'), ('Curriculum/terrain_levels', 'terrain')]:
            match = re.search(re.escape(label) + r':\s*([-+\d.eE]+)', blocks[j+2])
            if match:
                row[key] = float(match.group(1))
        values.append(row)
    trends[name] = values

events = [json.loads(line) for line in (ROOT / 'driver.log').read_text(encoding='utf-8').splitlines()
          if line.startswith('{')]
training_seconds = {e['job']: e['seconds'] for e in events
                    if e.get('stage') == 'JOB_COMPLETE' and e.get('job', '').startswith('train_')}
suite_seconds = sum(e['seconds'] for e in events
                    if e.get('stage') == 'JOB_COMPLETE' and e.get('job', '').startswith('eval_original_A_'))
assert late_upper < scores['F10_model_3000_A']['floor']
assert late['G6']['survival'] < original['G6']['survival']
result = dict(scores=scores, axis_weighted_mean_rmse=axes, scenarios=rows,
              late_checkpoint_known_floor=known, late_checkpoint_maximum_possible_floor=late_upper,
              ranking_winner_within_proxy='F10_model_3000_A', ranking_proven_despite_missing_G7=True,
              late_checkpoint_survival_gate_failed_on_G6=True,
              training_seconds=training_seconds, observed_original_A_suite_seconds=suite_seconds,
              training_trends=trends, required_heldout_and_stress_and_video_complete=False,
              official_score_prediction_valid=False,
              input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT / 'eval').glob('*/*.json')})
(BASE / 'result_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
report = f'''Go2 round20 결과 분석

판정: F10 3000회 체크포인트는 개선 후보이며, 공식 정책 교체는 보류한다.
학습 2종은 성공적으로 완료됐다. 검증 중 Isaac Sim 시작 오류로 B·추가 stress·영상이 생성되지 않았다.

완료된 A 평가 (공식 점수가 아닌 근사 지표)
기존 C0 5500: floor 58.128491 / ratio 58.079139 / 낙상 1
F00 3000: floor 57.519588 / ratio 57.112037 / 낙상 16
F00 3999: floor 57.670073 / ratio 57.208512 / 낙상 2
F10 3000: floor 58.419307 / ratio 58.752940 / 낙상 0
F10 3000 대 기존: floor +0.290816 / ratio +0.673801
각 시나리오 128환경 × 4평가시드 = 512 episode 기록. 하나의 학습 시드로 훈련했으며 독립 학습 반복은 아니다.
F00/F10은 각각 4000회, 기존 C0는 5500회이므로 같은 학습 길이의 대조 실험은 F00 대 F10이다.

어디서 개선됐나
시나리오 가중평균 RMSE 기준 vx {axes['vx']['reduction_percent']:.2f}% 감소,
vy {axes['vy']['reduction_percent']:.2f}% 감소, yaw {axes['yaw']['reduction_percent']:.2f}% 감소.
이는 시나리오별 RMSE의 가중평균이며 전체 샘플을 합쳐 계산한 RMSE와 다르다.
회전 오차는 G1~G7 모두 감소. 전방위·험지·경사에서 개선이 뚜렷하다.
계단(G5) vx는 0.0705→0.0791로 12.2% 증가, 밀침(G6) vx는 0.0947→0.1015로 7.2% 증가했다.
회전 보상 증가는 전반적으로 유망하지만 계단·밀침 선속도에서 교환 관계가 관측된다.
일반 A의 G6 밀침 시험은 완료됐고 생존율 1.0이다. 아직 없는 것은 추가 경계 스트레스·급전환 시험이다.

3999회 최종 모델보다 3000회가 확실히 낫다 (현재 proxy 안에서)
완료된 G1~G6 모두 최종 모델의 floor 기여도가 3000회보다 낮다.
최종 모델 G6에서는 512대 중 1대가 넘어져 기존/3000회의 생존율 1.0보다 낮다.
누락된 G7을 생존율 1.0, 모든 추종 오차 0으로 가정한 이론상 최고치를 더해도
최종 모델 floor 상한은 {late_upper:.6f}으로 3000회 58.419307보다 낮다.
따라서 승자 선정만을 위해 실패한 최종 모델 G7을 다시 평가할 필요는 없다.
이 상한은 공식 평가 점수의 상한이 아니며 오직 현재 floor 계산식의 수학적 상한이다.

학습 종료 및 자동 best 파일 해석
F10 action std는 3000회 0.90→3999회 1.04, 평균 학습 보상은 28.34→23.06이었다.
동시에 측정된 평가 성능도 후퇴했다. 원인(과적합/탐색 증가 등)은 이 기록만으로 확정할 수 없다.
F00와 F10의 보상 크기는 보상식이 달라 직접 성능 비교에 쓰면 안 된다.
제공 자동 model_best.pt는 학습 평균 보상의 최대값에 따라 1200회 모델을 선택했다.
1200회 모델은 이번 A 비교에서 평가되지 않았다. 현재 검증 대기 후보는 보존된 model_3000.pt와 그 env.yaml이다.

경제성
F00 학습 {training_seconds['train_F00']/3600:.3f}시간, F10 학습 {training_seconds['train_F10']/3600:.3f}시간.
이번 서버 사용 11시간57분 중 학습 2종이 약10시간21분을 사용했다.
단독6144환경 GPU90.2%; 동시 평가 시 GPU94.94%지만 iteration은5.228→6.406초(+22.5%)로 느려졌다.
동시 평가를 중단하고 순차 진행한 판단은 이번 실측에 부합한다. 짧은 초기 측정이므로 모든 상황의 벤치마크는 아니다.

다음 우선순위 (이번 분석에서는 서버를 다시 시작하지 않았다)
1. 새 학습보다 기존 C0와 F10 3000의 B 검증을 먼저 완료한다.
2. B가 통과하면 양 모델의 계단10/15cm, 급전환/밀침 추가 스트레스 및 후보 영상 생성·검토.
3. 통과 전에는 기존 공식 정책을 유지한다. 1200/2000회 탐색은 필요한 경우 다음 연구로 분리한다.
A 한 세트 실측 {suite_seconds/60:.2f}분이므로 B 두 세트는 비슷한 속도에서 약17~20분이다.
추가 스트레스·영상·회수까지 약40~60분을 계획치로 잡되, 영상 실행 시간은 미측정이라 보장된 ETA가 아니다.
90점에 필요한 공식 시뮬레이션 개선 +1.86은 아직 입증되지 않았다. 두 proxy의 +0.291/+0.674는 공식 상승량이나 신뢰구간이 아니다.
'''
(BASE / 'result_analysis.txt').write_text(report, encoding='utf-8')
print(json.dumps({k:result[k] for k in ['axis_weighted_mean_rmse', 'late_checkpoint_maximum_possible_floor',
                                      'ranking_proven_despite_missing_G7', 'observed_original_A_suite_seconds']}, indent=2))
