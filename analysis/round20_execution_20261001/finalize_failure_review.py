"""Review retrieved round20 evidence without starting Isaac Sim or training."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/extracted/round20_20261001'
sys.path.insert(0, str(ROOT))
import worker

labels = ['original_A', 'F00_model_3000_A', 'F00_model_3999_A', 'F10_model_3000_A']
data = {label: json.loads((ROOT / 'eval' / label / 'merged.json').read_text(encoding='utf-8')) for label in labels}
scores = {label: worker.proxy(value) for label, value in data.items()}
base = scores['original_A']
candidate = scores['F10_model_3000_A']
differences = {k: candidate[k] - base[k] for k in candidate}
checked = {}
for path in (ROOT / 'eval').glob('*/*.json'):
    value = json.loads(path.read_text(encoding='utf-8'))
    assert value['scenarios']
    assert all(s.get('nan_actions', 0) == 0 and s.get('crashes', 0) == 0 for s in value['scenarios'].values())
    checked[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

model = next((ROOT / 'preserved/F10').glob('*/model_3000.pt'))
review = BASE / 'review_candidate_pending_validation'
review.mkdir(exist_ok=True)
shutil.copy2(model, review / 'model_candidate.pt')
shutil.copy2(model.parent / 'params/env.yaml', review / 'env.yaml')

missing = ['F10_model_3999_A/dr.json', 'original_B', 'selected_B',
           'original_stress', 'selected_stress', '10cm/15cm stair stress', 'candidate video']
assert not list((ROOT / 'eval').glob('*B*'))
assert not list((ROOT / 'eval').glob('*stress*'))
assert not list(ROOT.rglob('*.mp4'))
result = dict(
    reviewed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    outcome='SAFE_FAILURE_RECOVERY_KEEP_EXISTING_POLICY',
    training_completed=True,
    screening_complete=False,
    scores=scores,
    candidate_minus_original=differences,
    completed_candidate_A_gate=worker.passes(data['F10_model_3000_A'], data['original_A']),
    accepted_for_submission=False,
    missing_validation=missing,
    evaluation_hashes=checked,
    candidate_checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    official_score=88.14,
    official_score_updated=False,
    local_pc_shutdown_eligible=False,
    local_pc_shutdown_blocker='Required B, stress and video validation did not complete.',
)
(BASE / 'final_review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

report = f'''NCRC Go2 20시간 실험 최종 회수·검토 기록

결론
두 학습은 완료됐지만 검증 시뮬레이터 시작 중 오류로 이번 실험은 검증 미완료 상태다.
기존 공식 제출 정책을 유지한다. 새 정책을 공식 제출하지 않았다.
원격 결과를 모두 회수하고 압축파일과 필수 파일을 검증한 뒤 서버를 실제 종료했다.
B·stress·영상 검증이 생성되지 않아 성공 또는 90점 달성을 주장하지 않는다.

학습과 오류
F00/F10 각각 seed42, 6144환경, 4000회 scratch 학습 완료. 중간 및 최종 체크포인트와 설정 보존.
F10 변경은 회전 추종 보상 0.75→1.5 하나뿐이다. 추가 학습, 재개, 동시 평가를 수행하지 않았다.
F10 3999회 모델의 A/G7 평가 시작 시 Isaac Sim이 uptime 0초에서 segmentation fault로 종료됐다.
backtrace에 telemetry plugin이 나타나지만 근본 원인은 확정하지 않았다.
이 실패는 모델의 점수나 NaN을 판정하는 평가 결과가 아니다. worker가 실패를 기록하고 원래 보상을 복원했다.

완료된 A 평가의 근사 지표 (공식 점수가 아님)
기존 C0 5500: floor {base['floor']:.6f}, ratio {base['ratio']:.6f}
F00 3000: floor {scores['F00_model_3000_A']['floor']:.6f}
F00 3999: floor {scores['F00_model_3999_A']['floor']:.6f}
F10 3000: floor {candidate['floor']:.6f}, ratio {candidate['ratio']:.6f}
F10 3000 대 기존: floor {differences['floor']:+.6f}, ratio {differences['ratio']:+.6f}
F10 3000은 A의 모든 시나리오 생존율 1.0이며 낙상 0이다.
회전 오차는 모든 시나리오에서 감소했다. G6 선속도·횡속도와 G5 선속도·횡속도 오차는 일부 증가했다.
A 단독 gate는 통과하지만 다른 시드 B 및 스트레스·영상의 검증이 없어 교체 근거로 충분하지 않다.
학습 시드는 하나이고 평가 시드 변경은 독립 학습 반복 실험이 아니다.

미완료 사항 검토
F10 3999 A/G7 결과, B 평가 2종, 밀침·급전환, 계단 10/15cm 스트레스, 영상이 없다.
파일 부재를 로컬에서 확인했다. 미실시를 통과 또는 검토 완료로 처리하지 않았다.
잠정 후보 F10 3000의 원래 모델과 자신의 env.yaml은 review_candidate_pending_validation에 복사했다.
이는 재검증용 학습 체크포인트이며 제출 가능한 JIT 정책으로 확정한 파일이 아니다.

회수와 사용 시간
전체 압축파일 566387722 bytes. 브라우저 편집기에서 17개 base64 조각을 회수해 조각별 해시를 확인했다.
재조합 압축파일 SHA-256: b66790910954f7661c5c73a8fb21f4dca8c75ecd9b5a79ae8c7908ea19282d1f
202개 archive member를 안전하게 추출했다. 최종 모델 2종의 해시·설정, 원래 모델·설정, 학습 로그, 평가 파일 28개 확인.
운영보드의 실제 종료 후 표시: 서버 사용가능, 서버 시작하기, VS Code 비활성.
누적 사용 60시간33분. 시작 전48시간36분 대비 이번11시간57분. 이번20시간 상한 대비8시간03분 미사용.
사용 시간은 운영보드의 분 단위 표시이며 보수적인 로컬 경과 시간과 차이가 있을 수 있다.
공식 Go2 점수는 기존88.14점(시뮬레이션58.14, 설계20, 리포트10) 그대로다.

컴퓨터 종료 조건
사용자의 작업 완료 후 PC 종료 요청은 보존했다. 그러나 필수 검증·영상이 미완료라 자동 종료 조건이 충족되지 않았다.
PC 종료 또는 고정 시각 종료 예약을 실행하지 않았다. 원격 서버 종료 완료와 로컬 PC 전원 종료를 구분한다.
서버 점검 heartbeat는 원격 서버 종료와 회수 완료로 더 이상 필요하지 않아 종료한다.
'''
(BASE / 'final_review.txt').write_text(report, encoding='utf-8')
ledger = json.loads((BASE / 'execution.json').read_text(encoding='utf-8'))
ledger.update(status='REMOTE_CLOSED_VALIDATION_INCOMPLETE',
              local_results_review_completed=True,
              missing_validation_reviewed=True,
              heldout_validation_complete=False,
              stress_validation_complete=False,
              video_validation_complete=False,
              final_report_saved=True,
              final_report=str(BASE / 'final_review.txt'),
              local_shutdown_eligible=False,
              local_shutdown_requested=False,
              local_shutdown_blocker=result['local_pc_shutdown_blocker'],
              last_heartbeat_result='RECOVERED_REMOTE_STOPPED_VALIDATION_INCOMPLETE')
(BASE / 'execution.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
print(json.dumps(dict(outcome=result['outcome'], candidate_delta=differences,
                      eval_files_verified=len(checked), local_shutdown_eligible=False)))
