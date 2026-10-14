from pathlib import Path
import json,datetime,hashlib
B=Path(__file__).resolve().parent; R=B/'stress_results/extracted/go2_stress_20261002'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
v=R/'candidate/videos/play/rl-video-step-0.mp4'
video=dict(reviewed_utc=now,duration_s=19.98,decoded_frames=999,width=1920,height=1080,fps=50,
    sample_interval_s=.2,samples_reviewed=100,visual_review_completed=True,
    observation='Visible robot walks and changes heading; no obvious collapse in sampled frames. Camera follows one visible robot.',
    limitation='Provided play is a qualitative replay, not a controlled GX1/GX5 stair comparison; does not prove all four environments survived or official score.',
    video=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest(),policy_equivalence=load(B/'video_policy_verification.json'))
assert video['policy_equivalence']['verified']
save(B/'video_review.json',video)
s=load(B/'stress_review.json');s.update(video_visual_review_complete=True,video_review=video);save(B/'stress_review.json',s)
b=load(B/'validation_review.json');assert b['B_complete'] and b['B_gate_passed'] is False
b.update(stress_complete=True,stress_survival_nonregression=s['survival_nonregression'],stress_comparison=s['comparisons'],videos=[str(v)],video_visual_review_complete=True,
    outcome='FULL_REVIEW_COMPLETE_KEEP_EXISTING_POLICY_B_SURVIVAL_GATE_REJECTED',official_submission=False);save(B/'validation_review.json',b)
lines=['NCRC Go2 최종 비교 — 2026-10-02',
    '결론: F10 3000은 회전 추종과 15cm 계단 생존을 개선했으나, B 밀침 생존 비회귀 게이트에 미달했다. 기존 C0 공식 정책을 유지한다.',
    '추가 학습과 공식 제출은 하지 않았다. 90점 달성은 입증되지 않았다.', '',
    'A: 기존 floor 58.128491 → 후보 58.419307 (+0.290816).',
    'B: 기존 floor 58.126598 → 후보 58.405983 (+0.279385). ratio 58.081804 → 58.740280 (+0.658476).',
    '두 채점 가정에서 점수 개선 방향은 재현됐다. 위 값은 내부 proxy이며 공식 점수나 공식 상승량 예측이 아니다.',
    'B 계단 G5 생존 511/512 →512/512, 밀침 G6 512/512 →511/512. 두 모델 총 낙상은 각각 1건.',
    'floor/ratio 각각 +0.15 점 조건은 통과했지만 모든 시나리오의 생존 비회귀 조건은 미달했다.',
    '1건 차이로 전체 생존성이 통계적으로 나빠졌다고 확정하지 않는다. 사전 고정한 보수적인 교체 기준에 따라 보류한다.', '',
    '추가 비교: seeds137/389/743/1201, 128 env, 각 조건512episode. 학습 없이 보존된 C0 5500과 F10 3000 비교.',
    'GX1 강한 밀침: 평면 속도 성분에 [-2,2]m/s 난수 증분, 1.5초 간격. GX5: 2초 명령 급전환 10구간.',
    'GX1/GX5는 transient=0; 계단 G5는 transient=0.5초. 같은 조건 내 모델만 직접 비교한다.',
    '계단 평가지형 생성 seed1234 고정; 평가 리셋 시드4개를 사용했다. 평가 시드는 독립 학습 반복이 아니다.']
for x in s['comparisons']:
    a,c=x['original'],x['selected']; lines.append(f"{x['test']} {x['scenario']}: 생존 {a['survived']}/512 →{c['survived']}/512; RMSE vx {a['rmse']['vx']:.4f}→{c['rmse']['vx']:.4f}, vy {a['rmse']['vy']:.4f}→{c['rmse']['vy']:.4f}, yaw {a['rmse']['yaw']:.4f}→{c['rmse']['yaw']:.4f}.")
lines+=['추가 stress 생존 비회귀는 통과했다. 15cm 낙상10→3건은 관찰상 개선이지만 B 실패를 사후에 지우거나 교체 게이트를 변경하지 않는다.',
    '강한 밀침과 급전환은 회전 오차 감소와 선속도 오차 증가를 함께 보였다. 15cm 계단도 생존은 좋아졌지만 vx RMSE 26.2% 증가.',
    '다음 연구 가설은 회전 보상 증가가 자세/회전 제어와 선속도 추종에 다른 영향을 준다는 것이다. 새 실험은 새 평가 시드를 사전 고정하고 별도 예산 승인 범위에서만 진행한다.', '',
    '영상과 내보낸 정책',
    '제공 play.py, 4env, video_length1000, headless, enable_cameras로 19.98초 H264 1920x1080/50fps 영상 생성. 999프레임 전부 디코딩 성공.',
    '0.2초 간격 100프레임을 5개 시트로 전체 구간 검토. 보이는 로봇의 보행과 방향 전환을 확인했고 뚜렷한 쓰러짐은 관찰되지 않았다.',
    '기본 카메라는 한 로봇을 따라가므로 4대 전부 무낙상이나 통제된 stress 영상 비교라고 주장하지 않는다. 정량 stress 판정은 별도 평가 JSON을 따른다.',
    'play 래퍼의 후처리는 staged export 경로를 찾지만 명시된 checkpoint의 실제 정책/영상은 candidate 아래 생성됐다. 그 원본을 압축에 보존해 검증했다.',
    'candidate/exported/policy.pt를 4096개 난수 관측에서 checkpoint actor와 비교: 최대 절대오차 5.722046e-6, iter3000 일치.', '',
    '회수/종료/예산',
    'stress 결과 28586565bytes, 54member, 6개 merged JSON 및 원본/후보 모델·설정·JIT·영상 회수.',
    'SHA256 f25a2925be630b51cb45d5cda69f6bd365d960e9877226d596b6e2fea19f5cc0; 원격-로컬 해시 일치, 모델/설정/보상 복원 검증.',
    '보호된 train.py/play.py/agent_cfg.py/env_cfg.py 해시는 실행기 finally에서 일치 확인했고 원본 reward는 복원했다.',
    '입력/결과 전송은 VSCode 파일 UI로 처리했다. 다운로드 이벤트 timeout과 별개로 실제 파일 존재와 전체 해시를 확인했다.',
    '27번 서버 종료 후 보드 사용가능/서버 시작/VSCode 비활성 확인. 누적61시간18분, 이번라운드12시간42분, 20시간 중7시간18분 미사용.',
    '이번 추가 비교17분. 절대 종료09:31:37과 보호 회수09:01:37 이전에 종료했다.',
    '기존 공식 Go2 88.14점 제출은 보존했다. 공식 제출은 수행하지 않았다.',
    '기존 heartbeat ncrc-20은 이미 삭제되어 있다. 모든 요청한 검증/회수/서버종료/로컬검토/최종보고서 저장을 완료했다.',
    '사용자 요청에 따라 마지막 동작으로 로컬 shutdown.exe /s /t 0을 요청한다. /f를 사용하지 않으며 실제 전원 꺼짐 확인은 별도 상태로 남긴다.']
(B/'final_comparison_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
# Preserve the earlier B-only report; final report is the combined review.
d=load(B/'execution.json');d.update(status='ALL_WORK_COMPLETE_PENDING_LOCAL_SHUTDOWN',video_validation_complete=True,
    local_results_review_completed=True,requested_full_validation_completed=True,stress_validation_complete=True,
    final_report_saved=True,final_review_saved=True,final_report=str(B/'final_comparison_review.txt'),
    local_shutdown_eligible=True,local_shutdown_blocker=None,cleanup_completed=True,active_job=None,
    completion_utc=now,stress_remote_shutdown_evidence=str(B/'stress_server_shutdown_confirmed.jpg'))
required=dict(results_retrieved_and_verified=d['results_verified'] and d['validation_results_verified'] and d['stress_results_verified'],
    remote_server_shutdown_confirmed=d['shutdown_completed'],heldout_and_stress_results_reviewed=d['heldout_validation_complete'] and s['complete'],
    video_reviewed=video['visual_review_completed'],final_report_saved=(B/'final_comparison_review.txt').exists(),heartbeat_stopped=d['heartbeat_deleted'])
assert all(required.values()),required
d['completion_conditions_verified']=required;save(B/'execution.json',d)
save(B/'completion_record.json',dict(completed_utc=now,conditions=required,policy_replacement=False,
    official_submission=False,remote_shutdown_confirmed=True,local_shutdown_requested=False,local_power_off_verified=False))
(B/'user_notification.txt').write_text('비교·영상 검토·회수·원격 서버 종료·최종 보고서 저장 완료. F10 회전/15cm 계단 생존 개선, B 밀침 생존 게이트 미달로 기존 정책 유지. 누적61시간18분, 이번12시간42분. 이전 사용자 요청에 따라 지금 로컬 PC 종료를 요청합니다. 실제 전원 꺼짐은 확인하지 못했습니다.\n',encoding='utf8')
print(json.dumps(required,indent=2));print(B/'final_comparison_review.txt')
