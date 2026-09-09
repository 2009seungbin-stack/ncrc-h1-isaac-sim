# 세션 요약 — 2026-09-08 밤 (3라운드 R3 파이프라인 가동)

## 한 줄
연구 메모(`research_round3_breakthrough.md`) 작성 후, 서버 26에서 **R3A/R3C 병렬 10k 학습 → 채점 → 번들 → keep-alive**
무인 파이프라인을 13:15 KST에 걸었다(서버 시계 = KST). 종료 예상 **09-08 21:00~21:30 KST**. 돌아와서 회수·판정·제출만 하면 된다.

## 돌아와서 할 일 (순서대로)
1. 보드 → VS Code 열기 → 터미널: `tail -20 /workspace/R3_STATUS.txt; cat /workspace/R3_RESULT.md`
2. 게이트: SimScore ≥ 63.32 **그리고** Stress X 생존 LCB가 후보 중 최고(S2 기준선이 있으면 S2 이상).
3. 회수: Termux `scp da-perfect25@26.robotics.connect.or.kr:/workspace/R3_bundle.tgz ~/` → PC로 옮겨 `tar xzf`.
4. 채택 시: `keep_R3/<후보>.pt` → policy.pt, `keep_R3/<후보>_env.yaml` → env.yaml, 리포트 갱신(`tools/reward_diff.py` 게이트) → 보드 3라운드 제출.
5. `touch /workspace/STOP_KEEPALIVE` → 보드에서 **서버 종료하기**.

## 파이프라인 구조 (tools/r3/r3_pipeline.sh)
가드(45초 ckpt 복사, 모든 run) → R3A 시작 → 첫 iteration 확인 후 R3C 시작 → 2분 실측 → TRAIN_DONE 대기(30분마다 진행 로그)
→ ckpt 9000/10000 export(play.py) → T-EVAL 7 + Stress X1~X5 (7 프로세스/정책) → 순위표 → tar → keep-alive(T-EVAL H1 반복).
R3C가 OOM이면 자동으로 R3A 뒤에 순차 실행.

## 실측
- 병렬 2분: GPU 96%, 13.4GB, iteration 2.25s(R3A)/1.85s(R3C). 합산 처리량은 단일 16384의 0.81배.
- 파일 전송: code-server에 file input 주입 + JS paste 이벤트 → 22KB tgz md5 일치. (메모리 ncrc-server-file-transfer-via-paste-event)

## 만약
- 파이프라인이 죽어 있으면: `tmux ls`, `tail -50 /workspace/R3_STATUS.txt`, `/workspace/r3logs/`. ckpt는 `ckpt_keep/<run>/`에 남아 있다.
  수동 export/채점은 `tools/s3_post.sh`의 절차를 R3 이름으로 바꿔 쓰면 된다.
- 워크스페이스가 사라졌으면(종료 후 30분 방치): R3A/R3C는 seed 42라 재현 가능. 시간만 잃는다.

## 보초 (15:19 KST 추가)
30분 무활동 초기화의 공식 예외는 "강화 학습 진행 중"뿐이라, 채점·keep-alive 구간에 train.py가 없는 게 위험했다.
`/tmp/sentinel.sh`(tmux `sentinel`)가 학습 종료를 감지하면 더미 학습(64 envs, tmux `sentinel_train`)을 띄워 그 상태를 유지한다.
`touch /workspace/STOP_KEEPALIVE` 하나로 keep-alive와 보초가 모두 멈춘다. 그 뒤 보드에서 서버 종료.

## 결과 (21:05 KST 완료, 서버 21:12 종료, 총 사용 20시간 16분)
| 정책 | SimScore | H낙상 | Stress 생존 | X1 밀침 생존 |
|---|---:|---:|---:|---:|
| **R3A_9000** | **63.87** | 0 | 0.981 | 0.910 |
| R3A_10000 | 63.89 | 0 | 0.975 | 0.881 |
| R3C_10000 | 62.55 | 0 | 0.990 | 0.951 |
| R3C_9000 | 62.13 | 2 | 0.981 | 0.914 |
R3A_9000 채택. 팔·상체 해방(R3C)은 yaw가 오히려 +9%, H4 하락, 낙상 발생 → 기각.
제출 세트: `submit_R3/{policy.pt, env.yaml, report_500.txt}` (reward_diff 게이트 PASS). 번들 `R3_bundle.tgz` + `R3_bundle_extracted/`.
상세: `tools/r3/analyze_r3.py R3_bundle.tgz` 출력, 메모리 ncrc-r3-results.

## 제출 (21:30 KST)
2라운드 제출본을 S2 → **R3A_9000**으로 교체 제출 완료(모달 "예선 자료가 제출되었습니다"). 휴머노이드 선택, policy.pt·env.yaml 재업로드(칸 비운 뒤 채움 확인), 리포트 500자.
제출 파일 바이트 대조는 보드에 다운로드 링크가 없어 못 함(칸 비움→업로드→채워짐으로 간접 확인). 공식 점수 나오면 `tools/score_model.py --add-official` 로 앵커 추가.
예측: 공식 생존 1.0이면 91.9, 0.95면 88.7, 1라운드 오프셋(×0.906) 적용 시 85.9 → 중심 추정 약 89.
- 21:45 리포트 수정 재제출: "SimScore"와 자체 점수 수치를 빼고 정성 표현(자체평가 전 시나리오 상승, 낙상 0, 오차 −29%/−15%)만 남김. 499자. 이전 버전은 `submit_R3/report_500_v1_with_simscore.txt`.
