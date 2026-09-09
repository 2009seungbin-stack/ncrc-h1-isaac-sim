# 세션 요약 — 2026-09-08 새벽 (마감 09-09 09:00)

## 한 줄
S2를 예선 2라운드에 **제출 완료**. S3(action_rate 실험)는 유망하나 미확인이라 **미제출**.
서버 **종료 확인**. 이 문서 + 메모리 + `s3_decision.md` 로 다음 세션 이어가면 됨.

## 완료
1. **S2 제출** (보드 → 휴머노이드 → policy.pt+env.yaml+481자 리포트 → 제출하기).
   보드가 "예선 2라운드 참가 중"으로 전환됨. 심사 전까지 수정 가능.
   제출물: `submit_S2/keep_S2/policy.pt` (md5 6e0b04a1…), `submit_S2/keep_S2/env.yaml`,
   리포트 원문 `submit_S2/report_500.txt`.
2. **S3 실험**: baseline+3개(ang2.0/term-200/action_rate-0.005), 16384×3000, seed42.
   run `2026-09-08_00-35-01`. 결과·판정은 `s3_decision.md`, 메모리 ncrc-s3-action-rate-finding.
   요지: 동일 지형서 yaw -24%·낙상 -9% 우수, 최대 지형 0.26 낮음, T-EVAL 미확인 → S2 유지.
3. **서버 종료 확인**: 배지 "사용가능", 사용량 12시간13분/100에서 정지. 워크스페이스 초기화됨
   (S3_bundle.tgz 소멸 — S3는 seed42로 재현 가능).

## 왜 t_eval.py를 서버에서 못 돌렸나
이 PC→서버 scp가 22포트 차단. code-server에 파일 입력 요소 없어 업로드 불가.
21KB base64 수기 붙여넣기는 긴 청크에서 전사 오류 발생. → 학습 지표(로그) 비교로 대체.
**다음 세션 해법**: Termux scp로 t_eval.py(또는 tools/t_eval_paste.sh) 서버 전송 후 채점.

## 다음 세션 (S3 확인 → 유리하면 3라운드 제출)
1. 서버 시작 → S3 재현: `set_reward` 또는 sed로 ang2.0/term-200/action_rate-0.005,
   16384×3000 scratch(≈85분). 학습 중 ckpt 가드 필수(POST가 초반 ckpt 오선택).
2. Termux로 t_eval.py 올려 7시나리오 채점. S2(62.32)와 대조.
3. S3 SimScore > 62.32 → 3라운드에 S3 제출(보드 동일 절차). 아니면 S2 유지.
4. (레버) Go2 신규 제출 검토 — 합산 200점 구조상 GPU 시간당 가치 높음.

## 파일
- `s3_decision.md` — S3 판정 프레임 + S2 궤적 + S3 최종결과 (상세)
- `research_plan_final_round2.md` — 이번 세션 계획
- `submit_S2/` — 제출한 S2 산출물 + 500자 리포트
- 메모리: ncrc-round2-submitted-s2, ncrc-s3-action-rate-finding, ncrc-status-page-is-placeholder
