# H1 Competition Weight Benchmark

공식 대회 evaluator는 서버에서 찾지 못했다. 아래 점수는 공식 점수가 아니라 같은 조건으로 비교하기 위한 대체 점수다.

대체 점수 = 생존율 × exp(-전체 속도 추종 RMSE)

## 1. Winner

최종 우승자는 현재 대회용 H1 학습 가중치 iteration 4600이다.

- 생존율: 0.890625
- 추종 점수: 0.634862
- 대체 점수: 0.566122
- 낙상: 7회

NVIDIA Rough와 HaoZi Rough가 속도는 조금 더 정확히 따라갔지만, 생존율이 더 낮았다. 대회 점수는 생존율과 추종 점수를 곱하므로 iteration 4600이 최종 점수에서 앞섰다.

## 2. Leaderboard

|순위|정책|생존율|추종 점수|대체 점수|낙상|호환성|
|---:|---|---:|---:|---:|---:|---|
|1|Competition H1 final, iteration 4600|0.890625|0.634862|0.566122|7|원래 환경|
|2|NVIDIA Isaac Lab H1 Rough|0.734375|0.748334|0.550468|17|최소 변환|
|3|HaoZi H1 Rough model_2999.pt|0.718750|0.747892|0.537548|24|최소 변환|
|4|HaoZi H1 Flat model_999.pt|0.250000|0.685393|0.171348|81|최소 변환|
|5|NVIDIA Isaac Lab H1 Flat|0.203125|0.727031|0.147607|71|최소 변환|
|6|Hazzzard velocity tracking|0.000000|0.377900|0.000000|388|최소 변환|
|7|Unitree motion.pt|0.000000|-|0.000000|-|하체 중심 구조|
|-|HOMIERL H1|-|-|-|-|PORTING_REQUIRED|

## 3. Why the winner won

iteration 4600은 가장 빠른 정책이 아니라 가장 오래 버틴 정책이다. 4개 seed와 같은 10단계 명령 순서에서 생존율 89.1%를 기록했다. NVIDIA Rough와 HaoZi Rough는 추종 점수가 약 0.748로 더 높았지만, 낙상 때문에 생존율이 각각 73.4%, 71.9%로 내려갔다. 곱셈 점수에서는 작은 추종 이득보다 큰 생존 손실이 더 크게 작용했다.

## 4. Best survival policy

Competition H1 final iteration 4600이다. 생존율 0.890625로 비교 가능한 정책 중 가장 높았다.

공개 저장소에서 말하는 100% 생존율은 같은 대회 실험 결과가 아니다. HaoZi의 100%는 저장소 자체 지형 단계 실험에서 나온 값이다. 이번 NCRC 명령 순서에는 좌우 이동, 반대 회전, 복합 명령, 급변 명령이 들어가므로 직접 비교하면 안 된다.

## 5. Best tracking policy

NVIDIA Isaac Lab H1 Rough가 추종 점수 0.748334로 가장 높았다. HaoZi Rough가 0.747892로 거의 같았다.

HaoZi Rough의 세부 RMSE:

- vx RMSE: 0.265118
- vy RMSE: 0.237838
- yaw RMSE: 0.355401
- 전체 RMSE: 0.290496
- 평균 절대 오차: 0.143021

## 6. Best overall competition policy

현재 기준에서는 Competition H1 final iteration 4600을 제출 후보로 유지한다. 공식 evaluator가 없는 상태이므로 0.566122는 공식 점수가 아니다. 다만 모든 정책을 같은 환경, 같은 명령, 같은 seed 수로 비교했을 때 가장 높은 대체 점수다.

## 7. Calibration results

순위표에는 command calibration을 적용하지 않은 원본 결과만 넣었다. 원본 결과를 덮어쓰지 않았다.

HaoZi Rough는 전진과 회전은 잘 따라갔지만, 좌우 명령에서 문제가 뚜렷했다. vy=+0.3과 vy=-0.3을 넣었을 때 실제 vy는 거의 0이었고, 대신 전진 속도가 생겼다. 단순 배율 조정만으로는 축이 섞이는 문제를 해결하기 어렵다. 다음 calibration은 vx, vy, yaw를 따로 조정하고 좌우 명령에 대한 2차원 보정표를 별도로 검증해야 한다.

## 8. Failure analysis

- NVIDIA Rough: 추종은 가장 좋았지만 급변 명령까지 가는 동안 생존율이 73.4%로 내려갔다.
- HaoZi Rough: 전진 0.3, 0.6, 1.0과 yaw ±0.5는 잘 따라갔다. 좌우 이동을 거의 만들지 못했고 복합·급변 구간에서 생존율이 더 내려갔다.
- HaoZi Flat: 출력은 정상이고 NaN도 없었지만 생존율이 25.0%였다. 평지 학습 조건과 현재 대회 환경의 차이가 컸다.
- Hazzzard: 유한한 action은 만들었지만 recurrent 상태와 관측 차이로 불안정했고 생존율이 0%였다.
- Unitree motion.pt: 하체 중심 정책 구조와 현재 19-action H1 인터페이스 차이가 컸다. 억지 zero padding은 하지 않았다.
- HOMIERL: 가중치 네 개는 로드됐지만 학습 때 사용한 reference motion 입력이 현재 환경에 없다. 큰 재구현이 필요해 PORTING_REQUIRED로 분류했다.
- 모든 완료 평가에서 simulation crash와 NaN action은 0회였다.

## 9. Recommended next step

다음 GPU 시간은 새 가중치를 더 찾는 것보다 우승자 iteration 4600을 이어서 30분 PPO 학습하는 데 쓰는 것이 점수 상승 가능성이 가장 높다.

우선순위는 다음과 같다.

1. 현재의 강한 생존 성향과 termination penalty -200을 유지한다.
2. yaw 추종 보상을 한 번에 하나의 값만 바꾸며 높인다.
3. negative lateral과 좌우 대칭 명령을 curriculum에 더 자주 넣는다.
4. 급변 명령 직후 0.5~1.0초 구간의 추종 보상을 강화한다.
5. 매 변경마다 같은 4-seed 10단계 benchmark를 다시 돌린다.

사이트 제출용 짧은 기술 개선 문장:

종료 보상을 -200으로 강화해 먼저 넘어짐을 줄였다. 그 결과 4개 seed에서 생존율 89.1%를 기록했고, 다음에는 회전과 반대쪽 이동 명령을 더 자주 학습해 추종 오차를 줄일 계획이다.

## 10. Added candidate: HaoZiOwO H1 release

Source: HaoZiOwO/rl-h1-locomotion, release h1-checkpoints-v1.0.0.

- Rough model_2999.pt: 256 observations, 19 actions
- Flat model_999.pt: 69 observations, 19 actions
- Rough: survival 0.718750, tracking 0.747892, proxy 0.537548, falls 24
- Flat: survival 0.250000, tracking 0.685393, proxy 0.171348, falls 81
- 두 정책 모두 NaN action 0회, simulation crash 0회
- 원본 weight는 그대로 보존했고 adapter용 state dictionary 복사본만 별도로 만들었다.

HaoZi 저장소가 공개한 100% 생존 결과는 자체 terrain sweep 기준이다. 이번 NCRC 명령 순서에서는 lateral, combined, rapid transition에서 생존율이 낮아졌다. 따라서 이 가중치가 무조건 100% 생존한다고 결론낼 수 없으며, 현재 우승자는 iteration 4600이다.
