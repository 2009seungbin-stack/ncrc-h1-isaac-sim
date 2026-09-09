# H1 기술 개선 연구 보고서 — 로컬 학습 정책만

## 0. 결론

이 문서는 외부 pretrained weight의 성능을 개선 성과로 포함하지 않는다. 세 실험은 모두 `seed: 42`, `resume: false`로 시작한 독립 scratch training이다. 초기 기준 설정과 최종 설정의 전체 YAML을 대조했을 때 보상 정의에서 바뀐 것은 `termination_penalty.weight: -5.0 -> -200.0` 한 항목뿐이다.

핵심 개선은 “더 빠른 보행”이 아니라 넘어짐의 비용을 40배 높여 대회식 `Survival × Tracking` 목적에서 생존을 먼저 확보하도록 학습 압력을 바꾼 것이다. 최종 선택 모델은 5,000-iteration 실험의 `model_4600.pt`이다. 별도 4-seed 평가에서 생존율 0.890625, tracking 0.634862, proxy scenario score 0.566122를 기록했다. 공식 evaluator는 확보하지 못했으므로 proxy를 공식 점수로 부르지 않는다.

## 1. 연구 질문과 가설

한 번 넘어지면 tracking이 좋아도 곱 점수가 크게 깎인다. 초기 병목을 단순 속도 부족이 아니라 “낙상에 비해 dense tracking reward를 취하는 것이 상대적으로 싼 보상 구조”로 진단했다.

- H1: 관측·동작·PD·timestep·tracking reward를 고정하고 termination penalty만 키우면 base-contact 종료가 줄고 timeout과 episode length가 늘어난다.
- H2: 생존 개선 뒤 held-out command sequence로 tracking을 평가해야 한다.
- H3: 보상 계수가 다른 run끼리 raw mean reward를 직접 순위화하면 안 된다.

## 2. 초기값

### 환경 및 제어 계약

| 항목 | 초기 | 최종 |
|---|---:|---:|
| observation | 256 | 256 |
| action | 19 | 19 |
| action scale | 0.5 | 0.5 |
| simulation dt | 0.005 s | 동일 |
| decimation | 4 | 동일 |
| control period | 0.020 s, 50 Hz | 동일 |
| recurrent state | 없음 | 없음 |
| seed / resume | 42 / false | 42 / false |

PD도 고정했다. hip yaw/roll은 stiffness/damping 150/5, hip pitch·knee·torso 200/5, ankle 20/4, shoulder·elbow 40/10이다.

### 초기 reward 설정

| reward | 초기 | 최종 |
|---|---:|---:|
| track_lin_vel_xy_exp | weight 0.5, std 0.5 | 동일 |
| track_ang_vel_z_exp | weight 0.5, std 0.5 | 동일 |
| ang_vel_xy_l2 | -0.05 | 동일 |
| dof_torques_l2 | 0.0 | 동일 |
| dof_acc_l2 | -1.25e-7 | 동일 |
| action_rate_l2 | -0.0005 | 동일 |
| feet_air_time | 0.2, threshold 0.4 | 동일 |
| flat_orientation_l2 | -1.5 | 동일 |
| dof_pos_limits | -1.0 | 동일 |
| termination_penalty | **-5.0** | **-200.0** |

전체 env.yaml diff에서 위 termination weight와 run별 log_dir 이외의 차이는 없었다. 현재 증거로 의도적으로 검증했다고 말할 수 있는 기술 변경은 이 한 변수뿐이다.

Run A TensorBoard step 0은 mean reward -0.192332, episode length 12.673, terrain 3.439, error_vel_xy 0.018878, error_vel_yaw 0.090137이었다. 이는 랜덤 초기 actor의 짧은 rollout 통계다. episode가 극단적으로 짧아 error가 작게 보일 수 있으므로 초기 tracking 숫자를 완성 정책 RMSE처럼 해석하지 않았다.

## 3. 통제 실험 과정

### Run A — 기준선

- `2026-09-01_10-32-21`, penalty -5, 3,000 iterations, 선택 2,999
- 선택점 mean reward 1.736737, ±50 평균 1.715130
- episode length 333.06, base-contact termination 0.924428, timeout 0.075572

reward는 양수가 됐지만 종료의 92% 이상이 base contact였다. “보상은 오르나 오래 생존하지 못하는 정책”으로 판정했다.

### Run B — 낙상 비용 탐침

- `2026-09-01_11-16-51`, penalty -200, 판단점 약 1,300
- mean reward -6.818102, ±50 평균 -6.851898
- episode length 861.60, terrain 4.0337, base contact 0.191732, timeout 0.808268

raw reward 하락은 악화 증거가 아니다. 계수가 40배 달라 reward 단위가 바뀌었기 때문이다. episode length는 A의 2.59배, timeout은 7.56%에서 80.83%로 증가해 가설을 지지했다. 이 run은 최종 선정이 아니라 방향 검증 probe다.

### Run C — 장기 학습

- `2026-09-01_11-57-48`, penalty -200, 5,000 iterations, 선택 4,600
- mean reward -4.584132, ±50 평균 -5.364609
- episode length 937.55, terrain 5.2519, base contact 0.188253, timeout 0.811747

A 대비 episode length는 2.81배, timeout은 약 10.7배이고 base-contact 비중은 92.44%에서 18.83%로 감소했다. training log에서 생존 구조가 바뀌었다는 직접 증거다.

| run | penalty | iter | reward | ±50 mean | ep length | terrain | base contact | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline | -5 | 2999 | 1.736737 | 1.715130 | 333.06 | 0.0892 | 0.9244 | 0.0756 |
| B probe | -200 | 1300 | -6.818102 | -6.851898 | 861.60 | 4.0337 | 0.1917 | 0.8083 |
| C final | -200 | 4600 | -4.584132 | -5.364609 | 937.55 | 5.2519 | 0.1883 | 0.8117 |

C의 내부 training 값은 error_vel_xy 0.523513, error_vel_yaw 4.040749, track_lin component 0.345975, track_yaw component 0.061099이다. 이는 training distribution과 reward aggregation 단위이므로 아래 held-out RMSE와 섞지 않았다.

## 4. 독립 명령 추종 평가

stand, 전진 0.3/0.6/1.0 m/s, 좌우 측면, 양방향 yaw, 결합 명령, 급전환을 동일 순서로 넣었다.

| 지표 | 4-seed 결과 |
|---|---:|
| survival rate | 0.890625 |
| falls | 7 |
| mean episode length | 14.362 s |
| vx RMSE | 0.252658 |
| vy RMSE | 0.154884 |
| yaw RMSE | 0.729310 |
| tracking score | 0.634862 |
| proxy score | 0.566122 |
| stability | 0.580342 |
| crash / NaN | 0 / 0 |

추가 single-environment 재실행은 survival 1.0, fall 0, tracking/proxy 0.729685, vx/vy/yaw RMSE 0.153517/0.095924/0.514951, 전체 tracking RMSE 0.315142였다. 단일 시연은 4-seed 집계를 대체하지 않는다.

## 5. Calibration 반증 실험

policy command gain `vx=1.12, vy=1.5, yaw=4.0`을 시험했다.

| 조건 | survival | tracking | proxy |
|---|---:|---:|---:|
| 원본 | 0.890625 | 0.634862 | 0.566122 |
| calibration | 0.875000 | 0.646146 | 0.565969 |

tracking은 0.011284 상승했지만 survival은 0.015625 하락했고 곱 점수는 0.000153 감소했다. 대회 목적에서는 기각했으며 원본 결과를 덮어쓰지 않았다.

## 6. 실제 개선 메커니즘

첫째, 낙상 비용을 무시할 수 없게 해 PPO가 짧은 episode에서 순간 tracking을 얻는 전략보다 episode를 유지하는 전략을 선호하게 했다.

둘째, 짧은 probe와 긴 run을 분리했다. B에서 방향을 확인하고 C에 GPU 시간을 투입했으므로 틀린 가설의 장기 학습 위험을 줄였다.

셋째, training reward가 아니라 held-out `survival × tracking`으로 선택했다. 보상 계수가 다른 A/C의 reward 부호를 성능 순위로 오독하지 않았다.

## 7. 실패 모드와 한계

가장 큰 약점은 yaw RMSE 0.729310이다. 회전 추종이 vx/vy보다 약하고, yaw 입력 gain을 크게 올리면 tracking 소폭 상승보다 낙상 증가가 커졌다. 입력 배율만의 문제가 아니라 회전 중 자세 안정성과 reward/curriculum의 결합 문제다.

학습 run은 모두 seed 42라 한 변수 통제에는 유리하지만 여러 training seed의 인과 반복 검증은 아직 없다. 평가는 여러 seed지만 학습 분산까지 검증한 것은 아니다. 공식 evaluator가 없으므로 proxy는 공식 70점 환산 점수가 아니다.

## 8. 다음 GPU 시간

최우선은 최종 설정 `termination_penalty=-200`을 유지해 3개 이상의 training seed로 재현성을 확인하는 것이다. 그 다음에는 한 번에 하나만 바꾼다.

1. yaw tracking weight 또는 yaw command curriculum 중 하나만 조정한다.
2. 동일 seed 집합·동일 wall-clock으로 비교한다.
3. survival이 0.890625 아래로 내려가면 tracking 상승만으로 채택하지 않는다.
4. yaw 뒤에 lateral negative-command 비대칭과 foot slip을 별도 실험한다.
5. 외부 weight나 PPO continuation은 이 scratch 연구와 분리한다.

현재 다음 GPU 시간을 가장 잘 쓰는 선택은 다른 weight 교체가 아니라 **생존 설정을 고정한 multi-seed yaw-specific 단일 변수 미세조정**이다.
