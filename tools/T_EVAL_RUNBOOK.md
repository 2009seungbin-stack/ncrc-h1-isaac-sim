# T-EVAL 운용 절차서

`tools/t_eval.py` 를 서버에서 돌리는 순서. **서버는 wall-clock 과금이고 30분 무활동이면
워크스페이스가 지워진다.** 아래를 순서대로 붙여넣기만 하면 되도록 써 놓았다.

작업 디렉터리는 항상 `/workspace/training/humanoid/` (train.py·play.py가 있는 곳).

---

## ⛔ 서버에서 절대 하면 안 되는 것 (2026-09-04 실측)

### 1. `train.py` 를 `--task` 없이 실행하지 마라 — `--help` 포함

이 서버의 `train.py` 는 인자가 부족하면 도움말을 찍고 끝나는 게 아니라
**학습 종료 후처리(POST)를 실행한다.** 실제로 `train.py --help` 한 번에 이런 일이 났다.

```
[ERROR] --task 인자가 없습니다. ...
[POST] 학습 종료 (exit=2) → best model 선별 → /workspace/training/humanoid
[INFO] Run: .../logs/rsl_rl/humanoid/2026-09-04_18-16-00      ← 진행 중이던 run
[OK]   model_best.pt → exported/model_best.pt (iter=5227)
[CLEAN] logs/ 의 intermediate .pt 71개 삭제 (484MB 회수)     ← 삭제
```

진행 중인 학습은 죽지 않았지만 **중간 체크포인트 71개가 지워졌다.**
(best-so-far인 `model_5200.pt` 는 보존됐고 이후 체크포인트는 정상 생성됐다.)

→ 인자를 확인하고 싶으면 **`README.md` 를 읽어라.** 공식 사용법이 전부 적혀 있다.
→ `screen_queue.sh probe` 는 이 사고 이후 `--help` 호출을 전부 제거했다.

### 2. 팀 보드의 "서버 종료하기" 를 누르지 마라

학습 중에 누르면 그 run 이 통째로 날아간다. VS Code 접속에는 종료가 필요 없다.

---

## 서버 환경 실측 (2026-09-04)

| 항목 | 값 |
|---|---|
| 접속 | 팀 보드 → **VS Code 열기** → `https://23.robotics.connect.or.kr/code` (code-server) |
| `https://23.robotics.connect.or.kr/` | Isaac Sim GUI 의 NVIDIA GFN 비디오 스트림. 터미널 아님 |
| **맨 `python` 은 없다** | Isaac 스크립트: `/workspace/IsaacLab/isaaclab.sh -p` / 순수 파이썬: `/isaac-sim/python.sh` |
| 작업 디렉터리 | `/workspace/training/humanoid` |
| 학습 로그 | `logs/rsl_rl/humanoid/<날짜>/` |
| 산출물 | `exported/` — `model_best.pt`, `env.yaml`, `report.html` |
| **`policy.pt` 는 학습만으로 안 생긴다** | `play.py` 를 돌려야 `exported/policy.pt` 가 만들어진다 |
| 셸 스크립트 줄바꿈 | **반드시 LF.** Windows에서 만든 CRLF 파일은 bash가 못 읽는다 |
| tmux | 학습은 반드시 tmux 안에서. code-server 터미널이 끊기면 학습이 멈춘다 |

### resume 은 지형 커리큘럼을 리셋한다 (README 명시)

> `resume` 을 하면 이어받을 때 **지형 난이도(curriculum)가 쉬운 단계로 리셋**됩니다.
> 그래서 제출본은 가능하면 `from scratch` 로 한 번에 길게 올리는 게 깔끔해요.

스크리닝 후보는 전부 똑같이 리셋되므로 **상대 비교는 유효**하다. 다만 H5(요철)·H6(경사)
점수가 전부 낮게 깔리므로, 지형 계열 후보의 음성은 특히 약한 증거다.

### `play.py --push` 가 있다 — H7 확인용

```
/workspace/IsaacLab/isaaclab.sh -p play.py --task Humanoid-v0 --num_envs 1     --headless --video --video_length 1000 --enable_cameras --push
```
`--push 0.8` 로 세기 조절(기본 0.5 m/s). README 원문: *"play 전용입니다 — 학습 환경에는
밀침이 없고 켤 수도 없습니다 (전 팀 동일 조건). 밀침 회복력은 별도 학습 없이 자세·안정성
계열 reward 가 결정합니다."* → 우리가 env.yaml 에서 추론한 결론을 대회측이 확인해 준다.

---

## 0. 설치 (붙여넣기 1회, 약 10초)

`tools/t_eval_paste.sh` **파일 전체**를 서버 셸에 붙여넣는다.
원본 3개(약 1,100줄)를 tar+gzip+base64 로 묶어 109줄이라 웹 터미널에서도 안전하다.

설치되는 것:

| 파일 | 용도 |
|---|---|
| `t_eval.py` | 7시나리오 평가기 |
| `set_reward.py` | `REWARD_WEIGHTS` 안전 편집기 — 항목명 오타를 막는다 |
| `screen_queue.sh` | `--resume` 스크리닝 큐 — 후보를 무인으로 연속 실행 (`experiment_queue.md` 참조) |

```
# 마지막 줄이 이렇게 나오면 성공
t_eval.py / set_reward.py / screen_queue.sh 설치 OK
```

로컬에서 도구를 고쳤다면 `python tools/_build_paste.py` 로 설치 스크립트를 다시 만든다.

### set_reward.py 를 반드시 쓰는 이유

가이드북 14강: *"REWARD_WEIGHTS에 없는 이름을 적으면 Isaac Lab 기본값을 쓰고 **경고만**
낸다."* 손으로 고치다 오타가 나면 학습은 정상으로 보이는 채 2시간을 태우고,
`env.yaml` 에도 변경이 안 남아 리포트와 불일치(제14조 감점)까지 난다.

```bash
python set_reward.py --show                                  # 현재 값 + 추천 범위
python set_reward.py track_ang_vel_z_exp=1.0                 # 항목명 검증 후 변경
python set_reward.py track_ang_vel_z_exp=1.0 --dry-run       # 쓰지 않고 확인만
python set_reward.py --restore                               # baseline 으로 복원
```

없는 항목명을 주면 실행이 **중단되고** 비슷한 이름을 제안한다. 첫 변경 시
`humanoid_rewards.baseline.bak` 을 자동 생성하고, 저장 후 재파싱으로 값을 검증한다.
추천 범위를 벗어난 값에는 `** 추천 범위 밖 **` 표시가 붙는다.

---

## 1. 프리플라이트 (약 1~2분) — **반드시 먼저**

지형 3종을 실제로 만들어 보고, API 가정이 맞는지 확인한다. 정책 파일이 없어도 된다.

```bash
python t_eval.py --preflight --headless --num_envs 16
```

확인할 출력 7가지:

| # | 나와야 하는 것 | 안 나오면 |
|---|---|---|
| 1 | `태스크 등록 OK` | `--task_module <모듈>` 로 train.py 상단 import 모듈을 넘긴다 |
| 2 | `지형 cfg 클래스 OK` | Isaac Lab 버전이 다르다. 클래스명을 확인해 `make_terrain_generator` 수정 |
| 3 | `flat/rough/slope: env OK` + obs shape | 아래 "실패 대응" 참조 |
| 5 | `root 속도 버퍼 OK` | `root_velocity()` 에 버퍼 이름 추가 |
| 6 | `write_root_velocity_to_sim OK` | H7 밀침 불가 → H7만 빼고 진행 |
| 7 | `flat: 실측 경사 ~0deg` | flat이 평평하지 않다 |
| 7 | `slope: 실측 경사 ~10deg` | **`--slope_deg` 로 보정한다** (아래 참조) |

### 경사각 보정

`slope` 지형의 실측값이 10도가 아니면 `HfPyramidSlopedTerrainCfg.slope_range` 의 단위
가정이 틀린 것이다. 실측값이 `m` 도로 나왔으면 다음부터 `--slope_deg $(python -c
"print(10*10/m)")` 처럼 비례 보정한 값을 **모든 후보에 동일하게** 쓴다.
(H6 점수는 후보 간 상대 비교용이므로 정확히 10도가 아니어도 되지만, **한 번 정하면 바꾸지 않는다.**)

### 실패 대응

- `flat` 만 실패 → `--flat_mode plane` 으로 재시도
- `rough`/`slope` 실패 → 해당 시나리오를 빼고 진행한다. 다만 H5+H6 배점이 0.30이라
  **커버리지가 0.70으로 떨어진다.** JSON의 `coverage_weight` 로 항상 확인할 것
- 전부 실패 → 서버 Isaac Lab 버전을 찍어 보고(`python -c "import isaaclab;print(isaaclab.__version__)"`)
  로컬에서 고쳐 다시 붙여넣는다. **서버에서 디버깅하지 마라**

---

## 2. 검증 게이트 — `model_4600` 으로 오프셋을 잰다 (약 5~10분)

**이걸 통과하기 전에는 어떤 후보도 이 평가기로 판정하지 않는다.** (spec 7.2)

`h1_current_training_bundle_20260901/policy.pt` 를 서버로 올려 `baseline_policy.pt` 로 둔다.
(이게 공식 42.34를 받은 바로 그 정책이다.)

```bash
python t_eval.py --headless \
  --policy baseline_policy.pt \
  --run_id H1_R1_BASELINE_4600 \
  --out eval_results/baseline_4600.json
```

끝나면 오프셋 표가 출력된다. 그 표를 `tools/scenario_eval_spec.md` 7.2절에 그대로 옮긴다.

| 오프셋 (공식 42.34 − 내부) | 판정 |
|---|---|
| ±5점 이내 | 정상. 평가기를 신뢰하고 진행 |
| ±5 ~ ±10 | 절대값은 못 믿지만 Δ 비교용으로는 쓴다. 리포트에 명시 |
| ±10 초과 | **시나리오 구성이 공식과 다르다. 평가기를 고치기 전에 학습 금지** |

시나리오별로도 본다. 특정 시나리오만 크게 어긋나면 그 시나리오 설계가 틀린 것이다.
특히 H4(공식 0.500)와 H2(공식 0.598)가 내부에서 크게 다르면 명령 시퀀스 설계를 의심한다.

---

## 3. 후보 채점 (약 5~10분/후보)

```bash
python t_eval.py --headless \
  --policy exported/policy.pt \
  --run_id H1_R2_E1 \
  --out eval_results/E1.json
```

**seed set·num_envs·seg_s·transient_s·slope_deg·rough_range 를 후보마다 바꾸지 마라.**
바꾸는 순간 과거 수치와 비교가 불가능하다 (CRN, spec 4절). JSON의 `config` 블록에
전부 기록되니 비교 전에 대조할 것.

### 채택 게이트

1. `survival_lcb >= 0.85` (Wilson 95% 하한. 점추정 아님) — S2 이후 `>= 0.90`
2. `sim_score` 가 baseline 대비 상승
3. `nan_actions == 0`

3개 중 하나라도 깨지면 채택하지 않는다.

### 판정 시 보는 것

- `scenarios.<ID>.survival` 과 `.tracking` 을 **따로** 본다. 곱만 보면 병목을 모른다.
- `segment_tracking` — 명령 구간별 추종. H2에서 vx=1.0 구간만 무너지는지 등이 보인다.
- `H2.score_forward_only` — 후진(-0.5) 구간을 뺀 값. 공식 H2가 후진을 포함하는지
  불명이라 두 가지로 집계해 둔다.

---

## 4. ⛔ 한 프로세스에 지형 그룹은 **반드시 하나** (선택이 아니라 필수)

**2026-09-04 서버 실측.** 한 프로세스에서 두 번째 환경을 만들면 이렇게 죽는다.

```
File ".../isaaclab/envs/manager_based_env.py", line 88, in __init__
    cfg.validate()
File ".../isaaclab/utils/configclass.py", line 291, in _validate
    missing_fields.extend(_validate(value, prefix=current_path))
  [같은 줄 무한 반복]
RecursionError: maximum recursion depth exceeded in __instancecheck__
```

지형 종류와 **무관하다.** 첫 번째 환경은 무엇이든 성공하고, 두 번째부터는 무엇이든 실패한다.
(mesh/height-field, 새로 만들기/deepcopy/제자리 수정 — 네 가지 다 시험했고 전부 같았다.)

`t_eval.py` 는 이제 그룹을 2개 이상 요청하면 **실행 전에 거부하고** 아래 절차를 안내한다.

```bash
L='/workspace/IsaacLab/isaaclab.sh -p'
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy exported/policy.pt   --scenarios H1,H2,H3,H4,H7 --run_id E1 --out eval_results/E1_flat.json
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy exported/policy.pt   --scenarios H5 --run_id E1 --out eval_results/E1_rough.json
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy exported/policy.pt   --scenarios H6 --run_id E1 --out eval_results/E1_slope.json

$L t_eval.py --merge eval_results/E1_*.json --run_id E1 --out eval_results/E1.json
```

`--merge` 는 Isaac 앱을 띄우지 않으므로 로컬에서 돌려도 된다.

### `PYTHONUNBUFFERED=1` 을 빼지 마라

`| tee` 나 리다이렉트를 쓰면 stdout 이 블록 버퍼링돼, 네이티브 크래시 때 **파이썬 print 가
통째로 사라진다.** 실제로 이것 때문에 첫 진단이 30분 늦어졌다.

### 지형 cfg 를 새로 만들지 마라

`make_terrain_generator` 는 학습 cfg 의 generator 를 **제자리에서** 고친다
(sub_terrain 의 `proportion` / `noise_range` / `slope_range` 만 바꾼다).
새 cfg 객체를 만들거나 deepcopy 하면 위 재귀에 더 쉽게 걸린다.

## 5. 결과를 잃지 않기 위해

서버는 30분 무활동이면 워크스페이스를 통째로 지운다. **이미 한 번 당했다.**

평가가 끝나면 즉시 로컬로 내린다.

```bash
tar czf eval_results.tgz eval_results/ exported/ logs/rsl_rl/humanoid/*/model_*.pt
```

규정 제2조가 결과물 다운로드를 명시적으로 허용한다.

---

## 설계상 알고 있어야 할 것

이 평가기가 **공식 채점기와 다른 점** — 리포트에 쓸 때 이대로 밝힌다.

| 항목 | 이 평가기 | 공식 |
|---|---|---|
| 명령 시퀀스 | 구간 4개 x 5초, 위 표대로 고정 | 비공개 |
| tracking 정의 | `exp(-RMSE)` | "오차 0 → 1.0" 만 공개 |
| 요철 강도 | `noise_range 0.02~0.06 m` | 비공개 ("약한 요철") |
| 경사각 | ±10° (오르막·내리막 50:50) | ±10° (규정 제7조에 명시) |
| 밀침 | 2.5초마다 ±1.0 m/s 수평 속도 주입 | 비공개 |
| 에피소드 수 | 128 x 4 seed = 512 | 비공개 |
| 관측 노이즈 | 학습과 동일 (켬) | 비공개 |

그래서 **절대값을 공식 점수라고 부르지 않는다.** 쓰는 방법은 두 가지뿐이다.

1. `model_4600` 대비 **Δ** 로 후보를 고른다
2. 제출할 때마다 `(내부 점수, 공식 점수)` 쌍을 쌓아 오프셋을 갱신한다
   → JSON의 `official_offset` 필드에 `공식 − 내부` 를 채워 넣는다

### 학습 설정과 평가 설정은 다르다 (규정 관계)

규정 제2조는 **학습** 설정 중 `REWARD_WEIGHTS` 만 수정을 허용한다. 지형·이벤트·커리큘럼은
학습에서 못 바꾼다. 이 평가기는 **제출물이 아니라 내부 계측 도구**이고, 학습된 정책을
읽기만 한다. 그래서 평가용으로 지형과 밀침을 구성한다.
**t_eval.py 를 제출 번들에 넣지 않는다.**

### 학습 환경에 push 이벤트가 없다는 사실

`env.yaml` 확인 결과 `events.push_robot: null`, `base_external_force_torque` 의
force/torque 범위도 전부 0이다. **H1 정책은 학습 중 밀침을 한 번도 겪지 않는다.**
그런데 공식 H7(밀침 회복)은 배점 0.15다. H7 점수(현재 0.620)의 해석에 이걸 반영해야 하고,
`REWARD_WEIGHTS` 만으로 H7을 올리려면 자세 안정 계열(`flat_orientation_l2`,
`ang_vel_xy_l2`, `termination_penalty`)을 통해 간접적으로 갈 수밖에 없다.
