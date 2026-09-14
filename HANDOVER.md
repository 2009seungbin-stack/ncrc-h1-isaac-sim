# 인수인계 — NCRC H1 프로젝트

> **2026-09-13 최신 인수인계는 [`HANDOVER_GO2_20260913.md`](HANDOVER_GO2_20260913.md) 다. 이 문서(09-05, H1)는 기록용.**

갱신: 2026-09-05 02:10 KST
이 문서를 먼저 읽어라. 계측 방법은 `tools/T_EVAL_RUNBOOK.md`, 점수 계산법은 `self_eval_S2.md`.

---

## 🔴 0. 학습은 반드시 GPU를 꽉 채워서 돌려라

**서버는 wall-clock 과금이다.** GPU가 놀면 그 시간은 그냥 버리는 것이다.
`--num_envs 4096`(가이드 기본값)은 이 GPU의 3분의 1도 안 쓴다. **실측이다.**

| | `--num_envs 4096` | **`--num_envs 16384`** |
|---|---:|---:|
| VRAM | 5,010 / 16,303 MiB (31%) | **10,246 MiB (63%)** |
| GPU 사용률 | 61~66% | **81%** |
| 전력 | 133 W (5080 TDP 약 360W) | **192 W** |
| iteration | 0.72s | 1.66s (4배 envs에 2.3배 시간) |
| **처리량** | 136,533 samples/s | **236,876 samples/s** |

**같은 벽시계 시간에 1.73배 더 학습한다.** 스케일링이 sublinear라서 그렇다 —
envs를 4배로 올려도 iteration 시간은 2.3배밖에 안 늘어난다.

그리고 이건 단순히 빠른 게 아니다. **업데이트당 지형·명령 다양성이 늘어서 지형
커리큘럼 진급에 직접 듣는다.** E1(4096)은 지형 3.74에서 정체했고 S2(16384)는 5.87까지 갔다.

### 상한 계산 (실측 2점으로 회귀)

```
VRAM(MiB) ≈ 3,265 + 0.426 × num_envs
```

| envs | 예상 VRAM | 판정 |
|---|---:|---|
| 16384 | 10.2 GB | **권장 운용점** (실측) |
| 24576 | 13.7 GB | 들어감. 처리량 +8% 정도. OOM 여유 2.5GB |
| 32768 | 17.2 GB | **OOM** |

24576까지 밀 수 있지만, 한 시간 돌다 OOM 나면 +8%보다 훨씬 비싸다.
**16384를 기본으로 쓰고, 급할 때만 24576을 시도하되 첫 2분 VRAM을 확인하라.**

### 확인 방법

학습 시작 후 2분 안에 반드시 재라. 기준에 못 미치면 envs를 올려라.

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
grep 'Iteration time' /tmp/<log> | tail -3
```

RAM은 64GB 중 60GB가 놀아서 병목이 아니다. **병목은 VRAM이고, 그건 채워야 한다.**

---

## 1. 현재 위치

| | |
|---|---|
| 공식 최고점 | **70.34** (2026-09-01 제출, 시뮬 42.34 + 설계의도 18 + 리포트 10) |
| 순위 | **76위** / 32위 컷 **79.25** (2026-09-04 14:59 기준) |
| 2라운드 마감 | **2026-09-09 (수) 09:00** |
| 서버 사용 | 약 10시간 / 100시간 (H1·Go2 공유) |
| 현재 상태 | 서버 종료함. 산출물은 로컬 번들에 있음 |

리더보드 상위: 1위 사족보행 88.2, 2위 사족보행 87.2, 3위 휴머노이드 87.01.
**전원이 로봇 하나만 냈다.** 순위는 H1+Go2 합산 200점이므로 Go2 신규 제출은 여전히
큰 지렛대다 — 다만 다른 팀도 Go2를 시작했으므로 차익은 줄고 있다.

---

## 2. 최선 후보 = S2 (아직 제출 안 됨)

### 설정 — baseline 대비 **정확히 2개**

| 보상 항목 | baseline | S2 |
|---|---:|---:|
| `track_ang_vel_z_exp` | 0.5 | **2.0** |
| `termination_penalty` | -5.0 | **-200.0** |

나머지 12개 항목 전부 baseline과 동일하다. 로컬에서 대조 검증 완료.
학습: 16384 envs · 3000 iteration · scratch · seed 42 → `model_2900.pt`.

### 자체 평가 결과

| ID | 공식(1R 챔피언) | E1 | **S2** | yaw RMSE E1→S2 |
|---|---:|---:|---:|---|
| H1 제자리 | 0.667 | 0.800 | **0.922** | 0.369 → 0.122 |
| H2 전진 | 0.598 | 0.636 | **0.897** | 0.780 → 0.146 |
| H3 좌우 | 0.661 | 0.789 | **0.873** | 0.401 → 0.144 |
| H4 회전 | 0.500 | 0.560 | **0.884** | 1.000 → 0.181 |
| H5 요철 | 0.562 | 0.636 | **0.887** | 0.780 → 0.156 |
| H6 경사 | 0.612 | 0.662 | **0.899** | 0.709 → 0.146 |
| H7 밀침 | 0.620 | 0.605 | **0.859** | 0.851 → 0.152 |
| **SimScore** | **42.34** | 46.73 | **62.32** | |

**7개 전부 상승, yaw RMSE 전부 −78~−82%.** 자체 총점 추정 90.3.

> **이 90.3을 그대로 믿지 마라.** 자체 평가는 생존율이 7개 전부 1.000으로 포화돼
> 생존 성능을 구분하지 못하고, 1라운드 챔피언을 이 평가기로 재면 공식 42.34보다 높게
> 나온다. **오프셋은 공식 점수를 받아야 확정된다.** 믿을 수 있는 건
> **E1 46.73 → S2 62.32 (+15.59)** 라는 방향과 크기다.

### 왜 이게 통했나 (리포트의 핵심 논거)

축별 감도 분해: **yaw 오차를 0으로 만들면 +19.21점, 선속도는 +0.08점.**
선속도 추종은 이미 끝난 문제였고 E1은 이미 풀린 축을 또 팠다.

yaw 오차 발생 조건 회귀 (R² 0.948):
```
yaw_RMSE = 0.390 + 0.701 × 전진속도 + 0.566 × |yaw명령|
```
배점가중 기여: 기본요동 0.390 + 걸음요동 0.316 vs **추종실패 0.055**.
즉 "못 돈다"가 아니라 **"똑바로 못 서 있고 못 걷는다"** 가 93%였다.

---

## 3. 이 상태를 복원하는 법 (Termux에서 resume 가능)

로컬 번들: **`C:\Users\2009s\Downloads\NCRC_S2_bundle.tgz`** (24,080,504 B)
검증됨 — `keep_S2/policy.pt` md5 `6e0b04a1cea3ba51bfa9b8abaefcf18f` (서버 값과 일치),
`keep_S2/model_2900.pt` md5 `1cfc92eedc5c5565032b5ffb513a1de4`.

담긴 것: `keep_S2/`(policy.pt · model_2900.pt · env.yaml · humanoid_rewards_S2.py),
`keep_E1/`, `eval_results/`(E1·S2 채점 전부), tfevents 4개, params.

### 복원 절차

```bash
# 1) 팀 보드에서 서버 시작 → VS Code 열기
# 2) 도구 설치: tools/t_eval_paste.sh 전체를 서버 셸에 붙여넣기 (113줄)
#    설치되는 것: t_eval.py, set_reward.py, screen_queue.sh

# 3) 번들을 서버로 올린다 — Termux에서 scp (폰에 파일을 먼저 옮겨둔다)
scp NCRC_S2_bundle.tgz da-perfect25@23.robotics.connect.or.kr:/workspace/

# 4) 풀기
cd /workspace/training/humanoid
tar xzf /workspace/NCRC_S2_bundle.tgz

# 5) 보상 설정 복원 (S2 그대로 이어가려면)
cp keep_S2/humanoid_rewards_S2.py humanoid_rewards.py
/isaac-sim/python.sh set_reward.py --show     # 2.0 / -200 확인

# 6) resume 기반 만들기
mkdir -p logs/rsl_rl/humanoid/from_S2
cp keep_S2/model_2900.pt logs/rsl_rl/humanoid/from_S2/

# 7) 이어서 학습 (GPU 꽉 채워서!)
/workspace/IsaacLab/isaaclab.sh -p train.py --task Humanoid-v0 \
  --num_envs 16384 --max_iterations 3000 --headless \
  --resume --load_run from_S2 --checkpoint model_2900.pt
```

규정 제14조상 **`model_best.pt` 업로드 후 `--resume` 은 공식적으로 허용된 절차**다
(대회 TIP 공지). 다른 PC에서 받은 체크포인트도 같은 방식으로 쓸 수 있다고 README가 명시한다.

### ⚠ resume 은 지형 커리큘럼을 리셋한다

README 명시: *"resume 을 하면 이어받을 때 지형 난이도(curriculum)가 쉬운 단계로
리셋됩니다. 그래서 제출본은 가능하면 from scratch 로 한 번에 길게 올리는 게 깔끔해요."*

S2는 지형 5.87까지 올라간 상태다. resume 하면 그게 날아간다.
→ **resume 은 스크리닝용. 제출본은 scratch 로 만든다.**

### scratch 로 S2를 재현하려면

설정이 2개뿐이라 재현이 쉽다. **16384 envs · 3000 iteration ≈ 85분.**

```bash
/isaac-sim/python.sh set_reward.py --restore
/isaac-sim/python.sh set_reward.py termination_penalty=-200.0 track_ang_vel_z_exp=2.0
/workspace/IsaacLab/isaaclab.sh -p train.py --task Humanoid-v0 \
  --num_envs 16384 --max_iterations 3000 --headless
```

---

## 4. 막혀 있는 것 — 제출창을 못 찾았다

규정 제5조: *"대회 사이트 내 팀 대시보드에서 제출한다."* 그런데 `/board` 의 상호작용
요소는 내비게이션·로그아웃·**1라운드 제출물 다운로드 링크**·"자세히" 버튼이 전부다.
업로드 컨트롤이 없다. `/status` 에도 없고 `/submit` 은 404다. 사용자도 못 찾았다.

**아직 2라운드 제출이 열리지 않은 것으로 보인다.** 마감이 9/9 09:00이라 시간은 있다.
창이 열리면 아래를 올린다.

| 제출물 | 위치 |
|---|---|
| `policy.pt` | 번들 `keep_S2/policy.pt` |
| `env.yaml` | 번들 `keep_S2/env.yaml` |
| 기술 개선 리포트 | `report_round2_S2.md` 의 "제출 본문" 절 |

**"학습 파일 내역"은 제출이 아니다.** 학습이 끝날 때마다 `submission_*.zip`(6.3MB)이
자동 업로드되는 **제14조 대조용 학습 이력**이다. 1라운드 실제 제출(09-01 23:19)은
그 목록에 없다 — 별개 경로가 맞다.

---

## 5. 서버에서 반드시 지킬 것 (전부 실측으로 데인 것들)

### ⛔ `train.py` 를 `--task` 없이 실행하지 마라 (`--help` 포함)

인자가 부족하면 도움말이 아니라 **학습 종료 후처리(POST)** 가 돈다. 진행 중이던 run에서
best를 뽑고 `logs/` 의 중간 체크포인트를 **삭제한다.** 실제로 E1 학습 중에 당했다
(체크포인트 71개 삭제). 인자를 확인하려면 **`README.md` 를 읽어라.**

### ⛔ POST의 `model_best.pt` 를 믿지 마라 — 학습 중 체크포인트 가드 필수

POST는 `Train/mean_reward` 최대 지점을 best로 고르는데, **이 값은 지형 커리큘럼 레벨을
건너 비교할 수 없다.** 어려운 지형에서는 더 나은 정책이 보상을 덜 번다.
- S1: 지형 5.90까지 갔는데 POST가 **iter 333**을 골랐다 → iter 3000 가중치를 잃었다
- S2: POST가 **iter 541**을 골랐다 → 가드가 `model_2900.pt` 를 살렸다

**항상 가드를 켜라.** POST는 정상 종료 때만 돌므로, 죽인 run의 체크포인트는 남는다.

```bash
cat > /tmp/guard.sh <<'EOF'
cd /workspace/training/humanoid
mkdir -p ckpt_keep
while true; do
  for r in $(ls -1dt logs/rsl_rl/humanoid/*/ | head -1); do
    b=$(basename $r); mkdir -p ckpt_keep/$b
    for f in $r/model_*.pt; do [ -f "$f" ] || continue; n=$(basename $f)
      [ -f ckpt_keep/$b/$n ] || cp "$f" ckpt_keep/$b/$n; done
    cp $r/params/env.yaml ckpt_keep/$b/env.yaml 2>/dev/null
  done
  sleep 45
done
EOF
tmux new -d -s guard 'bash /tmp/guard.sh'
```

### 그 밖에

- **맨 `python` 이 없다.** Isaac 스크립트는 `/workspace/IsaacLab/isaaclab.sh -p`,
  순수 파이썬은 `/isaac-sim/python.sh`
- **`PYTHONUNBUFFERED=1` 없이 리다이렉트하면** 네이티브 크래시 때 파이썬 print가
  통째로 사라진다. 이것 때문에 진단이 30분 늦어졌다
- **`play.py --checkpoint` 는 전체 경로**를 받는다 (파일명만 주면 FileNotFoundError).
  `train.py --resume` 은 `--load_run` + 파일명으로 받는다
- **`play.py` 가 `exported/policy.pt` 복사를 건너뛸 때가 있다.**
  실제 파일은 `logs/<run>/exported/policy.pt` 에 생긴다. 도구가 경고를 찍어주니 읽어라
- **t_eval 은 한 프로세스에 지형 그룹 하나만.** 두 번째 환경을 만들면 `cfg.validate()` 가
  무한 재귀한다. flat/rough/slope 를 나눠 돌리고 `--merge` 로 합쳐라
- **셸 스크립트는 LF.** Windows에서 만든 CRLF는 bash가 못 읽는다
- 학습은 반드시 tmux 안에서. code-server 터미널이 끊기면 학습이 멈춘다
- 30분 무활동이면 워크스페이스가 통째로 지워진다

---

## 6. 다음에 할 일

### 즉시

1. **제출창이 열렸는지 확인** → 열렸으면 S2 업로드 (policy.pt + env.yaml + 리포트)
2. 공식 점수를 받으면 `score_model.py --add-official` 로 앵커를 추가한다.
   앵커가 2건이 되는 순간 자체 평가기의 오프셋이 확정돼 이후 판단이 전부 정확해진다

### 다음 레버 — 자세 안정 계열

yaw를 잡고 나니 남은 손실이 **생존 쪽**으로 옮겨갔다. 학습 중 낙상률이
12.9%(E1) → 18.7%(S2)로 올랐다(지형이 3.7→5.9로 어려워진 상태이긴 하다).

| 후보 | 현재 | 권장 범위 | 근거 |
|---|---:|---|---|
| `action_rate_l2` | **-0.0005 (바닥)** | -0.008 ~ -0.0005 | 관절 명령 급변 억제 → 몸통 흔들림. **16배 여유** |
| `joint_deviation_hip` | -0.2 | -0.5 ~ -0.1 | hip_yaw가 몸통 요잉의 직접 구동원 |
| `flat_orientation_l2` | -1.5 | -3.0 ~ -0.2 | 자세 안정 |

프로토콜: **16384 envs · 3000 iteration · scratch · 한 번에 하나씩.**
판정은 `score_model.py` (tfevents만, GPU 0) → 유망하면 T-EVAL로 실측.

**지표는 반드시 에피소드 길이로 정규화해서 봐라.** `error_vel_*` 는 에피소드 동안
누적되는 값이라, 오래 살아남으면 자동으로 커진다. 정규화 안 하면 개선을 악화로 읽는다.
```
스텝당 평균오차 = metric × 500 / mean_episode_length
```

### 그다음 — Go2

H1을 80대 중반까지 올려도 컷이 계속 오른다. 합산 200점 구조에서 **Go2 신규 제출이
GPU 시간당 가치가 훨씬 높다.** 다만 Go2 baseline은 H1처럼 약화돼 있지 않다 —
항목이 5개뿐이고 `track_lin_vel_xy_exp` 는 이미 1.0, `termination_penalty` 는 목록에 없다.
`quadruped_rewards.py` 아래쪽에 주석 처리된 **[고급 옵션]** 이 있으니 그것부터 확인하라.

---

## 7. 자산 위치

### 로컬

| 파일 | 내용 |
|---|---|
| `C:\Users\2009s\Downloads\NCRC_S2_bundle.tgz` | **서버 산출물 전부** (24MB) |
| `report_round2_S2.md` | 2라운드 제출 리포트 (본문 + 체크리스트) |
| `self_eval_S2.md` | 자체 평가 점수와 계산법 전문 |
| `official_score_breakdown.md` | 1라운드 공식 점수·가중치·산식 |
| `eval_results/*.json` | E1·S2 채점 원자료 및 재채점 결과 |
| `tools/t_eval.py` | 7시나리오 평가기 |
| `tools/score_model.py` | tfevents만으로 점수 추정 (GPU 0) |
| `tools/set_reward.py` | 보상 안전 편집기 (오타 방지) |
| `tools/screen_queue.sh` | 스크리닝 큐 |
| `tools/t_eval_paste.sh` | 위 도구들을 113줄 붙여넣기로 서버 설치 |
| `tools/T_EVAL_RUNBOOK.md` | 서버 운용 절차 + 실측 제약 전부 |
| `tools/reward_diff.py` | 제출 게이트 (env.yaml ↔ 리포트 대조) |
| `experiment_queue.md` | 실험 설계 (resume 기반 — 일부 낡음) |
| `round2_plan.md` | 초기 계획 (예산표·실험설계 모두 낡음) |

### 사이트

- 팀 보드 https://robotics.connect.or.kr/board (팀 `team-2026-qn-019`, 계정 `s-2026-qn-002`)
- 진행 현황·리더보드 https://robotics.connect.or.kr/status
- 예선 규정 https://robotics.connect.or.kr/rules/preliminary_regulations.html
- 서버 https://23.robotics.connect.or.kr/code (code-server)

---

## 한 줄 요약

> S2(`track_ang_vel_z_exp` 0.5→2.0, `termination_penalty` -5→-200)가 자체 평가에서
> 7개 시나리오 전부를 올려 SimScore 46.73 → **62.32** 를 냈다. 제출창이 아직 안 열려
> 업로드는 못 했고, 산출물은 `Downloads\NCRC_S2_bundle.tgz` 에 전부 있다.
> **다시 학습할 때는 `--num_envs 16384` 로 GPU를 꽉 채워라 — 같은 시간에 1.73배 더 배운다.**
