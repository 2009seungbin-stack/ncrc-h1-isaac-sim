#!/usr/bin/env bash
# NCRC 스크리닝 큐 - S0 체크포인트에서 resume 하며 단일 변수 후보를 차례로 돌린다.
#
#   사용법 (서버 /workspace/training/humanoid/ 에서, tmux 안에서):
#     bash screen_queue.sh probe                 # train.py/play.py 인자 이름 확인 (GPU 0)
#     DRYRUN=1 BASE_RUN=... bash screen_queue.sh # 명령만 출력, 실행 안 함
#     BASE_RUN=... bash screen_queue.sh          # 실제 실행
#
# 서버는 30분 무활동이면 워크스페이스를 지운다. 큐를 통째로 걸어 공백을 없앤다.
# 반드시 tmux 안에서 돌린다:  tmux new -s q  ->  bash screen_queue.sh
set -uo pipefail

# ---------------------------------------------------------------- 설정
WORKDIR=${WORKDIR:-$(pwd)}
TASK=${TASK:-Humanoid-v0}
NUM_ENVS=${NUM_ENVS:-4096}
ITERS=${ITERS:-1000}                  # 스크리닝용. 확정 run은 이 스크립트를 쓰지 않는다
LOGROOT=${LOGROOT:-logs/rsl_rl/humanoid}
# 이 서버에는 맨 `python` 이 없다. Isaac 스크립트는 런처로, 순수 파이썬은 PY 로 돈다.
LAUNCH=${LAUNCH:-/workspace/IsaacLab/isaaclab.sh -p}
PY=${PY:-/isaac-sim/python.sh}
RESULTS=${RESULTS:-eval_results}
DRYRUN=${DRYRUN:-0}

# S0 - 모든 후보의 공통 출발점. E1이 남긴 run 디렉터리와 체크포인트 파일명.
BASE_RUN=${BASE_RUN:-}
BASE_CKPT=${BASE_CKPT:-}

# E1 확정 보상값. 후보 변경은 이 위에 하나만 얹는다.
BASE_WEIGHTS=${BASE_WEIGHTS:-"track_lin_vel_xy_exp=1.0 termination_penalty=-200.0"}

# 후보 큐 - "이름:항목=값" 을 공백으로 구분. 우선순위 순.
QUEUE=${QUEUE:-"\
E2:track_ang_vel_z_exp=1.0 \
E3:action_rate_l2=-0.005 \
E4:feet_air_time=0.25 \
E5:flat_orientation_l2=-1.0 \
E6:ang_vel_xy_l2=-0.15 \
E8:joint_deviation_hip=-0.1"}

# ---- 워크스페이스마다 다를 수 있는 인자 이름. `probe` 로 먼저 확인할 것. ----
TRAIN_RESUME_ARGS=${TRAIN_RESUME_ARGS:---resume --load_run BASE_RUN_SUB --checkpoint BASE_CKPT_SUB}
PLAY_LOAD_ARGS=${PLAY_LOAD_ARGS:---load_run RUN_SUB}

say() { printf '\n=== %s ===\n' "$*"; }
run() {
  echo "+ $*"
  if [ "$DRYRUN" = "1" ]; then return 0; fi
  "$@"
}

# ---------------------------------------------------------------- probe
if [ "${1:-}" = "probe" ]; then
  cd "$WORKDIR" || exit 1
  # !! train.py / play.py 를 인자 없이(또는 --help 로) 절대 실행하지 마라.
  #    이 서버의 train.py 는 --task 가 없으면 도움말이 아니라 **학습 종료 후처리**를 돌린다.
  #    진행 중인 run 에서 best 를 뽑고 logs/ 의 중간 체크포인트를 삭제한다. 실제로 당했다.
  say "README 의 공식 사용법 (이것이 유일한 근거)"
  grep -n -A3 -E 'isaaclab.sh -p (train|play).py' README.md | head -40
  say "최근 run 디렉터리 (S0 후보)"
  ls -1dt "$LOGROOT"/*/ 2>/dev/null | head -5
  say "가장 최근 run 안의 체크포인트"
  latest=$(ls -1dt "$LOGROOT"/*/ 2>/dev/null | head -1)
  [ -n "$latest" ] && ls -1t "$latest"model_*.pt 2>/dev/null | head -5
  say "exported/"
  ls -1 exported/ 2>/dev/null || echo "(아직 없음 - 학습이 끝나야 생긴다)"
  say "런처 확인"
  printf 'LAUNCH = %s
' "$LAUNCH"; printf 'PY     = %s
' "$PY"
  $PY -c "print('PY OK')" 2>&1 | tail -1
  exit 0
fi

# ---------------------------------------------------------------- 사전 점검
cd "$WORKDIR" || exit 1
fail=0
for f in train.py play.py humanoid_rewards.py set_reward.py t_eval.py; do
  [ -f "$f" ] || { echo "!! 없는 파일: $f"; fail=1; }
done
[ -n "$BASE_RUN" ]  || { echo "!! BASE_RUN 이 비었다. 'bash screen_queue.sh probe' 로 확인해 지정하라."; fail=1; }
[ -n "$BASE_CKPT" ] || { echo "!! BASE_CKPT 가 비었다."; fail=1; }
[ "$fail" = "0" ] || exit 1

mkdir -p "$RESULTS"
[ -f humanoid_rewards.baseline.bak ] || cp humanoid_rewards.py humanoid_rewards.baseline.bak

# .bak 은 매 후보마다 --restore 의 출발점이 된다. 원본 baseline 이 아니면
# BASE_WEIGHTS 로 덮이지 않는 항목이 후보마다 오염된 채 남는다.
$PY - <<'PYCHK'
import re, sys
t = open("humanoid_rewards.baseline.bak", encoding="utf-8").read()
m = re.search(r'"termination_penalty"\s*:\s*(-?[\d.]+)', t)
v = float(m.group(1)) if m else None
if v is None:
    print("!! .bak 에서 termination_penalty 를 못 찾았다. 눈으로 확인하라.")
elif abs(v + 5.0) > 1e-9:
    print(f"!! humanoid_rewards.baseline.bak 의 termination_penalty 가 {v} 다 "
          f"(원본 baseline 은 -5.0).")
    print("   이 .bak 은 원본이 아니다. 원본으로 바꾸거나 BASE_WEIGHTS 에 "
          "9개 값을 모두 명시하라.")
else:
    print("baseline.bak 확인 OK (termination_penalty = -5.0)")
PYCHK

start_all=$(date +%s)
echo "큐: $QUEUE"
echo "S0: $LOGROOT/$BASE_RUN/$BASE_CKPT   iterations=$ITERS   DRYRUN=$DRYRUN"
echo
echo "!! resume 은 지형 난이도(curriculum)를 쉬운 단계로 리셋한다 (README 명시)."
echo "   모든 후보가 똑같이 리셋되므로 상대 비교는 유효하지만, H5(요철)/H6(경사) 점수는"
echo "   전부 낮게 깔린다. 지형 계열 후보(E3/E4/E5)의 음성은 특히 약한 증거로 취급하라."
echo "   제출본은 반드시 scratch 로 만든다."

# ---------------------------------------------------------------- 본 루프
for item in $QUEUE; do
  name=${item%%:*}
  change=${item#*:}
  say "$name  ($change)"
  t0=$(date +%s)

  # 1) baseline 복원 -> E1 확정값 -> 후보 변경 하나. set_reward.py 가 항목명을 검증한다.
  run $PY set_reward.py --restore
  # shellcheck disable=SC2086
  run $PY set_reward.py $BASE_WEIGHTS
  run $PY set_reward.py "$change" || { echo "!! $name 보상 설정 실패 - 건너뛴다"; continue; }
  [ "$DRYRUN" = "1" ] || $PY set_reward.py --show 2>/dev/null | sed -n '2,20p'

  # 2) resume 학습
  resume_args=${TRAIN_RESUME_ARGS//BASE_RUN_SUB/$BASE_RUN}
  resume_args=${resume_args//BASE_CKPT_SUB/$BASE_CKPT}
  # shellcheck disable=SC2086
  run $LAUNCH train.py --task "$TASK" --num_envs "$NUM_ENVS" \
      --max_iterations "$ITERS" --headless $resume_args \
      || { echo "!! $name 학습 실패 - 건너뛴다"; continue; }

  # 3) 새로 생긴 run 디렉터리 찾기
  if [ "$DRYRUN" = "1" ]; then
    newrun="<DRYRUN-최신run>"
  else
    newrun=$(basename "$(ls -1dt "$LOGROOT"/*/ | head -1)")
    if [ "$newrun" = "$BASE_RUN" ]; then
      echo "!! 새 run 디렉터리가 안 생겼다 ($newrun). resume 인자를 확인하라."; continue
    fi
  fi
  echo "run: $newrun"

  # 4) 정책 export
  play_args=${PLAY_LOAD_ARGS//RUN_SUB/$newrun}
  # shellcheck disable=SC2086
  run $LAUNCH play.py --task "$TASK" --num_envs 32 --headless $play_args \
      || echo "!! $name export 실패 - 아래에서 policy.pt 를 직접 찾아본다"

  pol=""
  if [ "$DRYRUN" = "1" ]; then
    pol="<DRYRUN-policy.pt>"
  else
    pol=$(ls -1t "$LOGROOT/$newrun"/exported/policy.pt exported/policy.pt 2>/dev/null | head -1)
    [ -n "$pol" ] || { echo "!! $name policy.pt 를 못 찾았다 - 채점 건너뜀"; continue; }
  fi
  echo "policy: $pol"

  # 5) T-EVAL 채점
  run $LAUNCH t_eval.py --headless --task "$TASK" \
      --policy "$pol" --run_id "$name" --out "$RESULTS/$name.json"

  # 6) 결과물 보존 (30분 초기화 대비)
  run mkdir -p "$RESULTS/$name"
  [ "$DRYRUN" = "1" ] || {
    cp "$pol" "$RESULTS/$name/policy.pt" 2>/dev/null
    cp "$LOGROOT/$newrun/params/env.yaml" "$RESULTS/$name/env.yaml" 2>/dev/null
    cp humanoid_rewards.py "$RESULTS/$name/humanoid_rewards.py" 2>/dev/null
    echo "$name $change run=$newrun" >> "$RESULTS/manifest.txt"
  }

  echo "$name 완료 ($(( ($(date +%s) - t0) / 60 ))분)"
done

# ---------------------------------------------------------------- 정리
run $PY set_reward.py --restore
say "큐 전체 완료 ($(( ($(date +%s) - start_all) / 60 ))분)"
[ "$DRYRUN" = "1" ] || {
  tar czf screening_results.tgz "$RESULTS" 2>/dev/null
  echo "screening_results.tgz 생성. **지금 바로 로컬로 내려라.** 30분 놀면 사라진다."
  echo
  echo "요약:"
  for j in "$RESULTS"/*.json; do
    [ -f "$j" ] || continue
    $PY - "$j" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
sc = d["scenarios"]
worst = min(sc, key=lambda k: sc[k]["score"]) if sc else "-"
print(f'  {d["run_id"]:<6} sim={d["sim_score"]:>6.2f}  '
      f'커버리지={d["coverage_weight"]:.2f}  최저={worst}  '
      f'falls={d["falls"]}  nan={d["nan_actions"]}')
PY
  done
}
