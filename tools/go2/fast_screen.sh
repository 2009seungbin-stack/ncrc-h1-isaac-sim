#!/bin/bash
# fast_screen.sh — Go2 '전진 해금' 스크리닝. 팔 하나 = 8192 envs x 200 iter ≈ 23분, 순차, 로그로 판정.
#
# 왜 이게 되나 (2026-09-13 분석):
#   · GB 는 전진 명령을 무시했다. 평가 G1 구간 점수가 '제자리' 예측과 소수 셋째 자리까지 일치하고
#     (0.3/0.6/1.0 -> 0.833/0.706/0.561 vs 0.840/0.707/0.561), 학습 로그 xy 오차 0.71 m/s 가
#     '전진만 거부' 정책의 이론값 0.67 에 붙어 있다.
#   · 실패는 첫 250 iter 에 결정됐다(지형 3.5 -> 0.0). 그래서 200 iter 면 처방이 붕괴를 막는지 보인다.
#   · NCRC 가 IsaacLab Go2 기본값을 세 군데 약화시켰다: track_lin 1.5->1.0, lin_vel_z -2->-3,
#     ang_vel_xy -0.05->-0.08. 동시에 전진 명령 상한을 1.0->2.0 으로 넓혔다(파일로 못 바꿈).
#   · 12288 envs 는 PhysX 접촉 패치 버퍼(327,680)를 매 물리 스텝 넘겼다. 한도 내 최대 ~9,500 envs.
#
# 사용 (tmux 안에서):
#   bash fast_screen.sh                                   # 기본: S1 S0 S2 (약 75분)
#   ARMS="S1 S3" bash fast_screen.sh
#   AUTO_LONG=1 LONG_ITERS=5000 bash fast_screen.sh       # 끝나면 승자로 장기 학습 + 체크포인트 가드
#   touch /workspace/STOP_SCREEN                          # 다음 팔부터 중단
set -u
ROBOT_DIR=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$ROBOT_DIR" || { echo "!! $ROBOT_DIR 없음"; exit 1; }
PY=${PY:-/isaac-sim/python.sh}
L=${L:-"/workspace/IsaacLab/isaaclab.sh -p"}
TASK=${TASK:-Quadruped-v0}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-200}
ARMS=${ARMS:-"S1 S0 S2"}
AUTO_LONG=${AUTO_LONG:-0}
LONG_ITERS=${LONG_ITERS:-5000}
DRY_RUN=${DRY_RUN:-0}
TABLE=${TABLE:-$ROBOT_DIR/screen_table.py}
WS=${WS:-/workspace}
LOGD=$WS/screen
ST=$WS/SCREEN_STATUS.txt
RES=$WS/SCREEN_RESULT.md
REWARD_FILE=quadruped_rewards.py
BAK=quadruped_rewards.baseline.bak
mkdir -p "$LOGD"
status(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }

# ---- 팔 정의 --------------------------------------------------------------------------------
ARM_S0=""                                                                        # 대조군: NCRC 무수정 @8192
ARM_S1=${ARM_S1:-"track_lin_vel_xy_exp=1.5 lin_vel_z_l2=-2.0 ang_vel_xy_l2=-0.05"}  # IsaacLab Go2 원본 복원
ARM_S2=${ARM_S2:-"track_lin_vel_xy_exp=2.0 lin_vel_z_l2=-2.0 ang_vel_xy_l2=-0.05"}  # 복원 + 넓어진 명령 보상
ARM_S3=${ARM_S3:-"track_lin_vel_xy_exp=1.5"}                                        # 추종만 (S1 분해용)
ARM_S4=${ARM_S4:-"track_lin_vel_xy_exp=1.5 lin_vel_z_l2=-2.0 ang_vel_xy_l2=-0.05 feet_air_time=0.1"}

# ---- 0. 전제 확인 -----------------------------------------------------------------------------
for f in train.py "$REWARD_FILE"; do [ -f "$f" ] || { status "!! missing $f"; exit 1; }; done
[ -f "$TABLE" ] || { status "!! $TABLE 없음 — screen_table.py 를 먼저 올려라"; exit 1; }
[ -f "$BAK" ] || cp "$REWARD_FILE" "$BAK"

# 백업이 정말 배포 원본인지 확인한다. 이전 실험이 남긴 파일을 원본으로 착각하면 대조군이 오염된다.
$PY - "$BAK" <<'PEOF' || { status "!! $BAK 가 NCRC 원본이 아니다 — 원본으로 복구 후 재실행"; exit 1; }
import sys
ns = {}; exec(open(sys.argv[1], encoding="utf-8").read(), ns); w = ns["REWARD_WEIGHTS"]
want = {"track_lin_vel_xy_exp": 1.0, "feet_air_time": 0.01, "lin_vel_z_l2": -3.0,
        "ang_vel_xy_l2": -0.08, "action_rate_l2": -0.01}
if w != want:
    sys.exit(f"baseline mismatch: {sorted(w.items())}")
print("baseline OK:", sorted(w.items()))
PEOF

apply_arm(){ # $1=팔 $2=대상 파일 → 원본에서 복사한 뒤 활성 줄의 값만 바꾸고 재파싱으로 검증
  local name=$1 dst=$2 spec
  eval "spec=\${ARM_$name-UNSET}"
  [ "$spec" = "UNSET" ] && { status "!! 팔 $name 정의 없음"; return 1; }
  cp "$BAK" "$dst"
  [ -z "$spec" ] && { status "$name: 무수정"; return 0; }
  $PY - "$dst" $spec <<'PEOF'
import re, sys
path, kvs = sys.argv[1], sys.argv[2:]
src = open(path, encoding="utf-8").read()
want = {}
for kv in kvs:
    k, v = kv.split("=", 1)
    want[k] = float(v)
    pat = re.compile(r'^(\s*"%s"\s*:\s*)[-+0-9.eE]+' % re.escape(k), re.M)   # 주석(#) 줄은 안 걸린다
    src, n = pat.subn(lambda m: m.group(1) + v, src)
    if n != 1:
        sys.exit(f"!! {k}: 활성 줄이 {n}개 (정확히 1개여야 한다 — 오타이거나 주석 처리된 키)")
open(path, "w", encoding="utf-8", newline="").write(src)
ns = {}; exec(src, ns); got = ns["REWARD_WEIGHTS"]
bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
if bad:
    sys.exit(f"!! 재파싱 불일치 {bad}")
print("weights:", sorted(got.items()))
PEOF
}

# GPU 를 쓰기 전에 모든 팔을 임시 파일로 검증한다
for a in $ARMS; do
  apply_arm "$a" "$LOGD/validate_rewards.py" >> "$ST" 2>&1 || { status "!! 팔 $a 검증 실패 — 중단"; tail -3 "$ST"; exit 1; }
done
[ "$DRY_RUN" = "1" ] && { status "DRY_RUN — 검증만 하고 끝"; exit 0; }
status "=== screen start: arms=[$ARMS] envs=$NUM_ENVS iters=$ITERS task=$TASK ==="

# ---- 1. 순차 스크리닝 -------------------------------------------------------------------------
run_arm(){ # $1=팔 $2=iters $3=세션 접두사
  local name=$1 iters=$2 sess="$3$1" log t=0
  log="$LOGD/train_$3$1.log"
  apply_arm "$name" "$REWARD_FILE" 2>&1 | tee -a "$ST"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { status "!! $name 적용 실패"; return 1; }
  : > "$log"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess" "cd $ROBOT_DIR && PYTHONUNBUFFERED=1 $L train.py --task $TASK --num_envs $NUM_ENVS --max_iterations $iters --headless 2>&1 | tee $log; echo TRAIN_DONE >> $log"
  while ! grep -aq "Learning iteration" "$log"; do
    sleep 10; t=$((t+10))
    grep -aqiE "out of memory|illegal memory access" "$log" && { status "!! $name GPU 메모리 실패"; tmux kill-session -t "$sess"; return 2; }
    grep -aq TRAIN_DONE "$log" && { status "!! $name 이 iteration 없이 끝났다:"; tail -5 "$log" | tee -a "$ST"; return 1; }
    [ $t -ge 900 ] && { status "!! $name 15분 동안 첫 iteration 없음"; tmux kill-session -t "$sess"; return 1; }
  done
  status "$name first iteration (t=${t}s) — 보상 파일 읽음"
  cp "$BAK" "$REWARD_FILE"          # 이미 읽었으니 원본으로 돌려놓는다
  sleep 120
  status "$name nvidia-smi: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader) | overflow so far: $(grep -ac 'Patch buffer overflow' "$log")"
  t=0
  while ! grep -aq TRAIN_DONE "$log"; do
    sleep 30; t=$((t+30))
    tmux has-session -t "$sess" 2>/dev/null || { sleep 20; grep -aq TRAIN_DONE "$log" || { status "!! $name tmux 가 TRAIN_DONE 없이 죽음"; return 1; }; }
    [ $((t % 600)) -eq 0 ] && status "$name: $(grep -a 'Learning iteration' "$log" | tail -1 | tr -s ' ')"
  done
  status "$name TRAIN_DONE"
  return 0
}

DONE_ARMS=""
for a in $ARMS; do
  [ -f /workspace/STOP_SCREEN ] && { status "STOP_SCREEN — 중단"; break; }
  if run_arm "$a" "$ITERS" "scr_"; then
    DONE_ARMS="$DONE_ARMS $a"
  fi
  # 매 팔이 끝날 때마다 표를 갱신한다 — 중간에 서버가 죽어도 거기까지는 회수된다
  args=""; for d in $DONE_ARMS; do args="$args $d=$LOGD/train_scr_$d.log"; done
  [ -n "$args" ] && $PY "$TABLE" $args --every 25 --window 50 --out "$RES" > /dev/null 2> "$LOGD/winner.txt"
  status "표 갱신: $RES (완료 팔:$DONE_ARMS)"
done
cp "$BAK" "$REWARD_FILE"

WIN=$(grep -o 'WINNER=[A-Za-z0-9]*' "$LOGD/winner.txt" 2>/dev/null | cut -d= -f2)
status "=== screen done. winner=${WIN:-none} ==="
cat "$RES" | tee -a "$ST"

# ---- 2. (선택) 승자로 장기 학습 ----------------------------------------------------------------
if [ "$AUTO_LONG" = "1" ] && [ -n "$WIN" ]; then
  cat > /tmp/screen_guard.sh <<GEOF
cd $ROBOT_DIR; mkdir -p ckpt_keep
while [ ! -f /workspace/STOP_GUARD ]; do
  for r in logs/rsl_rl/*/*/; do
    [ -d "\$r" ] || continue; b=\$(basename "\$r"); mkdir -p "ckpt_keep/\$b"
    for f in "\$r"model_*.pt; do [ -f "\$f" ] || continue; n=\$(basename "\$f")
      [ -f "ckpt_keep/\$b/\$n" ] || cp "\$f" "ckpt_keep/\$b/\$n"; done
    [ -f "ckpt_keep/\$b/env.yaml" ] || cp "\$r/params/env.yaml" "ckpt_keep/\$b/env.yaml" 2>/dev/null
  done; sleep 45
done
GEOF
  tmux kill-session -t scrguard 2>/dev/null; tmux new -d -s scrguard 'bash /tmp/screen_guard.sh'
  status "guard started (model_best 결함 대비 45초 체크포인트 복사)"
  status "=== LONG: $WIN x $LONG_ITERS iter @ $NUM_ENVS envs ==="
  run_arm "$WIN" "$LONG_ITERS" "long_"
  cp "$BAK" "$REWARD_FILE"
  $PY "$TABLE" "long_$WIN=$LOGD/train_long_$WIN.log" --every 250 --window 100 --out /workspace/LONG_RESULT.md > /dev/null 2>&1
  status "LONG 끝. /workspace/LONG_RESULT.md, 체크포인트는 ckpt_keep/"
fi

# ---- 3. 보초: 학습이 없으면 64-env 더미로 워크스페이스를 살려둔다 --------------------------------
cat > /tmp/screen_sentinel.sh <<SEOF
cd $ROBOT_DIR
while [ ! -f /workspace/STOP_KEEPALIVE ]; do
  real=\$(pgrep -f "train.py --task $TASK --num_envs [0-9]" | while read p; do ps -o args= -p \$p; done | grep -v -- "--num_envs 64 " | wc -l)
  if [ "\$real" -gt 0 ]; then
    tmux kill-session -t sentinel_train 2>/dev/null
  elif ! tmux has-session -t sentinel_train 2>/dev/null; then
    tmux new -d -s sentinel_train "cd $ROBOT_DIR && $L train.py --task $TASK --num_envs 64 --max_iterations 100000 --headless > /tmp/sentinel_train.log 2>&1"
    echo "\$(date '+%m-%d %H:%M:%S') sentinel: dummy train up" >> $ST
  fi
  sleep 60
done
tmux kill-session -t sentinel_train 2>/dev/null
SEOF
tmux kill-session -t sentinel 2>/dev/null; tmux new -d -s sentinel 'bash /tmp/screen_sentinel.sh'
status "sentinel started — 끄려면 touch /workspace/STOP_KEEPALIVE"
