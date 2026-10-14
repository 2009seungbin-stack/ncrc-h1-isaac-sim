#!/bin/bash
# longrun_go2.sh — 설정 하나를 처음부터 길게 학습하고, 끝나자마자 채점·export·묶기까지 사람 없이 이어 간다.
#
# 왜 (2026-09-20 전략 v3):
#   · 가장 비싼 run(8~9시간)이 "끝난 뒤 30분 안에 사람이 접속해야 한다"에 걸려 있었다. 학습이 끝나도
#     채점(약 40분)과 export 가 이어 돌므로 그동안 워크스페이스가 살아 있고, 도착하면 tgz 하나만 받으면 된다.
#   · 중간 지점(800/2000/4000/마지막)을 같은 시험으로 채점한다 → 흔들림이 학습량으로 줄어드는지,
#     계단이 풀리는지, 800 iter 스크리닝이 장기를 예측하는지, 어디서 포화하는지가 한 번에 나온다.
#
# 사용:
#   bash longrun_go2.sh                       # 0단계(검증)만 — GPU 안 씀
#   CONFIRM=1 bash longrun_go2.sh             # C0(track_lin 2.0) 6000 iter
#   CONFIRM=1 SETTING=C0 ITERS=6000 MIDS="800 2000 4000" bash longrun_go2.sh
#   상태: cat /workspace/LONGRUN_STATUS.txt   (tmux 안에서 돌릴 것: tmux new -d -s lr 'CONFIRM=1 bash longrun_go2.sh')
set -u
R=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$R" || { echo "!! $R 없음"; exit 1; }
L=${L:-"/workspace/IsaacLab/isaaclab.sh -p"}
PY=${PY:-/isaac-sim/python.sh}
TASK=${TASK:-Quadruped-v0}
WS=${WS:-/workspace}
SETTING=${SETTING:-C0}
ITERS=${ITERS:-6000}
MIDS=${MIDS:-"800 2000 4000"}
ENVS=${ENVS:-6144}
CONFIRM=${CONFIRM:-0}
ST=$WS/LONGRUN_STATUS.txt
say(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }

for f in sweep_go2.sh teval_arms.sh t_eval_go2.py; do [ -f "$f" ] || { echo "!! $R/$f 없음 — all_install.tgz 부터"; exit 1; }; done
if [ "$CONFIRM" != "1" ]; then
  TRUNK_RUN=SKIP ARMS="$SETTING" bash sweep_go2.sh      # 보상 검증만 하고 멈춘다
  echo "=== 0단계 끝. 위 weights 줄을 확인하고 CONFIRM=1 로 다시 실행 ==="; exit 0
fi

# ---- 1. 학습 (scratch) + 고정 명령 영상 ----------------------------------------------------------
say "=== LONGRUN $SETTING $ITERS iter @ $ENVS envs ==="
TRUNK_RUN=SKIP ARMS="$SETTING" BRANCH_ITERS="$ITERS" ENVS="$ENVS" SCENARIOS="${SCENARIOS:-G1 G5}" \
  CONFIRM=1 bash sweep_go2.sh
RUN=$(grep "^$SETTING=" "$WS/SWEEP_RUNS.txt" 2>/dev/null | tail -1 | cut -d= -f2)
[ -n "$RUN" ] || { say "!! 학습 run 을 못 찾았다 — SWEEP_STATUS.txt 확인"; exit 1; }
LAST=$(ls "ckpt_keep/$RUN"/model_[0-9]*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
say "run=$RUN 마지막 ckpt=model_$LAST.pt"
[ "$LAST" -ge $((ITERS - 200)) ] || say "!! 마지막 ckpt 가 목표($ITERS)에 못 미친다 — 학습이 중간에 죽었을 수 있다"

# ---- 2. 제출용 export 를 채점보다 먼저 (시간이 모자라면 이것만은 남아야 한다) ---------------------
CKABS="$R/ckpt_keep/$RUN/model_$LAST.pt"
RD="logs/rsl_rl/quadruped/$RUN"
[ -d "$RD" ] || RD=$(ls -d logs/rsl_rl/*/"$RUN" | head -1)
mkdir -p exported; [ -f exported/model_best.pt ] || cp "$CKABS" exported/model_best.pt   # play.py 래퍼가 이 파일 없으면 거부
cat > /tmp/play_lr.sh <<EOF
cd $R
PYTHONUNBUFFERED=1 $L play.py --task $TASK --num_envs 32 --headless --load_run $RUN --checkpoint $CKABS > $WS/export_play.log 2>&1
EOF
# play.py 는 policy.pt 를 **체크포인트가 있는 폴더의 exported/** 에 만든다 (2026-09-21 실측: ckpt_keep/<run>/exported/).
t0=$(date +%s); pol=""; CANDS="$(dirname "$CKABS")/exported/policy.pt $RD/exported/policy.pt"
tmux kill-session -t play_lr 2>/dev/null; tmux new -d -s play_lr 'bash /tmp/play_lr.sh'
t=0
while [ $t -lt 600 ]; do sleep 5; t=$((t+5))
  for c in $CANDS; do
    [ -f "$c" ] && [ "$(stat -c %Y "$c")" -ge "$t0" ] && [ $(( $(date +%s) - $(stat -c %Y "$c") )) -ge 20 ] && pol=$c
  done
  [ -n "$pol" ] && break
done
for p in $(pgrep -f "play.py.*--load_run $RUN"); do kill "$p" 2>/dev/null; done   # play.py 는 헤드리스에서 스스로 안 끝난다
sleep 3; tmux kill-session -t play_lr 2>/dev/null
mkdir -p "$WS/submit"
if [ -n "$pol" ]; then
  cp "$pol" "$WS/submit/policy.pt"; cp "$RD/params/env.yaml" "$WS/submit/env.yaml"
  say "export OK policy.pt md5 $(md5sum "$WS/submit/policy.pt" | cut -c1-8) (ckpt model_$LAST md5 $(md5sum "$CKABS" | cut -c1-8))"
else
  say "!! export 실패 — export_play.log 끝:"; tail -12 "$WS/export_play.log" | tee -a "$ST"
  cp "$CKABS" "$WS/submit/model_$LAST.pt"      # 체크포인트라도 회수하면 다른 서버에서 export 할 수 있다
fi

# ---- 3. 중간 지점 + 마지막을 같은 시험으로 채점 ---------------------------------------------------
LABELS=""; RUNMAP=""
for m in $MIDS; do [ "$m" -lt "$LAST" ] && { LABELS="$LABELS ${SETTING}_$m"; RUNMAP="$RUNMAP ${SETTING}_$m=$RUN:$m"; }; done
LABELS="$LABELS ${SETTING}_$LAST"; RUNMAP="$RUNMAP ${SETTING}_$LAST=$RUN"
say "채점: $LABELS"
ARMS="$LABELS" RUNS="$RUNMAP" bash teval_arms.sh
cat "$WS/TEVAL_ARMS.txt" >> "$ST"

# ---- 4. 하나로 묶기 (다운로드는 오리진당 첫 1회만 확실하다) --------------------------------------
cp "$WS"/submit/* "$WS/out/" 2>/dev/null
cp "$ST" "$WS/SWEEP_STATUS.txt" "$WS/out/" 2>/dev/null
cd "$WS/out" && tar czf "$WS/longrun_out.tgz" . && say "BUNDLE $WS/longrun_out.tgz $(stat -c %s "$WS/longrun_out.tgz") bytes"
say "=== LONGRUN DONE — longrun_out.tgz 하나를 첫 다운로드로 받을 것 ==="
