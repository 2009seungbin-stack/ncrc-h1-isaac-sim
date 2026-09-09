#!/bin/bash
# r3_pipeline.sh — 3라운드 무인 파이프라인 (2026-09-08)
#   R3A / R3C 를 8192 envs 로 동시에 10k iter 학습 → 후반 ckpt export → T-EVAL(7) + Stress(X)
#   → 순위표 → /workspace/R3_bundle.tgz → keep-alive (GPU 를 계속 써서 워크스페이스 삭제 방지)
# 실행:  tmux new -d -s r3 'bash /workspace/training/humanoid/r3_pipeline.sh'
# 상태:  tail -20 /workspace/R3_STATUS.txt      결과: cat /workspace/R3_RESULT.md
# 종료:  touch /workspace/STOP_KEEPALIVE   (keep-alive 루프가 다음 주기에 멈춘다)
set -u
cd /workspace/training/humanoid || exit 1
PY=/isaac-sim/python.sh
L="/workspace/IsaacLab/isaaclab.sh -p"
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-10000}
EVAL_ITERS=${EVAL_ITERS:-"9000 10000"}
ST=/workspace/R3_STATUS.txt
RES=/workspace/R3_RESULT.md
LOGD=/workspace/r3logs
mkdir -p "$LOGD" keep_R3 eval_results ckpt_keep
status(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }
status "=== R3 pipeline start  envs=$NUM_ENVS iters=$ITERS eval_iters=[$EVAL_ITERS]"
for f in train.py play.py t_eval.py humanoid_rewards_R3A.py humanoid_rewards_R3C.py; do
  [ -f "$f" ] || { status "!! missing $f"; exit 1; }
done
[ -f humanoid_rewards.baseline.bak ] || cp humanoid_rewards.py humanoid_rewards.baseline.bak

# ---------------------------------------------------------------- 0. 체크포인트 가드 (모든 run)
cat > /tmp/guard.sh <<'GEOF'
cd /workspace/training/humanoid; mkdir -p ckpt_keep
while true; do
  for r in logs/rsl_rl/humanoid/*/; do
    [ -d "$r" ] || continue; b=$(basename "$r"); mkdir -p "ckpt_keep/$b"
    for f in "$r"model_*.pt; do [ -f "$f" ] || continue; n=$(basename "$f")
      [ -f "ckpt_keep/$b/$n" ] || cp "$f" "ckpt_keep/$b/$n"; done
    [ -f "ckpt_keep/$b/env.yaml" ] || cp "$r/params/env.yaml" "ckpt_keep/$b/env.yaml" 2>/dev/null
  done; sleep 45
done
GEOF
tmux kill-session -t guard 2>/dev/null; tmux new -d -s guard 'bash /tmp/guard.sh'
status "guard started"

show_weights(){ $PY - "$1" <<'PEOF'
import sys, yaml
class L(yaml.SafeLoader): pass
L.add_multi_constructor('tag:yaml.org,2002:python/', lambda l, s, n: l.construct_sequence(n) if isinstance(n, yaml.SequenceNode) else (l.construct_mapping(n) if isinstance(n, yaml.MappingNode) else l.construct_scalar(n)))
e = yaml.load(open(sys.argv[1]), Loader=L)
print({k: (v or {}).get('weight') for k, v in e['rewards'].items() if v})
PEOF
}

declare -A RUN
# ---------------------------------------------------------------- 1. run 시작
start_run(){ # $1=이름 $2=보상파일  → RUN[$1]=run dir 이름. 반환 0 ok / 1 no-run / 2 OOM
  local name=$1 rw=$2 before nd t=0
  before=$(ls -1d logs/rsl_rl/humanoid/*/ 2>/dev/null | sort)
  cp "$rw" humanoid_rewards.py
  $PY -c "exec(open('humanoid_rewards.py').read()); print('$name weights:', sorted(REWARD_WEIGHTS.items()))" | tee -a "$ST"
  tmux kill-session -t "$name" 2>/dev/null
  tmux new -d -s "$name" "cd /workspace/training/humanoid && PYTHONUNBUFFERED=1 $L train.py --task Humanoid-v0 --num_envs $NUM_ENVS --max_iterations $ITERS --headless 2>&1 | tee $LOGD/train_$name.log; echo TRAIN_DONE >> $LOGD/train_$name.log"
  while [ $t -lt 900 ]; do
    sleep 10; t=$((t+10))
    nd=$(comm -13 <(echo "$before") <(ls -1d logs/rsl_rl/humanoid/*/ 2>/dev/null | sort) | head -1)
    [ -n "$nd" ] && [ -f "$nd/params/env.yaml" ] && break
    nd=""
    grep -qi "out of memory" "$LOGD/train_$name.log" 2>/dev/null && { status "!! $name OOM at start"; tmux kill-session -t "$name"; return 2; }
  done
  [ -z "$nd" ] && { status "!! $name: run dir not created in 15 min"; return 1; }
  RUN[$name]=$(basename "$nd"); status "$name -> run ${RUN[$name]}"
  show_weights "$nd/params/env.yaml" | tee -a "$ST"
  # 첫 iteration 로그가 찍힐 때까지 기다린다 (이후엔 보상 파일을 바꿔도 이 run 에 영향 없음)
  t=0; while [ $t -lt 900 ]; do
    sleep 10; t=$((t+10))
    grep -qi "iteration" "$LOGD/train_$name.log" && break
    grep -qi "out of memory" "$LOGD/train_$name.log" && { status "!! $name OOM before first iteration"; tmux kill-session -t "$name"; return 2; }
  done
  status "$name first iteration seen (t=${t}s)"
  return 0
}

start_run R3A humanoid_rewards_R3A.py; rcA=$?
C_DEFERRED=0
if [ $rcA -eq 0 ]; then
  start_run R3C humanoid_rewards_R3C.py; rcC=$?
  [ $rcC -eq 2 ] && C_DEFERRED=1
else
  status "!! R3A failed to start (rc=$rcA) — aborting"; exit 1
fi

# ---------------------------------------------------------------- 2. 2분 실측
sleep 120
status "nvidia-smi: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader)"
for n in R3A R3C; do
  [ -f "$LOGD/train_$n.log" ] && status "$n: $(grep -i 'iteration time' "$LOGD/train_$n.log" | tail -1 | tr -s ' ')  |  $(grep -i 'learning iteration' "$LOGD/train_$n.log" | tail -1 | tr -s ' ')"
done
if [ $C_DEFERRED -eq 0 ] && grep -qi "out of memory" "$LOGD/train_R3C.log" 2>/dev/null; then
  status "!! R3C OOM after start — deferring to sequential"; tmux kill-session -t R3C 2>/dev/null; C_DEFERRED=1
fi

# ---------------------------------------------------------------- 3. 학습 종료 대기
wait_done(){ # $1=이름
  local n=$1 last=0
  while ! grep -q TRAIN_DONE "$LOGD/train_$n.log" 2>/dev/null; do
    sleep 60; last=$((last+1))
    if [ $((last % 30)) -eq 0 ]; then
      status "$n progress: $(grep -i 'learning iteration' "$LOGD/train_$n.log" | tail -1 | tr -s ' ') | ckpts=$(ls ckpt_keep/${RUN[$n]}/model_*.pt 2>/dev/null | wc -l)"
    fi
    tmux has-session -t "$n" 2>/dev/null || { sleep 30; grep -q TRAIN_DONE "$LOGD/train_$n.log" || { status "!! $n tmux died without TRAIN_DONE"; break; }; }
  done
  status "$n TRAIN_DONE  (ckpts kept: $(ls ckpt_keep/${RUN[$n]}/model_*.pt 2>/dev/null | wc -l))"
}
wait_done R3A
if [ $C_DEFERRED -eq 1 ]; then
  status "starting R3C sequentially"
  start_run R3C humanoid_rewards_R3C.py && wait_done R3C
elif [ -n "${RUN[R3C]:-}" ]; then
  wait_done R3C
fi
cp humanoid_rewards.baseline.bak humanoid_rewards.py

# ---------------------------------------------------------------- 4. export
export_ckpt(){ # $1=이름 $2=iter → keep_R3/<name>_<iter>.pt
  local n=$1 it=$2 run=${RUN[$n]} ck rd pol
  rd=logs/rsl_rl/humanoid/$run
  ck=ckpt_keep/$run/model_$it.pt
  if [ ! -f "$ck" ]; then
    ck=$(ls -1 ckpt_keep/$run/model_*.pt 2>/dev/null | sort -V | awk -v m="$it" '{split($0,a,"_"); sub(".pt","",a[2]); if (a[2]+0<=m+0) last=$0} END{print last}')
    [ -z "$ck" ] && { status "!! $n: no ckpt <= $it"; return 1; }
    status "$n: model_$it.pt missing, using $(basename $ck)"
  fi
  mkdir -p "$rd/params"; cp "$ck" "$rd/"
  PYTHONUNBUFFERED=1 $L play.py --task Humanoid-v0 --num_envs 32 --headless --load_run "$run" \
      --checkpoint "/workspace/training/humanoid/$rd/$(basename $ck)" > "$LOGD/export_${n}_$it.log" 2>&1
  pol=$(ls -1t "$rd/exported/policy.pt" exported/policy.pt 2>/dev/null | head -1)
  [ -z "$pol" ] && { status "!! $n $it: export produced no policy.pt (see $LOGD/export_${n}_$it.log)"; return 1; }
  cp "$pol" "keep_R3/${n}_$it.pt"; cp "$ck" "keep_R3/${n}_${it}_model.pt"
  cp "ckpt_keep/$run/env.yaml" "keep_R3/${n}_env.yaml" 2>/dev/null; cp "humanoid_rewards_$n.py" keep_R3/
  status "exported $n $it -> keep_R3/${n}_$it.pt md5=$(md5sum keep_R3/${n}_$it.pt | cut -c1-8) (src ckpt $(basename $ck))"
}
POLICIES=()
for n in R3A R3C; do
  [ -n "${RUN[$n]:-}" ] || continue
  for it in $EVAL_ITERS; do export_ckpt "$n" "$it" && POLICIES+=("${n}_$it"); done
done
[ -f keep_S2/policy.pt ] && { cp keep_S2/policy.pt keep_R3/S2_2900.pt; POLICIES+=("S2_2900"); status "S2 baseline found — will evaluate too"; }
status "policies to evaluate: ${POLICIES[*]}"

# ---------------------------------------------------------------- 5. T-EVAL(7) + Stress(X)
teval(){ # $1=policy id $2=scenarios $3=tag $4...=extra args
  local pid=$1 sc=$2 tag=$3; shift 3
  PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy "keep_R3/$pid.pt" --scenarios "$sc" --run_id "$pid" \
      --out "eval_results/R3_${pid}_$tag.json" "$@" > "$LOGD/teval_${pid}_$tag.log" 2>&1 \
    || status "!! t_eval $pid $tag failed (see $LOGD/teval_${pid}_$tag.log)"
}
for pid in "${POLICIES[@]}"; do
  status "T-EVAL start $pid"
  teval "$pid" H1,H2,H3,H4,H7 flat
  teval "$pid" H5 rough
  teval "$pid" H6 slope
  teval "$pid" X1,X4 xflat
  teval "$pid" X2 xrough --rough_min 0.06 --rough_max 0.10
  teval "$pid" X3 xslope --slope_deg 15
  teval "$pid" X5 xpush --rough_min 0.04 --rough_max 0.08
  $PY t_eval.py --merge eval_results/R3_${pid}_*.json --run_id "$pid" --out "eval_results/R3_${pid}.json" > "$LOGD/merge_$pid.log" 2>&1
  status "T-EVAL done $pid: $($PY -c "import json;d=json.load(open('eval_results/R3_$pid.json'));s=d.get('stress') or {};print('sim',d['sim_score'],'Hfalls',d['falls'],'Xsurv',s.get('survival'),'XLCB',s.get('survival_lcb'),'Xfalls',s.get('falls'))" 2>/dev/null)"
done

# ---------------------------------------------------------------- 6. 순위표
$PY - > "$RES" <<'PEOF'
import json, glob, os
rows = []
for p in sorted(glob.glob('eval_results/R3_*.json')):
    b = os.path.basename(p)
    if b.count('_') != 2 and not b.startswith('R3_S2'):
        pass
    if any(b.endswith(f'_{t}.json') for t in ('flat','rough','slope','xflat','xrough','xslope','xpush')):
        continue
    d = json.load(open(p)); s = d.get('stress') or {}
    rows.append((d['run_id'], d['sim_score'], d['falls'], s.get('survival'), s.get('survival_lcb'), s.get('falls'),
                 {k: v['score'] for k, v in d['scenarios'].items() if k.startswith('H')},
                 {k: v['survival'] for k, v in d['scenarios'].items() if k.startswith('X')}))
rows.sort(key=lambda r: (-(r[4] or 0), -r[1]))
print('# R3 결과 (정렬: Stress 생존 LCB ↓, SimScore ↓)\n')
print('| policy | SimScore | H낙상 | X생존 | X LCB | X낙상 | ' + ' | '.join(f'H{i}' for i in range(1,8)) + ' | ' + ' | '.join(f'X{i}' for i in range(1,6)) + ' |')
print('|---|---:|---:|---:|---:|---:|' + '---:|'*12)
for r in rows:
    h = ' | '.join(f"{r[6].get(f'H{i}', float('nan')):.3f}" for i in range(1,8))
    x = ' | '.join(f"{r[7].get(f'X{i}', float('nan')):.3f}" for i in range(1,6))
    print(f'| {r[0]} | {r[1]:.2f} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {h} | {x} |')
print('\n게이트: SimScore >= 63.32 그리고 X LCB >= S2_2900 의 X LCB (S2 미평가면 후보 중 최고 LCB).')
PEOF
status "RESULT written: $RES"; cat "$RES" | tee -a "$ST"

# ---------------------------------------------------------------- 7. 번들
TF=""
for n in R3A R3C; do [ -n "${RUN[$n]:-}" ] && TF="$TF logs/rsl_rl/humanoid/${RUN[$n]}/events.out.tfevents.* logs/rsl_rl/humanoid/${RUN[$n]}/params"; done
tar czf /workspace/R3_bundle.tgz keep_R3 eval_results/R3_*.json $RES $ST $LOGD/train_R3*.log $TF 2>/dev/null
status "bundle: $(ls -l /workspace/R3_bundle.tgz | awk '{print $5}') bytes md5=$(md5sum /workspace/R3_bundle.tgz | cut -c1-12)"
status "=== PIPELINE DONE. 회수: scp da-perfect25@23.robotics.connect.or.kr:/workspace/R3_bundle.tgz .   종료: touch /workspace/STOP_KEEPALIVE"

# ---------------------------------------------------------------- 8. keep-alive (GPU 사용 유지)
KA=${KEEPALIVE_POLICY:-${POLICIES[0]:-}}
while [ ! -f /workspace/STOP_KEEPALIVE ]; do
  if [ -n "$KA" ] && [ -f "keep_R3/$KA.pt" ]; then
    PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy "keep_R3/$KA.pt" --scenarios H1 --seeds 103 --num_envs 64 \
        --run_id keepalive --out /tmp/keepalive.json > /tmp/keepalive.log 2>&1
  fi
  echo "$(date '+%m-%d %H:%M:%S') keepalive tick" >> "$ST"
  for i in $(seq 1 60); do [ -f /workspace/STOP_KEEPALIVE ] && break; sleep 5; done
done
status "keep-alive stopped"
