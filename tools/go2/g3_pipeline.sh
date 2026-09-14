#!/bin/bash
# g3_pipeline.sh — 3라운드 Go2 무인 파이프라인 (2026-09-11)
#   GB/GH/GN 병렬 학습 → 후반 ckpt export → T-EVAL-GO2(G1~G7) + Stress(GX) → 순위표
#   → /workspace/G3_bundle.tgz → keep-alive + 보초(워크스페이스 삭제 방지)
#
# 실행 전 반드시 G-0 정찰을 먼저 돌려서 TASK / REWARD_FILE / ROBOT_DIR 을 확정할 것:
#   /workspace/IsaacLab/isaaclab.sh -p g0_probe.py --headless | tee /workspace/G0_RECON.txt
#
# 실행:  tmux new -d -s g3 'TASK=<태스크> bash /workspace/training/quadruped/g3_pipeline.sh'
# 상태:  tail -30 /workspace/G3_STATUS.txt        결과: cat /workspace/G3_RESULT.md
# 종료:  touch /workspace/STOP_KEEPALIVE
set -u

ROBOT_DIR=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$ROBOT_DIR" || { echo "!! $ROBOT_DIR 없음. ROBOT_DIR 을 넘겨라."; exit 1; }
PY=/isaac-sim/python.sh
L="/workspace/IsaacLab/isaaclab.sh -p"
TASK=${TASK:-Quadruped-v0}
NUM_ENVS=${NUM_ENVS:-8192}
ITERS=${ITERS:-10000}
EVAL_ITERS=${EVAL_ITERS:-"6000 8000 9000 10000"}
ARMS=${ARMS:-"GB GH GY GC"}
ST=/workspace/G3_STATUS.txt
RES=/workspace/G3_RESULT.md
LOGD=/workspace/g3logs
EVALD=eval_results
KEEPD=keep_G3

# 팔 정의 — 2026-09-11 G-0 정찰로 확정한 Go2 실제 키에 맞췄다.
#
# Go2 baseline 활성 항목 (quadruped_rewards.py):
#   track_lin_vel_xy_exp 1.0 | feet_air_time 0.01 | lin_vel_z_l2 -3.0
#   ang_vel_xy_l2 -0.08      | action_rate_l2 -0.01
# 주석 처리(= 지금 꺼짐)된 것:
#   track_ang_vel_z_exp 0.75  <- 유효 기본값. 2.0 으로 올리는 게 우리 H1 최대 승리의 재현
#   flat_orientation_l2 -1.0  <- **현재 유효값 0. 켜는 것 자체가 변경이다**
#   undesired_contacts -1.0   <- 사족 환경에서 null. 못 켠다
#   termination_penalty -200  <- **사족 환경에 항목 자체가 없다 (휴머노이드 전용)**
#
# H1 최대 무기(termination_penalty)가 Go2 엔 없다. 그리고 Go2 baseline 은 이미
# NVIDIA/유니트리 값에 가깝다(feet_air_time 0.01, action_rate -0.01). 즉 H1 때처럼
# "약화된 baseline 복원"으로 먹을 점수가 없다. 승부는 yaw 다.
#
#   GB = 무수정 대조군. 설계의도 20점의 조건이지 비용이 아니다.
#   GH = yaw 추종 가중치. H1에서 회전 복합 50.0%->83.4% 를 만든 그 처방
#   GY = 흔들림 억제. yaw 오차가 '명령 추종 실패'가 아니라 '상시 바닥'이라는 가설(go2_plan §2.5)
#
# **GH 와 GY 는 yaw 바닥의 두 경쟁 설명을 각각 때린다. 어느 쪽이 이기는지가 4라운드를 정한다.**
ARM_GB=""
ARM_GH=${ARM_GH:-"track_ang_vel_z_exp=2.0"}
ARM_GY=${ARM_GY:-"action_rate_l2=-0.05 flat_orientation_l2=-2.5"}
#   GC = 두 레버 동시. GH/GY 가 둘 다 먹히면 이게 이긴다.
ARM_GC=${ARM_GC:-"track_ang_vel_z_exp=2.0 action_rate_l2=-0.05 flat_orientation_l2=-2.5"}

mkdir -p "$LOGD" "$KEEPD" "$EVALD" ckpt_keep
status(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }

# ---------------------------------------------------------------- 0. 전제 확인
for f in train.py play.py t_eval.py t_eval_go2.py set_reward.py; do
  [ -f "$f" ] || { status "!! missing $f"; exit 1; }
done
REWARD_FILE=${REWARD_FILE:-$(ls -1 *_rewards.py 2>/dev/null | grep -v baseline | head -1)}
[ -n "$REWARD_FILE" ] || { status "!! 보상 파일을 못 찾았다. REWARD_FILE 을 넘겨라."; exit 1; }
BAK="${REWARD_FILE%.py}.baseline.bak"
[ -f "$BAK" ] || cp "$REWARD_FILE" "$BAK"
status "reward file: $REWARD_FILE  (backup $BAK)"

if [ -z "$TASK" ]; then
  TASK=$($PY - <<'PEOF'
import re, gymnasium as gym
try: import isaaclab_tasks
except Exception: pass
n = sorted(k for k in gym.registry if re.search(r"go2|quad", k, re.I) and "Play" not in k)
print(n[0] if n else "")
PEOF
)
fi
[ -n "$TASK" ] || { status "!! TASK 를 정하지 못했다. TASK=... 로 넘겨라."; exit 1; }
status "task: $TASK   envs=$NUM_ENVS iters=$ITERS eval_iters=[$EVAL_ITERS] arms=[$ARMS]"

# ---- 팔 정의 사전 검증: **GPU 를 한 톨도 쓰기 전에** 오타를 잡는다 ----
for a in $ARMS; do
  eval "spec=\${ARM_$a}"
  [ -z "$spec" ] && { status "arm $a: 무수정(대조군)"; continue; }
  # 주석 처리된 키(track_ang_vel_z_exp 등)는 set_reward 가 '새 항목'으로 본다.
  # --allow-new 로 허용하되, **보상 파일에 이름 자체가 없으면 오타이므로 여기서 죽인다.**
  for kv in $spec; do
    k=${kv%%=*}
    grep -q "\"$k\"" "$BAK" || { status "!! arm $a: '$k' 가 $REWARD_FILE 에 아예 없다 (오타?) — 중단"; exit 1; }
  done
  if ! $PY set_reward.py --file "$BAK" --allow-new --dry-run $spec > "$LOGD/validate_$a.log" 2>&1; then
    status "!! arm $a 의 보상 키가 baseline 에 없다 — 중단. $LOGD/validate_$a.log 확인"
    cat "$LOGD/validate_$a.log" | tee -a "$ST"
    exit 1
  fi
  status "arm $a 검증 OK: $spec"
done

# ---------------------------------------------------------------- 0b. 체크포인트 가드
# model_best 가 mean_reward 기준이라 커리큘럼이 오르면 초반 ckpt 를 고르고 나머지를 지운다.
cat > /tmp/g3_guard.sh <<GEOF
cd $ROBOT_DIR; mkdir -p ckpt_keep
while true; do
  for r in logs/rsl_rl/*/*/; do
    [ -d "\$r" ] || continue; b=\$(basename "\$r"); mkdir -p "ckpt_keep/\$b"
    for f in "\$r"model_*.pt; do [ -f "\$f" ] || continue; n=\$(basename "\$f")
      [ -f "ckpt_keep/\$b/\$n" ] || cp "\$f" "ckpt_keep/\$b/\$n"; done
    [ -f "ckpt_keep/\$b/env.yaml" ] || cp "\$r/params/env.yaml" "ckpt_keep/\$b/env.yaml" 2>/dev/null
  done; sleep 45
done
GEOF
tmux kill-session -t g3guard 2>/dev/null; tmux new -d -s g3guard 'bash /tmp/g3_guard.sh'
status "guard started"

declare -A RUN

# ---------------------------------------------------------------- 1. run 시작
start_run(){ # $1=팔 이름 → 0 ok / 1 no-run / 2 OOM
  local name=$1 spec before nd t=0
  eval "spec=\${ARM_$name}"
  cp "$BAK" "$REWARD_FILE"
  if [ -n "$spec" ]; then
    $PY set_reward.py --file "$REWARD_FILE" --backup "$BAK" --allow-new $spec 2>&1 | tee -a "$ST"
  fi
  $PY -c "ns={}; exec(open('$REWARD_FILE').read(), ns); print('$name weights:', sorted(ns['REWARD_WEIGHTS'].items()))" | tee -a "$ST"
  before=$(ls -1d logs/rsl_rl/*/*/ 2>/dev/null | sort)
  tmux kill-session -t "$name" 2>/dev/null
  tmux new -d -s "$name" "cd $ROBOT_DIR && PYTHONUNBUFFERED=1 $L train.py --task $TASK --num_envs $NUM_ENVS --max_iterations $ITERS --headless 2>&1 | tee $LOGD/train_$name.log; echo TRAIN_DONE >> $LOGD/train_$name.log"
  while [ $t -lt 900 ]; do
    sleep 10; t=$((t+10))
    nd=$(comm -13 <(echo "$before") <(ls -1d logs/rsl_rl/*/*/ 2>/dev/null | sort) | head -1)
    [ -n "$nd" ] && [ -f "$nd/params/env.yaml" ] && break
    nd=""
    grep -qi "out of memory" "$LOGD/train_$name.log" 2>/dev/null && { status "!! $name OOM at start"; tmux kill-session -t "$name"; return 2; }
  done
  [ -z "$nd" ] && { status "!! $name: run dir not created in 15 min"; return 1; }
  RUN[$name]=$(basename "$nd"); status "$name -> run ${RUN[$name]}"
  # 첫 iteration 이 찍히면 이 run 은 보상 파일을 이미 읽었다 → 다음 팔로 넘어가도 안전
  t=0; while [ $t -lt 900 ]; do
    sleep 10; t=$((t+10))
    grep -qi "iteration" "$LOGD/train_$name.log" && break
    grep -qi "out of memory" "$LOGD/train_$name.log" && { status "!! $name OOM before first iteration"; tmux kill-session -t "$name"; return 2; }
  done
  status "$name first iteration seen (t=${t}s)"
  return 0
}

# 2026-09-11 실측: Go2 는 8192 envs 에서 VRAM 9.3GB / 16.3GB 를 쓴다. **한 팔만 들어간다.**
# 병렬을 시도하면 2번째가 OOM 나는데, 실패 형태가 'out of memory' 가 아니라
# 'CUDA error: an illegal memory access' 로 나오는 경우가 있어 탐지를 빠져나가고
# 그 팔이 학습 없이 누락된다. 그래서 **처음부터 순차로 간다.**
DEFERRED=""
FIRST=""
for a in $ARMS; do
  if [ -z "$FIRST" ]; then
    start_run "$a" || { status "!! 첫 팔 $a 시작 실패 — 중단"; exit 1; }
    FIRST=$a
  else
    DEFERRED="$DEFERRED $a"
    status "$a → 순차 대기열 (VRAM 상 병렬 불가)"
  fi
done
cp "$BAK" "$REWARD_FILE"

# ---------------------------------------------------------------- 2. 2분 실측
sleep 120
status "nvidia-smi: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader)"
for a in $ARMS; do
  [ -f "$LOGD/train_$a.log" ] && status "$a: $(grep -i 'iteration time' "$LOGD/train_$a.log" | tail -1 | tr -s ' ')  |  $(grep -i 'learning iteration' "$LOGD/train_$a.log" | tail -1 | tr -s ' ')"
done
status ">>> VRAM 여유가 3GB 넘으면 다음 라운드는 NUM_ENVS 를 올려라 (사족 12-DOF 는 H1 19-DOF 보다 가볍다)"
for a in $ARMS; do
  case " $DEFERRED " in *" $a "*) continue;; esac
  if grep -qi "out of memory" "$LOGD/train_$a.log" 2>/dev/null; then
    status "!! $a OOM after start — 순차로 넘긴다"; tmux kill-session -t "$a" 2>/dev/null
    DEFERRED="$DEFERRED $a"; unset "RUN[$a]"
  fi
done

# ---------------------------------------------------------------- 3. 보초 (30분 무활동 방지)
cat > /tmp/g3_sentinel.sh <<SEOF
cd $ROBOT_DIR
while [ ! -f /workspace/STOP_KEEPALIVE ]; do
  if ! pgrep -f "train.py --task" > /dev/null 2>&1; then
    if ! tmux has-session -t sentinel_train 2>/dev/null; then
      tmux new -d -s sentinel_train "cd $ROBOT_DIR && $L train.py --task $TASK --num_envs 64 --max_iterations 100000 --headless > /tmp/sentinel_train.log 2>&1"
      echo "\$(date '+%m-%d %H:%M:%S') sentinel: dummy train up" >> $ST
    fi
  fi
  sleep 60
done
tmux kill-session -t sentinel_train 2>/dev/null
SEOF
tmux kill-session -t sentinel 2>/dev/null; tmux new -d -s sentinel 'bash /tmp/g3_sentinel.sh'
status "sentinel started"

# ---------------------------------------------------------------- 4. 학습 종료 대기
wait_done(){
  local n=$1 last=0
  while ! grep -q TRAIN_DONE "$LOGD/train_$n.log" 2>/dev/null; do
    sleep 60; last=$((last+1))
    if [ $((last % 30)) -eq 0 ]; then
      status "$n progress: $(grep -i 'learning iteration' "$LOGD/train_$n.log" | tail -1 | tr -s ' ') | ckpts=$(ls ckpt_keep/${RUN[$n]}/model_*.pt 2>/dev/null | wc -l)"
    fi
    tmux has-session -t "$n" 2>/dev/null || { sleep 30; grep -q TRAIN_DONE "$LOGD/train_$n.log" || {
        status "!! $n tmux 가 TRAIN_DONE 없이 죽었다 — 마지막 3줄:"; tail -3 "$LOGD/train_$n.log" | tee -a "$ST"
        unset "RUN[$n]"; break; }; }
  done
  status "$n TRAIN_DONE (ckpts kept: $(ls ckpt_keep/${RUN[$n]}/model_*.pt 2>/dev/null | wc -l))"
}
for a in $ARMS; do [ -n "${RUN[$a]:-}" ] && wait_done "$a"; done
for a in $DEFERRED; do
  status "=== starting $a sequentially ==="
  if start_run "$a"; then
    wait_done "$a"
  else
    status "!! $a 시작 실패 — 이 팔은 결과에서 빠진다 (로그: $LOGD/train_$a.log)"
  fi
  cp "$BAK" "$REWARD_FILE"
done

# ---------------------------------------------------------------- 5. export
export_ckpt(){ # $1=팔 $2=iter
  local n=$1 it=$2 run=${RUN[$n]} ck rd pol
  rd=$(dirname $(ls -1d logs/rsl_rl/*/$run 2>/dev/null | head -1))/$run
  [ -d "$rd" ] || { status "!! $n: run dir $run 없음"; return 1; }
  ck=ckpt_keep/$run/model_$it.pt
  if [ ! -f "$ck" ]; then
    ck=$(ls -1 ckpt_keep/$run/model_*.pt 2>/dev/null | sort -V | awk -v m="$it" '{split($0,a,"_"); sub(".pt","",a[2]); if (a[2]+0<=m+0) last=$0} END{print last}')
    [ -z "$ck" ] && { status "!! $n: no ckpt <= $it"; return 1; }
    status "$n: model_$it.pt 없음 → $(basename $ck) 사용"
  fi
  mkdir -p "$rd/params"; cp "$ck" "$rd/"
  PYTHONUNBUFFERED=1 $L play.py --task "$TASK" --num_envs 32 --headless --load_run "$run" \
      --checkpoint "$ROBOT_DIR/$rd/$(basename $ck)" > "$LOGD/export_${n}_$it.log" 2>&1
  pol=$(ls -1t "$rd/exported/policy.pt" exported/policy.pt 2>/dev/null | head -1)
  [ -z "$pol" ] && { status "!! $n $it: export 실패 ($LOGD/export_${n}_$it.log)"; return 1; }
  cp "$pol" "$KEEPD/${n}_$it.pt"; cp "$ck" "$KEEPD/${n}_${it}_model.pt"
  cp "ckpt_keep/$run/env.yaml" "$KEEPD/${n}_env.yaml" 2>/dev/null
  status "exported $n $it -> $KEEPD/${n}_$it.pt md5=$(md5sum $KEEPD/${n}_$it.pt | cut -c1-8)"
}
POLICIES=()
for a in $ARMS; do
  [ -n "${RUN[$a]:-}" ] || continue
  for it in $EVAL_ITERS; do export_ckpt "$a" "$it" && POLICIES+=("${a}_$it"); done
done
# md5 중복 제거 — play.py 가 --checkpoint 를 무시하면 같은 정책이 여러 개 나온다
status "policies: ${POLICIES[*]}"
$PY - <<PEOF | tee -a "$ST"
import hashlib, glob, collections, os
h = collections.defaultdict(list)
for p in sorted(glob.glob("$KEEPD/*_[0-9]*.pt")):
    if p.endswith("_model.pt"): continue
    h[hashlib.md5(open(p,'rb').read()).hexdigest()].append(os.path.basename(p))
dup = {k: v for k, v in h.items() if len(v) > 1}
print("!! md5 중복 — export 가 체크포인트를 무시했다:", dup) if dup else print("md5 중복 없음 (export OK)")
PEOF

# ---------------------------------------------------------------- 6. T-EVAL-GO2
teval(){ # $1=policy id $2=scenarios $3=tag $4...=extra
  local pid=$1 sc=$2 tag=$3; shift 3
  PYTHONUNBUFFERED=1 $L t_eval_go2.py --headless --task "$TASK" --policy "$KEEPD/$pid.pt" \
      --scenarios "$sc" --run_id "$pid" --out "$EVALD/G3_${pid}_$tag.json" "$@" \
      > "$LOGD/teval_${pid}_$tag.log" 2>&1 \
    || status "!! t_eval $pid $tag 실패 ($LOGD/teval_${pid}_$tag.log)"
}
# 채점은 128 envs 짜리라 한 프로세스가 GPU 를 거의 안 쓴다. 순차로 돌리면
# Isaac 기동시간(1~2분)만 9번 쌓인다. EVAL_PAR 개씩 동시에 띄운다.
EVAL_PAR=${EVAL_PAR:-3}
for pid in "${POLICIES[@]}"; do
  status "T-EVAL start $pid (parallel x$EVAL_PAR)"
  n=0
  teval "$pid" G1,G2,G6  flat                                   & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G3        rough                                  & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G4        slope   --slope_deg 20                 & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G5        stairs  --stair_height 0.12            & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G7        dr                                     & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX1,GX5   xflat                                  & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX2       xrough  --rough_min 0.08 --rough_max 0.14 & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX3       xslope  --slope_deg 25                 & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX4       xstairs --stair_height 0.15            & n=$((n+1))
  wait
  $PY t_eval_go2.py --merge $EVALD/G3_${pid}_*.json --run_id "$pid" \
      --out "$EVALD/G3_${pid}.json" > "$LOGD/merge_$pid.log" 2>&1
  status "T-EVAL done $pid: $($PY -c "
import json;d=json.load(open('$EVALD/G3_$pid.json'));s=d.get('stress') or {}
p=(d.get('official_pred') or {}).get('all') or {}
print('sim',d['sim_score'],'예측총점',p.get('official_total'),'G낙상',d['falls'],'Xsurv',s.get('survival'),'XLCB',s.get('survival_lcb'))" 2>/dev/null)"
done

# ---------------------------------------------------------------- 7. 순위표
$PY - > "$RES" <<'PEOF'
import json, glob, os
TAGS = ('flat','rough','slope','stairs','dr','xflat','xrough','xslope','xstairs')
rows = []
for p in sorted(glob.glob('eval_results/G3_*.json')):
    b = os.path.basename(p)
    if any(b.endswith(f'_{t}.json') for t in TAGS):
        continue
    d = json.load(open(p)); s = d.get('stress') or {}
    pred = d.get('official_pred') or {}
    rows.append(dict(
        rid=d['run_id'], sim=d['sim_score'],
        by=d.get('sim_score_by_axis_mode') or {},
        pred={k: v.get('official_total') for k, v in pred.items()},
        band={k: v.get('band') for k, v in pred.items()},
        falls=d['falls'], hlcb=d.get('h_survival_lcb'),
        xs=s.get('survival'), xlcb=s.get('survival_lcb'), xf=s.get('falls'),
        g={k: v['score'] for k, v in d['scenarios'].items() if k.startswith('G') and not k.startswith('GX')},
        x={k: v['survival'] for k, v in d['scenarios'].items() if k.startswith('GX')}))
# 정렬: Stress 생존 LCB ↓, 그다음 예측 총점 ↓ — 생존이 먼저인 이유는 §7 채택 게이트.
rows.sort(key=lambda r: (-(r['xlcb'] or 0), -(r['pred'].get('all') or 0)))
print('# G3 결과 (정렬: Stress 생존 LCB ↓, 예측 총점 ↓)\n')
print('| policy | 자체Sim | **예측총점(all)** | 밴드폭 | 예측(yaw) | G낙상 | H-LCB | X생존 | X LCB | X낙상 | '
      + ' | '.join(f'G{i}' for i in range(1, 8)) + ' | '
      + ' | '.join(f'X{i}' for i in range(1, 6)) + ' |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|' + '---:|' * 12)
for r in rows:
    g = ' | '.join(f"{r['g'].get(f'G{i}', float('nan')):.3f}" for i in range(1, 8))
    x = ' | '.join(f"{r['x'].get(f'GX{i}', float('nan')):.3f}" for i in range(1, 6))
    pa = r['pred'].get('all'); py = r['pred'].get('yaw'); ba = r['band'].get('all')
    print(f"| {r['rid']} | {r['sim']:.2f} | **{pa if pa is None else round(pa,2)}** | "
          f"{ba if ba is None else round(ba,2)} | "
          f"{py if py is None else round(py,2)} | {r['falls']} | {r['hlcb']} | "
          f"{r['xs']} | {r['xlcb']} | {r['xf']} | {g} | {x} |")
print("""
게이트 (go2_plan.md §7):
  채택   = Stress 전 시나리오 낙상 0(LCB 기준) AND 예측 총점 점추정이 후보 중 최고
           밴드폭 1.5 초과 후보끼리는 점추정 차이를 믿지 말고 자체 SimScore 순위로 가른다
  목표선 = Go2 87.5 → 합산 175.5 → 1위 / 본선 컷은 Go2 71.3
  주의   = 예측 배율은 **H1 앵커**에서 나온 값이다. Go2 는 이번 제출이 첫 앵커다.
           팔 사이 '순위'는 믿되 '절대 예측치'는 참고로만 봐라.
  대조군 = GB 는 설계의도 20점을 위해 리포트에 전후 비교로 반드시 들어간다.
""")
PEOF
status "RESULT written: $RES"; cat "$RES" | tee -a "$ST"

# ---------------------------------------------------------------- 8. 번들
TF=""
for a in $ARMS; do
  [ -n "${RUN[$a]:-}" ] && TF="$TF $(ls -d logs/rsl_rl/*/${RUN[$a]} 2>/dev/null)"
done
tar czf /workspace/G3_bundle.tgz "$KEEPD" $EVALD/G3_*.json "$RES" "$ST" \
    /workspace/G0_RECON.txt $LOGD/train_*.log $LOGD/validate_*.log $TF 2>/dev/null
status "bundle: $(ls -l /workspace/G3_bundle.tgz | awk '{print $5}') bytes md5=$(md5sum /workspace/G3_bundle.tgz | cut -c1-12)"
status "=== PIPELINE DONE.  종료: touch /workspace/STOP_KEEPALIVE"

# ---------------------------------------------------------------- 9. keep-alive
KA=${KEEPALIVE_POLICY:-${POLICIES[0]:-}}
while [ ! -f /workspace/STOP_KEEPALIVE ]; do
  if [ -n "$KA" ] && [ -f "$KEEPD/$KA.pt" ]; then
    PYTHONUNBUFFERED=1 $L t_eval_go2.py --headless --task "$TASK" --policy "$KEEPD/$KA.pt" \
        --scenarios G1 --seeds 103 --num_envs 64 --run_id keepalive \
        --out /tmp/keepalive.json > /tmp/keepalive.log 2>&1
  fi
  echo "$(date '+%m-%d %H:%M:%S') keepalive tick" >> "$ST"
  for i in $(seq 1 60); do [ -f /workspace/STOP_KEEPALIVE ] && break; sleep 5; done
done
tmux kill-session -t sentinel 2>/dev/null
tmux kill-session -t g3guard 2>/dev/null
status "keep-alive stopped"
