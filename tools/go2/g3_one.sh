#!/bin/bash
# g3_one.sh — 조건 하나만 export → 병렬 채점 → merge → 요약.
#
# 왜 따로 있나: g3_pipeline.sh 는 네 조건을 다 학습한 뒤에 채점한다. 오늘 안에 제출하려면
# GB 가 끝나자마자 GB 만 채점해야 하는데, Isaac 프로세스 고정 오버헤드가 6.4GB 라
# 학습(10.8GB)과 채점을 동시에 못 돌린다(합 17.2GB > 16.3GB). 그래서 학습을 잠시 멈추고
# 이 스크립트로 한 조건만 처리한 뒤, 남은 조건을 다시 건다.
#
# 사용:
#   bash g3_one.sh GB "1500 2000"
#   RUN=2026-09-11_13-44-14 bash g3_one.sh GB "1500 2000"   # run 을 직접 지정
set -u
ROBOT_DIR=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$ROBOT_DIR" || { echo "!! $ROBOT_DIR 없음"; exit 1; }
PY=/isaac-sim/python.sh
L="/workspace/IsaacLab/isaaclab.sh -p"
TASK=${TASK:-Quadruped-v0}
NAME=${1:?"조건 이름을 넘겨라 (예: GB)"}
ITS=${2:-"1500 2000"}
EVAL_PAR=${EVAL_PAR:-3}
KEEPD=keep_G3
EVALD=eval_results
LOGD=/workspace/g3logs
ST=/workspace/G3_STATUS.txt
mkdir -p "$KEEPD" "$EVALD" "$LOGD"
status(){ echo "$(date '+%m-%d %H:%M:%S') [one:$NAME] $*" | tee -a "$ST"; }

RUN=${RUN:-$(ls -1t ckpt_keep 2>/dev/null | head -1)}
[ -n "$RUN" ] && [ -d "ckpt_keep/$RUN" ] || { status "!! ckpt_keep 에 run 이 없다"; exit 1; }
RD=$(ls -1d logs/rsl_rl/*/"$RUN" 2>/dev/null | head -1)
[ -n "$RD" ] || { status "!! logs 에서 run $RUN 을 못 찾았다"; exit 1; }
status "run=$RUN  dir=$RD  iters=[$ITS]  ckpts=$(ls ckpt_keep/$RUN/model_*.pt 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- export
export_ckpt(){
  local it=$1 ck pol
  ck=ckpt_keep/$RUN/model_$it.pt
  if [ ! -f "$ck" ]; then
    ck=$(ls -1 ckpt_keep/$RUN/model_*.pt 2>/dev/null | sort -V \
         | awk -v m="$it" '{split($0,a,"_"); sub(".pt","",a[2]); if (a[2]+0<=m+0) last=$0} END{print last}')
    [ -z "$ck" ] && { status "!! model_$it.pt 이하 체크포인트가 없다"; return 1; }
    status "model_$it.pt 없음 -> $(basename $ck) 사용"
  fi
  mkdir -p "$RD/params"; cp "$ck" "$RD/"
  PYTHONUNBUFFERED=1 $L play.py --task "$TASK" --num_envs 32 --headless --load_run "$RUN" \
      --checkpoint "$ROBOT_DIR/$RD/$(basename $ck)" > "$LOGD/export_${NAME}_$it.log" 2>&1
  pol=$(ls -1t "$RD/exported/policy.pt" exported/policy.pt 2>/dev/null | head -1)
  [ -z "$pol" ] && { status "!! $it export 실패 ($LOGD/export_${NAME}_$it.log)"; return 1; }
  cp "$pol" "$KEEPD/${NAME}_$it.pt"; cp "$ck" "$KEEPD/${NAME}_${it}_model.pt"
  cp "ckpt_keep/$RUN/env.yaml" "$KEEPD/${NAME}_env.yaml" 2>/dev/null
  status "exported $it -> $KEEPD/${NAME}_$it.pt md5=$(md5sum $KEEPD/${NAME}_$it.pt | cut -c1-8)"
}
POL=()
for it in $ITS; do export_ckpt "$it" && POL+=("${NAME}_$it"); done
[ ${#POL[@]} -gt 0 ] || { status "!! export 된 정책이 없다"; exit 1; }

# play.py 가 --checkpoint 를 무시하면 같은 정책이 여러 개 나온다. 여기서 잡는다.
$PY - <<PEOF | tee -a "$ST"
import hashlib, glob, collections, os
h = collections.defaultdict(list)
for p in sorted(glob.glob("$KEEPD/${NAME}_[0-9]*.pt")):
    if p.endswith("_model.pt"): continue
    h[hashlib.md5(open(p,'rb').read()).hexdigest()].append(os.path.basename(p))
d = {k: v for k, v in h.items() if len(v) > 1}
print("!! md5 중복 - export 가 체크포인트를 무시했다:", d) if d else print("md5 중복 없음 (export OK)")
PEOF

# ---------------------------------------------------------------- 채점 (병렬)
teval(){ local pid=$1 sc=$2 tag=$3; shift 3
  PYTHONUNBUFFERED=1 $L t_eval_go2.py --headless --task "$TASK" --policy "$KEEPD/$pid.pt" \
      --scenarios "$sc" --run_id "$pid" --out "$EVALD/G3_${pid}_$tag.json" "$@" \
      > "$LOGD/teval_${pid}_$tag.log" 2>&1 || status "!! t_eval $pid $tag 실패"
}
for pid in "${POL[@]}"; do
  status "T-EVAL $pid (parallel x$EVAL_PAR)"
  n=0
  teval "$pid" G1,G2,G6 flat                                      & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G3       rough                                     & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G4       slope   --slope_deg 20                    & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G5       stairs  --stair_height 0.12               & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" G7       dr                                        & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX1,GX5  xflat                                     & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX2      xrough  --rough_min 0.08 --rough_max 0.14 & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX3      xslope  --slope_deg 25                    & n=$((n+1)); [ $((n%EVAL_PAR)) -eq 0 ] && wait
  teval "$pid" GX4      xstairs --stair_height 0.15               & n=$((n+1))
  wait
  $PY t_eval_go2.py --merge $EVALD/G3_${pid}_*.json --run_id "$pid" \
      --out "$EVALD/G3_${pid}.json" > "$LOGD/merge_$pid.log" 2>&1
  status "done $pid: $($PY -c "
import json;d=json.load(open('$EVALD/G3_$pid.json'));s=d.get('stress') or {}
p=(d.get('official_pred') or {}).get('all') or {}
print('sim',d['sim_score'],'pred',p.get('official_total'),'band',p.get('band'),
      'falls',d['falls'],'Xsurv',s.get('survival'),'XLCB',s.get('survival_lcb'))" 2>/dev/null)"
done

# ---------------------------------------------------------------- 요약
$PY - <<'PEOF' | tee -a /workspace/G3_STATUS.txt
import json, glob, os
TAGS = ('flat','rough','slope','stairs','dr','xflat','xrough','xslope','xstairs')
rows = []
for p in sorted(glob.glob('eval_results/G3_*.json')):
    b = os.path.basename(p)
    if any(b.endswith(f'_{t}.json') for t in TAGS): continue
    d = json.load(open(p)); s = d.get('stress') or {}
    pr = (d.get('official_pred') or {})
    rows.append((d['run_id'], d['sim_score'],
                 (pr.get('all') or {}).get('official_total'),
                 (pr.get('all') or {}).get('band'),
                 d['falls'], d.get('h_survival_lcb'), s.get('survival_lcb'),
                 {k: v['score'] for k, v in d['scenarios'].items()
                  if k.startswith('G') and not k.startswith('GX')}))
rows.sort(key=lambda r: -(r[2] or 0))
print('\n| policy | 자체Sim | 예측총점 | 밴드폭 | 낙상 | G-LCB | X-LCB | ' +
      ' | '.join(f'G{i}' for i in range(1, 8)) + ' |')
print('|---|---:|---:|---:|---:|---:|---:|' + '---:|' * 7)
for r in rows:
    g = ' | '.join(f"{r[7].get(f'G{i}', float('nan')):.3f}" for i in range(1, 8))
    print(f"| {r[0]} | {r[1]:.2f} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {g} |")
PEOF
status "=== g3_one 완료. 제출 후보: $KEEPD/${NAME}_*.pt + ${NAME}_env.yaml ==="
