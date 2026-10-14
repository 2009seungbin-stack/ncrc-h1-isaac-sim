#!/bin/bash
# sweep_go2.sh — 여러 보상값을 한 트렁크에서 분기 학습하고, **같은 명령으로** 영상까지 찍어 회수한다.
#
# 설계 근거 (2026-09-16):
#   · scratch 로 후보 8개를 돌리면 24시간이다. 남은 예산이 69시간인데 4~7라운드를 다 써야 한다.
#     그래서 공통 트렁크를 한 번 돌리고 --resume 으로 분기한다 (후보당 약 50분, 8개 6.7시간).
#   · ⚠️ resume 정책은 스크리닝 전용이다. 제출본은 반드시 우승 값으로 scratch 재학습한다 —
#     연쇄 fine-tune 은 env.yaml 한 장으로 설명이 안 돼 설계의도 20점이 무너진다.
#   · envs 는 6144 이하. 8192 는 iter 14 부터 PhysX 접촉 패치 한도(327,680)를 매 스텝 넘겨서
#     후보 비교에 '오버플로 여부'라는 교란 변수를 섞는다.
#   · 판정은 2단계다. 영상(눈) = 누움·제자리·뒷걸음 같은 퇴화를 걷어내는 예선,
#     t_eval_go2(숫자) = 살아남은 2~3개를 줄 세우는 결선. 바꿔 쓰지 말 것.
#
# 사용:
#   bash sweep_go2.sh                        # 0단계만: train.py 인자 정찰 + 보상 검증 (GPU 안 씀)
#   TRUNK_ITERS=1500 CONFIRM=1 bash sweep_go2.sh
#   TRUNK_RUN=2026-09-17_10-00-00 CONFIRM=1 ARMS="A1 A2 A3" bash sweep_go2.sh
#   touch /workspace/STOP_SWEEP              # 다음 팔부터 중단
#
# ⚠️ train.py 를 인자 없이/모르는 인자로 부르지 마라. 2026-09-05 에 `--help` 한 번으로
#    진행 중이던 run 의 중간 체크포인트 71개가 후처리에 삭제됐다. 그래서 이 스크립트는
#    train.py 를 실행해 보지 않고 **소스를 읽어** resume 인자 이름을 확인한다.
set -u
ROBOT_DIR=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$ROBOT_DIR" || { echo "!! $ROBOT_DIR 없음"; exit 1; }
PY=${PY:-/isaac-sim/python.sh}
L=${L:-"/workspace/IsaacLab/isaaclab.sh -p"}
TASK=${TASK:-Quadruped-v0}
ENVS=${ENVS:-6144}
TRUNK_ITERS=${TRUNK_ITERS:-0}          # >0 이면 트렁크를 scratch 로 학습
TRUNK_RUN=${TRUNK_RUN:-}               # 이미 있는 run 이름을 트렁크로 쓸 때
BRANCH_ITERS=${BRANCH_ITERS:-800}
RENDER_ENVS=${RENDER_ENVS:-32}
SCENARIOS=${SCENARIOS:-"G1 G5"}        # G1 평지 전진(병목), G5 계단(생존). 지형 그룹이 달라 따로 돈다
SEED=${SEED:-0}
CONFIRM=${CONFIRM:-0}
WS=${WS:-/workspace}
FILMS=${FILMS:-$WS/films}
ST=$WS/SWEEP_STATUS.txt
REWARD_FILE=quadruped_rewards.py
BAK=quadruped_rewards.baseline.bak
RENDER=${RENDER:-$ROBOT_DIR/render_go2.py}
mkdir -p "$FILMS" ckpt_keep
status(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }

# ---- 팔 정의 ---------------------------------------------------------------------------------
# 전진 거부가 병목이다(2026-09-13 로그 + 09-16 영상). 흔들 축은 '가는 이득 vs 버티는 이득'.
ARMS=${ARMS:-"C0 C1 C2"}
# --- R4 (2026-09-17~): "빠르고 조용한 정책" 축 ---------------------------------------------
# 근거: score_reverse_20260916.md — 3축이면 A2 56.34, 6축이면 46.32 로 순위가 뒤집힌다.
#       A2 의 6축 MSE 86.6% 가 roll+pitch 다. 추종을 유지한 채 흔들림만 GB 수준으로 낮추면
#       최솟값 53.41 로 어느 모형이든 최고. **모형에 베팅하지 않는 유일한 방향이다.**
# flat_orientation_l2 는 배포값 0.0(꺼짐) — REWARD_WEIGHTS 에 없던 키지만 로더가 임의 키를 적용한다(실측).
ARM_C0=${ARM_C0:-"track_lin_vel_xy_exp=2.0"}                                          # A2 재현(대조군)
ARM_C1=${ARM_C1:-"track_lin_vel_xy_exp=2.0 flat_orientation_l2=-2.5"}                 # 자세 직격(신규 키)
ARM_C2=${ARM_C2:-"track_lin_vel_xy_exp=2.0 flat_orientation_l2=-2.5 ang_vel_xy_l2=-0.15"}
ARM_C3=${ARM_C3:-"track_lin_vel_xy_exp=2.0 flat_orientation_l2=-1.0"}                 # C1 이 얼면 약하게
# --- R5 준비 (2026-09-21): 6축 오차의 90%가 roll·pitch 각속도다(longrun_C0/RESULT.md). 추종 보상을 2배로 올리며
#     절반이 된 흔들림 벌점의 상대 비중을 원래대로 되돌린다. 각속도를 직접 벌하는 항이라 6축이 재는 양과 같다.
ARM_D1=${ARM_D1:-"track_lin_vel_xy_exp=2.0 ang_vel_xy_l2=-0.16"}
ARM_D2=${ARM_D2:-"track_lin_vel_xy_exp=2.0 action_rate_l2=-0.02"}                      # D1 이 기각되면
# --- 구 R3 축 (track_lin 스윕, 2026-09-16 종료. A2=2.0 채택) --------------------------------
ARM_A0=""
ARM_A1=${ARM_A1:-"track_lin_vel_xy_exp=1.5"}
ARM_A2=${ARM_A2:-"track_lin_vel_xy_exp=2.0"}

# ---- 0. 정찰 + 검증 (GPU 안 씀) ---------------------------------------------------------------
: > "$ST"
status "=== sweep preflight ==="
for f in train.py "$REWARD_FILE"; do [ -f "$f" ] || { status "!! missing $f"; exit 1; }; done
[ -f "$RENDER" ] || status "!! $RENDER 없음 — render_go2.py 를 먼저 올려라 (영상 단계는 건너뛴다)"
[ -f "$BAK" ] || cp "$REWARD_FILE" "$BAK"

status "--- train.py 가 받는 인자 (소스에서 읽음, 실행 안 함) ---"
grep -n "add_argument" train.py | sed 's/^/    /' | tee -a "$ST"
RESUME_ARGS=${RESUME_ARGS:-}
if [ -z "$RESUME_ARGS" ]; then
  if grep -q '"--resume"\|'\''--resume'\''' train.py; then
    RESUME_ARGS="--resume --load_run TRUNK"
    status "resume 인자 발견 → 기본값 사용: $RESUME_ARGS  (TRUNK 는 아래에서 치환)"
  else
    status "!! train.py 에 --resume 이 안 보인다. 위 목록을 보고 RESUME_ARGS 로 직접 넘겨라."
    status "   예: RESUME_ARGS='--resume --load_run TRUNK --checkpoint model_1500.pt'"
  fi
fi

# 배포 원본 확인 — 이전 실험이 남긴 파일을 원본으로 착각하면 대조군이 오염된다
$PY - "$BAK" <<'PEOF' | tee -a "$ST" || { status "!! $BAK 가 NCRC 원본이 아니다"; exit 1; }
import sys
ns = {}; exec(open(sys.argv[1], encoding="utf-8").read(), ns); w = ns["REWARD_WEIGHTS"]
want = {"track_lin_vel_xy_exp": 1.0, "feet_air_time": 0.01, "lin_vel_z_l2": -3.0,
        "ang_vel_xy_l2": -0.08, "action_rate_l2": -0.01}
print("baseline OK:" if w == want else "baseline MISMATCH:", sorted(w.items()))
sys.exit(0 if w == want else 1)
PEOF

apply_arm(){ # $1=팔 $2=대상파일 — 원본에서 복사한 뒤 활성 줄 값만 바꾸고 재파싱으로 검증
  local name=$1 dst=$2 spec
  eval "spec=\${ARM_$name-UNSET}"
  [ "$spec" = "UNSET" ] && { status "!! 팔 $name 정의 없음"; return 1; }
  cp "$BAK" "$dst"
  [ -z "$spec" ] && { status "$name: 무수정(대조군)"; return 0; }
  $PY - "$dst" $spec <<'PEOF'
import re, sys
path, kvs = sys.argv[1], sys.argv[2:]
src = open(path, encoding="utf-8").read()
want = {}
for kv in kvs:
    k, v = kv.split("=", 1)
    want[k] = float(v)
    pat = re.compile(r'^(\s*"%s"\s*:\s*)[-+0-9.eE]+' % re.escape(k), re.M)
    src, n = pat.subn(lambda m: m.group(1) + v, src)
    if n > 1:
        sys.exit(f"!! {k}: 활성 줄이 {n}개 (정확히 1개여야 한다)")
    if n == 0:
        # 2026-09-16 확인: go2_task/_apply_common_overrides 가 **임의 키를 적용한다**.
        # env.yaml 에 항이 있지만 REWARD_WEIGHTS 에 없는 키(flat_orientation_l2 등)를 새로 켠다.
        m = re.search(r'(REWARD_WEIGHTS\s*(?::[^=\n]+)?=\s*\{)', src)
        if not m:
            sys.exit("!! REWARD_WEIGHTS 블록을 못 찾았다")
        src = src[:m.end()] + f'\n    "{k}": {v},' + src[m.end():]
        print(f"   + 신규 키 추가: {k} = {v}")
open(path, "w", encoding="utf-8", newline="").write(src)
ns = {}; exec(src, ns); got = ns["REWARD_WEIGHTS"]
bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
if bad:
    sys.exit(f"!! 재파싱 불일치 {bad}")
print("weights:", sorted(got.items()))
PEOF
}

for a in $ARMS; do
  apply_arm "$a" "$FILMS/_validate.py" >> "$ST" 2>&1 || { status "!! 팔 $a 검증 실패 — 중단"; exit 1; }
done
rm -f "$FILMS/_validate.py"
status "팔 검증 통과: [$ARMS]"

if [ "$CONFIRM" != "1" ]; then
  status "=== 0단계까지만 했다. 위 인자 목록을 확인하고 CONFIRM=1 로 다시 실행하라. ==="
  exit 0
fi

# ---- 체크포인트 가드 (model_best 선택 결함 대비) ------------------------------------------------
cat > /tmp/sweep_guard.sh <<GEOF
cd $ROBOT_DIR; mkdir -p ckpt_keep
while [ ! -f $WS/STOP_GUARD ]; do
  for r in logs/rsl_rl/*/*/; do
    [ -d "\$r" ] || continue; b=\$(basename "\$r"); mkdir -p "ckpt_keep/\$b"
    for f in "\$r"model_*.pt; do [ -f "\$f" ] || continue; n=\$(basename "\$f")
      [ -f "ckpt_keep/\$b/\$n" ] || cp "\$f" "ckpt_keep/\$b/\$n"; done
    [ -f "ckpt_keep/\$b/env.yaml" ] || cp "\$r/params/env.yaml" "ckpt_keep/\$b/env.yaml" 2>/dev/null
  done; sleep 45
done
GEOF
rm -f "$WS/STOP_GUARD"
tmux kill-session -t swguard 2>/dev/null; tmux new -d -s swguard 'bash /tmp/sweep_guard.sh'
status "guard 시작 (45초마다 ckpt_keep/ 로 복사)"

newest_run(){ ls -td logs/rsl_rl/*/*/ 2>/dev/null | head -1 | sed 's:/*$::'; }
last_ckpt(){ # $1=run 디렉터리 — 번호가 가장 큰 model_N.pt (model_best 는 믿지 않는다)
  ls "ckpt_keep/$(basename "$1")"/model_[0-9]*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2
}

train_one(){ # $1=세션 $2=로그 $3...=train.py 인자
  local sess=$1 log=$2; shift 2
  : > "$log"
  tmux kill-session -t "$sess" 2>/dev/null
  tmux new -d -s "$sess" "cd $ROBOT_DIR && PYTHONUNBUFFERED=1 $L train.py $* 2>&1 | tee $log; echo TRAIN_DONE >> $log"
  local t=0
  while ! grep -aq "Learning iteration" "$log"; do
    sleep 10; t=$((t+10))
    grep -aqiE "out of memory|illegal memory access" "$log" && { status "!! GPU 메모리 실패"; tmux kill-session -t "$sess"; return 2; }
    grep -aq TRAIN_DONE "$log" && { status "!! iteration 없이 끝났다:"; tail -8 "$log" | tee -a "$ST"; return 1; }
    [ $t -ge 900 ] && { status "!! 15분 동안 첫 iteration 없음"; tmux kill-session -t "$sess"; return 1; }
  done
  status "  첫 iteration (t=${t}s) — 보상 파일을 읽었다"
  cp "$BAK" "$REWARD_FILE"        # 읽었으니 즉시 원본 복구 (다음 팔 오염 방지)
  t=0
  while ! grep -aq TRAIN_DONE "$log"; do
    sleep 30; t=$((t+30))
    tmux has-session -t "$sess" 2>/dev/null || { sleep 20; grep -aq TRAIN_DONE "$log" || { status "!! tmux 가 TRAIN_DONE 없이 죽음"; return 1; }; }
    [ $((t % 600)) -eq 0 ] && status "  $(grep -a 'Learning iteration' "$log" | tail -1 | tr -s ' ')"
  done
  status "  TRAIN_DONE"
  return 0
}

render_one(){ # $1=라벨 $2=체크포인트 — 시나리오마다 프로세스를 새로 띄운다(지형 그룹 제약)
  local label=$1 ck=$2 sc
  [ -f "$RENDER" ] || return 0
  [ -f "$ck" ] || { status "  !! 체크포인트 없음: $ck"; return 1; }
  for sc in $SCENARIOS; do
    status "  render $label/$sc"
    $L "$RENDER" --task "$TASK" --checkpoint "$ROBOT_DIR/$ck" --scenario "$sc" \
        --label "$label" --num_envs "$RENDER_ENVS" --render_seed "$SEED" \
        --video_dir "$FILMS" --headless --enable_cameras \
        > "$FILMS/render_${label}_${sc}.log" 2>&1
    grep -a "전진 거부\|영상 \|!!" "$FILMS/render_${label}_${sc}.log" | tail -3 | sed 's/^/    /' | tee -a "$ST"
  done
}

# ---- 1. 트렁크 --------------------------------------------------------------------------------
# TRUNK_RUN=SKIP → 트렁크 없이 각 설정을 scratch 로 학습한다. 이 분기가 없으면 `--load_run SKIP` 이
# train.py 로 넘어간다(2026-09-20 발견 — train.py 에 --resume 이 있는 서버에서는 학습이 시작도 못 한다).
SCRATCH=0
if [ "$TRUNK_RUN" = "SKIP" ]; then
  SCRATCH=1; RESUME_ARGS=""
  status "SCRATCH 모드 — 트렁크·resume 없이 설정마다 처음부터 학습"
elif [ -z "$TRUNK_RUN" ]; then
  [ "$TRUNK_ITERS" -gt 0 ] || { status "!! TRUNK_RUN 도 TRUNK_ITERS 도 없다"; exit 1; }
  status "=== TRUNK: 배포 원본 $TRUNK_ITERS iter @ $ENVS envs ==="
  cp "$BAK" "$REWARD_FILE"
  train_one sw_trunk "$FILMS/train_trunk.log" \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$TRUNK_ITERS" --headless || exit 1
  TRUNK_RUN=$(basename "$(newest_run)")
fi
if [ "$SCRATCH" != "1" ]; then
  status "TRUNK = $TRUNK_RUN"
  TRUNK_CK=$(last_ckpt "$TRUNK_RUN")
  [ -n "$TRUNK_CK" ] && status "TRUNK 마지막 ckpt: $TRUNK_CK"
  render_one "TRUNK" "$TRUNK_CK"
fi

# ---- 2. 분기 ---------------------------------------------------------------------------------
for a in $ARMS; do
  [ -f "$WS/STOP_SWEEP" ] && { status "STOP_SWEEP — 중단"; break; }
  status "=== ARM $a : $(eval echo \${ARM_$a}) | resume $TRUNK_RUN + $BRANCH_ITERS iter ==="
  apply_arm "$a" "$REWARD_FILE" 2>&1 | tee -a "$ST"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { status "!! $a 적용 실패 — 건너뜀"; continue; }
  RA=$(echo "$RESUME_ARGS" | sed "s/TRUNK/$TRUNK_RUN/")
  train_one "sw_$a" "$FILMS/train_$a.log" \
    --task "$TASK" --num_envs "$ENVS" --max_iterations "$BRANCH_ITERS" --headless $RA \
    || { status "!! $a 학습 실패 — 다음 팔"; cp "$BAK" "$REWARD_FILE"; continue; }
  cp "$BAK" "$REWARD_FILE"
  RUN=$(basename "$(newest_run)")
  sleep 50                         # guard 주기(45초)를 한 번 넘겨 마지막 ckpt 까지 ckpt_keep 에 들어오게 한다
  CK=$(last_ckpt "$RUN")
  echo "$a=$RUN" >> "$WS/SWEEP_RUNS.txt"   # teval_arms.sh·longrun_go2.sh 가 설정↔run 을 여기서 찾는다
  status "  $a run=$RUN ckpt=$CK"
  render_one "$a" "$CK"
done
cp "$BAK" "$REWARD_FILE"
touch "$WS/STOP_GUARD"

# ---- 3. 한 번에 회수 ---------------------------------------------------------------------------
cd "$WS" || exit 1
tar czf films.tgz -C "$WS" "$(basename "$FILMS")" 2>/dev/null
status "=== sweep done. $WS/films.tgz $(stat -c %s films.tgz 2>/dev/null) bytes ==="
status "브라우저에서 remote-resource fetch 로 films.tgz 하나만 받으면 된다"
ls -la "$FILMS" | tee -a "$ST"
