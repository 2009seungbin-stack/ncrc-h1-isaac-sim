#!/bin/bash
# teval_arms.sh — 스윕이 끝난 뒤 각 팔을 T-EVAL-GO2 7시나리오로 정식 채점한다.
#
# sweep_go2.sh 가 만든 ckpt_keep/<run>/ 의 **번호가 가장 큰 model_N.pt** 를 쓴다
# (model_best 는 선택 결함 때문에 초기 체크포인트일 수 있다 — 믿지 않는다).
#
# 사용:
#   ARMS="C0 C1 C2" bash teval_arms.sh                 # run 은 /workspace/SWEEP_RUNS.txt 에서 찾는다
#   RUNS="C0=2026-09-17_10-00-00 C1=..." bash teval_arms.sh   # run 이름을 직접 지정
#   ARMS="C0_800 C0_last" RUNS="C0_800=<run>:800 C0_last=<run>" bash teval_arms.sh   # 한 run 의 중간 지점들
#
# 지형 그룹은 한 프로세스에 하나만 돌 수 있다(cfg.validate 무한 재귀) → 5개로 나눈 뒤 --merge.
# 팔당 약 9분. 결과: eval_results/<팔>_ALL.json
set -u
R=${ROBOT_DIR:-/workspace/training/quadruped}
cd "$R" || { echo "!! $R 없음"; exit 1; }
L=${L:-"/workspace/IsaacLab/isaaclab.sh -p"}
TASK=${TASK:-Quadruped-v0}
ARMS=${ARMS:-"C0 C1 C2"}
RUNS=${RUNS:-}
WS=${WS:-/workspace}
ST=$WS/TEVAL_ARMS.txt
: > "$ST"
say(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$ST"; }
mkdir -p eval_results

resolve_ckpt(){ # $1=팔 이름 → 체크포인트 절대경로
  local a=$1 run="" upto=""
  for kv in $RUNS; do [ "${kv%%=*}" = "$a" ] && run="${kv#*=}"; done
  # sweep_go2.sh 가 남긴 설정↔run 표. run 디렉터리는 시각 이름이라 설정 이름으로는 못 찾는다.
  [ -z "$run" ] && [ -f "$WS/SWEEP_RUNS.txt" ] && run=$(grep "^$a=" "$WS/SWEEP_RUNS.txt" | tail -1 | cut -d= -f2)
  # '가장 최근 run' 으로 떨어지면 모든 설정이 같은 체크포인트로 채점된다(무증상) → 못 찾으면 실패시킨다.
  [ -z "$run" ] && return 1
  # RUNS="C0_800=<run>:800" → 번호가 800 이하인 것 중 가장 큰 model_N.pt (장기 run 의 중간 지점 채점)
  case "$run" in *:*) upto="${run#*:}"; run="${run%%:*}";; esac
  ls "ckpt_keep/$run"/model_[0-9]*.pt 2>/dev/null \
    | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n \
    | awk -v u="$upto" 'u=="" || $1+0<=u+0' | tail -1 | cut -d' ' -f2
}

group(){ # $1=팔 $2=태그 $3=시나리오
  local a=$1 tag=$2 scs=$3
  say "  $a/$tag"
  PYTHONUNBUFFERED=1 $L t_eval_go2.py --task "$TASK" --checkpoint "$CK" \
    --scenarios "$scs" --run_id "$a" --out "$R/eval_results/${a}_${tag}.json" \
    --headless > "$WS/te_${a}_${tag}.log" 2>&1
  [ -f "$R/eval_results/${a}_${tag}.json" ] || { say "  !! $a/$tag 실패"; tail -6 "$WS/te_${a}_${tag}.log" >> "$ST"; }
}

for a in $ARMS; do
  CK=$(resolve_ckpt "$a")
  if [ -z "$CK" ] || [ ! -f "$CK" ]; then say "!! $a 체크포인트 못 찾음 — 건너뜀"; continue; fi
  CK="$R/$CK"
  say "=== $a  ckpt=$(basename "$CK")  md5 $(md5sum "$CK" | cut -c1-8) ==="
  group "$a" flat   "G1,G2,G6"
  group "$a" rough  "G3"
  group "$a" slope  "G4"
  group "$a" stairs "G5"
  group "$a" dr     "G7"
  $L t_eval_go2.py --merge \
    "$R/eval_results/${a}_flat.json" "$R/eval_results/${a}_rough.json" \
    "$R/eval_results/${a}_slope.json" "$R/eval_results/${a}_stairs.json" "$R/eval_results/${a}_dr.json" \
    --out "$R/eval_results/${a}_ALL.json" --run_id "$a" > "$WS/te_${a}_merge.log" 2>&1
  say "$a merged $(stat -c %s "$R/eval_results/${a}_ALL.json" 2>/dev/null) bytes"
done

# ---- 회수용으로 작게 묶는다 (다운로드는 오리진당 1회뿐이다) --------------------------------
FF=$(/isaac-sim/python.sh -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null | tail -1)
mkdir -p "$WS/out"; rm -f "$WS/out"/*
cp "$R"/eval_results/*_ALL.json "$WS/out/" 2>/dev/null
cp "$WS"/films/*.json "$WS/out/" 2>/dev/null
if [ -n "$FF" ] && [ -x "$FF" ]; then
  for f in "$WS"/films/*.mp4; do [ -f "$f" ] || continue
    "$FF" -v error -i "$f" -vf scale=800:-2 -c:v libx264 -crf 32 -preset veryfast -an -y "$WS/out/$(basename "$f")"
  done
  say "영상 축소 완료 (imageio-ffmpeg)"
else
  say "!! ffmpeg 못 찾음 — 영상은 원본 크기"
  cp "$WS"/films/*.mp4 "$WS/out/" 2>/dev/null
fi
cd "$WS/out" && tar czf "$WS/r4_out.tgz" . && say "BUNDLE $WS/r4_out.tgz $(stat -c %s "$WS/r4_out.tgz") bytes"
say "ALL DONE — 이 tgz 하나를 그 오리진의 **첫 다운로드**로 받을 것"
