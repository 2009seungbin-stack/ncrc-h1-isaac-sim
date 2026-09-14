#!/bin/bash
# tr2_run.sh — R3A_9000 을 표준 조건(과도 0.5s, 4 seeds)으로 7 시나리오 재채점 + 진단 축(vz,wx,wy)·MAE 기록
set -u
cd /workspace/training/humanoid || exit 1
mkdir -p keep_TR eval_results
S=/workspace/TR2_STATUS.txt; echo "start $(date)" > "$S"
[ -f keep_TR/policy.pt ] || curl -sSL 'https://kr.object.ncloudstorage.com/neotwin/HUMANOID/202609/260908213842565d5b0c0703da2481dbfeb223ae6399f1f.pt' -o keep_TR/policy.pt
md5sum keep_TR/policy.pt | grep -q 4f6540020d964e9e9c261a942b028bdd || { echo "!! policy md5 mismatch" >> "$S"; exit 1; }
L="/workspace/IsaacLab/isaaclab.sh -p"
run() { # id scenarios
  echo "== $1 begin $(date)" >> "$S"
  PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy keep_TR/policy.pt --scenarios "$2" \
    --run_id "$1" --out "eval_results/$1.json" > "/workspace/$1.log" 2>&1
  echo "== $1 end rc=$? $(date)" >> "$S"
  grep '\[t_eval\] H[1-7]:' "/workspace/$1.log" >> "$S"
}
run TR2_flat H1,H2,H3,H4,H7
run TR2_rough H5
run TR2_slope H6
echo "ALL DONE $(date)" >> "$S"
