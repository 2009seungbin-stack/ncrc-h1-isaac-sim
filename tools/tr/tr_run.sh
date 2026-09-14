#!/bin/bash
# tr_run.sh — 공식 채점 바닥 역산 실험: 리셋/전환 과도구간을 오차에 포함해 R3A_9000 을 재채점한다.
# 결과: /workspace/tr_out.tgz (eval_results/TR_*.json + 로그), 상태: /workspace/TR_STATUS.txt
set -u
cd /workspace/training/humanoid || { echo "!! no humanoid dir"; exit 1; }
mkdir -p keep_TR eval_results
S=/workspace/TR_STATUS.txt
echo "start $(date)" > "$S"
URL='https://kr.object.ncloudstorage.com/neotwin/HUMANOID/202609/260908213842565d5b0c0703da2481dbfeb223ae6399f1f.pt'
curl -sSL "$URL" -o keep_TR/policy.pt
md5sum keep_TR/policy.pt >> "$S"
md5sum keep_TR/policy.pt | grep -q 4f6540020d964e9e9c261a942b028bdd || { echo "!! policy md5 mismatch" >> "$S"; exit 1; }
L="/workspace/IsaacLab/isaaclab.sh -p"
run() { # id transient_s seg_s
  echo "== $1 begin $(date)" >> "$S"
  PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy keep_TR/policy.pt --scenarios H1,H2,H3,H4,H7 \
    --seeds 103,280 --transient_s "$2" --seg_s "$3" --run_id "$1" --out "eval_results/TR_$1.json" \
    > "/workspace/TR_$1.log" 2>&1
  echo "== $1 end rc=$? $(date)" >> "$S"
  grep -E '"(survival|tracking|sim_score)"' "eval_results/TR_$1.json" | head -20 >> "$S"
}
run T0_seg5 0.0 5.0
run T0_seg25 0.0 2.5
tar czf /workspace/tr_out.tgz eval_results/TR_*.json /workspace/TR_*.log "$S" 2>/dev/null
echo "ALL DONE $(date)" >> "$S"
