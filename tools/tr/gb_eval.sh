#!/bin/bash
# gb_eval.sh — 보드 백업 zip(GB, 09-11 18:51)의 model_best.pt 를 6축 진단 평가기로 7 시나리오 채점 + 명령 스윕.
set -u
cd /workspace/training/quadruped || { echo "!! no quadruped dir"; exit 1; }
S=/workspace/GB_STATUS.txt; echo "start $(date)" > "$S"
PY=/isaac-sim/python.sh; L="/workspace/IsaacLab/isaaclab.sh -p"
mkdir -p keep_GB eval_results
ZIP='https://kr.object.ncloudstorage.com/neotwin/STUDY_FILE/202609/2609111851148214507b2d8b2d24878bda455ed66fe28e8.zip'
[ -f keep_GB/gb.zip ] || curl -sSL "$ZIP" -o keep_GB/gb.zip
$PY -m zipfile -l keep_GB/gb.zip >> "$S" 2>&1
$PY -m zipfile -e keep_GB/gb.zip keep_GB/x >> "$S" 2>&1
CK=$(find keep_GB/x -name 'model_best*.pt' | head -1)
[ -f "$CK" ] || { echo "!! no model_best in zip" >> "$S"; exit 1; }
md5sum "$CK" >> "$S"      # 1ba6cc3c... 이면 submit_G3/policy.pt(GB_2000) 와 같은 체크포인트
run() { # tag scenarios extra...
  local tag=$1 sc=$2; shift 2
  echo "== $tag begin $(date)" >> "$S"
  PYTHONUNBUFFERED=1 $L t_eval_go2.py --headless --task Quadruped-v0 --checkpoint "$CK" --scenarios "$sc" \
    --run_id GB --out "eval_results/GB6_$tag.json" "$@" > "/workspace/GB6_$tag.log" 2>&1
  echo "== $tag end rc=$? $(date)" >> "$S"
  grep '\[t_eval\] G[X0-9]*[0-9]:' "/workspace/GB6_$tag.log" >> "$S"
  grep -iE 'error|traceback' "/workspace/GB6_$tag.log" | head -3 >> "$S"
}
run flat   G1,G2,G6
run rough  G3
run slope  G4  --slope_deg 20
run stairs G5  --stair_height 0.12
run dr     G7
run flat_c05 G1,G2  --cmd_scale 0.5
run flat_c15 G1,G2  --cmd_scale 1.5
$PY t_eval_go2.py --merge eval_results/GB6_flat.json eval_results/GB6_rough.json eval_results/GB6_slope.json eval_results/GB6_stairs.json eval_results/GB6_dr.json --run_id GB --out eval_results/GB6_all.json >> "$S" 2>&1
tar czf /workspace/gb6_out.tgz eval_results/GB6_*.json /workspace/GB6_*.log "$S" 2>/dev/null
echo "ALL DONE $(date)" >> "$S"
