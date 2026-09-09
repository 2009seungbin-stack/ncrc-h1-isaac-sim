# ===== S3 kickoff — /workspace/training/humanoid 에서, t_eval_paste.sh 설치 후 붙여넣기 =====
cd /workspace/training/humanoid || exit 1
PY=/isaac-sim/python.sh
L="/workspace/IsaacLab/isaaclab.sh -p"
for f in train.py play.py humanoid_rewards.py set_reward.py t_eval.py; do [ -f "$f" ] || echo "!! missing $f"; done
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
cat > /tmp/guard.sh <<'GEOF'
cd /workspace/training/humanoid
mkdir -p ckpt_keep
while true; do
  for r in $(ls -1dt logs/rsl_rl/humanoid/*/ 2>/dev/null | head -1); do
    b=$(basename $r); mkdir -p ckpt_keep/$b
    for f in $r/model_*.pt; do [ -f "$f" ] || continue; n=$(basename $f)
      [ -f ckpt_keep/$b/$n ] || cp "$f" ckpt_keep/$b/$n; done
    cp $r/params/env.yaml ckpt_keep/$b/env.yaml 2>/dev/null
  done
  sleep 45
done
GEOF
tmux kill-session -t guard 2>/dev/null; tmux new -d -s guard 'bash /tmp/guard.sh'
$PY set_reward.py termination_penalty=-200.0 track_ang_vel_z_exp=2.0 action_rate_l2=-0.005
$PY set_reward.py --show | grep -E 'termination_penalty|track_ang_vel_z_exp|action_rate_l2|track_lin_vel_xy_exp'
tmux kill-session -t train 2>/dev/null
tmux new -d -s train "cd /workspace/training/humanoid && PYTHONUNBUFFERED=1 $L train.py --task Humanoid-v0 --num_envs 16384 --max_iterations 3000 --headless 2>&1 | tee /tmp/train_S3.log; echo TRAIN_DONE >> /tmp/train_S3.log"
sleep 2; tmux ls; date
echo "== 2분 후: nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader; grep -iE 'iteration' /tmp/train_S3.log | tail -3"
