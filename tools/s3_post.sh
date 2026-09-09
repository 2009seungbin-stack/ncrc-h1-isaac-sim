# ===== S3 post — 학습 종료(TRAIN_DONE) 후 붙여넣기: export + T-EVAL 3그룹(tmux) =====
cd /workspace/training/humanoid || exit 1
PY=/isaac-sim/python.sh
L="/workspace/IsaacLab/isaaclab.sh -p"
tail -3 /tmp/train_S3.log
RUN=$(basename $(ls -1dt ckpt_keep/*/ | head -1)); echo "RUN=$RUN"; ls ckpt_keep/$RUN | sort -V | tail -4
CK=ckpt_keep/$RUN/model_2900.pt; [ -f "$CK" ] || CK=$(ls -1 ckpt_keep/$RUN/model_*.pt | sort -V | tail -1); echo "CK=$CK"
mkdir -p logs/rsl_rl/humanoid/$RUN/params keep_S3
cp "$CK" logs/rsl_rl/humanoid/$RUN/; cp "$CK" keep_S3/
cp ckpt_keep/$RUN/env.yaml keep_S3/env.yaml; cp humanoid_rewards.py keep_S3/humanoid_rewards_S3.py
PYTHONUNBUFFERED=1 $L play.py --task Humanoid-v0 --num_envs 32 --headless --load_run $RUN --checkpoint /workspace/training/humanoid/logs/rsl_rl/humanoid/$RUN/$(basename $CK) 2>&1 | tail -12
POL=$(ls -1t logs/rsl_rl/humanoid/$RUN/exported/policy.pt exported/policy.pt 2>/dev/null | head -1); echo "POL=$POL"
cp "$POL" keep_S3/policy.pt; md5sum keep_S3/policy.pt keep_S3/$(basename $CK)
cat > /tmp/s3_teval.sh <<'TEOF'
cd /workspace/training/humanoid
L="/workspace/IsaacLab/isaaclab.sh -p"
POL=keep_S3/policy.pt; mkdir -p eval_results
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy $POL --scenarios H1,H2,H3,H4,H7 --run_id S3 --out eval_results/S3_flat.json 2>&1 | tee /tmp/teval_S3_flat.log
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy $POL --scenarios H5 --run_id S3 --out eval_results/S3_rough.json 2>&1 | tee /tmp/teval_S3_rough.log
PYTHONUNBUFFERED=1 $L t_eval.py --headless --policy $POL --scenarios H6 --run_id S3 --out eval_results/S3_slope.json 2>&1 | tee /tmp/teval_S3_slope.log
RUN=$(basename $(ls -1dt ckpt_keep/*/ | head -1))
tar czf /workspace/S3_bundle.tgz keep_S3 eval_results/S3_*.json logs/rsl_rl/humanoid/$RUN/events.out.tfevents.* logs/rsl_rl/humanoid/$RUN/params 2>/dev/null
ls -l /workspace/S3_bundle.tgz; echo TEVAL_DONE >> /tmp/teval_S3_slope.log; echo TEVAL_DONE
TEOF
tmux kill-session -t teval 2>/dev/null; tmux new -d -s teval 'bash /tmp/s3_teval.sh'
echo "T-EVAL started $(date)  확인: tail -n2 /tmp/teval_S3_*.log; grep -c TEVAL_DONE /tmp/teval_S3_slope.log"
