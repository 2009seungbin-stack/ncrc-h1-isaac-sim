# R3 킥오프 — r3_paste.sh 설치 후 붙여넣기. 학습 2개 + 채점 + 번들 + keep-alive 전부 tmux 'r3' 안에서 돈다.
cd /workspace/training/humanoid || exit 1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
rm -f /workspace/STOP_KEEPALIVE
tmux kill-session -t r3 2>/dev/null
tmux new -d -s r3 'bash /workspace/training/humanoid/r3_pipeline.sh'
sleep 15; tmux ls; tail -5 /workspace/R3_STATUS.txt
echo "== 3분 뒤: tail -8 /workspace/R3_STATUS.txt ; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader"
