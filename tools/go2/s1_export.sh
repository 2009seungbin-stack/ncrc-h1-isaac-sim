#!/bin/bash
# s1_export.sh — 보드 '학습 파일 내역' 백업 zip(model_best.pt + env.yaml)에서 제출용 policy.pt 를 만든다.
#
# 워크스페이스가 지워져도 train.py 종료 후처리가 올린 백업이 공개 스토리지에 남는다.
# 6.9MB 체크포인트를 붙여넣기로 보낼 필요 없이 curl 한 줄로 되살린다.
#
#   ZIP_URL=... WANT_MD5=<로컬 model_best md5> bash s1_export.sh
#   상태: cat /workspace/EXPORT_STATUS.txt
set -u
: "${ZIP_URL:?ZIP_URL 을 넘겨라}"
ROBOT_DIR=/workspace/training/quadruped
RUN=${RUN:-2026-09-13_20-15-36}
CK_NAME=${CK_NAME:-model_199.pt}   # 백업 model_best md5 ec129990 = ckpt_keep model_199.pt
WANT_MD5=${WANT_MD5:-}
OUT=/workspace/s1x
L="/workspace/IsaacLab/isaaclab.sh -p"
PY=/isaac-sim/python.sh
ST=/workspace/EXPORT_STATUS.txt
status(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$ST"; }

rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT" || exit 1
curl -sSL -o s1.zip "$ZIP_URL" || { status "!! curl 실패"; exit 1; }
status "zip $(stat -c %s s1.zip) bytes"
$PY -m zipfile -l s1.zip >> "$ST" 2>&1
$PY -m zipfile -e s1.zip x >> "$ST" 2>&1
MB=$(find x -name 'model_best*.pt' | head -1)
EY=$(find "$OUT/x" -name 'env*.yaml' | head -1)   # 절대경로 — 아래에서 cd 한 뒤 [ -f ] 로 검사한다
[ -f "$MB" ] || { status "!! zip 안에 model_best*.pt 가 없다"; exit 1; }
got=$(md5sum "$MB" | cut -d' ' -f1)
status "model_best md5 $got  want ${WANT_MD5:-미지정}"
if [ -n "$WANT_MD5" ] && [ "$got" != "$WANT_MD5" ]; then status "!! md5 불일치 — S1 체크포인트가 아니다. 중단"; exit 1; fi

cd "$ROBOT_DIR" || exit 1
RD=logs/rsl_rl/quadruped/$RUN
mkdir -p "$RD/params"
cp "$OUT/$MB" "$RD/$CK_NAME"
[ -f "$EY" ] && cp "$EY" "$RD/params/env.yaml"
# 서버 play.py 래퍼는 --checkpoint 를 줘도 exported/model_best.pt 가 없으면 "재생할 정책이 없다"며 실행을 거부한다
# (2026-09-13 실측). 초기화된 워크스페이스에서는 같은 체크포인트를 거기에도 둔다.
mkdir -p exported
cp "$OUT/$MB" exported/model_best.pt
[ -f "$EY" ] && cp "$EY" exported/env.yaml

# play.py 는 헤드리스에서 export 후에도 안 끝난다. 커맨드라인에 play.py 가 안 보이게 스크립트 파일로 띄우고
# 이 run 의 프로세스만 죽인다(pkill -f play.py 통째로 쓰면 셸까지 죽는다).
cat > /tmp/play_s1.sh <<EOF
cd $ROBOT_DIR
PYTHONUNBUFFERED=1 $L play.py --task Quadruped-v0 --num_envs 32 --headless --load_run $RUN --checkpoint $ROBOT_DIR/$RD/$CK_NAME > /workspace/export_play.log 2>&1
EOF
t0=$(date +%s)
tmux kill-session -t play_s1 2>/dev/null
tmux new -d -s play_s1 'bash /tmp/play_s1.sh'
status "play.py 시작 (run $RUN, $CK_NAME)"
pol=$RD/exported/policy.pt
t=0
while [ $t -lt 600 ]; do
  sleep 5; t=$((t+5))
  if [ -f "$pol" ] && [ "$(stat -c %Y "$pol")" -ge "$t0" ] && [ $(( $(date +%s) - $(stat -c %Y "$pol") )) -ge 20 ]; then break; fi
done
for p in $(pgrep -f "play.py.*--load_run $RUN"); do kill "$p" 2>/dev/null; done
sleep 3; tmux kill-session -t play_s1 2>/dev/null
if ! { [ -f "$pol" ] && [ "$(stat -c %Y "$pol")" -ge "$t0" ]; }; then
  status "!! export 실패 (${t}s) — export_play.log 끝:"; tail -20 /workspace/export_play.log | tee -a "$ST"; exit 1
fi
status "policy.pt $(stat -c %s "$pol") bytes, md5 $(md5sum "$pol" | cut -c1-32), ${t}s"

# JIT 정책이 체크포인트 actor 가중치 그대로인지 순수 torch 로 재구성해 비교한다.
$PY - "$pol" "$RD/$CK_NAME" <<'PEOF' 2>&1 | tee -a "$ST"
import sys, torch
jit = torch.jit.load(sys.argv[1], map_location="cpu").eval()
ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
sd = next((ck[k] for k in ("actor_state_dict", "model_state_dict") if k in ck), None)
print("ckpt keys:", list(ck.keys()))
if sd is None:
    sys.exit("!! actor 가중치 키를 못 찾았다")
ws = [n for n, v in sd.items() if n.endswith("weight") and v.dim() == 2 and "critic" not in n]
layers = [(sd[n], sd[n[:-6] + "bias"]) for n in ws]
print("actor layers:", [tuple(w.shape) for w, _ in layers])
x = torch.randn(64, layers[0][0].shape[1])
h = x
for i, (w, b) in enumerate(layers):
    h = h @ w.T + b
    if i < len(layers) - 1:
        h = torch.nn.functional.elu(h)
with torch.no_grad():
    y = jit(x)
d = (y - h).abs().max().item()
print(f"JIT vs ckpt maxdiff {d:.3e}", "OK" if d < 1e-4 else "!! 불일치")
PEOF

cp "$pol" /workspace/S1_policy.pt
cp "$RD/params/env.yaml" /workspace/S1_env.yaml 2>/dev/null
md5sum /workspace/S1_policy.pt /workspace/S1_env.yaml | tee -a "$ST"
status "=== EXPORT DONE ==="
