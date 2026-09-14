#!/bin/bash
# v1_crossval.sh — 축 모델 교차검증 (2026-09-11)
#
# 무엇: 1라운드 제출 정책(공식 42.34, 시나리오별 q 0.500~0.667)을 v3 평가기로 재채점해서
#       **두 번째 per-시나리오 앵커**를 만든다. 2라운드 앵커(q 0.793~0.842)와 합치면
#       14개 점이 되고, 그걸로 축 모델(yaw 단독 vs 3축 동일)을 교차검증한다.
#
# 왜:   yaw 단독이 2라운드 앵커에서 변동계수 0.0130 으로 3축 동일(0.0231)의 절반이었다.
#       그런데 **앵커 1개(7점)에 적합한 결과다.** 점수 범위가 완전히 다른 1라운드에서도
#       같은 결론이 나오면 채택하고, 아니면 과적합으로 기록하고 `all` 을 유지한다.
#
# 비용: 학습 0. 평가만 약 20분.
#
# 준비: 1라운드 제출 정책을 서버에 올려둘 것 (로컬 h1_upload_ready/policy.pt).
#         → /workspace/training/humanoid/keep_V1/R1_policy.pt
#
# 실행: tmux new -d -s v1 'bash /workspace/training/humanoid/v1_crossval.sh'
# 결과: cat /workspace/V1_RESULT.md
set -u
DIR=${ROBOT_DIR:-/workspace/training/humanoid}
cd "$DIR" || exit 1
L="/workspace/IsaacLab/isaaclab.sh -p"
PY=/isaac-sim/python.sh
TASK=${TASK:-Humanoid-v0}
POL=${POL:-keep_V1/R1_policy.pt}
ST=/workspace/V1_STATUS.txt
RES=/workspace/V1_RESULT.md
LOGD=/workspace/v1logs
mkdir -p "$LOGD" eval_results
status(){ echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$ST"; }

[ -f "$POL" ] || { status "!! 정책이 없다: $POL  (로컬 h1_upload_ready/policy.pt 를 올려라)"; exit 1; }
status "V1 start — policy=$POL md5=$(md5sum "$POL" | cut -c1-12)"

ev(){ local sc=$1 tag=$2; shift 2
  PYTHONUNBUFFERED=1 $L t_eval.py --headless --task "$TASK" --policy "$POL" \
      --scenarios "$sc" --run_id V1_R1policy --out "eval_results/V1_$tag.json" \
      --official_ref R1 "$@" > "$LOGD/$tag.log" 2>&1 \
    || status "!! $tag 실패 ($LOGD/$tag.log)"
}
ev H1,H2,H3,H4,H7 flat
ev H5 rough
ev H6 slope
$PY t_eval.py --merge eval_results/V1_*.json --run_id V1_R1policy \
    --out eval_results/V1.json --official_ref R1 > "$LOGD/merge.log" 2>&1

# 두 앵커를 같이 놓고 축 모드별 비율 일관성을 비교한다.
$PY - > "$RES" <<'PEOF'
import json, math, os
import t_eval as T

R1 = T.OFFICIAL_R1
R2 = T.OFFICIAL_R2
anchors = []
p1 = "eval_results/V1.json"
if os.path.exists(p1):
    anchors.append(("1R (공식 42.34)", json.load(open(p1)), R1))
for cand in ("eval_results/R3_R3A_9000.json", "eval_results/R2_anchor.json"):
    if os.path.exists(cand):
        anchors.append(("2R (공식 58.00)", json.load(open(cand)), R2))
        break

print("# V1 — 축 모델 교차검증\n")
if len(anchors) < 2:
    print("!! 앵커가 2개가 아니다. 2라운드 결과 JSON(R3_R3A_9000.json)을 eval_results/ 에 넣어라.")
    print("   찾은 앵커:", [a[0] for a in anchors])

print("| 축 모드 | 앵커 | 비율 평균 | 표준편차 | 범위 | **변동계수** |")
print("|---|---|---:|---:|---:|---:|")
summary = {}
for mode in T.AXIS_MODES:
    cvs = []
    for label, d, off in anchors:
        rs = []
        for sid in T.SCENARIO_ORDER:
            sc = d["scenarios"].get(sid)
            if not sc or sid not in off:
                continue
            mse = {a: float(sc["rmse"][a]) ** 2 for a in ("vx", "vy", "yaw")}
            t = T.tracking_from_mse(mse, T.SCENARIOS[sid], mode)
            score = sc["survival"] * t
            if score > 0:
                rs.append(off[sid] / score)
        if len(rs) < 2:
            continue
        m = sum(rs) / len(rs)
        sd = math.sqrt(sum((x - m) ** 2 for x in rs) / len(rs))
        cvs.append(sd / m)
        print(f"| {mode} | {label} | {m:.4f} | {sd:.5f} | {max(rs)-min(rs):.4f} | **{sd/m:.4f}** |")
    if cvs:
        summary[mode] = sum(cvs) / len(cvs)

print("\n## 판정\n")
if summary:
    best = min(summary, key=summary.get)
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"- `{k}`: 앵커 평균 변동계수 **{v:.4f}**")
    print(f"\n→ 최소는 `{best}`.")
    if best == "yaw" and len(anchors) >= 2:
        print("  **두 앵커 모두에서 yaw 단독이 최소면 채택한다.** t_eval.py 의 --axis_mode 기본값을 바꿔라.")
    elif len(anchors) >= 2:
        print(f"  yaw 단독은 2라운드 단독 적합에서만 좋았다 → **과적합으로 기록하고 `all` 을 유지한다.**")
    print("\n  ⚠️ 어느 쪽이든 순위 판정은 세 모드를 다 보고 한다. 셋이 같은 순위를 주면")
    print("     축 모드 논쟁과 무관하게 그 후보가 이긴 것이다.")
PEOF
status "V1 결과: $RES"; cat "$RES" | tee -a "$ST"
