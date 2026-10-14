#!/usr/bin/env python3
"""T-EVAL — NCRC H1 7시나리오 내부 평가기.

`tools/scenario_eval_spec.md` 의 구현체다. 서버(`/workspace/training/humanoid/`)에
그대로 복사해서 Isaac Lab 파이썬으로 실행한다.

    python t_eval.py --task Humanoid-v0 --policy exported/policy.pt \
        --run_id H1_R2_003 --out eval_results/H1_R2_003.json --headless

주의 — 이 평가기는 **공식 점수를 예측하지 못한다.** 공식 명령 시퀀스·지형 파라미터는
비공개다. 이 도구의 목적은 후보 정책 사이의 **상대 비교(Δ)** 이며, 절대값은
`model_4600`(공식 42.34)과의 오프셋을 기록해 보정 기준으로만 쓴다.

채점 (규정 제8조):
    scenario_score_i = survival_i x tracking_i
    SimScore         = 70 x sum_i( w_i x scenario_score_i )
    tracking_i       = exp(-RMSE_i),  RMSE는 시나리오 해당 축만
"""

import argparse
import json
import math
import os
import re
import sys

# 표가 한글·기호를 쓴다. 윈도우 콘솔 기본 코덱(cp949)에서 죽지 않게 먼저 UTF-8로 돌린다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# 시나리오 정의 (spec 3절)
# ---------------------------------------------------------------------------
# cmds: 동일 길이 구간 4개. 각 원소는 (vx, vy, yaw_rate) 명령 (body frame).
# axes: RMSE를 계산할 축. terrain: 'flat' | 'rough' | 'slope'.
SCENARIOS = {
    "H1": dict(w=0.15, terrain="flat", axes=("vx", "vy", "yaw"),
               cmds=[(0.0, 0.0, 0.0)] * 4,
               desc="제자리 균형"),
    "H2": dict(w=0.20, terrain="flat", axes=("vx",),
               cmds=[(0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)],
               desc="전진 속도 추종 (4번째 구간은 후진)"),
    "H3": dict(w=0.10, terrain="flat", axes=("vy",),
               cmds=[(0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.5, 0.0), (0.0, -0.5, 0.0)],
               desc="좌우 속도 추종"),
    "H4": dict(w=0.10, terrain="flat", axes=("yaw", "vx"),
               cmds=[(0.5, 0.0, 0.5), (0.5, 0.0, -0.5), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)],
               desc="회전 복합 추종"),
    "H5": dict(w=0.15, terrain="rough", axes=("vx", "yaw"),
               cmds=[(0.5, 0.0, 0.0), (0.5, 0.0, 0.3), (0.5, 0.0, 0.0), (0.5, 0.0, -0.3)],
               desc="약한 요철"),
    "H6": dict(w=0.15, terrain="slope", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               desc="완만한 경사 +-10도 (오르막/내리막 50:50)"),
    "H7": dict(w=0.15, terrain="flat", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               push=True,
               desc="밀침 회복"),
    # ---- Stress 묶음 X — w=0 이라 SimScore 에 안 들어간다. 생존 포화를 풀어 후보 간
    #      생존 순위를 가르는 용도. 시나리오별 push_mag/push_period_s/seg_s 를 덮어쓴다.
    #      지형 파라미터는 프로세스 단위(CLI)라 X2/X3/X5 는 각각 따로 실행한다.
    "X1": dict(w=0.0, terrain="flat", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               push=True, push_mag=2.0, push_period_s=1.5,
               desc="STRESS 강한 밀침 2.0 m/s 주기 1.5 s"),
    "X2": dict(w=0.0, terrain="rough", axes=("vx", "yaw"),
               cmds=[(0.8, 0.0, 0.0), (0.8, 0.0, 0.5), (0.8, 0.0, 0.0), (0.8, 0.0, -0.5)],
               desc="STRESS 거친 요철 (--rough_min 0.06 --rough_max 0.10 으로 실행)"),
    "X3": dict(w=0.0, terrain="slope", axes=("vx",),
               cmds=[(0.8, 0.0, 0.0), (0.8, 0.0, 0.0), (-0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)],
               desc="STRESS 경사 15도 + 후진 (--slope_deg 15 로 실행)"),
    "X4": dict(w=0.0, terrain="flat", axes=("vx", "vy", "yaw"), seg_s=2.0,
               cmds=[(1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, -1.0),
                     (1.0, 0.0, 1.0), (0.0, -0.5, 0.0), (0.0, 0.0, 1.0), (-0.5, 0.0, -1.0),
                     (1.0, 0.5, 0.0), (0.0, 0.0, 0.0)],
               desc="STRESS 2초 급전환 x10"),
    "X5": dict(w=0.0, terrain="rough", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               push=True, push_mag=1.5, push_period_s=2.5,
               desc="STRESS 밀침 1.5 + 요철 (--rough_min 0.04 --rough_max 0.08 로 실행)"),
}
SCENARIO_ORDER = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
STRESS_ORDER = ["X1", "X2", "X3", "X4", "X5"]

# run() 이 순회할 지형 그룹. Go2 모듈이 stairs/dr 을 덧붙인다.
TERRAIN_ORDER = ["flat", "rough", "slope"]

# spec 4절 — 절대 바꾸지 않는다.
EVAL_SET_A = [103, 280, 512, 991]

# 공식 1라운드 실측 (official_score_breakdown.md) — 오프셋 계산용
OFFICIAL_R1 = {"H1": 0.667, "H2": 0.598, "H3": 0.661,
               "H4": 0.500, "H5": 0.562, "H6": 0.612, "H7": 0.620}
OFFICIAL_R1_SIM = 42.34

# 공식 2라운드 실측 (R3A_9000, 보드 2026-09-11 확인). 받은점수/배점.
OFFICIAL_R2 = {"H1": 8.84 / 10.5, "H2": 11.68 / 14.0, "H3": 5.86 / 7.0,
               "H4": 5.84 / 7.0, "H5": 8.33 / 10.5, "H6": 8.70 / 10.5,
               "H7": 8.75 / 10.5}
OFFICIAL_R2_SIM = 58.00

OFFICIAL_REFS = {"R1": (OFFICIAL_R1, OFFICIAL_R1_SIM),
                 "R2": (OFFICIAL_R2, OFFICIAL_R2_SIM)}

# E1 — 자체→공식 환산. **깨끗한 앵커 2개 확보 (2026-09-11 V1 교차검증).**
#
#   평지 5종(H1,H2,H3,H4,H7 / 가중치 0.70) 동일 기준 비교:
#     1R model_4600 : 자체 0.7034 -> 공식 0.6125   (비율 0.8707)
#     2R R3A_9000   : 자체 0.9122 -> 공식 0.8361   (비율 0.9166)
#
#   1R 정책은 보드에서 내려받아 md5 57a68b4ebffd9fe7f7604f5bbe803b4b 로 동일성을 증명했고
#   (로컬 h1_upload_ready/policy.pt 와 SHA-256 일치), 현재 평가기로 재채점한 값이다.
#   과거 "1R 배율 0.906" 은 서로 다른 정책(42.34 / 46.73)을 나눈 잘못된 값이었다.
#
#   2앵커 직선:  공식q = 1.0714 * 자체q - 0.1411
#   기울기가 1보다 크다 = **공식이 정책 간 차이를 증폭한다.** 우리 시험이 더 쉬워
#   좋은 정책과 나쁜 정책이 압축돼 보이는 것과 일치한다.
#
#   검정 결과: 거듭제곱 모형(공식q = 자체q**m)은 기각됐다.
#     V1 실측 대비 잔차 RMS  직선/비율 0.0335  vs  거듭제곱 0.0915 (2.7배)
CALIB_LINE = (1.0714, -0.1411)     # (기울기, 절편) — q 공간
CALIB_ANCHORS_N = 2

# 목표 총점 -> 필요 자체 SimScore (CALIB_LINE 역산)
#   88 -> 63.36   89 -> 64.29   90 -> 65.22   **91 -> 66.16**   93 -> 68.02
#   현재 H1 최고 R3A_9000 = 63.87 (자체 q 0.9124, 공식 88)
TARGET_TOTAL = 91.0

# Go2 모듈이 True 로 바꾼다 — 다른 로봇의 앵커를 빌려 쓰는 중이라는 표시.
CALIB_BORROWED = False
NONSIM_POINTS = 30.0   # 설계의도 20 + 리포트 10. 2라운드에 만점 확인.

AXIS_MODES = ("all", "scenario", "yaw")


# ---------------------------------------------------------------------------
# 통계 (spec 5절)
# ---------------------------------------------------------------------------
def wilson_lcb(k, n, z=1.96):
    """생존율의 Wilson 95% 하한. 점추정 대신 이걸로 게이트를 건다."""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def axis_weights(spec, mode):
    """RMSE에 쓸 축 가중치 (vx, vy, yaw).

    mode="all"      : 세 축 동일 가중. **기본값.**
    mode="scenario" : 시나리오별 축 선택 (구버전).
    mode="yaw"      : yaw 단독. 2026-09-11 2라운드 앵커 재적합에서 공식과 가장 잘 맞았다
                      (비율 변동계수 0.0130 vs all 0.0231, 범위 0.041 vs 0.077).
                      ⚠️ 앵커 1개(7점)에 적합한 결과다. 1라운드 정책 재채점(V1)으로
                      교차검증하기 전까지 **단독 채택 금지** — 병기만 한다.

    왜 all 이 기본인가 — 2026-09-04 E1 실측:
      시나리오별 축 선택은 공식 대비 오프셋 표준편차 0.137, 범위 0.356.
      3축 동일 가중은 표준편차 0.048, 범위 0.148 로 2.9배 조여진다.
      이 환경은 heading_command=True 라 모든 시나리오에서 방향 유지가 요구되고,
      직진 명령(H2/H6)에서도 yaw RMSE가 0.71~0.78 로 선속도(0.05)의 10배가 넘는다.
      공식 추종 점수가 yaw를 항상 반영한다고 보는 것이 측정과 맞는다.
    """
    if mode == "all":
        return {"vx": 1.0, "vy": 1.0, "yaw": 1.0}
    if mode == "yaw":
        return {"vx": 0.0, "vy": 0.0, "yaw": 1.0}
    used = spec["axes"]
    return {a: (1.0 if a in used else 0.0) for a in ("vx", "vy", "yaw")}


def tracking_from_mse(mse_by_axis, spec, mode):
    w = axis_weights(spec, mode)
    tot = sum(w.values())
    if tot <= 0:
        return 0.0
    mse = sum(w[a] * mse_by_axis[a] for a in w) / tot
    return math.exp(-math.sqrt(mse))


# ---------------------------------------------------------------------------
# --merge 모드 — Isaac 앱 없이 동작한다 (그룹별로 나눠 돌린 결과 합치기)
# ---------------------------------------------------------------------------
def do_recompute(paths, out_path, run_id, mode):
    """저장된 축별 RMSE로 tracking/score 를 다시 계산한다. 시뮬 재실행 없음.

    축 정의를 바꿔도 과거 결과를 버리지 않아도 되는 이유가 이것이다.
    """
    scenarios, meta = {}, {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        meta.setdefault("checkpoint", d.get("checkpoint"))
        meta.setdefault("config", d.get("config"))
        meta.setdefault("run_id", d.get("run_id"))
        for sid, sc in d.get("scenarios", {}).items():
            if sid not in SCENARIOS:
                continue
            mse = {a: float(sc["rmse"][a]) ** 2 for a in ("vx", "vy", "yaw")}
            t = tracking_from_mse(mse, SCENARIOS[sid], mode)
            sc = dict(sc)
            sc["tracking"] = round(t, 4)
            sc["score"] = round(sc["survival"] * t, 4)
            sc["axis_mode"] = mode
            sc["axis_weights"] = axis_weights(SCENARIOS[sid], mode)
            scenarios[sid] = sc
    cfg = dict(meta.get("config") or {})
    cfg["axis_mode"] = mode
    cfg["recomputed"] = True
    result = finalize(scenarios, run_id or meta.get("run_id", "recomputed"),
                      meta.get("checkpoint"), cfg)
    write_result(result, out_path)
    return result


def do_merge(paths, out_path, run_id):
    scenarios, meta = {}, {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        scenarios.update(d.get("scenarios", {}))
        meta.setdefault("checkpoint", d.get("checkpoint"))
        meta.setdefault("config", d.get("config"))
        meta.setdefault("run_id", d.get("run_id"))
    result = finalize(scenarios, run_id or meta.get("run_id", "merged"),
                      meta.get("checkpoint"), meta.get("config", {}))
    write_result(result, out_path)
    return result


def sim_scores_by_axis_mode(scenarios):
    """저장된 축별 RMSE로 세 축 모드의 SimScore 를 한꺼번에 계산한다 (E2).

    시뮬 재실행이 없다. 판정할 때 한 모드를 고르지 말고 셋을 다 보고, 셋이 같은
    순위를 주면 축 모드 논쟁과 무관하게 그 후보가 이긴 것이다.
    """
    out = {}
    for mode in AXIS_MODES:
        sim, cov = 0.0, 0.0
        for sid in SCENARIO_ORDER:
            s = scenarios.get(sid)
            if not s or "rmse" not in s:
                continue
            mse = {a: float(s["rmse"][a]) ** 2 for a in ("vx", "vy", "yaw")}
            t = tracking_from_mse(mse, SCENARIOS[sid], mode)
            sim += s["w"] * s["survival"] * t
            cov += s["w"]
        if cov > 0:
            out[mode] = round(70.0 * sim, 3)
    return out


def predict_official(self_sim, mode=None, line=None):
    """자체 SimScore -> 공식 총점. 앵커 2개로 적합한 직선(CALIB_LINE)을 쓴다.

    밴드는 두 앵커의 '비율 모형'(원점 통과) 배율 0.8707 / 0.9166 을
    보수적 대안으로 같이 계산해 최소~최대를 보여준다.
    """
    m, c = line or CALIB_LINE
    qs = self_sim / 70.0
    qo = m * qs + c
    alts = [0.8707 * qs, 0.9166 * qs]
    lo, hi = min([qo] + alts), max([qo] + alts)
    tgt = ((TARGET_TOTAL - NONSIM_POINTS) / 70.0 - c) / m * 70.0
    return {"self_q": round(qs, 4),
            "official_sim": round(70 * qo, 2),
            "official_total": round(70 * qo + NONSIM_POINTS, 2),
            "total_lo": round(70 * lo + NONSIM_POINTS, 2),
            "total_hi": round(70 * hi + NONSIM_POINTS, 2),
            "band": round(70 * (hi - lo), 2),
            "anchor": "2R R3A_9000 + 1R model_4600 (V1)",
            "target_total": TARGET_TOTAL,
            "target_self_sim": round(tgt, 2)}


def official_prediction(sim_by_mode, borrowed=False):
    """E1 — 자체 SimScore 로 공식 총점을 밴드와 함께 예측한다.

    점추정만 믿지 말고 밴드를 봐라. 밴드가 좁으면(앵커 근처) 절대값도 쓸 만하고,
    넓으면(원거리 외삽) 순위 비교로만 써야 한다.

    borrowed=True 는 **다른 로봇의 앵커를 빌려 쓰는 중**이라는 뜻이다 (Go2 는 앵커가 없다).
    이때는 λ 불확실성 위에 '로봇 전이' 불확실성이 더 얹히는데 그건 정량화할 근거가 없다.
    → 절대값을 쓰지 말고 후보 사이 순위만 봐라. 3라운드 제출이 Go2 의 첫 앵커가 된다.
    """
    pred = {}
    for mode, sim in sim_by_mode.items():
        r = predict_official(sim)
        if r is None:
            continue
        r["borrowed_anchor"] = borrowed
        r["ratio"] = r.get("lambda_point", r.get("ratio"))  # 하위 호환
        pred[mode] = r
    return pred


def finalize(scenarios, run_id, checkpoint, config):
    sim = 0.0
    covered_w = 0.0
    for sid in SCENARIO_ORDER:
        s = scenarios.get(sid)
        if not s:
            continue
        sim += s["w"] * s["score"]
        covered_w += s["w"]
    sim_score = 70.0 * sim
    h_ids = [k for k in SCENARIO_ORDER if k in scenarios]
    x_ids = [k for k in STRESS_ORDER if k in scenarios]
    # Stress 요약 — 생존만 본다 (추종은 참고용). 에피소드 수를 모르는 옛 JSON 은 512 로 가정.
    stress = None
    if x_ids:
        ep = sum(int(scenarios[k].get("episodes", 512)) for k in x_ids)
        sv = sum(int(scenarios[k].get("survived",
                     round(scenarios[k]["survival"] * scenarios[k].get("episodes", 512))))
                 for k in x_ids)
        stress = {
            "episodes": ep, "survived": sv, "falls": ep - sv,
            "survival": round(sv / max(1, ep), 4),
            "survival_lcb": round(wilson_lcb(sv, ep), 4),
            "per_scenario": {k: {"survival": scenarios[k]["survival"],
                                 "survival_lcb": scenarios[k]["survival_lcb"],
                                 "falls": scenarios[k]["falls"],
                                 "tracking": scenarios[k]["tracking"]} for k in x_ids},
        }
    # E6 — H 시나리오 전체의 생존 Wilson 하한. 점추정은 1.0으로 포화하므로 게이트는 이걸 쓴다.
    h_ep = sum(int(scenarios[k].get("episodes", 0)) for k in h_ids)
    h_sv = sum(int(scenarios[k].get("survived", 0)) for k in h_ids)
    sim_by_mode = sim_scores_by_axis_mode(scenarios)
    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "eval_seed_set": "EVAL_SET_A",
        "scenarios": {k: scenarios[k] for k in h_ids + x_ids},
        "coverage_weight": round(covered_w, 4),
        "sim_score": round(sim_score, 3),
        "sim_score_pct": round(100.0 * sim / covered_w, 2) if covered_w else 0.0,
        "sim_score_by_axis_mode": sim_by_mode,
        "official_pred": official_prediction(sim_by_mode, CALIB_BORROWED),
        "h_survival": round(h_sv / h_ep, 4) if h_ep else None,
        "h_survival_lcb": round(wilson_lcb(h_sv, h_ep), 4) if h_ep else None,
        "official_offset": None,
        "falls": sum(scenarios[k]["falls"] for k in h_ids),
        "nan_actions": sum(scenarios[k].get("nan_actions", 0) for k in scenarios),
        "crashes": sum(scenarios[k].get("crashes", 0) for k in scenarios),
        "stress": stress,
        "config": config,
    }


def write_result(result, out_path):
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def print_offset_table(result, ref="R2"):
    """검증 게이트 표. **비율 열이 E3′ 역보정의 계기판이다.**

    비율 = 공식 q / 자체 점수. 7개가 한 상수로 모이면 평가기가 공식과 같은 걸 재고 있다는
    뜻이고, 그때부터 자체 SimScore 가 공식 점수의 단일 배율 예측치가 된다.
    튀는 시나리오가 그 시나리오의 평가 환경이 공식과 다르다는 신호다
    (2026-09-11 실측: H5 요철 0.874 = 우리가 너무 쉽다, H7 밀침 0.951 = 우리가 너무 세다).
    """
    off, off_sim = OFFICIAL_REFS.get(ref, (None, None))
    if off is None:
        off = {}
    print(f"\n| 시나리오 | 공식 실측({ref}) | 내부 평가기 | 오프셋 | 비율 |")
    print("|---|---:|---:|---:|---:|")
    ratios = []
    for sid in SCENARIO_ORDER:
        s = result["scenarios"].get(sid)
        o = off.get(sid)
        if not s or o is None:
            print(f"| {sid} | {'-' if o is None else format(o, '.3f')} | "
                  f"{'-' if not s else format(s['score'], '.3f')} | - | - |")
            continue
        r = o / s["score"] if s["score"] > 0 else float("nan")
        ratios.append(r)
        print(f"| {sid} | {o:.3f} | {s['score']:.3f} | {o - s['score']:+.3f} | {r:.4f} |")
    if off_sim is not None and result["sim_score"]:
        print(f"| **SimScore** | **{off_sim:.2f}** | **{result['sim_score']:.2f}** | "
              f"**{off_sim - result['sim_score']:+.2f}** | "
              f"**{off_sim / result['sim_score']:.4f}** |")
    if len(ratios) >= 2:
        m = sum(ratios) / len(ratios)
        sd = math.sqrt(sum((x - m) ** 2 for x in ratios) / len(ratios))
        print(f"\n비율 평균 {m:.4f}  표준편차 {sd:.5f}  범위 {max(ratios) - min(ratios):.4f}  "
              f"변동계수 {sd / m:.4f}   ← **E3′는 이 변동계수를 최소화하는 지형·밀침 값을 찾는다**")

    # E1/E2 — 축 모드별 SimScore 와 공식 점수 예측
    by_mode = result.get("sim_score_by_axis_mode") or {}
    pred = result.get("official_pred") or {}
    if by_mode:
        print("")
        print("| 축 모드 | 자체 SimScore | **예측 총점** | 보수 밴드 | 목표까지 |")
        print("|---|---:|---:|:---:|---:|")
        nan = float("nan")
        for mode in AXIS_MODES:
            if mode not in by_mode:
                continue
            pr = pred.get(mode) or {}
            star = " <-기본" if mode == result.get("config", {}).get("axis_mode") else ""
            need = pr.get("target_self_sim", nan)
            print(f"| {mode}{star} | {by_mode[mode]:.2f} | "
                  f"**{pr.get('official_total', nan):.2f}** | "
                  f"{pr.get('total_lo', nan):.2f} ~ {pr.get('total_hi', nan):.2f} | "
                  f"{need - by_mode[mode]:+.2f} |")
        any_p = next((v for v in pred.values() if v), None)
        if any_p:
            m, c = CALIB_LINE
            print("")
            print(f"보정: 공식q = {m:.4f} x 자체q {c:+.4f}  (앵커 2개: {any_p['anchor']})")
            print(f"목표 총점 {TARGET_TOTAL:.0f} = 자체 SimScore {any_p['target_self_sim']:.2f}")
            if any_p.get("borrowed_anchor"):
                print("(!) 다른 로봇(H1) 앵커를 빌려 쓰는 중 - Go2 절대값 금지, 순위만.")
    if result.get("h_survival_lcb") is not None:
        print(f"\nH 전체 생존 {result['h_survival']:.4f}  "
              f"**LCB {result['h_survival_lcb']:.4f}** (게이트는 LCB로 건다)")
    print("\n(커버리지 가중치 합계 = %.2f — 1.00 미만이면 일부 시나리오를 건너뛴 것)"
          % result["coverage_weight"])
    st = result.get("stress")
    if st:
        print("\n| STRESS | 생존율 | LCB | 낙상 | 추종 |")
        print("|---|---:|---:|---:|---:|")
        for k, v in st["per_scenario"].items():
            print(f"| {k} | {v['survival']:.3f} | {v['survival_lcb']:.3f} | {v['falls']} | {v['tracking']:.3f} |")
        print(f"| **X 합계** | **{st['survival']:.3f}** | **{st['survival_lcb']:.3f}** | "
              f"**{st['falls']}** / {st['episodes']} | |")


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="NCRC H1 7시나리오 내부 평가기 (T-EVAL)")
    p.add_argument("--task", default="Humanoid-v0")
    p.add_argument("--policy", default="exported/policy.pt",
                   help="play.py가 내보낸 TorchScript 정책 (.pt)")
    p.add_argument("--num_envs", type=int, default=128,
                   help="seed당 동시 에피소드 수. 총 에피소드 = num_envs x seed 수. "
                        "128x4=512면 생존율 0.891의 Wilson 하한이 0.861까지 좁혀진다")
    p.add_argument("--seeds", default=",".join(str(s) for s in EVAL_SET_A),
                   help="CRN seed set. 기본 EVAL_SET_A. 바꾸면 과거 수치와 비교 불가")
    p.add_argument("--scenarios", default="all",
                   help="쉼표 구분 (예: H5,H6) 또는 all")
    p.add_argument("--seg_s", type=float, default=5.0, help="명령 구간 길이(초)")
    p.add_argument("--transient_s", type=float, default=0.5,
                   help="명령이 바뀐 직후 RMSE에서 제외할 시간(초). 모든 구간에 동일 적용")
    p.add_argument("--run_id", default="UNSET")
    p.add_argument("--out", default=None, help="결과 JSON 경로")
    p.add_argument("--rough_min", type=float, default=0.02, help="H5 요철 noise 하한(m)")
    p.add_argument("--rough_max", type=float, default=0.06, help="H5 요철 noise 상한(m)")
    p.add_argument("--slope_deg", type=float, default=10.0, help="H6 경사각(도)")
    p.add_argument("--stair_height", type=float, default=0.12,
                   help="Go2 G5 계단 한 단 높이(m). 공식은 10~15cm라 기본 0.12")
    p.add_argument("--stair_width", type=float, default=0.30,
                   help="Go2 G5 계단 한 단 폭(m)")
    p.add_argument("--dr_friction", default="0.4,1.2",
                   help="Go2 G7 도메인 랜덤화 마찰 범위 'min,max'")
    p.add_argument("--dr_mass", default="-1.0,3.0",
                   help="Go2 G7 도메인 랜덤화 몸통 질량 가감 범위 'min,max' (kg)")
    p.add_argument("--flat_mode", choices=("mesh", "plane"), default="mesh",
                   help="평지 구현. mesh=평평한 메쉬 패치(기본, height_scanner 안전), "
                        "plane=TerrainImporter 무한 평면")
    p.add_argument("--push_period_s", type=float, default=2.5, help="H7 밀침 주기(초)")
    p.add_argument("--push_mag", type=float, default=1.0, help="H7 밀침 속도 크기(m/s)")
    p.add_argument("--no_obs_noise", action="store_true",
                   help="관측 노이즈를 끈다 (분산은 줄지만 공식 환경과 달라진다)")
    p.add_argument("--checkpoint", default=None,
                   help="rsl_rl 체크포인트(model_*.pt)를 직접 로드해 actor MLP(ELU)로 평가한다. "
                        "play.py export 없이 보드 백업 zip 의 model_best.pt 를 바로 잰다 (2026-09-13)")
    p.add_argument("--task_module", default=None,
                   help="태스크를 등록하는 모듈명. 생략하면 train.py에서 자동 추출")
    p.add_argument("--train_script", default="train.py",
                   help="태스크 등록 import를 긁어올 스크립트")
    p.add_argument("--axis_mode", choices=AXIS_MODES, default="all",
                   help="RMSE 축 가중. all=3축 동일(기본), scenario=시나리오별(구버전), "
                        "yaw=yaw 단독(2R 앵커 재적합 최적, V1 교차검증 전까지 병기만). "
                        "세 모드 SimScore 는 어차피 전부 출력된다")
    p.add_argument("--official_ref", choices=tuple(OFFICIAL_REFS), default="R2",
                   help="오프셋·비율 표에 쓸 공식 실측 기준 라운드")
    p.add_argument("--cmd_scale", type=float, default=1.0,
                   help="E5 — 모든 시나리오 명령을 이 배수로 키운다(기본 1.0=그대로). "
                        "공식 채점이 우리보다 빠른 명령을 쓰는지 가르는 진단용. "
                        "NCRC 학습 명령 범위 lin_x 상한은 1.5로 유니트리 Go2(1.0)보다 넓다")
    p.add_argument("--recompute", nargs="+", default=None,
                   help="기존 결과 JSON을 저장된 축별 RMSE로 다시 채점한다 (GPU 0)")
    p.add_argument("--merge", nargs="+", default=None,
                   help="JSON들을 합치기만 한다 (Isaac 앱 미기동)")
    p.add_argument("--predict", nargs="+", type=float, metavar="SIMSCORE",
                   help="자체 SimScore 값들을 받아 예측 총점과 밴드만 출력한다 (GPU 0, Isaac 불필요). "
                        "목표 역산·후보 선별에 쓴다. 예: --predict 62 64 66 68 70")
    p.add_argument("--preflight", action="store_true",
                   help="본 평가 전 API 가정만 검증하고 끝낸다. 지형 3종을 실제로 만들고 "
                        "height_scanner로 경사각을 실측해 보여준다. 30초 이내")
    return p


def print_prediction_table(sims, mode="all", borrowed=False):
    """자체 SimScore 목록 → 예측 총점 밴드. 시뮬레이터 없이 돈다."""
    m, c = CALIB_LINE
    print("")
    print(f"보정 직선: 공식q = {m:.4f} x 자체q {c:+.4f}  (앵커 2개, 2026-09-11 V1)")
    print(f"목표 총점 {TARGET_TOTAL:.0f}")
    print("")
    print("| 자체 SimScore | **예측 총점** | 보수 밴드 | 목표까지 |")
    print("|---:|---:|:---:|---:|")
    for x in sims:
        r = predict_official(x)
        print(f"| {x:.2f} | **{r['official_total']:.2f}** | "
              f"{r['total_lo']:.2f} ~ {r['total_hi']:.2f} | {r['target_self_sim'] - x:+.2f} |")
    if borrowed:
        print("")
        print("(!) H1 앵커를 빌려 쓰는 중. 로봇 전이 불확실성은 밴드에 없다.")


def main():
    parser = build_parser()

    # --predict 는 JSON 도 시뮬레이터도 필요 없다.
    if "--predict" in sys.argv:
        args, _ = parser.parse_known_args()
        print_prediction_table(args.predict, args.axis_mode, CALIB_BORROWED)
        return

    # --recompute / --merge 는 시뮬레이터 없이 처리한다.
    if "--recompute" in sys.argv:
        args, _ = parser.parse_known_args()
        r = do_recompute(args.recompute, args.out,
                         args.run_id if args.run_id != "UNSET" else None,
                         args.axis_mode)
        print_offset_table(r, args.official_ref)
        return

    if "--merge" in sys.argv:
        args, _ = parser.parse_known_args()
        r = do_merge(args.merge, args.out, args.run_id if args.run_id != "UNSET" else None)
        print_offset_table(r, args.official_ref)
        return

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    run(args)

    simulation_app.close()


# ---------------------------------------------------------------------------
def load_checkpoint_actor(path, device):
    """rsl_rl 체크포인트의 actor MLP 를 순수 torch 로 재구성한다 (s1_export.sh 검증 코드와 동일 규칙:
    JIT export 와 maxdiff 7e-7 로 일치 확인됨 — 관측 정규화 없음, 은닉 ELU)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = next((ck[k] for k in ("actor_state_dict", "model_state_dict") if k in ck), None)
    if sd is None:
        raise SystemExit(f"[t_eval] actor 가중치 키를 못 찾았다: {list(ck.keys())}")
    ws = [n for n, v in sd.items() if n.endswith("weight") and v.dim() == 2 and "critic" not in n]
    layers = [(sd[n].to(device), sd[n[:-6] + "bias"].to(device)) for n in ws]
    print(f"[t_eval] ckpt iter={ck.get('iter')} actor layers={[tuple(w.shape) for w, _ in layers]}")

    def policy(x):
        h = x
        for i, (w, b) in enumerate(layers):
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = torch.nn.functional.elu(h)
        return h
    return policy


def import_task_module(args):
    """Humanoid-v0 를 gym 레지스트리에 올린다.

    NCRC 워크스페이스마다 태스크 등록 모듈 이름이 다를 수 있어, train.py의 import를
    그대로 재현하는 방식으로 맞춘다.
    """
    import importlib
    import gymnasium as gym

    def registered():
        try:
            gym.spec(args.task)
            return True
        except Exception:
            return False

    for mod in ("isaaclab_tasks", args.task_module):
        if not mod:
            continue
        try:
            importlib.import_module(mod)
        except Exception as e:
            print(f"[t_eval] import {mod} 실패: {e}")
        if registered():
            return

    # train.py 의 import 문을 그대로 실행해 본다.
    if os.path.exists(args.train_script):
        with open(args.train_script, encoding="utf-8") as f:
            src = f.read()
        for line in re.findall(r"^\s*(?:import|from)\s+\S+.*$", src, re.M):
            if "isaaclab.app" in line or "argparse" in line:
                continue
            try:
                exec(line.strip(), {})
            except Exception:
                pass
            if registered():
                print(f"[t_eval] 태스크 등록 성공: `{line.strip()}`")
                return

    if not registered():
        raise SystemExit(
            f"[t_eval] 태스크 '{args.task}' 를 찾지 못했습니다.\n"
            f"        train.py 상단의 태스크 등록 import 를 --task_module 로 넘겨주세요.")


def make_terrain_generator(kind, args, base_tg=None):
    """평가용 지형. 학습 cfg 의 generator 를 **제자리에서** 고친다.

    새 cfg 객체를 만들거나 deepcopy 하면 `cfg.validate()` 가 무한 재귀한다
    (RecursionError at isaaclab/utils/configclass.py:_validate). 서버 실측 2026-09-04.
    """
    import isaaclab.terrains as tr

    if kind == "flat" and args.flat_mode == "plane":
        return None
    if base_tg is None:
        raise RuntimeError("base_tg 가 필요하다 (학습 cfg 의 terrain_generator)")
    tg = base_tg

    if kind == "flat":
        tg.sub_terrains = {"flat": tr.MeshPlaneTerrainCfg(proportion=1.0)}
    elif kind == "rough":
        for k, v in tg.sub_terrains.items():
            v.proportion = 1.0 if k == "random_rough" else 0.0
        tg.sub_terrains["random_rough"].noise_range = (args.rough_min, args.rough_max)
    elif kind == "slope":
        sl = math.radians(args.slope_deg)
        keep = ("hf_pyramid_slope", "hf_pyramid_slope_inv")
        for k, v in tg.sub_terrains.items():
            v.proportion = 0.5 if k in keep else 0.0
        tg.sub_terrains["hf_pyramid_slope"].slope_range = (sl, sl)
        tg.sub_terrains["hf_pyramid_slope_inv"].slope_range = (sl, sl)
    elif kind == "stairs":
        # Go2 G5 — 오르막/내리막 계단 50:50. 서브지형 이름이 워크스페이스마다 다를 수 있어
        # 'stairs' 를 포함하는 키를 전부 찾는다. G-0 정찰에서 실제 이름을 확인할 것.
        keep = [k for k in tg.sub_terrains if "stairs" in k or "stair" in k]
        if not keep:
            raise RuntimeError(
                "계단 서브지형을 못 찾았다. 있는 키: %s" % sorted(tg.sub_terrains))
        for k, v in tg.sub_terrains.items():
            v.proportion = (1.0 / len(keep)) if k in keep else 0.0
        h, w = args.stair_height, args.stair_width
        for k in keep:
            v = tg.sub_terrains[k]
            if hasattr(v, "step_height_range"):
                v.step_height_range = (h, h)
            if hasattr(v, "step_width"):
                v.step_width = w
    elif kind == "dr":
        # Go2 G7 — 험지 + 도메인 랜덤화. 지형은 rough 와 같게 두고, 마찰·질량 랜덤화는
        # make_env 가 이벤트 항으로 켠다 (지형 생성기에서 할 수 있는 게 아니다).
        for k, v in tg.sub_terrains.items():
            v.proportion = 1.0 if k == "random_rough" else 0.0
        tg.sub_terrains["random_rough"].noise_range = (args.rough_min, args.rough_max)
    else:
        raise ValueError(kind)

    tg.curriculum = False
    tg.difficulty_range = (1.0, 1.0)
    tg.num_rows = 4
    tg.num_cols = 4
    tg.seed = int(getattr(args, "terrain_seed", 1234))
    tg.use_cache = False
    return tg


def make_env(args, kind):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    # 에피소드 길이는 요청된 시나리오 중 가장 긴 것(구간 수 × 구간 길이) 이상이어야 한다.
    # 짧으면 time_out 리셋이 시나리오 중간에 끼어든다. run() 이 _horizon_s 를 채운다.
    horizon_s = float(getattr(args, "_horizon_s", args.seg_s * 4))
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    env_cfg.seed = int(getattr(args, "startup_seed", 42))
    env_cfg.episode_length_s = horizon_s
    env_cfg.scene.num_envs = args.num_envs

    # 지형
    tg = make_terrain_generator(kind, args, env_cfg.scene.terrain.terrain_generator)
    env_cfg.scene.terrain.terrain_type = "generator" if tg is not None else "plane"
    env_cfg.scene.terrain.terrain_generator = tg
    if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
        env_cfg.scene.terrain.max_init_terrain_level = None
    if tg is None and hasattr(env_cfg.scene.terrain, "use_terrain_origins"):
        env_cfg.scene.terrain.use_terrain_origins = False

    # 지형 커리큘럼은 평가에서 무의미하고, 난이도를 흔들어 CRN을 깬다.
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    # 밀침은 H7에서 우리가 직접 스케줄대로 넣는다 (재현성).
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # Go2 G7 — 도메인 랜덤화는 이 지형 그룹에서만 켠다. 다른 그룹은 손대지 않아야
    # 기존 H1 결과와 비교가 유지된다.
    if kind == "dr":
        def _rng(text):
            lo, hi = (float(x) for x in str(text).split(","))
            return (lo, hi)
        fr = _rng(args.dr_friction)
        ms = _rng(args.dr_mass)
        ev = env_cfg.events
        applied = []
        for name in ("physics_material", "randomize_rigid_body_material"):
            t = getattr(ev, name, None)
            if t is not None and getattr(t, "params", None) is not None:
                for key in ("static_friction_range", "dynamic_friction_range"):
                    if key in t.params:
                        t.params[key] = fr
                        applied.append(f"{name}.{key}")
        for name in ("add_base_mass", "randomize_rigid_body_mass"):
            t = getattr(ev, name, None)
            if t is not None and getattr(t, "params", None) is not None:
                for key in ("mass_distribution_params", "mass_range"):
                    if key in t.params:
                        t.params[key] = ms
                        applied.append(f"{name}.{key}")
        print("[t_eval] DR 적용: " + (", ".join(applied) if applied else
                                      "없음 — 이벤트 항 이름을 G-0에서 확인하라"))

    if args.no_obs_noise:
        env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    return env


def pin_command_term(env):
    """명령 항을 고정한다.

    UniformVelocityCommand 는 heading_command=True 라 매 스텝 yaw 명령을 heading
    오차로 덮어쓰고, resampling_time_range 마다 새 명령을 뽑는다. 둘 다 막아야
    우리가 설계한 명령 시퀀스가 그대로 정책에 들어간다.
    """
    cm = env.unwrapped.command_manager
    term = cm.get_term("base_velocity") if hasattr(cm, "get_term") else cm._terms["base_velocity"]
    if hasattr(term.cfg, "heading_command"):
        term.cfg.heading_command = False
    term._resample_command = lambda env_ids: None
    term._update_command = lambda: None
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    return term


def set_command(term, cmd):
    term.vel_command_b[:, 0] = cmd[0]
    term.vel_command_b[:, 1] = cmd[1]
    term.vel_command_b[:, 2] = cmd[2]


def root_velocity(asset):
    for name in ("root_vel_w", "root_com_vel_w"):
        if hasattr(asset.data, name):
            return getattr(asset.data, name)
    raise AttributeError("root velocity 버퍼를 찾지 못했습니다")


def run_scenario(env, term, policy, sid, args, device):
    """한 시나리오를 EVAL_SET_A 전체 seed로 굴리고 집계 결과를 돌려준다."""
    import torch

    spec = SCENARIOS[sid]
    seeds = [int(s) for s in args.seeds.split(",")]
    # E5 — 명령 크기 스윕. 1.0 이면 아무것도 안 바뀐다(기존 결과 재현 보장).
    cmd_scale = float(getattr(args, "cmd_scale", 1.0) or 1.0)
    if cmd_scale != 1.0:
        spec = dict(spec)
        spec["cmds"] = [tuple(c * cmd_scale for c in cmd) for cmd in spec["cmds"]]
    n_seg = len(spec["cmds"])
    dt = env.unwrapped.step_dt
    seg_s = float(spec.get("seg_s", args.seg_s))            # X4 는 2초 구간
    push_mag = float(spec.get("push_mag", args.push_mag))     # X1/X5 는 시나리오 고유값
    push_period_s = float(spec.get("push_period_s", args.push_period_s))
    steps_per_seg = int(round(seg_s / dt))
    transient = int(round(args.transient_s / dt))
    asset = env.unwrapped.scene["robot"]

    axis_idx = {"vx": 0, "vy": 1, "yaw": 2}
    # 축별/구간별 제곱오차 합과 표본 수 (모든 seed 누적)
    sq = {a: torch.zeros(n_seg, dtype=torch.float64, device=device) for a in axis_idx}
    cnt = torch.zeros(n_seg, dtype=torch.float64, device=device)
    # 진단 축 (명령 0 가정): vz, roll rate, pitch rate — 공식 채점기가 이 축도 재는지 역산용 (2026-09-13)
    extra_idx = {"vz": 0, "wx": 1, "wy": 2}
    sq_x = {a: torch.zeros(n_seg, dtype=torch.float64, device=device) for a in extra_idx}
    ab = {a: torch.zeros(n_seg, dtype=torch.float64, device=device) for a in axis_idx}  # 절대오차 합
    # 구간별 실제 평균 속도 (vx, vy, yaw). RMSE 만으로는 '제자리'와 '2배 과속'이 구분 안 된다
    # — Go2 GB 가 전진 명령에서 정확히 제자리 RMSE 를 냈다(2026-09-13).
    vsum = torch.zeros(n_seg, 3, dtype=torch.float64, device=device)
    falls_seg = [0] * n_seg                                   # 구간별 낙상 (원인 추적용)
    survived_total, episodes_total = 0, 0
    nan_actions = 0

    push_steps = None
    if spec.get("push"):
        period = max(1, int(round(push_period_s / dt)))
        push_steps = list(range(period, steps_per_seg * n_seg, period))

    for seed in seeds:
        env.unwrapped.seed(seed)
        torch.manual_seed(seed)
        set_command(term, spec["cmds"][0])
        obs, _ = env.reset()
        obs = obs["policy"] if isinstance(obs, dict) else obs
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[:] = False

        # 밀침 속도는 미리 뽑아둔다 — 후보가 달라도 동일한 외란을 받게 (CRN)
        pushes = None
        if push_steps:
            g = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
            pushes = (torch.rand(len(push_steps), args.num_envs, 2, generator=g) * 2 - 1) \
                * push_mag
            pushes = pushes.to(device)

        alive = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        step = 0
        for seg in range(n_seg):
            set_command(term, spec["cmds"][seg])
            cmd_t = torch.tensor(spec["cmds"][seg], dtype=torch.float32, device=device)
            for k in range(steps_per_seg):
                with torch.inference_mode():
                    act = policy(obs)
                if torch.isnan(act).any():
                    nan_actions += 1
                    act = torch.nan_to_num(act)
                obs, _, terminated, truncated, _ = env.step(act)
                obs = obs["policy"] if isinstance(obs, dict) else obs

                # 명령 항이 리셋된 env에서 되살아나지 않도록 매 스텝 다시 박아둔다
                set_command(term, spec["cmds"][seg])

                newly = alive & terminated.bool()
                falls_seg[seg] += int(newly.sum().item())
                alive &= ~terminated.bool()

                if k >= transient:
                    lin = asset.data.root_lin_vel_b
                    ang = asset.data.root_ang_vel_b
                    actual = torch.stack([lin[:, 0], lin[:, 1], ang[:, 2]], dim=-1)
                    err = (actual - cmd_t) ** 2
                    m = alive.double()
                    cnt[seg] += m.sum()
                    vsum[seg] += (actual.double() * m[:, None]).sum(dim=0)
                    for a, i in axis_idx.items():
                        sq[a][seg] += (err[:, i].double() * m).sum()
                        ab[a][seg] += ((actual[:, i] - cmd_t[i]).abs().double() * m).sum()
                    extra = torch.stack([lin[:, 2], ang[:, 0], ang[:, 1]], dim=-1)
                    for a, i in extra_idx.items():
                        sq_x[a][seg] += ((extra[:, i] ** 2).double() * m).sum()

                if push_steps and step in push_steps:
                    pi = push_steps.index(step)
                    vel = root_velocity(asset).clone()
                    vel[:, 0:2] += pushes[pi]
                    asset.write_root_velocity_to_sim(vel)
                step += 1

        survived_total += int(alive.sum().item())
        episodes_total += args.num_envs

    # ---- 집계 ----
    c = torch.clamp(cnt, min=1.0)
    mse_seg = {a: (sq[a] / c).cpu().numpy() for a in axis_idx}          # 축별 x 구간별
    rmse_axis = {a: float(math.sqrt(float((sq[a].sum() / c.sum()).item()))) for a in axis_idx}

    mse_all = {a: float((sq[a].sum() / c.sum()).item()) for a in axis_idx}
    used = spec["axes"]
    tracking = tracking_from_mse(mse_all, spec, args.axis_mode)
    rmse = -math.log(max(tracking, 1e-12))

    survival = survived_total / max(1, episodes_total)
    lcb = wilson_lcb(survived_total, episodes_total)

    # 구간별 tracking (H2 후진 구간 분리 집계용, spec 3절)
    seg_tracking = []
    for si in range(n_seg):
        seg_tracking.append(round(tracking_from_mse(
            {a: float(mse_seg[a][si]) for a in axis_idx}, spec, args.axis_mode), 4))

    out = {
        "w": spec["w"],
        "desc": spec["desc"],
        "terrain": spec["terrain"],
        "episodes": episodes_total,
        "survived": survived_total,
        "survival": round(survival, 4),
        "survival_lcb": round(lcb, 4),
        "tracking": round(tracking, 4),
        "score": round(survival * tracking, 4),
        "rmse": {a: round(rmse_axis[a], 4) for a in axis_idx},
        "rmse_used": round(rmse, 4),
        "rmse_extra": {a: round(math.sqrt(float((sq_x[a].sum() / c.sum()).item())), 4) for a in extra_idx},
        "mae": {a: round(float((ab[a].sum() / c.sum()).item()), 4) for a in axis_idx},
        "mse_seg": {a: [round(float(x), 5) for x in mse_seg[a].tolist()] for a in axis_idx},
        "axis_mode": args.axis_mode,
        "axis_weights": axis_weights(spec, args.axis_mode),
        "segment_cmds": [list(c_) for c_ in spec["cmds"]],
        "segment_tracking": seg_tracking,
        "segment_mean_actual": [[round(float(x), 3) for x in (vsum[si] / c[si]).tolist()]
                                for si in range(n_seg)],
        "segment_falls": falls_seg,
        "seg_s": seg_s,
        "push_mag": push_mag if push_steps else None,
        "push_period_s": push_period_s if push_steps else None,
        "falls": episodes_total - survived_total,
        "nan_actions": nan_actions,
        "crashes": 0,
    }
    if sid == "H2":
        # 후진(-0.5) 구간을 뺀 값도 남긴다. 공식 H2가 후진을 포함하는지 불명이므로.
        fwd = [i for i, c_ in enumerate(spec["cmds"]) if c_[0] >= 0]
        t_fwd = tracking_from_mse(
            {a: sum(float(mse_seg[a][i]) for i in fwd) / len(fwd) for a in axis_idx},
            spec, args.axis_mode)
        out["tracking_forward_only"] = round(t_fwd, 4)
        out["score_forward_only"] = round(survival * t_fwd, 4)
    return out


def preflight(args):
    """본 평가 전에 서버 API 가정을 전부 확인한다.

    여기서 통과하면 t_eval 본체가 돌 수 있다는 뜻이다. 확인 항목:
      1. 태스크 등록          2. 지형 cfg 클래스 존재     3. 지형 3종 실제 생성
      4. 명령 항 고정 가능    5. root 속도 버퍼 이름       6. 밀침 API
      7. H6 경사각 실측 (height_scanner ray hit로 z~x 최소자승)
    """
    import torch

    import_task_module(args)
    print("[preflight] 1. 태스크 등록 OK")

    import isaaclab.terrains as tr
    for name in ("MeshPlaneTerrainCfg", "HfRandomUniformTerrainCfg", "HfPyramidSlopedTerrainCfg"):
        assert hasattr(tr, name), f"isaaclab.terrains.{name} 없음"
    print("[preflight] 2. 지형 cfg 클래스 OK")

    args.num_envs = min(args.num_envs, 16)
    ok = True
    for kind in ("flat", "rough", "slope"):
        try:
            env = make_env(args, kind)
            term = pin_command_term(env)
            set_command(term, (0.5, 0.0, 0.0))
            obs, _ = env.reset()
            obs = obs["policy"] if isinstance(obs, dict) else obs
            asset = env.unwrapped.scene["robot"]
            print(f"[preflight] 3.{kind}: env OK  obs={tuple(obs.shape)}  "
                  f"step_dt={env.unwrapped.step_dt}  cmd={term.vel_command_b[0].tolist()}")

            v = root_velocity(asset)
            print(f"[preflight] 5.{kind}: root 속도 버퍼 OK shape={tuple(v.shape)}")
            asset.write_root_velocity_to_sim(v.clone())
            print(f"[preflight] 6.{kind}: write_root_velocity_to_sim OK (밀침 가능)")

            hs = env.unwrapped.scene.sensors.get("height_scanner") \
                if hasattr(env.unwrapped.scene, "sensors") else None
            if hs is not None:
                hits = hs.data.ray_hits_w                      # (N, R, 3)
                x = hits[..., 0] - hits[..., 0].mean(dim=1, keepdim=True)
                z = hits[..., 2] - hits[..., 2].mean(dim=1, keepdim=True)
                slope = (x * z).sum(dim=1) / torch.clamp((x * x).sum(dim=1), min=1e-8)
                deg = torch.rad2deg(torch.atan(slope)).abs()
                print(f"[preflight] 7.{kind}: 실측 경사 |median|={deg.median().item():.2f}deg "
                      f"max={deg.max().item():.2f}deg  "
                      f"고저차 p50={(hits[...,2].max(1).values - hits[...,2].min(1).values).median().item():.3f}m")
            else:
                print(f"[preflight] 7.{kind}: height_scanner 없음 (경사 실측 생략)")
            env.close()
        except Exception as e:
            ok = False
            import traceback; traceback.print_exc()
            print(f"[preflight] !! {kind} 실패: {type(e).__name__}: {e}")

    print("\n[preflight] " + ("전부 통과 — 본 평가를 돌려도 된다." if ok
                              else "실패 항목 있음 — 위 메시지를 보고 고쳐야 한다."))
    print("[preflight] 7번 slope 실측이 --slope_deg(%.1f)와 크게 다르면 "
          "slope_range 단위 가정이 틀린 것이니 --slope_deg 로 보정하라." % args.slope_deg)


def run(args):
    import torch

    if args.preflight:
        preflight(args)
        return

    import_task_module(args)

    device = args.device if getattr(args, "device", None) else "cuda:0"
    if args.scenarios.lower() == "all":
        want = list(SCENARIO_ORDER)
    elif args.scenarios.lower() == "stress":
        want = list(STRESS_ORDER)
    else:
        want = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    for s in want:
        if s not in SCENARIOS:
            raise SystemExit(f"[t_eval] 알 수 없는 시나리오: {s}")
    # 가장 긴 시나리오에 맞춰 에피소드 길이를 정한다 (make_env 가 읽는다)
    args._horizon_s = max(len(SCENARIOS[s]["cmds"]) * float(SCENARIOS[s].get("seg_s", args.seg_s))
                          for s in want)

    if getattr(args, "checkpoint", None):
        policy = load_checkpoint_actor(args.checkpoint, device)
        args.policy = args.checkpoint
        print(f"[t_eval] 체크포인트 actor 로드: {args.checkpoint}")
    else:
        if not os.path.exists(args.policy):
            raise SystemExit(
                f"[t_eval] 정책 파일이 없습니다: {args.policy}\n"
                f"        play.py 를 먼저 돌려 exported/policy.pt 를 만드세요.")
        policy = torch.jit.load(args.policy, map_location=device)
        policy.eval()
        print(f"[t_eval] 정책 로드: {args.policy}")

    # 지형이 같은 시나리오는 환경을 재사용한다 (환경 생성이 가장 비싸다).
    groups = {}
    for s in want:
        groups.setdefault(SCENARIOS[s]["terrain"], []).append(s)

    if len(groups) > 1:
        msg = [
            "[t_eval] 한 프로세스에서는 지형 그룹을 하나만 돌릴 수 있다.",
            "        두 번째 환경을 만들면 cfg.validate() 가 무한 재귀한다 (서버 실측).",
            "        요청한 그룹: " + ", ".join(sorted(groups)),
            "        나눠 돌린 뒤 --merge 로 합쳐라:",
            "          t_eval.py ... --scenarios H1,H2,H3,H4,H7 --out r_flat.json",
            "          t_eval.py ... --scenarios H5 --out r_rough.json",
            "          t_eval.py ... --scenarios H6 --out r_slope.json",
            "          t_eval.py --merge r_flat.json r_rough.json r_slope.json --out r.json",
        ]
        raise SystemExit(chr(10).join(msg))

    scenarios = {}
    for kind in TERRAIN_ORDER:
        ids = groups.get(kind)
        if not ids:
            continue
        print(f"\n[t_eval] === terrain='{kind}' 환경 생성: {', '.join(ids)} ===")
        env = make_env(args, kind)
        term = pin_command_term(env)
        try:
            for sid in ids:
                print(f"[t_eval] --- {sid} ({SCENARIOS[sid]['desc']}) ---")
                r = run_scenario(env, term, policy, sid, args, device)
                scenarios[sid] = r
                print(f"[t_eval] {sid}: survival={r['survival']:.3f} "
                      f"(LCB {r['survival_lcb']:.3f})  tracking={r['tracking']:.3f}  "
                      f"score={r['score']:.3f}")
        finally:
            env.close()

    config = {
        "seg_s": args.seg_s, "transient_s": args.transient_s,
        "num_envs": args.num_envs, "seeds": args.seeds,
        "rough_range": [args.rough_min, args.rough_max],
        "slope_deg": args.slope_deg,
        "push_period_s": args.push_period_s, "push_mag": args.push_mag,
        "obs_noise": not args.no_obs_noise,
        "tracking_def": "exp(-RMSE)",
        "axis_mode": args.axis_mode,
        "cmd_scale": float(getattr(args, "cmd_scale", 1.0) or 1.0),
        "stair_height": getattr(args, "stair_height", None),
        "stair_width": getattr(args, "stair_width", None),
    }
    result = finalize(scenarios, args.run_id, args.policy, config)
    write_result(result, args.out)
    print_offset_table(result, getattr(args, "official_ref", "R2"))


if __name__ == "__main__":
    main()
