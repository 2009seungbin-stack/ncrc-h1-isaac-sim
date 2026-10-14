"""Analyze saved JSON only. No policy loading, simulator, server, or submission."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SUMMARY = ROOT / "screen_summary.json"
ANCHOR = WORKSPACE / "analysis/score_reverse_20261001/analysis.json"
EVIDENCE = ROOT / "portal_evidence.json"
AUDIT = ROOT / "audit.json"
CALIBRATION = WORKSPACE / "analysis/calibration_review_20261008/calibration_audit.json"

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

models = read(SUMMARY)["models"]
official = read(ANCHOR)["anchors"]["Go2_R4"]
points = {r["sid"]: r["official_point"] for r in official["rows"]}
weights = {r["sid"]: r["w"] for r in official["rows"]}
sim = official["official_sim"]
assert math.isclose(sum(points.values()), sim)
assert math.isclose(sum(weights.values()), 1)
assert set(models) == {"C0_5500", "F10_3000", "S1_tracking_pair", "R1_tracking_double"}
assert read(EVIDENCE)["board"]["Go2_official_total"] == sim + 30

def quality(model, error_fraction=1.0, mse_floor=0.0, error_gain=1.0, no_falls=False):
    result = 0.0
    for sid, row in model["scenarios"].items():
        s = 1.0 if no_falls else row["survival"]
        mse = sum((row["rmse"][a] * error_fraction * error_gain) ** 2
                  for a in ("vx", "vy", "yaw")) / 3
        result += weights[sid] * s * math.exp(-math.sqrt(mse + mse_floor))
    return result

for m in models.values():
    assert set(m["scenarios"]) == set(weights)
    assert math.isclose(quality(m), m["weighted_quality"], abs_tol=1e-12)
    assert sum(r["episodes"] - r["survived"] for r in m["scenarios"].values()) == m["falls"]

base = models["C0_5500"]
f10 = models["F10_3000"]
q0 = quality(base)
params = read(CALIBRATION)["parameters"]
c = params["legacy_mse_floor"]
gain = params["illustrative_error_gain"]
assert math.isclose(70 * quality(base, mse_floor=c), sim, abs_tol=1e-10)
assert math.isclose(70 * quality(base, error_gain=gain), sim, abs_tol=1e-10)

def outputs(model, fraction=1.0):
    return {
        "legacy_fixed_mse_floor": 30 + 70 * quality(model, fraction, mse_floor=c),
        "fixed_quality_ratio": 30 + sim * quality(model, fraction) / q0,
        "illustrative_error_gain": 30 + 70 * quality(model, fraction, error_gain=gain),
    }

comparisons = {}
for name, m in models.items():
    hypothetical = outputs(m)
    loss_vs_f10 = quality(f10) - quality(m)
    removed_falls = quality(m, no_falls=True) - quality(m)
    comparisons[name] = {
        "measured_internal_quality": quality(m),
        "falls_of_1792": m["falls"],
        "conditional_formula_outputs_NOT_official_predictions": hypothetical,
        "conditional_delta_from_official_C0": {k: v - (sim + 30) for k, v in hypothetical.items()},
        "counterfactual_no_falls_quality": quality(m, no_falls=True),
        "fall_component_fraction_of_gap_vs_F10": removed_falls / loss_vs_f10 if loss_vs_f10 > 0 else None,
        "weighted_three_axis_rmse": math.sqrt(sum(
            weights[s] * sum(r["rmse"][a] ** 2 for a in ("vx", "vy", "yaw")) / 3
            for s, r in m["scenarios"].items())),
    }

counterfactuals = [{"uniform_RMSE_reduction": r, "outputs": outputs(f10, 1-r)}
                   for r in (0, .10, .20, .25, .30, .40, .50)]
required = {}
for target in (90, 92, 93):
    required[str(target)] = {}
    for method in ("fixed_quality_ratio", "illustrative_error_gain"):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if outputs(f10, mid)[method] > target:
                lo = mid
            else:
                hi = mid
        required[str(target)][method] = 1 - (lo + hi) / 2

out = {
    "status": "OFFLINE_PROSPECT_ASSESSMENT_CONDITIONAL_SENSITIVITY_NOT_VALIDATED_FORECAST",
    "official_anchor_count": 1,
    "qualitative_30_assumed_for_new_policies": True,
    "primary_official_prediction": None,
    "probability_of_93": None,
    "long_training_score_forecast": None,
    "server_access_policy_loading_or_new_evaluation": False,
    "models": comparisons,
    "F10_uniform_error_reduction_counterfactuals_NOT_measured": counterfactuals,
    "target_required_reductions_under_unvalidated_assumptions": required,
    "official_target_arithmetic": {
        "anchor_total": sim + 30, "target_total": 93, "target_sim": 63,
        "additional_points_required": 63 - sim,
        "official_sim_relative_increase_required": 63 / sim - 1,
        "remaining_official_sim_loss_recovery_fraction": (63 - sim) / (70 - sim),
    },
    "limitations": [
        "One official policy anchor cannot identify a prediction function; C0 fit is by construction.",
        "The three outputs are alternative assumptions, not a prediction interval or calibrated bounds.",
        "Fixed MSE-floor artificial ceiling is not the competition ceiling and is unsuitable for target feasibility.",
        "The error-gain function is arbitrary and is not preferred because its numbers are higher.",
        "Uniform error reductions keep survival fixed and do not describe a trained or achievable policy.",
        "Only one training seed; unequal training lengths and historical runtimes; no matched1500 control.",
        "S1 evaluated1499 differs from recovered400; recovered400 has no performance measurement.",
        "Full current archive/raw traces/events absent; summary was exactly copied but not raw-recomputed locally.",
        "Existing F10 selection gains are already in the measured F10 baseline and cannot be counted again.",
        "Older F10 push survival regression remains relevant; latest0/1792 does not erase all prior gates.",
    ],
    "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (SUMMARY, ANCHOR, EVIDENCE, AUDIT, CALIBRATION, ROOT/"final_analysis.txt")],
}
(ROOT / "prospect_assessment_20261009.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "Go2 오프라인 유망성 평가 — 2026-10-09 KST",
    "서버 접근·정책 실행·학습·재평가·공식 제출 없이 저장 JSON만 재계산했다.",
    "",
    "판정: F10은 현재 후보 중 유지할 근거가 가장 강하다. 현재 S1/R1의 장기학습 우선순위는 낮다.",
    "F10의 내부 개선은 작은 상승 가능성을 뒷받침하지만 공식93 또는 +4.86점 상승의 근거는 아니다.",
    "검증된 공식 예측값/93 성공확률/추가 학습시간별 점수 예측은 산출할 수 없다.",
    "",
    "조건부 환산(모두 C0 공식88.14 하나에 맞춘 미검증 가정; 예측구간이 아님)",
    "모델 | 고정MSE식 | 고정품질비율식 | 임의오차배율식 | 유망성 판단",
]
verdicts = {"C0_5500": "기존 공식 기준", "F10_3000": "내부 비교 최선; 소폭 상승 가설",
            "S1_tracking_pair": "현재1499는 낮음; 400은 미평가", "R1_tracking_double": "현재1499는 낮음; 역전 미입증"}
for name, row in comparisons.items():
    v = row["conditional_formula_outputs_NOT_official_predictions"]
    lines.append(f"{name} | {v['legacy_fixed_mse_floor']:.2f} | {v['fixed_quality_ratio']:.2f} | {v['illustrative_error_gain']:.2f} | {verdicts[name]}")
lines += [
    "F10 상승폭의 식별 가능한 실측은 내부 품질 상대+0.8411%다. +0.22/+0.49/+0.96점은 각 가정의 출력이다.",
    "약+0.5점이라는 표현을 쓴다면 품질비율 가정의 예시이며 검증된 대표 예측값이 아니다.",
    "고정MSE식은 완벽한 내부정책도89.72로 제한한다. 이를 비관적 하한이나93 불가능 근거로 쓰지 않는다.",
    "",
    "R1이 F10보다 낮은 내부 품질 차이 중 낙상 부분은 약12.22%다. 낙상을 없앤 가상R1도 품질0.905417로F10보다 낮다.",
    "S1의 해당 비중은 약4.23%다. 낙상만 해결하는 조치로 현재 추종 손실 대부분이 회복되지는 않는다.",
    "1500→3000/5000 학습만으로 역전할 것이라는 예측은 없다. 현재runtime1500 대조군과 학습곡선도 부족하다.",
    "",
    "93 목표의 확정 산술: 기존88.14→93은+4.86점; 정성30 유지 시 시뮬58.14→63이다.",
    "공식 시뮬 품질 상대+8.36%, 남은 공식 감점의40.98% 회수가 필요하다.",
    "",
    "F10 모든 시나리오·3축 RMSE를 균등 감소시키고 생존 유지한 반사실",
    "RMSE 감소 | 품질비율식 출력 | 임의오차배율식 출력",
]
for row in counterfactuals:
    if row["uniform_RMSE_reduction"] in (0, .10, .20, .25, .40):
        v = row["outputs"]
        lines.append(f"{row['uniform_RMSE_reduction']:.0%} | {v['fixed_quality_ratio']:.2f} | {v['illustrative_error_gain']:.2f}")
lines += [
    "93을 출력하는 균등RMSE 감소는 가정에 따라82.7% 또는37.8%다. 이 숫자 차이는 예측 불확실성을 보여주며 달성 가능 범위가 아니다.",
    "특정 미학습 보상 후보에 +2~5점을 붙이거나 성공확률을 제시할 근거는 없다.",
    "",
    "연구 우선순위 갱신",
    "1. F10의 측정된 성능을 기준으로 유지. 이미 선택한3000 저장본의 개선을 미래의 무료 상승으로 중복 계산하지 않는다.",
    "2. 전환·계단·밀침의 전진 오차를 줄이되 yaw와 생존을 유지하는 방법. 병목 위치는 근거가 있으나 해결 조작의 효과는 미입증이다.",
    "3. 토크/보행/자세 단독 변경은 아직 가설. S1/R1 실패 후 추종강도 확대 또는 비용 완화만으로 크게 개선한다는 기대는 낮춘다.",
    "본 평가로 새 실험이나 장기학습을 자동 실행하지 않는다. 서버OFF/자동화PAUSED 상태를 유지한다.",
    "",
    "한계: 독립training seed 반복 없음; 학습길이/runtime 다름; 현재 raw archive 미회수; 이전F10의 밀침 생존 퇴행 기록도 존재.",
    "최신0/1792만으로 모든 과거 생존 게이트를 통과했다고 주장하지 않는다. 공식 점수의 일반화는 별도 검증이 필요하다.",
    "수치와 입력 해시는 prospect_assessment_20261009.json, 재현은 postprocess/assess_offline_prospects.py.",
]
(ROOT / "prospect_assessment_20261009.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"models": comparisons, "official_target": out["official_target_arithmetic"],
                  "output": str(ROOT/"prospect_assessment_20261009.txt")}, ensure_ascii=False))
