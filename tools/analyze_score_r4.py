"""Offline inverse-score analysis; never changes the evaluator or calibration ledger.

Run: python tools/analyze_score_r4.py
Official values were read from the authenticated NCRC board on 2026-10-01.
The fitted models translate local diagnostics into official scores; they do not
identify the private evaluator's implementation or its survival component.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/score_reverse_20261001"
ANCHORS = {
    "H1_R2": {
        "path": "R3_bundle_extracted/eval_results/R3_R3A_9000.json",
        "prefix": "H", "points": [8.84, 11.68, 5.86, 5.84, 8.33, 8.70, 8.75],
        "weights": [.15, .20, .10, .10, .15, .15, .15],
        "sim": 58.00, "qualitative": 30,
    },
    "Go2_R4": {
        "path": "longrun_C0/C0_5500_ALL.json",
        "prefix": "G", "points": [8.80, 8.66, 11.69, 8.79, 8.56, 5.79, 5.85],
        "weights": [.15, .15, .20, .15, .15, .10, .10],
        "sim": 58.14, "qualitative": 30,
    },
}


def rows_for(anchor):
    raw = (ROOT / anchor["path"]).read_bytes()
    data = json.loads(raw)
    assert math.isclose(sum(anchor["weights"]), 1)
    assert math.isclose(sum(anchor["points"]), anchor["sim"])
    rows = []
    for i, (point, weight) in enumerate(zip(anchor["points"], anchor["weights"]), 1):
        sid = anchor["prefix"] + str(i)
        s = data["scenarios"][sid]
        assert math.isclose(s["w"], weight)
        mse3 = sum(float(s["rmse"][axis])**2 for axis in ("vx", "vy", "yaw")) / 3
        # The current high-performing anchors have no observed local falls.
        # This fact provides no estimate of the official survival probability.
        assert s["survival"] == 1.0
        extra = s.get("rmse_extra")
        mse6 = None if extra is None else (
            3*mse3 + sum(float(extra[axis])**2 for axis in ("vz", "wx", "wy"))
        ) / 6
        rows.append(dict(sid=sid, w=weight, official_point=point,
                         q=point/(70*weight), survival=s["survival"], mse3=mse3,
                         mse6=mse6, q3=math.exp(-math.sqrt(mse3)),
                         q6=None if mse6 is None else math.exp(-math.sqrt(mse6)),
                         inferred_ratio=point/(70*weight)/math.exp(-math.sqrt(mse3)),
                         effective_c=(-math.log(point/(70*weight)))**2-mse3))
    return rows, hashlib.sha256(raw).hexdigest()


def metrics(rows, pred):
    norm = sum(r["w"] for r in rows)
    score = 70*sum(r["w"]*p for r, p in zip(rows, pred))
    actual = 70*sum(r["w"]*r["q"] for r in rows)
    return dict(sim=score, sim_error=score-actual,
                q_rmse=math.sqrt(sum(r["w"]*(p-r["q"])**2 for r, p in zip(rows, pred))/norm),
                maximum_scenario_point_error=max(abs(70*r["w"]*(p-r["q"]))
                                                 for r, p in zip(rows, pred)), prediction_q=pred)


def fit_scale(rows, axis=3):
    key = "q" + str(axis)
    return sum(r["w"]*r[key]*r["q"] for r in rows)/sum(r["w"]*r[key]**2 for r in rows)


def fit_floor(rows):
    def loss(c):
        return sum(r["w"]*(math.exp(-math.sqrt(r["mse3"]+c))-r["q"])**2 for r in rows)
    left, right = 0., .25
    for _ in range(120):
        a = left + (right-left)/3
        b = right - (right-left)/3
        if loss(a) <= loss(b):
            right = b
        else:
            left = a
    return (left+right)/2


def fit_affine(rows, axis):
    key = "q" + str(axis)
    norm = sum(r["w"] for r in rows)
    x = sum(r["w"]*r[key] for r in rows)/norm
    y = sum(r["w"]*r["q"] for r in rows)/norm
    slope = sum(r["w"]*(r[key]-x)*(r["q"]-y) for r in rows)/sum(r["w"]*(r[key]-x)**2 for r in rows)
    return slope, y-slope*x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"date": "2026-10-01", "official_source": "https://robotics.connect.or.kr/board",
              "rules_source": "https://robotics.connect.or.kr/rules/preliminary_regulations.html",
              "status": "EMPIRICAL_PROXY_NOT_IDENTIFIED_OFFICIAL_FORMULA", "anchors": {},
              "transfer": {}, "verification": {}}
    all_rows = {}
    for label, anchor in ANCHORS.items():
        rows, sha = rows_for(anchor)
        all_rows[label] = rows
        models = {
            "raw_3axis": metrics(rows, [r["q3"] for r in rows]),
            "prior_3axis_times_0.905": metrics(rows, [.905*r["q3"] for r in rows]),
            "prior_3axis_affine_1.0714_minus_0.1411": metrics(rows, [1.0714*r["q3"]-.1411 for r in rows]),
        }
        rho = fit_scale(rows)
        c = fit_floor(rows)
        models["fitted_3axis_scale"] = dict(rho=rho, **metrics(rows, [rho*r["q3"] for r in rows]))
        models["fitted_3axis_mse_floor"] = dict(c=c, sigma=math.sqrt(c),
            **metrics(rows, [math.exp(-math.sqrt(r["mse3"]+c)) for r in rows]))
        if all(r["q6"] is not None for r in rows):
            models["prior_raw_6axis"] = metrics(rows, [r["q6"] for r in rows])
            rho6 = fit_scale(rows, 6)
            models["fitted_6axis_scale"] = dict(rho=rho6,
                **metrics(rows, [rho6*r["q6"] for r in rows]))
            for axis in (3, 6):
                a, b = fit_affine(rows, axis)
                models[f"fitted_{axis}axis_affine"] = dict(slope=a, intercept=b,
                    **metrics(rows, [a*r[f"q{axis}"]+b for r in rows]))
        result["anchors"][label] = dict(input=anchor["path"], sha256=sha,
                                        official_sim=anchor["sim"], rows=rows, models=models)
    for train, test in (("H1_R2", "Go2_R4"), ("Go2_R4", "H1_R2")):
        source, target = all_rows[train], all_rows[test]
        c, rho = fit_floor(source), fit_scale(source)
        result["transfer"][f"{train}_to_{test}"] = {
            "scale": dict(rho=rho, **metrics(target, [rho*r["q3"] for r in target])),
            "mse_floor": dict(c=c, **metrics(target, [math.exp(-math.sqrt(r["mse3"]+c)) for r in target])),
        }
    c_joint = fit_floor(all_rows["H1_R2"]+all_rows["Go2_R4"])
    result["joint_floor"] = dict(c=c_joint, sigma=math.sqrt(c_joint))
    for name in ("official_r4", "official_h1"):
        p = OUT / name / "verification.json"
        if p.exists():
            result["verification"][name] = json.loads(p.read_text(encoding="utf-8"))
            assert all(v.get("matches") is True for v in result["verification"][name].values())
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    g = result["anchors"]["Go2_R4"]
    lines = ["# 채점식 역산 — 2026-10-01", "",
        "결론: 종합 구조·배점은 공식 규정으로 확정된다. 추종 함수의 구현·축·생존 항은 식별되지 않았다.",
        "현재 자체→공식 점수 예측에는 3축 평균제곱오차에 공통 보정 c≈0.02685를 더하는 근사가 가장 잘 전달된다.",
        "기존에 제시했던 무보정 6축 모형은 새 Go2 공식 결과를 과소예측한다. 다만 보정한 6축 모형까지 기각할 근거는 없다.", "",
        "## 확인된 자료", "",
        "- 공식 규정 v1.1 제6~9조: 총점 = 70×Σ(w×생존율×추종점수) + 설계 20 + 리포트 10.",
        "- 공식 Go2 가중치 = [0.15,0.15,0.20,0.15,0.15,0.10,0.10]. 기존 자체 가중치와 일치.",
        "- R2 H1: 58.00+30=88.00. R4 Go2: 58.14+30=88.14.",
        "- 보드에서 내려받은 두 제출본의 policy.pt와 env.yaml은 해당 로컬 파일과 SHA-256 및 전체 바이트가 일치한다.",
        "- 공식 생존율/축별 오차/명령 시퀀스/실행 시드는 공개되지 않았다.",
        "- 로컬 두 제출본은 각 시나리오 생존율 1.0. 이것을 공식 생존율 1.0으로 옮길 수는 없다.", "",
        "## Go2: 기존에 정해 둔 가설을 새 공식 결과로 비교", "",
        "| 가설 | 예측 시뮬 | 실제 대비 | 시나리오 가중 q RMSE |", "|---|---:|---:|---:|"]
    labels = [("raw_3axis", "무보정 3축"), ("prior_3axis_times_0.905", "3축 × 0.905"),
              ("prior_3axis_affine_1.0714_minus_0.1411", "3축 × 1.0714 − 0.1411"),
              ("prior_raw_6axis", "무보정 6축")]
    for key, title in labels:
        m = g["models"][key]
        lines.append(f"| {title} | {m['sim']:.5f} | {m['sim_error']:+.5f} | {m['q_rmse']:.6f} |")
    lines += ["", "기존 가설 비교는 0.905 보정을 지지한다. 무보정 6축과의 차이는 총점뿐 아니라 시나리오별로도 나타난다.",
        "", "## 새로 적합한 3축 MSE 보정", "",
        "E3²=(RMSE_vx²+RMSE_vy²+RMSE_yaw²)/3", "",
        "q_hat = S_local × exp(−sqrt(E3²+c))", "",
        "P_hat = D + R + 70×Σ(w×q_hat)", "",
        "c는 공식 평가기 상수로 확인된 값이 아니라, 로컬 진단량과 공식 결과 사이의 차이를 흡수한 유효 보정량이다.",
        "수치 단위는 자체 평가기의 혼합 축 관례를 따른다. sqrt(c)를 물리적인 센서 노이즈 단위로 해석하지 않는다.", "",
        "| 학습 자료 | c | 다른 로봇 예측 | 실제 | 시나리오 q RMSE |", "|---|---:|---:|---:|---:|"]
    for train, test in (("H1_R2", "Go2_R4"), ("Go2_R4", "H1_R2")):
        m = result["transfer"][f"{train}_to_{test}"]["mse_floor"]
        lines.append(f"| {train} → {test} | {m['c']:.9f} | {m['sim']:.6f} | {ANCHORS[test]['sim']:.2f} | {m['q_rmse']:.6f} |")
    lines += ["", f"두 자료 공동 적합: c={c_joint:.9f}, sqrt(c)={math.sqrt(c_joint):.6f}.", "",
        "모형 가족은 Go2 결과를 본 뒤 추가했다. 로봇별 분리 적합/전달은 탐색적 검증이며, 이후 새 제출본에 대한 사전 예측 검증이 필요하다.",
        "시나리오 14개가 독립적인 정책 14개인 것은 아니다. 실제 고성능 제출 정책은 2개뿐이다.", "",
        "공식 점수는 소수 둘째 자리, 로컬 축별 RMSE는 넷째 자리로 반올림된 자료다. c의 여러 소수 자리는 계산 재현용이며 추정 정밀도를 뜻하지 않는다.",
        "H1→Go2의 총점 오차는 약0.002점이지만 시나리오 하나의 최대 오차는 약0.129점이다. 총점의 일치에는 시나리오별 잔차 상쇄도 포함된다.", "",
        "## Go2 시나리오별 대조", "",
        "| ID | 공식 점수 | 0.905 보정 | 무보정 6축 | H1에서 적합한 MSE 보정 | 공식/자체3축 | 유효 c_i |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    c_h = fit_floor(all_rows["H1_R2"])
    for r in all_rows["Go2_R4"]:
        scale = 70*r["w"]
        lines.append(f"| {r['sid']} | {r['official_point']:.2f} | {scale*.905*r['q3']:.3f} | {scale*r['q6']:.3f} | "
                     f"{scale*math.exp(-math.sqrt(r['mse3']+c_h)):.3f} | {r['inferred_ratio']:.5f} | {r['effective_c']:.6f} |")
    lines += ["", "## 식별 한계와 다음 실험에 대한 의미", "",
        "- 공식은 q=S_official×T_official만 공개한다. 각 시나리오에서 하나의 q로 S와 T를 따로 풀 수 없다.",
        "- 예를 들어 T_official=T_local이면 S_official=q/T_local(Go2 0.896~0.927)이 모든 점수를 그대로 설명한다. 실제 낙상률이라는 뜻은 아니다.",
        "- 다른 설명은 공식 생존율 1.0, 공식 추종점수=q인 경우다. 추가 오차 바닥은 이 경우를 요약한 보정량이다.",
        "- 6축에 새 배율을 적합하면 Go2 q RMSE가 약 0.00774로 낮아져 3축 배율의 약 0.00735와 가까워진다. 보정된 6축의 가능성은 남는다.",
        "- 0.905를 공식 채점기에 박힌 감점 또는 공식 만점 93.35의 근거로 해석하면 안 된다. 오차 0→추종 1.0은 공식 규정이다.",
        "- 마찬가지로 MSE 보정 모델의 로컬 오차 0 상한(약89.4점)은 이 경험식의 외삽값이며, 실제 대회 상한이 아니다.",
        "- 흔들림이 공식 점수의 주축이라고 확정하거나, 반대로 아예 채점되지 않는다고 확정하지 않는다.",
        "- D1은 동일 학습량 비교에서 추종·생존까지 악화되어 변수 통제용 공식 검증본으로 부적절하다.",
        "- 다음 제출 전에 q_hat 모델의 시나리오별 예측을 동결한 뒤 결과를 비교한다. 흔들림만 바뀌고 추종·생존이 유사한 정책이 추가되면 축 가설의 판별력이 높아진다.",
        "- 낮은 성능의 1R 앵커를 E1 파일과 직접 연결하지 않는다. 기존 코드가 지적하듯 E1과 실제 1R 제출본은 동일 정책이 아니다.",
        "- 기존 t_eval.py, t_eval_go2.py, calibration.json 및 학습/제출 파일은 변경하지 않았다.", "",
        "## 재현", "", "`python tools/analyze_score_r4.py`", "",
        "원자료·SHA-256·전체 계산값: analysis/score_reverse_20261001/analysis.json.", "",
        "출처: https://robotics.connect.or.kr/board 및 https://robotics.connect.or.kr/rules/preliminary_regulations.html", ""]
    (ROOT / "score_reverse_20261001.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"prior_models_go2": {k: {t:v for t,v in m.items() if t!='prediction_q'}
                       for k,m in g["models"].items()}, "transfer": result["transfer"],
                       "joint_floor": result["joint_floor"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
