"""Offline score headroom audit. Does not train, alter policies, or launch a server."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ROUND = ROOT / "analysis/round20_execution_20261001"
RUN = ROUND / "results/extracted/round20_20261001"
VALIDATION = ROUND / "validation_results/extracted/go2_validation_20261002"
WEIGHTS = dict(G1=.15, G2=.15, G3=.20, G4=.15, G5=.15, G6=.10, G7=.10)
AXES = ("vx", "vy", "yaw")
C = .02684614570005177
PATHS = {
    "C0_A": RUN / "eval/original_A/merged.json",
    "F10_A": RUN / "eval/F10_model_3000_A/merged.json",
    "C0_B": VALIDATION / "eval/original_B/merged.json",
    "F10_B": VALIDATION / "eval/selected_B/merged.json",
}
DATA = {k: json.loads(p.read_text(encoding="utf-8"))["scenarios"] for k, p in PATHS.items()}


def contribution(s, weight, family, scales=None):
    scales = scales or {}
    mse = sum((s["rmse"][a] * scales.get(a, 1)) ** 2 for a in AXES) / 3
    offset, multiplier = (C, 1) if family == "mse_offset" else (0, .905)
    return 70 * weight * s["survival"] * math.exp(-math.sqrt(mse + offset)) * multiplier


def score(data, family, scales=None):
    return sum(contribution(data[k], w, family, scales) for k, w in WEIGHTS.items())


result = {
    "date": "2026-10-02",
    "scope": "Offline analysis of existing measurements only",
    "limitations": [
        "Both score families are empirical proxies, not official evaluator implementations.",
        "Counterfactual RMSE reductions hold all other quantities fixed; not forecasts of reward changes.",
        "Reset seeds are not independent training seeds; terrain seed is fixed in these evaluations.",
        "A and B have both been inspected and must not serve as untouched final holdouts in new research.",
        "Last recorded official scores and usage were not refreshed from the account dashboard.",
    ],
    "official_rules_checked_in_browser": {
        "url": "https://robotics.connect.or.kr/rules/preliminary_regulations.html",
        "version": "v1.1",
        "article_2": "Change reward weights only; training on the provided cloud environment.",
        "article_10": "One robot per round; sum each robot's historical maximum score.",
    },
    "measurements": {},
}
for name, data in DATA.items():
    mse = {a: sum(w * data[k]["rmse"][a] ** 2 for k, w in WEIGHTS.items()) for a in AXES}
    result["measurements"][name] = {
        "proxy": {f: score(data, f) for f in ("mse_offset", "ratio")},
        "falls": sum(s["falls"] for s in data.values()),
        "axis_share_of_weighted_mse": {a: x / sum(mse.values()) for a, x in mse.items()},
    }

result["B_scenario_proxy_deltas"] = {
    k: {f: contribution(DATA["F10_B"][k], w, f) - contribution(DATA["C0_B"][k], w, f)
        for f in ("mse_offset", "ratio")}
    for k, w in WEIGHTS.items()
}
result["counterfactuals_from_F10_B"] = []
for axes in (("vx", "vy"), ("yaw",), AXES):
    for factor in (.75, .5, 0):
        result["counterfactuals_from_F10_B"].append({
            "axes": axes, "rmse_multiplier": factor,
            "proxy_delta": {f: score(DATA["F10_B"], f, dict.fromkeys(axes, factor)) - score(DATA["F10_B"], f)
                            for f in ("mse_offset", "ratio")},
        })

result["one_G6_fall_in_512"] = {
    "arithmetic_score_cost_upper_bound_if_tracking_fixed": 70 * .1 / 512,
    "warning": "Not a bound on true population risk; only the contribution of one observed failure in the sample.",
}
result["mse_offset_model_ceiling_if_all_errors_zero_and_survival_one"] = 30 + 70 * math.exp(-math.sqrt(C))
result["ceiling_warning"] = "This is a ceiling of the fitted formula, not a ceiling of the competition."
result["preserved_checkpoint_counts"] = {
    n: len(list((RUN / "preserved" / n).glob("*/model_*.pt"))) for n in ("F00", "F10")
}
result["proposed_unevaluated_F10_screen"] = [1200, 1800, 2200, 2600, 2800, 3200]
result["planning_hours"] = {
    "six_checkpoint_suites_at_8_5_minutes_each_excludes_overhead": 6 * 8.5 / 60,
    "one_3000_iteration_run_at_previous_F10_average_excludes_overhead": 5.170 * 3000 / 4000,
    "last_recorded_round_budget_unused_not_new_authorization": 7 + 18 / 60,
}
result["input_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS.values()}

assert abs(result["measurements"]["F10_B"]["proxy"]["mse_offset"] - 58.405982757841855) < 1e-9
assert all(p.is_file() for p in PATHS.values())
(OUT / "findings.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ("measurements", "one_G6_fall_in_512", "planning_hours")}, ensure_ascii=False, indent=2))
