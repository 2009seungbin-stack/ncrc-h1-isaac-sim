"""CPU-only metrics and predeclared 4h admission rule. Never loads a policy.

Input models[name]['seeds'][seed]['scenarios'][G1..G7] require:
episodes, survived, valid_samples, sum_squared_error[vx,vy,yaw],
sum_absolute_error[vx,vy,yaw], expected_steps_per_episode, completed_control_steps,
per_episode_valid_samples, first_terminal_steps, first_timeout_steps,
and per_episode_metrics_reported=true. Event indices are zero-based.
No rounded legacy score is used. A report failing completeness is not ranked.
"""
import math
import re

WEIGHTS = {"G1": .15, "G2": .15, "G3": .20, "G4": .15,
           "G5": .15, "G6": .10, "G7": .10}
SEEDS = ("2039", "2063")
AXES = ("vx", "vy", "yaw")
PROTOCOL = {"axis_mode": "all", "transient_seconds": 0.0, "command_scale": 1.0,
            "observation_noise_enabled": True, "num_envs": 128,
            "startup_seed": 409, "terrain_seed": 11003,
            "segment_seconds": 5.0, "control_step_seconds": .02}
EXPECTED_STEPS = 1000


def at_most(a, b):
    # Resolve roundoff at a predeclared boundary, not meaningful regression.
    return a <= b or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def validate_models(models):
    if set(models) != {"candidate", "C0_5500", "F10_3000"}:
        raise ValueError("Require exactly candidate and both frozen references")
    for model in models.values():
        if model.get("protocol") != PROTOCOL:
            raise ValueError("Full-window shared validation protocol mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", model.get("checkpoint_sha256", "")):
            raise ValueError("Require actual evaluated checkpoint SHA256")
        if model.get("source_hashes_verified") is not True or model.get("diagnostics_complete") is not True:
            raise ValueError("Require hash verification and diagnostic completeness")
        if set(model["seeds"]) != set(SEEDS):
            raise ValueError("Validation seed set incomplete")
        for seed in SEEDS:
            sc = model["seeds"][seed]["scenarios"]
            if set(sc) != set(WEIGHTS):
                raise ValueError("Seven-scenario coverage incomplete")
            for row in sc.values():
                if type(row["episodes"]) is not int or row["episodes"] != 128 or type(row["survived"]) is not int or not 0 <= row["survived"] <= 128:
                    raise ValueError("Episode counts invalid")
                n = row["valid_samples"]
                if type(n) is not int or n <= 0:
                    raise ValueError("Require positive integer valid sample count")
                if row.get("expected_steps_per_episode") != EXPECTED_STEPS or row.get("completed_control_steps") != EXPECTED_STEPS:
                    raise ValueError("Scheduled full control-step coverage incomplete")
                counts, terminals, timeouts = (row.get(k) for k in
                    ("per_episode_valid_samples", "first_terminal_steps", "first_timeout_steps"))
                if any(not isinstance(v, list) or len(v) != 128 for v in (counts, terminals, timeouts)):
                    raise ValueError("Require all per-episode sample/event records")
                for count, terminal, timeout in zip(counts, terminals, timeouts):
                    if type(count) is not int or not 0 <= count <= EXPECTED_STEPS:
                        raise ValueError("Invalid per-episode sample count")
                    for event in (terminal, timeout):
                        if event is not None and (type(event) is not int or not 0 <= event < EXPECTED_STEPS):
                            raise ValueError("Invalid event step")
                    if timeout is not None and timeout != EXPECTED_STEPS-1:
                        raise ValueError("Unexpected early timeout invalidates fixed protocol")
                    events = [e for e in (terminal, timeout) if e is not None]
                    expected_valid = min(events) if events else EXPECTED_STEPS
                    if count != expected_valid:
                        raise ValueError("Missing full-window samples or incorrect reset mask")
                if n != sum(counts) or row["survived"] != sum(t is None for t in terminals):
                    raise ValueError("Aggregate count disagrees with episode audit")
                if row.get("nonfinite_count") != 0:
                    raise ValueError("Nonfinite observations/actions invalidate result")
                if row.get("per_episode_metrics_reported") is not True:
                    raise ValueError("Require per-episode diagnostics before admission")
                for key in ("sum_squared_error", "sum_absolute_error"):
                    if set(row[key]) != set(AXES):
                        raise ValueError("Missing error axis")
                    if any(not math.isfinite(v) or v < 0 for v in row[key].values()):
                        raise ValueError("Error sums must be finite and nonnegative")
                for a in AXES:
                    # Cauchy-Schwarz is a necessary consistency check for sums.
                    if row["sum_absolute_error"][a] ** 2 > n * row["sum_squared_error"][a] + 1e-7:
                        raise ValueError("MAE sum exceeds squared-error consistency bound")


def aggregate(model, seeds=SEEDS):
    scenarios = {}
    for sid in WEIGHTS:
        rows = [model["seeds"][seed]["scenarios"][sid] for seed in seeds]
        n = sum(r["valid_samples"] for r in rows)
        rmse = {a: math.sqrt(sum(r["sum_squared_error"][a] for r in rows) / n) for a in AXES}
        mae = {a: sum(r["sum_absolute_error"][a] for r in rows) / n for a in AXES}
        ep = sum(r["episodes"] for r in rows)
        survived = sum(r["survived"] for r in rows)
        scenarios[sid] = {"rmse": rmse, "mae": mae, "xy": math.sqrt((rmse["vx"]**2 + rmse["vy"]**2)/2),
                          "survival": survived/ep, "falls": ep-survived, "samples": n}
    return {
        "scenarios": scenarios,
        "vx": math.sqrt(sum(WEIGHTS[s]*r["rmse"]["vx"]**2 for s, r in scenarios.items())),
        "xy": math.sqrt(sum(WEIGHTS[s]*r["xy"]**2 for s, r in scenarios.items())),
        "yaw": math.sqrt(sum(WEIGHTS[s]*r["rmse"]["yaw"]**2 for s, r in scenarios.items())),
        "Q3_internal": sum(WEIGHTS[s]*r["survival"]*math.exp(-math.sqrt(sum(r["rmse"][a]**2 for a in AXES)/3))
                           for s, r in scenarios.items()),
        "falls": sum(r["falls"] for r in scenarios.values())
    }


def decide(models):
    validate_models(models)
    metrics = {name: aggregate(model) for name, model in models.items()}
    c, c0, f = (metrics[n] for n in ("candidate", "C0_5500", "F10_3000"))
    checks = {
        "complete_data": True,
        "zero_candidate_falls": c["falls"] == 0,
        "XY_practical_gain_10pct": at_most(c["xy"], .90*f["xy"]),
        "VX_aggregate_no_regression_F10": at_most(c["vx"], f["vx"]),
        "critical_scenario_VX_tolerance_5pct_F10": all(at_most(c["scenarios"][s]["rmse"]["vx"], 1.05*f["scenarios"][s]["rmse"]["vx"]) for s in ("G1", "G3", "G5", "G6")),
        "yaw_aggregate_tolerance_2pct": at_most(c["yaw"], 1.02*f["yaw"]),
        "Q3_no_regression": at_most(f["Q3_internal"], c["Q3_internal"]),
        "rough_stairs_XY_no_regression_C0": all(at_most(c["scenarios"][s]["xy"], c0["scenarios"][s]["xy"]) for s in ("G3", "G5")),
        "other_scenario_XY_tolerance_10pct_C0": all(at_most(c["scenarios"][s]["xy"], 1.10*c0["scenarios"][s]["xy"])
                                                  for s in WEIGHTS if s not in ("G3", "G5")),
        "scenario_yaw_tolerance_10pct_F10": all(at_most(c["scenarios"][s]["rmse"]["yaw"], 1.10*f["scenarios"][s]["rmse"]["yaw"]) for s in WEIGHTS),
        "seed_XY_consistent": all(at_most(aggregate(models["candidate"], (seed,))["xy"], aggregate(models["F10_3000"], (seed,))["xy"]) for seed in SEEDS),
        "scenario_seed_falls_no_regression": all(
            models["candidate"]["seeds"][seed]["scenarios"][s]["survived"] >= models[ref]["seeds"][seed]["scenarios"][s]["survived"]
            for seed in SEEDS for s in WEIGHTS for ref in ("C0_5500", "F10_3000"))
    }
    return {"admit_separate_confirmation_budget": all(checks.values()), "checks": checks, "metrics": metrics,
            "official_score_prediction": None, "automatic_training": False,
            "rule_type": "practical_exploratory_gate_not_statistical_test"}
