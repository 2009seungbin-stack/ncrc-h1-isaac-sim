"""Generate one review-only reward literal and verify the offline design.
Standard library only. Never imports reward/IsaacLab/Torch, loads checkpoints,
starts processes other than this script, or accesses a server/network.
"""
import ast
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from decision_rules import AXES, EXPECTED_STEPS, PROTOCOL, SEEDS, WEIGHTS, aggregate, decide

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
ORIGINAL = WS / "analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002/fresh_rewards_original.py"
OLD = WS / "analysis/high_score_search_20261009"
EXPECTED_ORIGINAL_SHA = "09630e6f548b16a2fa0a81eef73aa393cf5f12eadccd0bad8a144b924e8c1be7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def reward_node(tree):
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in n.targets)]
    if len(nodes) != 1 or not isinstance(nodes[0].value, ast.Dict):
        raise ValueError("Expected exactly one literal REWARD_WEIGHTS dictionary")
    return nodes[0]


def stripped_ast(tree):
    t = copy.deepcopy(tree)
    reward_node(t).value = ast.Constant(None)
    return ast.dump(t, include_attributes=False)


def byte_span(data, node):
    lines = data.splitlines(keepends=True)
    return (sum(map(len, lines[:node.lineno-1]))+node.col_offset,
            sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset)


def estimate(plan, iterations, rate=None):
    b = plan["budget"]
    rate = b["planning_iteration_seconds"] if rate is None else rate
    train = rate*iterations*b["planning_multiplier"]+b["training_boot_seconds"]
    fixed = sum(b[k] for k in ("setup_seconds", "selection_seconds", "validation_seconds", "retrieval_shutdown_seconds"))
    return {"iterations": iterations, "training_seconds": train, "non_training_seconds": fixed,
            "total_seconds": train+fixed, "headroom_seconds": plan["user_budget_seconds"]-train-fixed,
            "fits": train+fixed <= plan["user_budget_seconds"]+1e-9}


def synthetic_model(xy=.1, yaw=.1):
    n = 128*EXPECTED_STEPS
    seeds = {}
    for seed in SEEDS:
        seeds[seed] = {"scenarios": {s: {
            "episodes": 128, "survived": 128, "valid_samples": n,
            "expected_steps_per_episode": EXPECTED_STEPS, "completed_control_steps": EXPECTED_STEPS,
            "per_episode_valid_samples": [EXPECTED_STEPS]*128,
            "first_terminal_steps": [None]*128, "first_timeout_steps": [None]*128,
            "nonfinite_count": 0, "per_episode_metrics_reported": True,
            "sum_squared_error": {"vx": xy*xy*n, "vy": xy*xy*n, "yaw": yaw*yaw*n},
            "sum_absolute_error": {"vx": xy*n, "vy": xy*n, "yaw": yaw*n}
        } for s in WEIGHTS}}
    return {"protocol": dict(PROTOCOL), "checkpoint_sha256": "a"*64,
            "source_hashes_verified": True, "diagnostics_complete": True, "seeds": seeds}


def metric_contract_checks():
    passed = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    def rejects(name, models):
        try:
            decide(models)
        except (ValueError, KeyError):
            passed.append(name)
        else:
            raise AssertionError(name)

    models = {"candidate": synthetic_model(.09), "C0_5500": synthetic_model(),
              "F10_3000": synthetic_model()}
    good = decide(models)
    check("10pct_XY_boundary_passes_without_roundoff_rejection", good["admit_separate_confirmation_budget"])
    check("no_official_forecast_or_automatic_training", good["official_score_prediction"] is None and not good["automatic_training"])
    bad = copy.deepcopy(models)
    bad["candidate"] = synthetic_model(.090001)
    check("XY_less_than_10pct_gain_does_not_admit", not decide(bad)["checks"]["XY_practical_gain_10pct"])
    bad = copy.deepcopy(models)
    bad["candidate"] = synthetic_model(.09, .102)
    check("2pct_yaw_boundary_passes", decide(bad)["checks"]["yaw_aggregate_tolerance_2pct"])
    bad["candidate"] = synthetic_model(.09, .102001)
    check("yaw_above_2pct_regression_rejects", not decide(bad)["checks"]["yaw_aggregate_tolerance_2pct"])
    bad = copy.deepcopy(models)
    row = bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G5"]
    row["survived"] = 127
    row["first_terminal_steps"][0] = 500
    row["per_episode_valid_samples"][0] = 500
    row["valid_samples"] -= 500
    for key in ("sum_squared_error", "sum_absolute_error"):
        row[key] = {a: v*row["valid_samples"]/(128*EXPECTED_STEPS) for a, v in row[key].items()}
    check("one_fall_rejects_even_with_average_gain", not decide(bad)["admit_separate_confirmation_budget"])
    bad = copy.deepcopy(models)
    del bad["candidate"]["seeds"][SEEDS[1]]["scenarios"]["G7"]
    rejects("missing_scenario_is_incomplete", bad)
    bad = copy.deepcopy(models)
    bad["candidate"]["protocol"]["transient_seconds"] = .5
    rejects("transient_excluded_run_cannot_pass_as_full_window", bad)
    bad = copy.deepcopy(models)
    bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]["sum_squared_error"]["vx"] = float("nan")
    rejects("nonfinite_error_rejected", bad)
    bad = copy.deepcopy(models)
    bad["candidate"]["diagnostics_complete"] = False
    rejects("missing_diagnostics_rejected", bad)
    bad = copy.deepcopy(models)
    del bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]["nonfinite_count"]
    rejects("missing_nonfinite_audit_rejected", bad)
    bad = copy.deepcopy(models)
    bad["candidate"]["checkpoint_sha256"] = "unknown"
    rejects("missing_actual_checkpoint_hash_rejected", bad)
    bad = copy.deepcopy(models)
    row = bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]
    row["valid_samples"] = 128
    row["per_episode_valid_samples"] = [1]*128
    rejects("one_sample_per_survivor_cannot_claim_full_window", bad)
    bad = copy.deepcopy(models)
    row = bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]
    row["completed_control_steps"] -= 1
    rejects("shortened_command_horizon_rejected", bad)
    bad = copy.deepcopy(models)
    row = bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]
    row["first_timeout_steps"] = [EXPECTED_STEPS-1]*128
    row["per_episode_valid_samples"] = [EXPECTED_STEPS-1]*128
    row["valid_samples"] = (EXPECTED_STEPS-1)*128
    for key in ("sum_squared_error", "sum_absolute_error"):
        row[key] = {a: v*(EXPECTED_STEPS-1)/EXPECTED_STEPS for a, v in row[key].items()}
    check("explicit_final_timeout_mask_accepted", decide(bad)["admit_separate_confirmation_budget"])
    row["first_timeout_steps"][0] = 10
    rejects("unexpected_early_timeout_rejected", bad)
    bad = copy.deepcopy(models)
    bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]["valid_samples"] -= 1
    rejects("aggregate_sample_sum_disagreement_rejected", bad)
    # A large vy benefit must not mask globally worse forward tracking.
    hidden = {"candidate": synthetic_model(.09), "C0_5500": synthetic_model(.1),
              "F10_3000": synthetic_model(.1)}
    for model_name, errors in (("F10_3000", (.1, .2, .1)), ("candidate", (.11, 0, .1))):
        for seed in SEEDS:
            for row in hidden[model_name]["seeds"][seed]["scenarios"].values():
                n = row["valid_samples"]
                row["sum_squared_error"] = {a: v*v*n for a, v in zip(AXES, errors)}
                row["sum_absolute_error"] = {a: v*n for a, v in zip(AXES, errors)}
    decision = decide(hidden)
    check("vy_benefit_cannot_hide_VX_regression", decision["checks"]["XY_practical_gain_10pct"]
          and not decision["checks"]["VX_aggregate_no_regression_F10"] and not decision["admit_separate_confirmation_budget"])
    bad = copy.deepcopy(models)
    bad["candidate"]["seeds"][SEEDS[0]]["scenarios"]["G1"]["sum_absolute_error"]["vx"] *= 100
    rejects("inconsistent_MAE_error_sums_rejected", bad)
    # Pool error sums by sample count; never average rounded per-seed RMSEs.
    pool = synthetic_model()
    first = pool["seeds"][SEEDS[0]]["scenarios"]["G1"]
    first["survived"] = 0
    first["per_episode_valid_samples"] = [100]*128
    first["first_terminal_steps"] = [100]*128
    first["valid_samples"] = 100*128
    first["sum_squared_error"] = {a: .01*first["valid_samples"] for a in AXES}
    first["sum_absolute_error"] = {a: .1*first["valid_samples"] for a in AXES}
    row = pool["seeds"][SEEDS[1]]["scenarios"]["G1"]
    row["survived"] = 0
    row["per_episode_valid_samples"] = [900]*128
    row["first_terminal_steps"] = [900]*128
    row["valid_samples"] = 900*128
    row["sum_squared_error"] = {a: .04*row["valid_samples"] for a in AXES}
    row["sum_absolute_error"] = {a: .2*row["valid_samples"] for a in AXES}
    pooled = aggregate(pool)["scenarios"]["G1"]
    check("unequal_sample_counts_pool_correctly", math.isclose(pooled["rmse"]["vx"], math.sqrt(.037)) and math.isclose(pooled["mae"]["vx"], .19))
    return passed


def main():
    plan = read(ROOT/"design.json")
    originals = [ORIGINAL, OLD/"execution.json", OLD/"screen_summary.json",
                 OLD/"execution/previous_source_hashes_20261002.json",
                 OLD/"execution/reference_models.json", OLD/"execution/runner.py",
                 OLD/"execution/t_eval.py", OLD/"execution/t_eval_go2_core.py",
                 WS/"analysis/score_reverse_20261001/analysis.json"]
    before = {str(p.relative_to(WS)): sha(p) for p in originals}
    assert sha(ORIGINAL) == EXPECTED_ORIGINAL_SHA
    assert plan["status"] == "OFFLINE_DESIGN_ONLY" and not plan["execution_authorized"] and not plan["server_started"]
    assert plan["user_budget_seconds"] == 14400
    assert plan["training"] == {**plan["training"], "policies": 1, "iterations": 1500, "num_envs": 6144, "seed": 42,
                                "initialization": "scratch", "resume": False, "second_candidate": False, "automatic_extension": False}
    assert set(plan["candidates"]) == {"B1_selective_cost", "P1_posture_reserve", "U0_upstream_like_reference"}
    assert [n for n, c in plan["candidates"].items() if c["role"] == "only_scheduled_policy_in_future_design"] == ["B1_selective_cost"]
    assert plan["evaluation"]["scenario_weights"] == WEIGHTS and math.isclose(sum(WEIGHTS.values()), 1)
    assert plan["evaluation"]["transient_seconds"] == 0 and plan["evaluation"]["axis_mode"] == "all"
    assert plan["evaluation"]["expected_control_steps_per_scenario_seed"] == EXPECTED_STEPS
    assert math.isclose(plan["evaluation"]["segment_seconds"]*plan["evaluation"]["segments_per_scenario"]
                        /plan["evaluation"]["required_control_step_seconds"], EXPECTED_STEPS)
    assert set(map(str, plan["evaluation"]["validation"]["reset_seeds"])) == set(SEEDS)
    assert not set(plan["evaluation"]["selection"]["reset_seeds"]) & set(plan["evaluation"]["validation"]["reset_seeds"])
    assert plan["evaluation"]["selection"]["candidates"] == ["model_1000.pt", "model_1499.pt"]
    assert plan["official_target"]["candidate_predicted_official_score"] is None
    assert plan["official_target"]["candidate_predicted_score_gain"] is None
    assert math.isclose(93-88.14, plan["official_target"]["required_gain"], abs_tol=1e-12)

    weights = plan["candidates"][plan["primary_candidate"]]["weights"]
    assert len(weights) == 10 and set(weights) == set(plan["candidates"]["U0_upstream_like_reference"]["weights"])
    for key, value in weights.items():
        assert isinstance(value, (int, float)) and math.isfinite(value)
        assert value >= 0 if key in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time") else value <= 0
    data = ORIGINAL.read_bytes()
    tree = ast.parse(data.decode("utf-8"))
    lo, hi = byte_span(data, reward_node(tree).value)
    literal = "{\n"+"".join(f'    "{key}": {value!r},\n' for key, value in weights.items())+"}"
    candidate = data[:lo]+literal.encode("utf-8")+data[hi:]
    candidate_tree = ast.parse(candidate.decode("utf-8"))
    assert stripped_ast(tree) == stripped_ast(candidate_tree)
    assert ast.literal_eval(reward_node(candidate_tree).value) == weights
    newlo, newhi = byte_span(candidate, reward_node(candidate_tree).value)
    assert data[:lo] == candidate[:newlo] and data[hi:] == candidate[newhi:]
    compile(candidate.decode("utf-8"), "B1_candidate", "exec")  # syntax only, never exec
    path = ROOT/"candidates/B1_selective_cost_rewards.py"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(candidate)

    old = read(OLD/"execution.json")
    assert weights != old["s1_reward_weights"] and weights != old["r1_reward_weights"]
    f10 = {**old["s1_reward_weights"], "track_lin_vel_xy_exp": 2.0, "track_ang_vel_z_exp": 1.5}
    assert weights != f10
    schedule = [estimate(plan, n) for n in (1500, 1600, 1800)]
    assert schedule[0]["fits"] and schedule[1]["fits"] and not schedule[2]["fits"]
    assert estimate(plan, 1500, 5.2)["fits"] and not estimate(plan, 1500, 5.2001)["fits"]
    deadlines = plan["budget"]["deadlines_relative_to_actual_allocation_start"]
    assert list(deadlines.values()) == [900, 9600, 10500, 11700, 13800, 14400]
    assert deadlines["compute_end"]+plan["budget"]["retrieval_shutdown_seconds"] == 14400
    assert deadlines["training_end_latest"]+900 == deadlines["selection_end_latest"]
    assert deadlines["selection_end_latest"]+1200 == deadlines["compute_end"]
    assert schedule[0]["training_seconds"]+900 <= deadlines["training_end_latest"]
    assert not plan["execution_readiness"]["future_runner_ready"]

    tests = metric_contract_checks()
    after = {str(p.relative_to(WS)): sha(p) for p in originals}
    assert before == after
    summary = read(OLD/"screen_summary.json")
    audit = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "OFFLINE_STATIC_DESIGN_VALIDATED_NOT_EXECUTED",
        "server_accessed": False, "simulator_executed": False, "checkpoint_deserialized": False,
        "protected_runtime_sources_live_verified": False,
        "protected_provided_source_actual_files_verified_here": False,
        "protected_hash_manifests_and_existing_artifacts_unchanged": before == after,
        "input_sha256": before,
        "candidate": {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": len(candidate),
                      "ten_finite_weights_and_signs": True, "syntax_valid": True,
                      "AST_identical_outside_REWARD_WEIGHTS": True,
                      "bytes_identical_before_and_after_dictionary": True,
                      "differs_from_S1_R1_F10_exact_weights": True,
                      "novelty_scope": "new untrained exact combination, not new reward principle or framework"},
        "budget_calculations": schedule,
        "maximum_1500_training_rate_with_fixed_reservations_seconds": 5.2,
        "recorded_last_run_iteration_seconds": {
            "S1": old["s1_training_seconds"]/1500,
            "R1": old["r1_training_seconds"]/1500,
            "combined": (old["s1_training_seconds"]+old["r1_training_seconds"])/3000},
        "recorded_last_run_evaluation_seconds": sum(v["seconds"] for v in old["evaluations"].values()),
        "historical_quality": {n: {"Q3": m["weighted_quality"], "falls": m["falls"]} for n, m in summary["models"].items()},
        "metric_contract_checks_passed": len(tests), "metric_contract_check_names": tests,
        "metric_helper_sha256": sha(ROOT/"decision_rules.py"),
        "design_artifact_sha256": {name: sha(ROOT/name) for name in
            ("design.json", "research_design.txt", "evidence_sources.json", "RUNNER_REQUIREMENTS.txt", "build_and_validate.py")},
        "future_runner_implemented": False, "new_evaluator_diagnostics_implemented": False,
        "official_score_forecast": None,
        "scope": "syntax, artifact identity, budget arithmetic and synthetic metric/gate boundaries only"}
    write(ROOT/"design_validation.json", audit)
    print(json.dumps({"status": audit["status"], "checks": len(tests), "candidate_sha256": sha(path),
                      "planned_total_seconds": schedule[0]["total_seconds"],
                      "headroom_seconds": schedule[0]["headroom_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
