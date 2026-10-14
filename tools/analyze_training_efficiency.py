"""Read-only local diagnostics and untrained reward candidates for NCRC.

Run: python -X utf8 tools/analyze_training_efficiency.py
Writes only analysis/training_efficiency_20261001; never starts Isaac, trains,
changes an evaluator, edits an original reward file, or contacts the server.
PyTorch is used only to inspect local full checkpoints on CPU.
"""
import ast
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/training_efficiency_20261001"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing REWARD_WEIGHTS")


def change(source, key, value):
    # Change a numeric value only; optionally activate the provided advanced key.
    active = rf'^(\s*"{re.escape(key)}"\s*:\s*)([-+\d.eE]+)'
    result, count = re.subn(active, lambda m: m[1] + str(value), source, flags=re.M)
    if not count:
        commented = rf'^(\s*)#\s*("{re.escape(key)}"\s*:\s*)([-+\d.eE]+)'
        result, count = re.subn(commented, lambda m: m[1] + m[2] + str(value), source, flags=re.M)
    assert count == 1, (key, count)
    assert weights(result)[key] == value
    return result


def nonreward_ast(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS"
                                         for t in n.targets))]
    return ast.dump(tree, include_attributes=False)


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + "/" + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensors(item, prefix + "/" + str(i))


def inspect_checkpoint(relative):
    path = ROOT / relative
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    bad, count, distributions = [], 0, {}
    for name, value in tensors(checkpoint):
        count += 1
        if not torch.isfinite(value).all().item():
            bad.append(name)
        if name.endswith("distribution.std_param"):
            distributions[name] = {"dimensions": value.numel(), "min": value.min().item(),
                                   "max": value.max().item(), "mean": value.mean().item()}
    return {"path": relative, "sha256": sha(path), "iter": checkpoint.get("iter"),
            "top_level_keys": list(checkpoint), "tensor_count": count,
            "nonfinite_tensors": bad, "std_parameters": distributions,
            "limitation": "Finite checkpoint does not establish safe future PPO updates."}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inverse = module("inverse_score", "tools/analyze_score_r4.py")
    parser = module("event_parser", "tools/score_model.py")
    inputs = [ROOT / a["path"] for a in inverse.ANCHORS.values()]
    inputs += [ROOT / "longrun_D1/cfg/quadruped_rewards.py",
               ROOT / "tools/r3/humanoid_rewards_R3A.py",
               ROOT / "submit_C0/env.yaml", ROOT / "submit_R3/env.yaml"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    all_rows = [r for a in inverse.ANCHORS.values() for r in inverse.rows_for(a)[0]]
    c = inverse.fit_floor(all_rows)
    result = {
        "date": "2026-10-01", "status": "UNTRAINED_CANDIDATES_AND_CONDITIONAL_PLAN",
        "server_started": False, "official_total": 176.14, "leader_gap_at_read": .25,
        "remaining_server_hours_at_board_read": 51 + 25/60,
        "rules": "Only provided REWARD_WEIGHTS values; supported CLI options require wrapper verification.",
        "sources": {
            "competition_rules": "https://robotics.connect.or.kr/rules/preliminary_regulations.html",
            "reward_functions": "https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py",
            "resume_cli": "https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/scripts/reinforcement_learning/rsl_rl/cli_args.py",
            "runner_reference_not_verified_server_version": "https://github.com/leggedrobotics/rsl_rl/blob/v5.0.0/rsl_rl/runners/on_policy_runner.py",
            "off_policy_research_outside_current_rules": "https://arxiv.org/abs/2512.01996",
        },
        "input_sha256": before, "checkpoints": {}, "robots": {}, "candidates": [],
        "proxy_c": c,
        "proxy_warning": "Two policies only. Counterfactuals assume unchanged survival and all other axes; they are not measured gains or confidence intervals.",
    }
    for name, relative in {
        "Go2_C0": "longrun_C0/model_5500.pt",
        "H1_R3A": "R3_bundle_extracted/keep_R3/R3A_9000_model.pt",
    }.items():
        result["checkpoints"][name] = inspect_checkpoint(relative)

    events = sorted((ROOT / "longrun_D1/logs").glob("events.out.tfevents.*"))
    assert len(events) == 1
    tags = parser.read_tfevents(events[0])
    perf = {"source": str(events[0].relative_to(ROOT)), "sha256": sha(events[0]),
            "steps": ">=500", "robot": "Go2_D1", "num_envs": 6144, "steps_per_env": 24}
    for tag in ("Perf/total_fps", "Perf/collection_time", "Perf/learning_time"):
        samples = [v for s, v in tags[tag].items() if s >= 500 and math.isfinite(v)]
        perf[tag] = {"count": len(samples), "median": statistics.median(samples),
                     "mean": statistics.mean(samples)}
    collection = perf["Perf/collection_time"]["mean"]
    learning = perf["Perf/learning_time"]["mean"]
    perf["collection_fraction_of_reported_iteration_time"] = collection / (collection + learning)
    perf["maximum_speedup_removing_all_reported_learning_time"] = (collection + learning) / collection
    perf["transitions_per_iteration"] = 6144 * 24
    result["performance"] = perf

    for name, anchor in inverse.ANCHORS.items():
        raw = json.loads((ROOT / anchor["path"]).read_text(encoding="utf-8"))
        rows = list(raw["scenarios"].values())
        axis_mse = {axis: sum(r["w"] * r["rmse"][axis]**2 for r in rows)
                    for axis in ("vx", "vy", "yaw")}

        def score(factors, family):
            total = 0
            for row in rows:
                mse = sum((row["rmse"][axis]*factors.get(axis, 1))**2
                          for axis in ("vx", "vy", "yaw")) / 3
                q = math.exp(-math.sqrt(mse + c)) if family == "floor" else .905*math.exp(-math.sqrt(mse))
                total += 70 * row["w"] * row["survival"] * q
            return total

        conditions = {"yaw_rmse_minus_20pct": {"yaw": .8},
                      "yaw_rmse_minus_50pct": {"yaw": .5},
                      "xy_rmse_minus_50pct": {"vx": .5, "vy": .5},
                      "all3_rmse_minus_50pct": {"vx": .5, "vy": .5, "yaw": .5}}
        counterfactual = {k: {family: score(factors, family) - score({}, family)
                             for family in ("floor", "ratio_0.905")}
                          for k, factors in conditions.items()}
        result["robots"][name] = {"weighted_axis_mse": axis_mse,
                                   "axis_fraction_of_local_mse3": {k: v/sum(axis_mse.values()) for k, v in axis_mse.items()},
                                   "counterfactual_sim_score_delta": counterfactual}

    g = (ROOT / "longrun_D1/cfg/quadruped_rewards.py").read_text(encoding="utf-8")
    # This source is the original baseline dictionary, not D1's effective env.
    g = change(change(g, "track_lin_vel_xy_exp", 2.0), "track_ang_vel_z_exp", .75)
    h = (ROOT / "tools/r3/humanoid_rewards_R3A.py").read_text(encoding="utf-8")
    for robot, control in (("Go2", g), ("H1", h)):
        env_path = "submit_C0/env.yaml" if robot == "Go2" else "submit_R3/env.yaml"
        # BaseLoader returns strings without instantiating Python-tagged objects.
        actual = yaml.load((ROOT / env_path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["rewards"]
        for key, value in weights(control).items():
            assert float(actual[key]["weight"]) == value, (robot, key)
        path = OUT / f"{robot}_CONTROL_rewards.py"
        path.write_text(control, encoding="utf-8", newline="\n")
        result["candidates"].append({"id": f"{robot}_CONTROL", "robot": robot,
                                     "path": str(path.relative_to(ROOT)), "changes_from_current": {},
                                     "weights": weights(control), "sha256": sha(path)})
    specs = [
        ("GY10", "Go2", "track_ang_vel_z_exp", 1.0, 1, "Increase underweighted yaw tracking."),
        ("GY15", "Go2", "track_ang_vel_z_exp", 1.5, 1, "Restore original yaw to linear reward ratio with linear=2."),
        ("HY30", "H1", "track_ang_vel_z_exp", 3.0, 2, "Exploratory yaw gain; exceeds suggested range, not the editable-value rule."),
        ("HO10", "H1", "flat_orientation_l2", -1.0, 2, "Test more posture freedom on rough terrain and slopes."),
        ("GF00", "Go2", "feet_air_time", 0.0, 3, "Test whether signed airtime shaping at threshold .5 hinders gait."),
        ("GF03", "Go2", "feet_air_time", .03, 3, "Small gait-shaping increase; not a direct foot-height target."),
        ("HX15", "H1", "track_lin_vel_xy_exp", 1.5, 3, "Test linear tracking if yaw variants plateau."),
        ("HA15", "H1", "feet_air_time", .15, 3, "Reduce biped gait constraint if terrain/video evidence warrants it."),
    ]
    for cid, robot, key, value, priority, hypothesis in specs:
        control = g if robot == "Go2" else h
        candidate = change(control, key, value)
        base, new = weights(control), weights(candidate)
        diff = {k: {"from": base.get(k), "to": new.get(k)} for k in base.keys() | new.keys()
                if base.get(k) != new.get(k)}
        assert list(diff) == [key]
        assert nonreward_ast(candidate) == nonreward_ast(control)
        path = OUT / f"{cid}_rewards.py"
        path.write_text(candidate, encoding="utf-8", newline="\n")
        result["candidates"].append({"id": cid, "robot": robot, "path": str(path.relative_to(ROOT)),
                                     "changes_from_current": diff, "priority": priority,
                                     "hypothesis_not_result": hypothesis, "weights": new, "sha256": sha(path)})

    seconds_per_iter = 4.8  # Go2 historical planning estimate, not an H1 measurement.
    result["cost_model"] = {
        "seconds_per_iteration_planning_estimate_go2": seconds_per_iter,
        "hours_by_additional_iterations": {str(n): n*seconds_per_iter/3600 for n in (100, 300, 500, 800, 1200, 2000, 2500, 5500)},
        "four_scratch_2500_hours": 4*2500*seconds_per_iter/3600,
        "control_plus_four_resume_800_hours": 5*800*seconds_per_iter/3600,
        "conditional_training_cost_ratio": (4*2500)/(5*800),
        "scope": "Training only; unequal learning histories, startup/evaluation/curriculum recovery excluded. Validate resume quality before using this saving.",
    }
    result["execution_plan"] = [
        "Read competition train wrapper, imported CLI module and README. Do not call train.py --help: local runbook records checkpoint cleanup on this path.",
        "Verify provided --resume/--checkpoint semantics. Upstream max_iterations means additional iterations; competition wrapper remains unverified.",
        "Run same-reward 50-100 iteration resume pilot first. C0 failed at 5528; finite model5500 is not proof it will survive another 28 updates.",
        "If C0 pilot fails, train one fresh C0 trunk for 2000-2500 iterations and preserve its full checkpoint; old C0 model2000 is not locally available.",
        "Branch CONTROL, GY10 and GY15 independently from identical parent and reset curriculum. Use 300 preview then 800-1200 total additional updates; do not discard terrain variants before curriculum recovers.",
        "Verify saved env.yaml actual reward weights and tensor health. Keep full checkpoints outside wrapper cleanup every available save interval; do not edit checkpoint tensors.",
        "Compare all seven scenarios at same env count and seeds. Screen 128 env x1 seed; finalists 128 env x4 seeds. Confirm training stochasticity with an independent parent/seed where wrapper supports it.",
        "Rank by survival, paired axis RMSE and both 3/6-axis diagnostics plus frozen score proxies, never raw training reward across changed weights. New falls or serious rough/stair/push regressions block promotion pending expanded evaluation.",
        "Extend only consistent winners by 1500-2000 updates or retrain one winner from scratch if curriculum reset is harmful. Render videos for finalists; keep contact/slip/gait evidence for airtime changes.",
        "If provided num_envs CLI allows it, benchmark 2048/4096/6144 one job at a time; match transition budget and measure eval improvement per wall hour. Prior sweep records 8192 contact-patch overflow and 12288 overflow; the previous above-6144 suggestion is withdrawn for this configuration.",
        "Initial campaign soft budget 8-12 hours, reserve at least 20 hours for later rounds/validation. Stop/rescope after failed pilot or absent paired improvement.",
    ]
    result["excluded_from_current_execution"] = [
        "Algorithm, architecture, learning rate, entropy, normalization or std parameterization edits.",
        "Training push/randomization/physics/terrain changes; local guide says push is play-only.",
        "Large airtime jump .01->.25 without gait evidence; airtime function rewards duration, not height.",
        "Repeating D1 angular penalty -0.08->-0.16: existing equal-budget comparison failed all predeclared gates.",
        "FastSAC/FastTD3 and symmetry augmentation: useful research directions outside current competition edit scope.",
        "Go2 num_envs 8192/12288 under unchanged current PhysX buffer settings: overflow recorded in earlier experiments, not an eligible default speedup candidate.",
    ]
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    result["original_input_hashes_unchanged"] = True
    result["controls_match_actual_submitted_reward_weights"] = True
    result["candidate_nonreward_code_unchanged"] = True
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "candidates": len(result["candidates"]),
                      "collection_fraction": perf["collection_fraction_of_reported_iteration_time"],
                      "yaw_mse_fraction": {k: v["axis_fraction_of_local_mse3"]["yaw"] for k, v in result["robots"].items()},
                      "original_inputs_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
