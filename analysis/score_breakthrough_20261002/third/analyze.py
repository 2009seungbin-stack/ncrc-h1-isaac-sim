"""Offline reward-objective and error-tail audit; no training or remote access.

Uses rounded saved summaries. Moment identities concern scored, alive samples,
not complete episodes. Synthetic reward examples are not robot performance.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / "analysis/round20_execution_20261001"
C = 0.02684614570005177
INPUTS = {}


def read(path):
    path = ROOT / path
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8-sig")


def json_file(path):
    return json.loads(read(path))


def axes(s):
    return {**s["rmse"], **s.get("rmse_extra", {})}


def score(data, family):
    result = 0.0
    for s in data["scenarios"].values():
        if not s["w"]:
            continue
        mse = sum(s["rmse"][a] ** 2 for a in ("vx", "vy", "yaw")) / 3
        if family == "floor":
            t = math.exp(-math.sqrt(mse + C))
        elif family == "ratio":
            t = .905 * math.exp(-math.sqrt(mse))
        elif family == "raw3":
            t = math.exp(-math.sqrt(mse))
        else:
            if "rmse_extra" not in s:
                return None
            mse6 = (3*mse + sum(v*v for v in s["rmse_extra"].values())) / 6
            t = math.exp(-math.sqrt(mse6))
        result += 70*s["w"] * (s["survived"] / s["episodes"]) * t
    return result


def moments(s):
    out = {}
    for a, mae in s.get("mae", {}).items():
        rmse = s["rmse"][a]
        mse = rmse * rmse
        # All files round RMSE and MAE to four decimals.
        lo_mse, hi_mse = max(0, rmse - .00005)**2, (rmse+.00005)**2
        lo_mae, hi_mae = max(0, mae - .00005), mae+.00005
        assert mae <= rmse + .0001
        var_abs = mse - mae*mae
        out[a] = {
            "rmse": rmse, "mae": mae, "mse": mse,
            "variance_of_absolute_error": var_abs,
            "variance_of_absolute_error_rounding_interval":
                [max(0, lo_mse-hi_mae**2), hi_mse-lo_mae**2],
            # This is NOT a fraction of bad samples or effective sample size.
            "mae_squared_over_mse_concentration_index": mae*mae/mse,
            "tail_mse_fraction_lower_bounds": {},
        }
        for t in (.1, .2, .5, 1.0):
            # e^2 1[|e| <= t] <= t |e|, so tail MSE >= MSE - t MAE.
            central = max(0, 1-t*mae/mse)
            conservative = max(0, 1-t*hi_mae/lo_mse)
            out[a]["tail_mse_fraction_lower_bounds"][str(t)] = {
                "central": central, "conservative_for_rounding": conservative,
            }
    return out


def load_evaluations():
    paths = {
        "C0_A": RUN / "results/extracted/round20_20261001/eval/original_A/merged.json",
        "F10_A": RUN / "results/extracted/round20_20261001/eval/F10_model_3000_A/merged.json",
        "C0_B": RUN / "validation_results/extracted/go2_validation_20261002/eval/original_B/merged.json",
        "F10_B": RUN / "validation_results/extracted/go2_validation_20261002/eval/selected_B/merged.json",
        "C0_stress": RUN / "stress_results/extracted/go2_stress_20261002/eval/original_stress/merged.json",
        "F10_stress": RUN / "stress_results/extracted/go2_stress_20261002/eval/selected_stress/merged.json",
        "C0_stairs15": RUN / "stress_results/extracted/go2_stress_20261002/eval/original_stairs_0.15/merged.json",
        "F10_stairs15": RUN / "stress_results/extracted/go2_stress_20261002/eval/selected_stairs_0.15/merged.json",
        "C0_2000": ROOT / "longrun_C0/C0_2000_ALL.json",
        "D1_2000": ROOT / "longrun_D1/eval/D1_2000_ALL.json",
        "H1_R3A9000": ROOT / "R3_bundle_extracted/eval_results/R3_R3A_9000.json",
        "H1_R3A9900": ROOT / "R3_bundle_extracted/eval_results/R3_R3A_10000.json",
        "H1_R3C9000": ROOT / "R3_bundle_extracted/eval_results/R3_R3C_9000.json",
        "H1_R3C9900": ROOT / "R3_bundle_extracted/eval_results/R3_R3C_10000.json",
    }
    data, summary = {}, {}
    for name, path in paths.items():
        d = json_file(path)
        data[name] = d
        summary[name] = {
            "file": str(path.relative_to(ROOT)),
            "scores": {f: score(d, f) for f in ("raw3", "raw6", "floor", "ratio")},
            "scenarios": {sid: {
                "survived": s["survived"], "episodes": s["episodes"],
                "axis_rmse": axes(s), "moments": moments(s),
            } for sid, s in d["scenarios"].items()},
        }
        main = [s for s in d["scenarios"].values() if s["w"] > 0]
        if main:
            common_axes = set.intersection(*(set(axes(s)) for s in main))
            summary[name]["weighted_axis_mse"] = {
                a: sum(s["w"]*axes(s)[a]**2 for s in main) for a in sorted(common_axes)
            }
    comparisons = {}
    for suffix in ("A", "B", "stress", "stairs15"):
        b, f = summary["C0_"+suffix], summary["F10_"+suffix]
        diffs = {}
        for sid in b["scenarios"]:
            bs, fs = b["scenarios"][sid], f["scenarios"][sid]
            diffs[sid] = {}
            for a in bs["moments"]:
                bm, fm = bs["moments"][a], fs["moments"][a]
                diffs[sid][a] = {
                    "mae_change_percent": 100*(fm["mae"]/bm["mae"]-1),
                    "rmse_change_percent": 100*(fm["rmse"]/bm["rmse"]-1),
                    "absolute_error_variance_change_percent": 100*(
                        fm["variance_of_absolute_error"]/bm["variance_of_absolute_error"]-1),
                }
        comparisons[suffix] = diffs
    return data, summary, comparisons


def reward_functions():
    # Execute only the two pure tensor reward functions read from the reference
    # source, not its imports/module initialization or any simulation code.
    text = read(OUT / "sources/rewards.py")
    wanted = {"track_lin_vel_xy_exp", "track_ang_vel_z_exp"}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 2
    for node in nodes:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    namespace = {"torch": torch, "SceneEntityCfg": lambda name: NS(name=name)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "isolated_reference_rewards", "exec"), namespace)
    return namespace


def kernel_audit():
    env = yaml.load(read(RUN / "results/extracted/round20_20261001/exports/F10/env.yaml"), Loader=yaml.BaseLoader)
    cfg = env["rewards"]["track_lin_vel_xy_exp"]
    std, weight = float(cfg["params"]["std"]), float(cfg["weight"])
    assert std == .5 and weight == 2.0
    fun = reward_functions()["track_lin_vel_xy_exp"]

    def case(error):
        err = torch.tensor(error, dtype=torch.float64)
        v = torch.zeros((len(error), 3), dtype=torch.float64)
        v[:, 0] = err
        e = NS(scene={"robot": NS(data=NS(root_lin_vel_b=v))},
               command_manager=NS(get_command=lambda _: torch.zeros_like(v)))
        r = fun(e, std, "base_velocity")
        return {"mae": err.abs().mean().item(), "rmse": err.square().mean().sqrt().item(),
                "mean_tracking_kernel": r.mean().item(), "weighted_reward_before_dt": weight*r.mean().item()}

    constant = case([.2]*100)
    intermittent = case([0.0]*90+[.8]*10)
    assert intermittent["rmse"] > constant["rmse"]
    assert intermittent["mean_tracking_kernel"] > constant["mean_tracking_kernel"]
    vals = []
    for err in (0, .05, .1, .2, std/math.sqrt(2), .5, .8, 1, 1.5, 2):
        r = math.exp(-err*err/std**2)
        vals.append({"planar_error_norm": err, "kernel": r,
                     "weighted_reward": weight*r,
                     "absolute_slope_wrt_error_norm": weight*2*err/std**2*r})
    return {"std": std, "weight": weight, "constant_error_example": constant,
            "intermittent_error_example": intermittent, "kernel_values": vals,
            "limits": "Static equal-survival examples; this is not a PPO gradient or a robot rollout. Future recovery rewards still provide credit."}


def training_tradeoff():
    d = json_file(OUT.parent / "deeper/parsed_training.json")
    reward_tags = sorted(k for k in d["F10"][0] if k.startswith("Episode_Reward/"))
    windows = {}
    for name, rows in d.items():
        windows[name] = []
        for low, high in ((2000, 2800), (3000, 4000)):
            sub = rows[low:high]
            med = {k: float(np.median([r[k] for r in sub])) for k in reward_tags}
            angular_weight = 1.5 if name == "F10" else .75
            med["lin_tracking_div_weight"] = med["Episode_Reward/track_lin_vel_xy_exp"]/2
            med["yaw_tracking_div_weight"] = med["Episode_Reward/track_ang_vel_z_exp"]/angular_weight
            med["negative_term_magnitude_sum_of_medians"] = -sum(min(0, med[k]) for k in reward_tags)
            med["torque_plus_acc_penalty_magnitude"] = -sum(med[k] for k in (
                "Episode_Reward/dof_torques_l2", "Episode_Reward/dof_acc_l2"))
            windows[name].append({"start": low, "stop_exclusive": high, "medians": med})
    return windows


def scale_audit():
    src = read(OUT.parent / "deeper/sources/ppo.py")
    assert "st.advantages.std() + 1e-8" in src
    assert "clip_grad_norm_(self.actor.parameters()" in src
    assert "clip_grad_norm_(self.critic.parameters()" in src
    a = torch.tensor([-4., -1., -.1, .05, .7, 2., 3.], dtype=torch.float64)
    def norm(a):
        return (a-a.mean())/(a.std()+1e-8)
    base = norm(a)
    results = {str(k): (norm(a*k)-base).abs().max().item() for k in (.25, 1., 4., 10.)}
    return {"scaled_advantage_max_difference": results,
            "conditions": "Fixed samples and advantages scaled exactly; real critic fit, absolute value clipping and KL adaptation break exact training invariance.",
            "actor_critic_grad_norm_clipped_separately": True}


def target_math():
    # Hypothetical common RMSE reduction on both current official anchor policies.
    names = ["longrun_C0/C0_5500_ALL.json", "R3_bundle_extracted/eval_results/R3_R3A_9000.json"]
    baseline = [json_file(p) for p in names]
    results = []
    for factor in (1, .75, .5, .25, 0):
        values = {}
        for family in ("floor", "ratio", "raw3"):
            total = 60.
            for data in baseline:
                copied = json.loads(json.dumps(data))
                for s in copied["scenarios"].values():
                    s["rmse"] = {a: v*factor for a, v in s["rmse"].items()}
                total += score(copied, family)
            values[family] = total
        results.append({"all_three_axis_rmse_multiplier": factor, "combined_score_counterfactual": values})
    return {"counterfactuals": results, "floor_model_combined_limit": 60+140*math.exp(-math.sqrt(C)),
            "warning": "Neither empirical calibration family identifies the official implementation. These are NOT attainable-score predictions or competition limits."}


def prepare_conditional_candidates():
    source = read(RUN / "results/extracted/round20_20261001/F10_applied_rewards.py")
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in n.targets))
    weights = ast.literal_eval(assignment.value)
    env = yaml.load(read(RUN / "results/extracted/round20_20261001/exports/F10/env.yaml"), Loader=yaml.BaseLoader)
    assert float(env["rewards"]["dof_acc_l2"]["weight"]) == -2.5e-7
    assert weights["action_rate_l2"] == -.01
    def outside_ast(text):
        parsed = ast.parse(text)
        for n in parsed.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REWARD_WEIGHTS" for t in n.targets):
                n.value = ast.Constant(None)
        return ast.dump(parsed, include_attributes=False)
    plans = {}
    for name, term, value in (("R_action", "action_rate_l2", -.0075),
                              ("R_acc", "dof_acc_l2", -2e-7)):
        candidate = dict(weights)
        candidate[term] = value
        lines = source.splitlines(keepends=True)
        lines[assignment.lineno-1:assignment.end_lineno] = ["REWARD_WEIGHTS = " + repr(candidate) + "\n"]
        output = "".join(lines)
        assert outside_ast(output) == outside_ast(source)
        effective = {k: float(v["weight"]) for k,v in env["rewards"].items() if isinstance(v, dict)}
        actual = dict(effective)
        actual.update(candidate)
        changes = {k: [effective[k], actual[k]] for k in effective if effective[k] != actual[k]}
        assert changes == {term: [effective[term], value]}
        fn = OUT / f"{name}_rewards_PROPOSAL.py"
        fn.write_text(output, encoding="utf-8")
        plans[name] = {"status": "UNTRAINED_CONDITIONAL_PROPOSAL",
                       "effective_weight_change": changes, "outside_reward_ast_unchanged": True,
                       "file": str(fn.relative_to(ROOT)),
                       "sha256": hashlib.sha256(fn.read_bytes()).hexdigest()}
    return plans


def main():
    data, evaluations, changes = load_evaluations()
    result = {"status": "OFFLINE_RESEARCH_NOT_TRAINING_OR_OFFICIAL_SCORE",
              "evaluations": evaluations, "F10_vs_C0_moment_changes": changes,
              "tracking_kernel": kernel_audit(), "training_terms": training_tradeoff(),
              "global_scale": scale_audit(), "target_math": target_math(),
              "conditional_candidates": prepare_conditional_candidates()}
    # Identity bytes are handled separately because policy.pt is binary.
    p = ROOT / "analysis/score_reverse_20261001/official_h1/policy.pt"
    q = ROOT / "submit_R3/policy.pt"
    if p.exists() and q.exists():
        hp, hq = [hashlib.sha256(v.read_bytes()).hexdigest() for v in (p,q)]
        assert hp == hq
        INPUTS[str(p.relative_to(ROOT))] = hp
        INPUTS[str(q.relative_to(ROOT))] = hq
        result["H1_official_download_matches_R3A_submission"] = {"sha256": hp, "matches": True}
    else:
        result["H1_historical_identity_verification"] = json_file("analysis/score_reverse_20261001/official_h1/verification.json")
    result["input_sha256"] = INPUTS
    (OUT/"analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "G6_B_moment_change": changes["B"]["G6"],
        "GX1_moment_change": changes["stress"]["GX1"],
        "nominal_B_weighted_axis_mse": {k: evaluations[k]["weighted_axis_mse"] for k in ("C0_B","F10_B")},
        "kernel_counterexample": {k: v for k,v in result["tracking_kernel"].items() if k.endswith("example")},
        "target_math": result["target_math"],
        "H1_and_D1_scores": {k: evaluations[k]["scores"] for k in evaluations if "H1" in k or "2000" in k},
        "input_files": len(INPUTS),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
