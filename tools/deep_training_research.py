"""Offline follow-up: error mechanisms, learning curves and policy sensitivity.

Uses existing evaluation summaries, event logs, full checkpoints and env YAML.
CPU probes use synthetic observations, NOT simulator rollouts or score estimates.
Writes only analysis/deep_research_20261001. No network, training or submission.
Run: python -X utf8 tools/deep_training_research.py
"""
import copy
import json
import math
import statistics
from pathlib import Path

import torch
import yaml

from analyze_training_efficiency import module, sha, weights, change, nonreward_ast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/deep_research_20261001"
C = .02684614570005177
AXES = ("vx", "vy", "yaw")


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def score(data, family="floor"):
    result = 0.
    for s in data["scenarios"].values():
        if s["w"] == 0:
            continue
        mse = sum(s["rmse"][a]**2 for a in AXES)/3
        if family == "floor":
            q = math.exp(-math.sqrt(mse + C))
        elif family == "raw3":
            q = math.exp(-math.sqrt(mse))
        elif family == "ratio":
            q = .905*math.exp(-math.sqrt(mse))
        else:
            extra = s.get("rmse_extra")
            if extra is None:
                return None
            q = math.exp(-math.sqrt((3*mse + sum(extra[a]**2 for a in ("vz", "wx", "wy")))/6))
        result += 70*s["w"]*s["survival"]*q
    return result


def summarize(data):
    main = {k: s for k, s in data["scenarios"].items() if s["w"] > 0}
    return {"scores": {f: score(data, f) for f in ("raw3", "raw6", "floor", "ratio")},
            "weighted_survival": sum(s["w"]*s["survival"] for s in main.values()),
            "weighted_axis_rmse": {a: sum(s["w"]*s["rmse"][a] for s in main.values()) for a in AXES},
            "stress": {k: {"survival": s["survival"], "episodes": s["episodes"],
                            "falls": s.get("falls")} for k, s in data["scenarios"].items() if not s["w"]},
            "scenario_scores": {k: s["score"] for k, s in main.items()}, "config": data.get("config")}


def decomposition(data):
    totals = {a: dict(mse=0., mean_bias_squared=0., zero_command_mse=0.) for a in AXES}
    per_scenario = {}
    for sid, s in data["scenarios"].items():
        if not s["w"]:
            continue
        # C0 has no falls; all four equal-duration segments contribute equally.
        assert s["survival"] == 1 and len(s["segment_cmds"]) == 4
        per_scenario[sid] = {}
        for a, j in zip(AXES, range(3)):
            mse = sum(s["mse_seg"][a])/4
            b = sum((cmd[j]-actual[j])**2 for cmd, actual in
                    zip(s["segment_cmds"], s["segment_mean_actual"]))/4
            z = sum(m for cmd, m in zip(s["segment_cmds"], s["mse_seg"][a]) if cmd[j] == 0)/4
            assert 0 <= b < mse
            per_scenario[sid][a] = dict(mse=mse, mean_bias_squared=b,
                                        within_segment_variance=mse-b, zero_command_mse=z)
            for key, value in (("mse", mse), ("mean_bias_squared", b), ("zero_command_mse", z)):
                totals[a][key] += s["w"]*value
    for t in totals.values():
        t["mean_bias_fraction"] = t["mean_bias_squared"]/t["mse"]
        t["within_segment_variance_fraction"] = 1-t["mean_bias_fraction"]
        t["zero_command_mse_fraction"] = t["zero_command_mse"]/t["mse"]
    cf = {}
    for condition in ("yaw_bias_removed", "all_bias_removed", "nonzero_yaw_error_removed"):
        changed, reference = copy.deepcopy(data), copy.deepcopy(data)
        for sid, s in changed["scenarios"].items():
            if not s["w"]:
                continue
            for a in AXES:
                t = per_scenario[sid][a]
                new = t["mse"]
                reference["scenarios"][sid]["rmse"][a] = math.sqrt(new)
                if condition == "all_bias_removed" or condition == "yaw_bias_removed" and a == "yaw":
                    new -= t["mean_bias_squared"]
                if condition == "nonzero_yaw_error_removed" and a == "yaw":
                    new = t["zero_command_mse"]
                s["rmse"][a] = math.sqrt(max(0, new))
        cf[condition] = {f: score(changed, f)-score(reference, f) for f in ("floor", "ratio")}
    def zero_yaw_score(factor, family):
        changed = copy.deepcopy(data)
        for s in changed["scenarios"].values():
            if not s["w"]:
                continue
            s["rmse"]["yaw"] = math.sqrt(sum(m*(factor**2 if cmd[2] == 0 else 1)
                for m, cmd in zip(s["mse_seg"]["yaw"], s["segment_cmds"]))/4)
            for a in ("vx", "vy"):
                s["rmse"][a] = math.sqrt(sum(s["mse_seg"][a])/4)
        return score(changed, family)
    targets = {}
    for family in ("floor", "ratio"):
        baseline = zero_yaw_score(1, family)
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo+hi)/2
            if zero_yaw_score(mid, family)-baseline > .25:
                lo = mid
            else:
                hi = mid
        targets[family] = {"zero_command_yaw_rmse_reduction_for_proxy_gain_0.25": 1-(lo+hi)/2,
                           "gain_if_zero_command_yaw_rmse_minus20pct": zero_yaw_score(.8, family)-baseline,
                           "gain_if_zero_command_yaw_rmse_minus30pct": zero_yaw_score(.7, family)-baseline}
    return {"weighted_axis": totals, "by_scenario": per_scenario, "counterfactuals": cf,
            "conditional_zero_command_targets": targets,
            "limitations": "Rounded segment means/MSE. Variance includes gait, transients, environment variation, observation-noise response and other effects; no frequency or noise-cause identification."}


def event_stats(parser, path, start):
    tags = parser.read_tfevents(ROOT / path)
    out = {}
    for key, values in tags.items():
        if key.startswith(("Episode_Reward/", "Perf/", "Loss/", "Policy/", "Curriculum/")):
            finite = [v for s, v in values.items() if s >= start and math.isfinite(v)]
            if finite:
                out[key] = {"median": statistics.median(finite), "min": min(finite),
                            "max": max(finite), "samples": len(finite)}
    return {"path": path, "sha256": sha(ROOT / path), "step_min": start, "tags": out}


def policy_probe(robot, checkpoint, jit_path, env_path):
    env = yaml.load((ROOT / env_path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    ck = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=True)
    sd = ck["actor_state_dict"]
    width, joints = sd["mlp.0.weight"].shape[1], sd["mlp.6.weight"].shape[0]
    net = torch.nn.Sequential(torch.nn.Linear(width, 512), torch.nn.ELU(),
                              torch.nn.Linear(512, 256), torch.nn.ELU(),
                              torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, joints))
    net.load_state_dict({k.removeprefix("mlp."): v for k, v in sd.items() if k.startswith("mlp.")})
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    dims = [3, 3, 3, 3, joints, joints, joints, width-(12+3*joints)]
    names = ["base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
             "joint_pos", "joint_vel", "actions", "height_scan"]
    actual_order = [k for k, v in env["observations"]["policy"].items() if isinstance(v, dict) and "func" in v]
    assert names == actual_order and dims[-1] == 187
    slices, offset = {}, 0
    for name, size in zip(names, dims):
        slices[name] = slice(offset, offset+size)
        offset += size
    commands = [(0,0,0), (.5,0,0), (-.5,0,0), (0,.3,0), (0,0,.5), (0,0,-.5), (.5,0,.5)]
    observations = []
    for height in (-.2, 0., .3, .5):
        for command in commands:
            x = torch.zeros(width)
            x[slices["base_lin_vel"]][:2] = torch.tensor(command[:2])
            x[slices["base_ang_vel"]][2] = command[2]
            x[slices["projected_gravity"]][2] = -1
            x[slices["velocity_commands"]] = torch.tensor(command)
            x[slices["height_scan"]] = height
            observations.append(x)
    x = torch.stack(observations)
    with torch.inference_mode():
        baseline = net(x)
        jit = torch.jit.load(str(ROOT / jit_path), map_location="cpu").eval()
        exported = jit(x)
    parity = float((baseline-exported).abs().max())
    assert parity < 1e-5 and torch.isfinite(baseline).all()
    action_scale = float(env["actions"]["joint_pos"]["scale"])
    noise_terms = {}
    for name in names:
        term = env["observations"]["policy"][name]
        if term.get("noise") not in (None, "null"):
            noise_terms[name] = term["noise"]
    results = {}
    generator = torch.Generator().manual_seed(20261001)
    for selected in list(noise_terms) + ["all_observation_noise"]:
        batch = x.repeat_interleave(128, dim=0)
        for name, cfg in noise_terms.items():
            if selected not in (name, "all_observation_noise"):
                continue
            lo, hi = float(cfg["n_min"]), float(cfg["n_max"])
            sl = slices[name]
            batch[:, sl] += lo+(hi-lo)*torch.rand(batch[:, sl].shape, generator=generator)
            clip = env["observations"]["policy"][name].get("clip")
            if isinstance(clip, list):
                batch[:, sl].clamp_(float(clip[0]), float(clip[1]))
        with torch.inference_mode():
            delta = net(batch)-baseline.repeat_interleave(128, dim=0)
        assert torch.isfinite(delta).all()
        per_state = delta.reshape(len(x), 128, joints).square().mean((1,2)).sqrt()
        results[selected] = {"raw_action_rms": delta.square().mean().sqrt().item(),
                             "joint_target_delta_rms_rad": action_scale*delta.square().mean().sqrt().item(),
                             "max_state_joint_target_delta_rms_rad": action_scale*per_state.max().item()}
    std2 = float(sd["distribution.std_param"].square().sum())
    action_rate = float(env["rewards"]["action_rate_l2"]["weight"])
    return {"robot": robot, "checkpoint": checkpoint, "checkpoint_sha256": sha(ROOT / checkpoint),
            "jit": jit_path, "jit_forward_parity_max_abs": parity, "state_count": len(x),
            "noise_samples_per_state": 128, "seed": 20261001, "input_width": width,
            "action_scale": action_scale, "noise_probe": results,
            "independent_exploration_noise_reference": {"sum_sigma_squared": std2,
                "expected_delta_action_squared_fixed_mean": 2*std2,
                "weighted_action_rate_term_fixed_mean": action_rate*2*std2},
            "limitations": "Synthetic upright/default-joint observations, not rollout states. Action sensitivity is not body-velocity error or official score. Independent exploration estimate assumes fixed mean and omits policy feedback correlations."}


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    parser = module("event_parser", "tools/score_model.py")
    paths = ["longrun_C0/C0_800_ALL.json", "longrun_C0/C0_2000_ALL.json",
             "longrun_C0/C0_4000_ALL.json", "longrun_C0/C0_5500_ALL.json",
             "longrun_D1/eval/D1_1000_ALL.json", "longrun_D1/eval/D1_2000_ALL.json",
             "longrun_D1/eval/D1_2499_ALL.json"]
    paths += [f"R3_bundle_extracted/eval_results/R3_{config}_{step}.json"
              for config in ("R3A", "R3C") for step in (9000,10000)]
    hashes = {p: sha(ROOT/p) for p in paths}
    datasets = {p: load(p) for p in paths}
    curves = {Path(p).stem: summarize(d) for p, d in datasets.items()}
    c0 = datasets["longrun_C0/C0_5500_ALL.json"]
    result = {"date": "2026-10-01", "status": "OFFLINE_RESEARCH_NOT_TRAINING_RESULTS",
              "input_sha256": hashes, "learning_curves": curves,
              "go2_error_decomposition": decomposition(c0), "event_stats": {},
              "policy_sensitivity": {}, "proxy_warning": "Two-policy calibration; all floor/ratio estimates are local empirical proxies, not identified official evaluator.",
              "server_started": False}
    event_paths = {"Go2_D1": ("longrun_D1/logs/events.out.tfevents.1789946121.da-perfect13.11143.0",2000),
                   "H1_R3A": ("R3_bundle_extracted/logs/rsl_rl/humanoid/2026-09-08_13-15-40/events.out.tfevents.1788840956.da-perfect28.16598.0",9000)}
    for name, (path, start) in event_paths.items():
        result["event_stats"][name] = event_stats(parser, path, start)
    for robot, checkpoint, jit, env in (
        ("Go2", "longrun_C0/model_5500.pt", "longrun_C0/C0_5500_policy.pt", "submit_C0/env.yaml"),
        ("H1", "R3_bundle_extracted/keep_R3/R3A_9000_model.pt", "R3_bundle_extracted/keep_R3/R3A_9000.pt", "submit_R3/env.yaml")):
        result["policy_sensitivity"][robot] = policy_probe(robot, checkpoint, jit, env)
    result["observed_marginal_learning"] = {
        "Go2_4000_to_5500": {"floor_delta": curves["C0_5500_ALL"]["scores"]["floor"]-curves["C0_4000_ALL"]["scores"]["floor"],
                             "additional_iterations": 1500, "hours_planning": 2.0},
        "H1_R3A_9000_to_10000": {"floor_delta": curves["R3_R3A_10000"]["scores"]["floor"]-curves["R3_R3A_9000"]["scores"]["floor"],
                                   "additional_iterations": 1000,
                                   "strong_push_survival_change": curves["R3_R3A_10000"]["stress"]["X1"]["survival"]-curves["R3_R3A_9000"]["stress"]["X1"]["survival"]},
        "limitation": "Historical single runs, no extrapolation or training-seed CI; hardware/runtime differ by robot/run."}
    result["performance_plan_correction"] = {
        "withdrawn": "Go2 8192/12288 env count benchmark under current configuration",
        "evidence": {"tools/go2/sweep_go2.sh:9": "8192 env: contact patch capacity 327680 exceeded from iteration14, according to the recorded experiment comment",
                     "r4_sweep_results_20260916.md:5": "12288-env training comparator had overflow, according to the experiment report"},
        "safe_candidate_counts_pending_live_check": [2048, 4096, 6144],
        "equal_transition_example": {"2048_env_iterations": 3000, "4096_env_iterations": 1500,
                                     "6144_env_iterations": 1000, "total_transitions": 147456000},
        "limitation": "Recorded overflow not reproduced this turn. Matched transitions do not match PPO update count or curriculum exposure; evaluate improvement per wall hour as well as FPS. Do not change physics buffers or host settings under reward-only scope."}
    base = (ROOT/"analysis/training_efficiency_20261001/Go2_CONTROL_rewards.py").read_text(encoding="utf-8")
    candidates = []
    for cid, yaw, action in (("F00", .75, -.01), ("F10", 1.5, -.01),
                             ("F01", .75, -.015), ("F11", 1.5, -.015)):
        candidate = change(change(base, "track_ang_vel_z_exp", yaw), "action_rate_l2", action)
        assert nonreward_ast(candidate) == nonreward_ast(base)
        path = OUT/f"{cid}_rewards.py"
        path.write_text(candidate, encoding="utf-8", newline="\n")
        candidates.append({"id": cid, "yaw_reward": yaw, "action_rate_penalty": action,
                           "path": str(path.relative_to(ROOT)), "weights": weights(candidate),
                           "sha256": sha(path), "status": "NOT_TRAINED"})
    result["go2_factorial_candidates"] = candidates
    result["factorial_contrasts"] = {"yaw_effect": "((F10-F00)+(F11-F01))/2",
        "action_effect": "((F01-F00)+(F11-F10))/2", "interaction": "F11-F10-F01+F00",
        "restriction": "Same parent, update count, env count, curriculum reset and evaluation seeds. One realization per cell is exploratory; interaction needs independent confirmation."}
    result["decision_plan"] = [
        "First diagnose C0 unchanged with paired G1/G2/G5 evaluation, observation noise on/off using existing --no_obs_noise; noise-off is diagnostic only, never a submission-ranking score.",
        "Preserve old body-frame evaluation. H1 training uses yaw-aligned linear and world-z angular reward; supplemental frame-aware telemetry is needed before interpreting H1 yaw/tilt changes.",
        "Check wrapper-supported resume, read source/imports, do not run train.py --help. C0 50-100 unchanged-update pilot tests immediate health only, not 1000-update safety.",
        "If C0 pilot works: four factorial cells independently from original C0 parent, 1000 extra iterations each, approximately 5.33 training hours at historical Go2 speed. Track critic loss, reward spikes, action std and preserved checkpoint health.",
        "If pilot fails: create one fresh C0 common trunk 2000-2500 updates, save full state, then branch all cells. Never transplant/reset optimizer, std or critic tensors.",
        "Screen all seven scenarios identically; record zero/nonzero command yaw, segment means/variance, action variation, falls and 3/6-axis metrics. Low-fidelity negative result cannot eliminate terrain candidates until curriculum recovery is checked.",
        "Extend only winner and matched control by 1000 further updates. Restarting a run resets curriculum again, so this is another matched stage, not seamless continuation. Independent seed/parent confirmation precedes promotion.",
        "Add one gait variant only when contact/video evidence supports it; avoid a blind giant airtime jump. H1 angular/noise diagnostics precede HY30, and unchanged H1 long training is low priority.",
        "Freeze two proxy predictions before official submission. Judge actual official uplift after scoring; do not treat noise-off, synthetic policy sensitivity or stress score as the official objective.",
        "For speed testing use only 2048/4096/6144 envs with existing supported CLI; above-6144 suggestion withdrawn due recorded overflow. Match transitions and record update count, curriculum and score per wall hour.",
    ]
    result["research_sources"] = {
        "CAPS_temporal_spatial_smoothness_not_equivalent_to_reward_penalty": "https://arxiv.org/html/2012.06644v2",
        "Hyperband_adaptive_budget_not_a_guarantee_for_this_curriculum": "https://www.jmlr.org/papers/v18/16-558.html",
        "reward_functions": "https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/envs/mdp/rewards.py",
        "H1_reward_frames": "https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py",
        "Go2_value_loss_failure_report_not_our_root_cause_proof": "https://github.com/isaac-sim/IsaacLab/issues/1999",
        "action_penalty_clipping_did_not_fix_reported_failure": "https://github.com/isaac-sim/IsaacLab/issues/4570",
        "reward_interactions_and_safety_tracking_tradeoff": "https://arxiv.org/html/2309.14246v2",
    }
    assert hashes == {p: sha(ROOT/p) for p in paths}
    result["original_eval_input_hashes_unchanged"] = True
    (OUT/"analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "decomposition": result["go2_error_decomposition"]["weighted_axis"],
                      "counterfactuals": result["go2_error_decomposition"]["counterfactuals"],
                      "policy_probes": {k: v["noise_probe"] for k,v in result["policy_sensitivity"].items()},
                      "marginal_learning": result["observed_marginal_learning"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
