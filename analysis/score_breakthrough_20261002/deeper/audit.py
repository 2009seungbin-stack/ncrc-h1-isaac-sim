"""Offline audit of logs, saved tensors, and the actual local evaluator contract.

Mock-environment and ideal-trajectory results are diagnostics, not robot rollouts.
No policy is modified and no Isaac Sim server is started.
"""
import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / "analysis/round20_execution_20261001/results/extracted/round20_20261001"
torch.set_num_threads(2)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def training_audit():
    report, all_rows = {}, {}
    tags = ["Mean action std", "Mean reward", "Mean value loss", "Curriculum/terrain_levels",
            "Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/track_ang_vel_z_exp",
            "Episode_Reward/lin_vel_z_l2", "Episode_Reward/ang_vel_xy_l2",
            "Episode_Reward/dof_torques_l2", "Episode_Reward/dof_acc_l2",
            "Episode_Reward/action_rate_l2", "Episode_Reward/feet_air_time",
            "Episode_Termination/base_contact"]
    for run in ["F00", "F10"]:
        p = RUN / f"logs/train_{run}.log"
        blocks = re.split(r"Learning iteration\s+(\d+)/(\d+)", p.read_text(encoding="utf-8", errors="replace"))
        rows = []
        for j in range(1, len(blocks), 3):
            row = {"iteration": int(blocks[j])}
            row.update({k: float(v) for k, v in re.findall(r"^\s*([^\n:]+):\s*([-+\d.eE]+)\s*$", blocks[j+2], re.M)})
            rows.append(row)
        assert len(rows) == 4000 and [r["iteration"] for r in rows] == list(range(4000))
        all_rows[run] = rows
        windows = []
        for start in range(1000, 4000, 500):
            sub = rows[start:start+500]
            windows.append({"start": start, "stop_exclusive": start+500,
                            "medians": {k: float(np.median([r[k] for r in sub])) for k in tags}})
        tail = rows[1000:]
        value = np.array([r["Mean value loss"] for r in tail])
        z = np.array([r["Episode_Reward/lin_vel_z_l2"] for r in tail])
        z_spike = z < -1
        near_spike = np.convolve(z_spike.astype(int), np.ones(7, dtype=int), mode="same") > 0
        high = value > 10
        extremes = {}
        for key in ["Mean value loss", "Episode_Reward/lin_vel_z_l2", "Episode_Reward/action_rate_l2"]:
            extreme = sorted(rows, key=lambda r: abs(r[key]), reverse=True)[:8]
            extremes[key] = [{"iteration": r["iteration"], "value": r[key]} for r in extreme]
        report[run] = {
            "rows": len(rows), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "windows": windows, "extremes": extremes,
            "cooccurrence_after_1000": {
                "value_loss_above_10": int(high.sum()),
                "vertical_reward_below_minus1": int(z_spike.sum()),
                "high_value_loss_within_3_iterations_of_vertical_spike": int((high & near_spike).sum()),
                "fraction_of_high_value_loss_near_vertical_spike": float((high & near_spike).sum()/max(high.sum(), 1)),
                "fraction_of_iterations_near_vertical_spike": float(near_spike.mean()),
                "warning": "Post-hoc descriptive overlap; smoothed/delayed logging and common causes prevent causal inference.",
            },
        }
    return report, all_rows


def checkpoint_audit():
    out = {}
    for run in ["F00", "F10"]:
        rows = []
        for p in sorted((RUN / "preserved" / run).glob("*/model_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
            ck = torch.load(p, map_location="cpu", weights_only=True)
            actor, critic = ck["actor_state_dict"], ck["critic_state_dict"]
            std = actor["distribution.std_param"]
            finite = all(torch.isfinite(v).all().item() for d in [actor, critic] for v in d.values() if isinstance(v, torch.Tensor))
            rows.append({"iteration": ck["iter"], "actor_critic_finite": finite,
                         "std_min": std.min().item(), "std_mean": std.mean().item(), "std_max": std.max().item(),
                         "optimizer_lr": [g["lr"] for g in ck["optimizer_state_dict"]["param_groups"]],
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        out[run] = rows
    return out


def stress_decomposition():
    base = ROOT / "analysis/round20_execution_20261001/stress_results/extracted/go2_stress_20261002/eval"
    output = {}
    for name in ["original_stairs_0.15", "selected_stairs_0.15", "original_stress", "selected_stress"]:
        path = base / name / "merged.json"
        scenarios = json.loads(path.read_text(encoding="utf-8"))["scenarios"]
        entries = {}
        for sid, sc in scenarios.items():
            axes = {}
            for axis, idx in [("vx", 0), ("vy", 1), ("yaw", 2)]:
                mse = float(np.mean(sc["mse_seg"][axis]))
                bias2 = float(np.mean([(c[idx]-v[idx])**2 for c, v in zip(sc["segment_cmds"], sc["segment_mean_actual"])]))
                axes[axis] = {"reported_rmse": sc["rmse"][axis], "equal_segment_mse": mse,
                              "equal_segment_mean_actual": float(np.mean([v[idx] for v in sc["segment_mean_actual"]])),
                              "equal_segment_bias_squared": bias2, "equal_segment_bias_fraction": bias2/mse}
            entries[sid] = dict(axes=axes, survived=sc["survived"], episodes=sc["episodes"])
        output[name] = dict(scenarios=entries, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output["warning"] = "Equal segment weighting of rounded aggregates; surviving sample counts vary and are not available. Residual includes time variation and between-environment differences, not proven oscillation."
    return output


class MockEnv:
    """Perfect command following with snapshots and timeout auto-reset, no physics."""
    def __init__(self, n, term, timeout):
        self.unwrapped = self
        self.step_dt = .02
        self.n, self.term, self.timeout = n, term, timeout
        self.robot = NS(data=NS(root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3)))
        self.scene = {"robot": self.robot}
        self.command_mismatches, self.timeouts = [], 0

    def seed(self, seed):
        pass

    def reset(self):
        self.i = 0
        self.robot.data.root_lin_vel_b.zero_()
        self.robot.data.root_ang_vel_b.zero_()
        return {"policy": self.term.vel_command_b.clone()}, {}

    def step(self, action):
        if not torch.equal(action, self.term.vel_command_b):
            self.command_mismatches.append(self.i)
        self.robot.data.root_lin_vel_b[:, :2] = action[:, :2]
        self.robot.data.root_ang_vel_b[:, 2] = action[:, 2]
        self.i += 1
        terminated = torch.zeros(self.n, dtype=torch.bool)
        truncated = torch.full((self.n,), self.timeout and self.i == 1000, dtype=torch.bool)
        if truncated.any():
            self.timeouts += 1
            self.robot.data.root_lin_vel_b.zero_()
            self.robot.data.root_ang_vel_b.zero_()
        return {"policy": self.term.vel_command_b.clone()}, torch.zeros(self.n), terminated, truncated, {}


def evaluator_contract():
    # Load the exact evaluator functions used for preserved results, not a rewritten scoring loop.
    evaluator_path = RUN / "t_eval.py"
    evaluator = load_module("audit_teval", evaluator_path)
    cases = {}
    for sid, cmds, seg_s in [
        ("constant", [(0.4, 0., 0.)]*4, 5.),
        ("switching", [(1., 0., 0.), (-.5, 0., 0.), (0., .5, 0.), (0., 0., -1.),
                       (1., 0., 1.), (0., -.5, 0.), (0., 0., 1.), (-.5, 0., -1.),
                       (1., .5, 0.), (0., 0., 0.)], 2.)]:
        for timeout in [False, True]:
            term = NS(vel_command_b=torch.zeros(4, 3), is_standing_env=torch.zeros(4, dtype=torch.bool))
            env = MockEnv(4, term, timeout)
            spec = dict(w=1., desc="contract test", terrain="flat", axes=("vx", "vy", "yaw"), cmds=cmds, seg_s=seg_s)
            evaluator.SCENARIOS = {sid: spec}
            args = NS(seeds="1", cmd_scale=1., seg_s=seg_s, push_mag=1., push_period_s=3.,
                      transient_s=.5 if sid == "constant" else 0., num_envs=4, axis_mode="all")
            value = evaluator.run_scenario(env, term, lambda obs: obs.clone(), sid, args, "cpu")
            cases[f"{sid}_timeout_{timeout}"] = dict(rmse=value["rmse"], score=value["score"],
                command_mismatch_steps=env.command_mismatches, timeout_calls=env.timeouts)
    assert cases["constant_timeout_False"]["rmse"]["vx"] == 0
    assert cases["constant_timeout_True"]["rmse"]["vx"] == .0133
    assert cases["switching_timeout_False"]["command_mismatch_steps"] == list(range(100, 1000, 100))
    # Validate the proposed command-refresh contract in the mock's 3-value observation.
    # This is not a ready-to-run robot patch; real observation term dimensions must be validated.
    tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_scenario")
    for node in ast.walk(fun):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "seg":
            node.body[1:1] = ast.parse("obs = obs.clone()\nobs[:, :3] = term.vel_command_b\n").body
    ns = vars(evaluator).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[])), "mock_contract_patch", "exec"), ns)
    for sid, cmds, seg_s in [("constant", [(0.4, 0., 0.)]*4, 5.),
                           ("switching", [(1.,0.,0.),(-.5,0.,0.),(0.,.5,0.),(0.,0.,-1.),
                                          (1.,0.,1.),(0.,-.5,0.),(0.,0.,1.),(-.5,0.,-1.),
                                          (1.,.5,0.),(0.,0.,0.)], 2.)]:
        term = NS(vel_command_b=torch.zeros(4, 3), is_standing_env=torch.zeros(4, dtype=torch.bool))
        env = MockEnv(4, term, timeout=False)  # Models padding episode timeout beyond the measured horizon.
        ns["SCENARIOS"] = {sid: dict(w=1., desc="corrected mock contract", terrain="flat",
                                    axes=("vx", "vy", "yaw"), cmds=cmds, seg_s=seg_s)}
        args = NS(seeds="1", cmd_scale=1., seg_s=seg_s, push_mag=1., push_period_s=3.,
                  transient_s=.5 if sid == "constant" else 0., num_envs=4, axis_mode="all")
        value = ns["run_scenario"](env, term, lambda obs: obs.clone(), sid, args, "cpu")
        cases[sid + "_corrected_mock_contract"] = dict(rmse=value["rmse"], score=value["score"],
                                                      command_mismatch_steps=env.command_mismatches)
        assert value["score"] == 1. and not env.command_mismatches
    return {"source": str(evaluator_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(), "cases": cases,
            "scope": "Verified evaluator behavior under a mock step/reset contract matching upstream. Real robot effect has not been replayed."}


def ideal_terrain_coverage():
    rng = np.random.default_rng(20261002)
    n = 32000
    # Equal origin allocation in a 4x4 grid, uniform heading and reset XY jitter.
    tile_idx = np.arange(n) % 16
    origins = np.stack(((tile_idx // 4 + .5)*8, (tile_idx % 4 + .5)*8), axis=1)
    start = origins + rng.uniform(-.5, .5, (n, 2))
    yaw = rng.uniform(-3.14, 3.14, n)
    direction = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
    num_steps = int((8 - 2*1 - 3)//(2*.3) + 1)
    central_half_width = (8 - 2*1 - 2*num_steps*.3)/2
    output = {}
    for name, speed in [("G5_stairs", .4), ("G4_slope", .5)]:
        inside_sum = stair_zone_sum = evaluated = 0
        left_origin = np.zeros(n, bool)
        end_outside = None
        for i in range(1, 1001):
            pos = start + direction*(speed*i*.02)
            inside = ((pos >= 0) & (pos < 32)).all(axis=1)
            left_origin |= (np.abs(pos-origins) > 4).any(axis=1)
            end_outside = ~inside
            # Match existing transient exclusion: first 25 ticks of every 250 tick segment.
            if (i-1) % 250 < 25:
                continue
            inside_sum += inside.sum()
            local_max = np.abs(pos % 8 - 4).max(axis=1)
            stair_zone_sum += (inside & (local_max >= central_half_width) & (local_max < 3)).sum()
            evaluated += n
        output[name] = {
            "nominal_distance_m": speed*20, "fraction_scored_outside_4x4_tile_grid": float(1-inside_sum/evaluated),
            "fraction_trajectories_ever_leaving_spawn_tile": float(left_origin.mean()),
            "fraction_trajectories_outside_grid_at_end": float(end_outside.mean()),
        }
        if name == "G5_stairs":
            output[name]["fraction_scored_inside_stair_ring_plan_area"] = float(stair_zone_sum/evaluated)
    return {"n": n, "rng_seed": 20261002, "stair_steps_from_config": num_steps,
            "central_platform_half_width_m": central_half_width, "cases": output,
            "warning": "Ideal straight trajectories, no dynamics, collisions, foot contacts, or real policy. Plan-area coverage is not measured stair exposure."}


def main():
    logs, rows = training_audit()
    report = dict(training=logs, checkpoints=checkpoint_audit(), stress=stress_decomposition(), evaluator_contract=evaluator_contract(),
                  ideal_terrain_coverage=ideal_terrain_coverage())
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/"sources").glob("*.py")}
    report["limitations"] = ["No new robot rollout, training, or official score.",
        "Training log medians are not deterministic-policy evaluation metrics.",
        "Mock tests identify sampling behavior, not its effect on current candidate ranking."]
    (OUT/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/"parsed_training.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cooccurrence": {r: logs[r]["cooccurrence_after_1000"] for r in logs},
                      "checkpoint_counts": {r: len(v) for r,v in report["checkpoints"].items()},
                      "evaluator": report["evaluator_contract"], "coverage": report["ideal_terrain_coverage"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
