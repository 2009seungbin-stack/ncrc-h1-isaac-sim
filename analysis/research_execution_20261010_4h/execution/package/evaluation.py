#!/usr/bin/env python3
"""Separate B1 evaluator. Importing this file never starts Isaac or loads a policy.

One terrain process at a time. Runtime reuses the frozen legacy Go2 commands and
environment constructor, not its rounded scoring/NaN substitution. --merge is
CPU only. Output is models.json plus explicit completion/failure evidence.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

AXES = ("vx", "vy", "yaw")
STEPS = 1000
DT = .02
PHASES = {
    "screen": dict(scenarios=("G1", "G3", "G5", "G6"), num_envs=64,
                   seeds=(2029,), startup_seed=404, terrain_seed=10037),
    "selection": dict(scenarios=("G1", "G3", "G5", "G6"), num_envs=64,
                      seeds=(2029,), startup_seed=404, terrain_seed=10037),
    "validation": dict(scenarios=("G1", "G2", "G3", "G4", "G5", "G6", "G7"),
                       num_envs=128, seeds=(2039, 2063), startup_seed=409, terrain_seed=11003),
}
TERRAINS = {"G1": "flat", "G2": "flat", "G3": "rough", "G4": "slope",
            "G5": "stairs", "G6": "flat", "G7": "dr"}
TRACE_FIELDS = ("projected_gravity", "body_quaternion_w", "foot_force_w", "foot_contact",
                "joint_position", "joint_velocity", "applied_torque", "body_linear_velocity",
                "body_angular_velocity", "world_linear_velocity", "world_angular_velocity",
                "raw_action", "clipped_action", "processed_joint_action")


class EvaluationInvalid(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    os.replace(tmp, path)


def require_finite(value, label):
    import numpy as np
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise EvaluationInvalid("Nonfinite " + label)
    return value


class ErrorAudit:
    """Exact float64 per-episode sums; terminal/timeout step itself is excluded."""
    def __init__(self, episodes):
        import numpy as np
        self.episodes = episodes
        self.active = np.ones(episodes, dtype=bool)
        self.counts = np.zeros(episodes, dtype=np.int64)
        self.sse = np.zeros((episodes, 3), dtype=np.float64)
        self.sae = np.zeros((episodes, 3), dtype=np.float64)
        self.segment_sse = np.zeros((4, episodes, 3), dtype=np.float64)
        self.segment_sae = np.zeros((4, episodes, 3), dtype=np.float64)
        self.segment_counts = np.zeros((4, episodes), dtype=np.int64)
        self.first_terminal = [None] * episodes
        self.first_timeout = [None] * episodes
        self.completed = 0
        self.error_history = []
        self.mask_history = []

    def step(self, actual, command, terminated, truncated):
        import numpy as np
        index = self.completed
        if index >= STEPS:
            raise EvaluationInvalid("Too many control steps")
        actual = require_finite(actual, "tracking velocity").astype(np.float64)
        command = require_finite(command, "command").astype(np.float64)
        if actual.shape != (self.episodes, 3) or command.shape != (3,):
            raise EvaluationInvalid("Tracking/command shape mismatch")
        term = require_finite(terminated, "termination flag").astype(bool)
        timeout = require_finite(truncated, "timeout flag").astype(bool)
        if term.shape != (self.episodes,) or timeout.shape != (self.episodes,):
            raise EvaluationInvalid("Event shape mismatch")
        new_terminal = self.active & term
        new_timeout = self.active & timeout
        for env_id in np.flatnonzero(new_terminal):
            self.first_terminal[int(env_id)] = index
        for env_id in np.flatnonzero(new_timeout):
            self.first_timeout[int(env_id)] = index
        if new_timeout.any() and index != STEPS - 1:
            raise EvaluationInvalid("Early timeout at control step " + str(index))
        self.active &= ~(term | timeout)
        mask = self.active.copy()
        error = actual - command
        self.error_history.append(error.copy())
        self.mask_history.append(mask.copy())
        squared, absolute = error ** 2, np.abs(error)
        squared[~mask] = 0
        absolute[~mask] = 0
        self.sse += squared
        self.sae += absolute
        self.counts += mask
        segment = index // 250
        self.segment_sse[segment] += squared
        self.segment_sae[segment] += absolute
        self.segment_counts[segment] += mask
        self.completed += 1
        return mask

    def result(self, spec=None):
        import numpy as np
        if self.completed != STEPS:
            raise EvaluationInvalid("Incomplete control horizon")
        for count, terminal, timeout in zip(self.counts, self.first_terminal, self.first_timeout):
            events = [e for e in (terminal, timeout) if e is not None]
            if int(count) != (min(events) if events else STEPS):
                raise EvaluationInvalid("Per-episode coverage mismatch")
        n = int(self.counts.sum())
        if n <= 0:
            raise EvaluationInvalid("No valid tracking samples")
        total_sse, total_sae = self.sse.sum(axis=0), self.sae.sum(axis=0)
        metrics = []
        for i in range(self.episodes):
            count = int(self.counts[i])
            metrics.append(dict(env_id=i, valid_samples=count,
                first_terminal_step=self.first_terminal[i], first_timeout_step=self.first_timeout[i],
                sum_squared_error=dict(zip(AXES, self.sse[i].tolist())),
                sum_absolute_error=dict(zip(AXES, self.sae[i].tolist())),
                rmse={a: math.sqrt(float(self.sse[i, j]) / count) if count else None for j, a in enumerate(AXES)},
                mae={a: float(self.sae[i, j]) / count if count else None for j, a in enumerate(AXES)}))
        distribution = {}
        for axis in AXES:
            values = [r["rmse"][axis] for r in metrics if r["valid_samples"]]
            distribution[axis] = dict(metric="per_episode_full_window_RMSE", units="rad/s" if axis == "yaw" else "m/s",
                quantiles={str(q): float(np.quantile(values, q)) for q in (0, .5, .9, .95, .99, 1)},
                episodes_with_samples=len(values), episodes_without_samples=self.episodes-len(values))
        segments = []
        for i in range(4):
            segments.append(dict(segment=i, valid_samples=int(self.segment_counts[i].sum()),
                sum_squared_error=dict(zip(AXES, self.segment_sse[i].sum(axis=0).tolist())),
                sum_absolute_error=dict(zip(AXES, self.segment_sae[i].sum(axis=0).tolist()))))
        survived = sum(t is None for t in self.first_terminal)
        return dict(episodes=self.episodes, survived=survived, falls=self.episodes-survived,
            survival=survived/self.episodes, valid_samples=n, sum_squared_error=dict(zip(AXES, total_sse.tolist())),
            sum_absolute_error=dict(zip(AXES, total_sae.tolist())),
            rmse={a: math.sqrt(float(total_sse[j])/n) for j, a in enumerate(AXES)},
            mae={a: float(total_sae[j])/n for j, a in enumerate(AXES)},
            expected_steps_per_episode=STEPS, planned_control_steps=STEPS, completed_control_steps=self.completed,
            per_episode_valid_samples=self.counts.tolist(), first_terminal_steps=self.first_terminal,
            first_timeout_steps=self.first_timeout, per_episode_metrics_reported=True,
            per_episode_metrics=metrics, per_episode_error_distribution=distribution,
            segment_error_sums=segments, nonfinite_count=0,
            tracking_diagnostics=tracking_diagnostics(self.error_history, self.mask_history, spec),
            mask="first terminal/timeout tick and all later reset trajectory excluded; initial/transition windows included")


def tracking_diagnostics(errors, valid_masks, spec=None):
    """Pure arithmetic on the exact full-window mask; never changes score inputs."""
    import numpy as np
    error = require_finite(errors, "recorded tracking errors").astype(np.float64)
    valid = np.asarray(valid_masks, dtype=bool)
    if error.shape[0] != STEPS or error.ndim != 3 or error.shape[2] != 3 or valid.shape != error.shape[:2]:
        raise EvaluationInvalid("Diagnostic error/mask shape mismatch")
    z = error[valid]
    if not len(z):
        raise EvaluationInvalid("No valid diagnostic samples")
    full_sse = (z**2).sum(0)
    distributions = {}
    for i, axis in enumerate(AXES):
        absolute = np.abs(z[:, i]); squared = z[:, i]**2
        n = max(1, math.ceil(len(z)*.01))
        top = np.partition(squared, -n)[-n:]
        distributions[axis] = dict(units="rad/s" if axis == "yaw" else "m/s",
            full_valid_samples=len(z), signed_mean_actual_minus_command=float(z[:, i].mean()),
            sample_absolute_error_quantiles={str(q): float(np.quantile(absolute, q)) for q in (.5, .9, .95, .99, .999)},
            top1pct_sample_SSE_fraction=float(top.sum()/full_sse[i]) if full_sse[i] > 0 else 0.)
    result = dict(full_window_sample_error_distribution=distributions,
        score_inputs_unchanged=True, diagnostic_windows_are_not_ranked=True,
        raw_error_definition="post-step body actual[vx,vy,yaw] minus command; terminal/timeout step excluded",
        command_boundary_semantics="legacy cached observation retained; first actor input after boundary may contain old command")
    if spec is None:
        return result
    if len(spec["cmds"]) != 4:
        raise EvaluationInvalid("Diagnostic command segment count changed")
    masks = {k: np.zeros(STEPS, bool) for k in ("initial", "true_command_transition", "post_push", "ordinary")}
    masks["initial"][:25] = True
    for segment in (1, 2, 3):
        if tuple(spec["cmds"][segment]) != tuple(spec["cmds"][segment-1]):
            masks["true_command_transition"][segment*250:segment*250+25] = True
    if spec.get("push"):
        for after in range(125, STEPS, 125):
            masks["post_push"][after+1:min(after+26, STEPS)] = True
    masks["ordinary"] = ~(masks["initial"] | masks["true_command_transition"] | masks["post_push"])
    if sum(int(m.sum()) for m in masks.values()) != STEPS:
        raise EvaluationInvalid("Diagnostic phase windows overlap")
    phases = {}
    for name, ticks in masks.items():
        phase_valid = valid & ticks[:, None]; samples = int(phase_valid.sum())
        phase_errors = error[phase_valid]; sse = (phase_errors**2).sum(0)
        phases[name] = dict(valid_samples=samples,
            full_window_sample_fraction=samples/len(z),
            sum_squared_error=dict(zip(AXES, sse.tolist())),
            full_axis_SSE_fraction={axis: float(sse[i]/full_sse[i]) if full_sse[i] > 0 else 0. for i, axis in enumerate(AXES)},
            signed_mean_actual_minus_command={axis: float(phase_errors[:, i].mean()) if samples else None for i, axis in enumerate(AXES)},
            RMSE={axis: math.sqrt(float(sse[i])/samples) if samples else None for i, axis in enumerate(AXES)})
    transitions = []
    for segment in (1, 2, 3):
        delta = float(spec["cmds"][segment][0]-spec["cmds"][segment-1][0])
        if delta == 0:
            continue
        start = segment*250; direction = 1. if delta > 0 else -1.; windows = {}
        for ticks in (25, 50):
            phase_valid = valid[start:start+ticks]
            e = error[start:start+ticks, :, 0][phase_valid]
            squared = e**2; total = float(squared.sum()); aligned = direction*e
            windows[str(ticks*DT)] = dict(valid_samples=len(e),
                mean_actual_minus_command=float(e.mean()) if len(e) else None,
                mean_direction_aligned_error=float(aligned.mean()) if len(e) else None,
                RMSE=math.sqrt(total/len(e)) if len(e) else None,
                lag_side_SSE_fraction=float(squared[aligned < 0].sum()/total) if total else 0.,
                past_target_side_SSE_fraction=float(squared[aligned > 0].sum()/total) if total else 0.)
        transitions.append(dict(boundary_step_index=start, previous_vx=spec["cmds"][segment-1][0],
            current_vx=spec["cmds"][segment][0], direction=direction, diagnostic_windows=windows))
    result.update(phases_0p5s=phases, vx_actual_command_changes=transitions,
        ordinary_yaw=phases["ordinary"]["RMSE"]["yaw"],
        phase_definition="Initial0..24; unequal command boundaries250/500/750 first25 ticks; G6 push response126/251/...876 first25 ticks; rest ordinary. Repeated identical commands are not transitions.",
        direction_definition="d=sign(newvx-oldvx); d*error<0 is lag-side, >0 is target-past side. Diagnostic sign labels are not identified dynamics or causal delay.")
    return result


def protocol(args):
    return dict(axis_mode="all", transient_seconds=0.0, command_scale=1.0,
                observation_noise_enabled=True, num_envs=args.num_envs, startup_seed=args.startup_seed,
                terrain_seed=args.terrain_seed, segment_seconds=5.0, control_step_seconds=DT)


def validate_args(args):
    frozen = PHASES[args.phase]
    seeds = tuple(int(v.strip()) for v in args.seeds.split(","))
    scenarios = tuple(v.strip().upper() for v in args.scenarios.split(","))
    if len(set(scenarios)) != len(scenarios) or not scenarios or not set(scenarios).issubset(frozen["scenarios"]):
        raise EvaluationInvalid("Scenario selection disagrees with frozen phase")
    if (args.num_envs, seeds, args.startup_seed, args.terrain_seed) != (
            frozen["num_envs"], frozen["seeds"], frozen["startup_seed"], frozen["terrain_seed"]):
        raise EvaluationInvalid("Environment/seed protocol changed")
    kinds = {TERRAINS[s] for s in scenarios}
    if len(kinds) != 1:
        raise EvaluationInvalid("A simulator process must contain exactly one terrain group")
    if args.task != "Quadruped-v0":
        raise EvaluationInvalid("Task registration differs from authorized task")
    return scenarios, seeds, next(iter(kinds))


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    models = manifest["models"] if isinstance(manifest, dict) else manifest
    if not isinstance(models, list) or not models:
        raise EvaluationInvalid("Manifest must contain model list")
    names = set()
    for item in models:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(item.get("name", ""))) or item["name"] in names:
            raise EvaluationInvalid("Invalid/duplicate model name")
        names.add(item["name"])
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        item["checkpoint"] = str(checkpoint.resolve())
        expected = item.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(item["checkpoint"]) != expected:
            raise EvaluationInvalid("Checkpoint SHA256 mismatch: " + item["name"])
        if item.get("actor_observation_normalization", False) is not False:
            raise EvaluationInvalid("Normalized actors are not supported by frozen legacy loader")
        clip = item.get("policy_clip_actions")
        if clip is not None and (not isinstance(clip, (float, int)) or not math.isfinite(clip) or clip <= 0):
            raise EvaluationInvalid("Invalid effective policy action clipping")
    return models


def source_audit(path, train_dir):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("ok") is not True or not value.get("provided") or not value.get("runtime"):
        raise EvaluationInvalid("Current provided/PPO source audit not verified")
    if set(value["provided"]) != {"train.py", "play.py", "go2_task/agent_cfg.py", "go2_task/env_cfg.py"}:
        raise EvaluationInvalid("Provided source audit does not cover the four protected sources")
    for relative, expected in value["provided"].items():
        if sha256(Path(train_dir) / relative) != expected:
            raise EvaluationInvalid("Current provided source changed: " + relative)
    for absolute, expected in value["runtime"].items():
        if not Path(absolute).is_absolute() or sha256(absolute) != expected:
            raise EvaluationInvalid("Current PPO/runtime source changed: " + absolute)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path), evidence=value)


def load_actor(item, legacy, device):
    import torch
    if sha256(item["checkpoint"]) != item["sha256"]:
        raise EvaluationInvalid("Checkpoint changed before actor load")
    # Hash-bound files are our own produced/reference checkpoints, never an arbitrary download.
    ck = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise EvaluationInvalid("Checkpoint is not an rsl_rl dictionary")
    sd_key = next((k for k in ("actor_state_dict", "model_state_dict") if k in ck), None)
    if sd_key is None or not isinstance(ck[sd_key], dict):
        raise EvaluationInvalid("Missing actor state dictionary")
    # Legacy loader has no normalizer; reject actual normalization evidence rather than silently ignore it.
    norm_keys = [k for k, v in ck.items() if ("normaliz" in k.lower() or "obs_norm" in k.lower()) and v is not None and v != {}]
    norm_sd_keys = [k for k in ck[sd_key] if "normaliz" in k.lower() or "obs_norm" in k.lower()]
    if norm_keys or norm_sd_keys:
        raise EvaluationInvalid("Normalization state unsupported: " + str(norm_keys + norm_sd_keys))
    weights = [n for n, v in ck[sd_key].items() if n.endswith("weight") and getattr(v, "ndim", -1) == 2 and "critic" not in n]
    shapes = [tuple(ck[sd_key][n].shape) for n in weights]
    if shapes != [(512, 235), (256, 512), (128, 256), (12, 128)]:
        raise EvaluationInvalid("Actor layer order/dimensions mismatch: " + str(shapes))
    for n in weights:
        bias = ck[sd_key].get(n[:-6] + "bias")
        if bias is None or tuple(bias.shape) != (ck[sd_key][n].shape[0],):
            raise EvaluationInvalid("Actor bias mismatch")
        if not torch.isfinite(ck[sd_key][n]).all() or not torch.isfinite(bias).all():
            raise EvaluationInvalid("Nonfinite checkpoint parameter")
    actor = legacy.load_checkpoint_actor(item["checkpoint"], device)
    if sha256(item["checkpoint"]) != item["sha256"]:
        raise EvaluationInvalid("Checkpoint changed during actor load")
    with torch.inference_mode():
        probe = actor(torch.zeros((2, 235), dtype=torch.float32, device=device))
    if tuple(probe.shape) != (2, 12) or not torch.isfinite(probe).all():
        raise EvaluationInvalid("Actor output probe failed")
    return actor, dict(state_key=sd_key, layers=[list(s) for s in shapes], iteration=ck.get("iter"),
                       normalization_state_keys=norm_keys + norm_sd_keys, output_probe_shape=list(probe.shape),
                       activation="ELU", actual_checkpoint_sha256=item["sha256"])


def cpu(value):
    return value.detach().cpu().numpy().copy()


def tensor_finite(value, label):
    import torch
    if not torch.isfinite(value).all():
        raise EvaluationInvalid("Nonfinite " + label)
    return value


def diagnostic_bindings(env):
    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene["contact_forces"]
    ids, names = sensor.find_bodies(".*_foot")
    if len(ids) != 4:
        raise EvaluationInvalid("Require exactly four foot contact sensor body IDs")
    action_manager = env.unwrapped.action_manager
    action_term = action_manager.get_term("joint_pos")
    if not hasattr(action_term, "processed_actions"):
        raise EvaluationInvalid("Processed joint action is not available")
    threshold = float(getattr(sensor.cfg, "force_threshold", 1.0))
    if not math.isclose(threshold, 1.0, abs_tol=1e-12):
        raise EvaluationInvalid("Contact sensor threshold differs from frozen effective config")
    for name in ("projected_gravity_b", "root_quat_w", "joint_pos", "joint_vel", "applied_torque",
                 "root_lin_vel_b", "root_ang_vel_b", "root_lin_vel_w", "root_ang_vel_w"):
        if not hasattr(robot.data, name):
            raise EvaluationInvalid("Required diagnostic missing: " + name)
    return robot, sensor, list(ids), list(names), action_term, threshold


def capture_diagnostics(bindings, raw, clipped):
    import torch
    robot, sensor, ids, _, term, threshold = bindings
    d = robot.data
    forces = sensor.data.net_forces_w[:, ids]
    result = dict(projected_gravity=d.projected_gravity_b, body_quaternion_w=d.root_quat_w,
        foot_force_w=forces, foot_contact=torch.linalg.vector_norm(forces, dim=-1) > threshold,
        joint_position=d.joint_pos, joint_velocity=d.joint_vel, applied_torque=d.applied_torque,
        body_linear_velocity=d.root_lin_vel_b, body_angular_velocity=d.root_ang_vel_b,
        world_linear_velocity=d.root_lin_vel_w, world_angular_velocity=d.root_ang_vel_w,
        raw_action=raw, clipped_action=clipped, processed_joint_action=term.processed_actions)
    for field, value in result.items():
        tensor_finite(value, "diagnostic " + field)
    return {field: cpu(value[:4]) for field, value in result.items()}


def policy_obs(value):
    result = value["policy"] if isinstance(value, dict) else value
    tensor_finite(result, "policy observation")
    if result.ndim != 2 or result.shape[1] != 235:
        raise EvaluationInvalid("Actor observation dimension mismatch")
    return result


def run_scenario(env, term, actor, item, sid, seed, legacy, bindings, trace_path, device):
    import numpy as np
    import torch
    spec = legacy.SCENARIOS[sid]
    if len(spec["cmds"]) != 4:
        raise EvaluationInvalid("Frozen command segment count changed")
    audit = ErrorAudit(env.unwrapped.num_envs)
    env.unwrapped.seed(seed)
    torch.manual_seed(seed)
    legacy.set_command(term, spec["cmds"][0])
    obs, _ = env.reset()
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    obs = policy_obs(obs)
    trace = {field: [] for field in TRACE_FIELDS}
    trace.update(command=[], valid_initial_episode=[], control_step=[])
    push_steps = list(range(125, STEPS, 125)) if spec.get("push") else []
    pushes = None
    if push_steps:
        generator = torch.Generator(device="cpu").manual_seed(seed * 7919 + 13)
        pushes = ((torch.rand(len(push_steps), audit.episodes, 2, generator=generator) * 2 - 1) * 1.0).to(device)
    for index in range(STEPS):
        command = spec["cmds"][index // 250]
        legacy.set_command(term, command)
        # Preserve legacy cached-observation stepping, including its command-boundary
        # one-step lag. Do not add observation compute calls/noise/history samples.
        with torch.inference_mode():
            raw = tensor_finite(actor(obs), "actor action")
        if tuple(raw.shape) != (audit.episodes, 12):
            raise EvaluationInvalid("Actor output dimension mismatch")
        clip = item.get("policy_clip_actions")
        applied = raw if clip is None else torch.clamp(raw, -clip, clip)
        obs, reward, terminated, truncated, _ = env.step(applied)
        tensor_finite(reward, "environment reward")
        obs = policy_obs(obs)
        legacy.set_command(term, command)
        robot = bindings[0]
        actual = torch.stack((robot.data.root_lin_vel_b[:, 0], robot.data.root_lin_vel_b[:, 1],
                              robot.data.root_ang_vel_b[:, 2]), dim=-1)
        mask = audit.step(cpu(actual), np.asarray(command), cpu(terminated), cpu(truncated))
        diagnostic = capture_diagnostics(bindings, raw, applied)
        for field in TRACE_FIELDS:
            trace[field].append(diagnostic[field])
        trace["command"].append(np.asarray(command, dtype=np.float64))
        trace["valid_initial_episode"].append(mask[:4].copy())
        trace["control_step"].append(index)
        if index in push_steps:
            velocity = legacy.root_velocity(robot).clone()
            velocity[:, :2] += pushes[push_steps.index(index)]
            tensor_finite(velocity, "push root velocity")
            robot.write_root_velocity_to_sim(velocity)
    row = audit.result(spec)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    arrays.update(tracking_error_full_envs=np.asarray(audit.error_history),
                  tracking_valid_initial_episode=np.asarray(audit.mask_history))
    np.savez_compressed(trace_path, **arrays)
    row.update(scenario=sid, terrain=spec["terrain"], reset_seed=seed, control_step_seconds=DT,
               segment_seconds=5.0, segment_commands=[list(v) for v in spec["cmds"]],
               detailed_trace=dict(path=str(trace_path.resolve()), sha256=sha256(trace_path),
                   bytes=trace_path.stat().st_size, episodes=[0, 1, 2, 3], control_steps=STEPS,
                   fields=list(TRACE_FIELDS), foot_names=bindings[3], contact_threshold_N=bindings[-1],
                   contact_definition="net foot force norm above threshold; actual vectors also recorded",
                   timing="post-env.step states; terminal tick can be autoreset and is marked invalid",
                   action_definition="raw actor output; clipped_action is exact env input after effective policy clip; processed_joint_action is action term after scale/offset/term clipping"))
    row["full_tracking_trace"] = dict(path=str(trace_path.resolve()), episodes=list(range(audit.episodes)),
        fields=["tracking_error_full_envs", "tracking_valid_initial_episode", "command", "control_step"],
        error_dtype="float64", coverage="All environments, exact post-terminal/timeout mask; first4 detailed mechanics remain a subset")
    return row


def merge_results(paths, out, phase):
    models = {}
    protocol_value = None
    for folder in paths:
        folder = Path(folder)
        status = json.loads((folder / "evaluation_complete.json").read_text(encoding="utf-8"))
        if status.get("complete") is not True or status.get("phase") != phase:
            raise EvaluationInvalid("Cannot merge incomplete/mismatched terrain output")
        part = json.loads((folder / "models.json").read_text(encoding="utf-8"))
        if models and set(part) != set(models):
            raise EvaluationInvalid("Model set differs between terrain processes")
        for name, item in part.items():
            if protocol_value is None:
                protocol_value = item["protocol"]
            if item["protocol"] != protocol_value:
                raise EvaluationInvalid("Protocol differs across models/processes")
            if name not in models:
                models[name] = dict(item)
                models[name]["seeds"] = {}
                models[name]["terrain_source_audits"] = []
            dest = models[name]
            for key in ("checkpoint_sha256", "protocol", "actor_validation", "runtime"):
                if dest[key] != item[key]:
                    raise EvaluationInvalid("Model metadata mismatch: " + key)
            if item.get("source_hashes_verified") is not True or item.get("diagnostics_complete") is not True:
                raise EvaluationInvalid("Incomplete protected source/diagnostic verification")
            dest["terrain_source_audits"].append(item["source_audit"])
            for seed, seed_result in item["seeds"].items():
                dest_seed = dest["seeds"].setdefault(seed, {"scenarios": {}})["scenarios"]
                for sid, row in seed_result["scenarios"].items():
                    if sid in dest_seed:
                        raise EvaluationInvalid("Duplicate scenario/seed in merge")
                    dest_seed[sid] = row
    frozen = PHASES[phase]
    if not models:
        raise EvaluationInvalid("No terrain results")
    for item in models.values():
        if set(item["seeds"]) != {str(s) for s in frozen["seeds"]}:
            raise EvaluationInvalid("Merged seed coverage incomplete")
        for seed in item["seeds"].values():
            if set(seed["scenarios"]) != set(frozen["scenarios"]):
                raise EvaluationInvalid("Merged scenario coverage incomplete")
    out = Path(out)
    write_json(out / "models.json", models)
    write_json(out / "evaluation_complete.json", dict(complete=True, phase=phase, merged=True,
               models=list(models), scenarios=list(frozen["scenarios"]), seeds=list(frozen["seeds"]),
               ended_epoch=time.time(), official_score_prediction=None))
    return models


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--scenarios")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--startup_seed", type=int)
    parser.add_argument("--terrain_seed", type=int)
    parser.add_argument("--task", default="Quadruped-v0")
    parser.add_argument("--task-module", dest="task_module", default="go2_task")
    parser.add_argument("--train-dir", type=Path, default=Path("/workspace/training/quadruped"))
    parser.add_argument("--merge", nargs="+", type=Path)
    return parser


def run(args, simulation_app):
    import torch
    import t_eval_go2_core
    legacy = t_eval_go2_core.install()
    scenarios, seeds, terrain = validate_args(args)
    entries = read_manifest(args.manifest)
    audit = source_audit(args.source_audit, args.train_dir)
    args.device = getattr(args, "device", None) or "cuda:0"
    args.seg_s, args.transient_s, args._horizon_s = 5.0, 0.0, 20.0
    args.axis_mode, args.cmd_scale, args.no_obs_noise = "all", 1.0, False
    args.rough_min, args.rough_max, args.slope_deg = .04, .10, 20.0
    args.stair_height, args.stair_width, args.flat_mode = .12, .30, "mesh"
    args.dr_friction, args.dr_mass = "0.4,1.2", "-1.0,3.0"
    args.push_period_s, args.push_mag = 2.5, 1.0
    args.train_script = str(args.train_dir / "train.py")
    sys.path.insert(0, str(args.train_dir))
    legacy.import_task_module(args)
    env = legacy.make_env(args, terrain)
    models = {}
    try:
        if not math.isclose(float(env.unwrapped.step_dt), DT, abs_tol=1e-12, rel_tol=0):
            raise EvaluationInvalid("Protected effective step_dt differs from .02s")
        if env.unwrapped.num_envs != args.num_envs:
            raise EvaluationInvalid("Actual environment count mismatch")
        if int(env.unwrapped.cfg.seed) != args.startup_seed:
            raise EvaluationInvalid("Actual startup seed mismatch")
        generator = env.unwrapped.cfg.scene.terrain.terrain_generator
        if generator is None or int(generator.seed) != args.terrain_seed:
            raise EvaluationInvalid("Actual terrain seed mismatch")
        if not env.unwrapped.cfg.observations.policy.enable_corruption:
            raise EvaluationInvalid("Provided observation corruption is not enabled")
        # Startup seeds are shared; one environment reused within a terrain process.
        term = legacy.pin_command_term(env)
        bindings = diagnostic_bindings(env)
        runtime = dict(torch=torch.__version__, rsl_rl_lib=importlib.metadata.version("rsl-rl-lib"),
                       python=sys.version.split()[0], device=args.device)
        for entry in entries:
            actor, actor_meta = load_actor(entry, legacy, args.device)
            model = dict(checkpoint=str(Path(entry["checkpoint"]).resolve()), checkpoint_sha256=entry["sha256"],
                protocol=protocol(args), source_hashes_verified=True, diagnostics_complete=False,
                source_audit=audit, runtime=runtime, actor_validation=actor_meta, seeds={},
                policy_clip_actions=entry.get("policy_clip_actions"),
                command_observation_refresh="none: matches legacy cached-observation boundary behavior, including possible one-step old command",
                shared_startup_randomization="same reusable environment; reset seed restored before each scenario")
            model["effective_seed_readback"] = dict(startup_seed=int(env.unwrapped.cfg.seed),
                                                    terrain_seed=int(generator.seed))
            models[entry["name"]] = model
            for seed in seeds:
                model["seeds"][str(seed)] = {"scenarios": {}}
                for sid in scenarios:
                    if not simulation_app.is_running():
                        raise EvaluationInvalid("Simulator stopped before planned evaluation completed")
                    trace = args.out / "traces" / (entry["name"] + "_" + str(seed) + "_" + sid + ".npz")
                    row = run_scenario(env, term, actor, entry, sid, seed, legacy, bindings, trace, args.device)
                    model["seeds"][str(seed)]["scenarios"][sid] = row
                    write_json(args.out / "models.partial.json", models)
                    print("[B1-eval]", entry["name"], seed, sid, "falls", row["falls"], "RMSE", row["rmse"], flush=True)
            model["diagnostics_complete"] = True
            del actor
        if sha256(args.source_audit) != audit["sha256"]:
            raise EvaluationInvalid("Source audit evidence changed during evaluation")
        source_audit(args.source_audit, args.train_dir)
    finally:
        env.close()
    write_json(args.out / "models.json", models)
    write_json(args.out / "evaluation_complete.json", dict(complete=True, phase=args.phase, terrain=terrain,
               scenarios=list(scenarios), seeds=list(seeds), models=list(models), ended_epoch=time.time(),
               official_score_prediction=None))


def main():
    parser = build_parser()
    # CPU merge does not import Isaac, torch, a checkpoint, or the legacy prediction code.
    if "--merge" in sys.argv:
        args = parser.parse_args()
        merge_results(args.merge, args.out, args.phase)
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not all((args.manifest, args.source_audit, args.scenarios, args.num_envs, args.seeds,
                args.startup_seed, args.terrain_seed)):
        parser.error("Simulator evaluation requires manifest, source audit, scenarios, env count and all seeds")
    validate_args(args)
    if args.out.exists():
        raise EvaluationInvalid("Refuse existing evaluation output; no retry/resume or mixing prior results")
    args.out.mkdir(parents=True)
    launcher = None
    try:
        launcher = AppLauncher(args)
        run(args, launcher.app)
    except BaseException as exc:
        write_json(args.out / "evaluation_complete.json", dict(complete=False, phase=args.phase,
                   error_type=type(exc).__name__, error=str(exc), ended_epoch=time.time(),
                   official_score_prediction=None))
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
