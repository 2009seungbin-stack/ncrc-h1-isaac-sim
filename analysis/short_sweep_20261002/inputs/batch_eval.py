"""Versioned internal diagnostic: one terrain boot, multiple preserved policies.

Only this analysis driver changes. Provided training/algorithm sources unchanged.
Command observations synchronized without resampling other observation noise;
timeout padded past measured horizon; unexpected truncation invalidates a run.
"""
import hashlib
import json
import math
from pathlib import Path
import time

VERSION = "short-sweep-v2-20261002"


def command_slice(env):
    u = env.unwrapped
    manager = u.observation_manager
    assert manager.group_obs_concatenate["policy"]
    names = manager.active_terms["policy"]
    dims = manager.group_obs_term_dim["policy"]
    assert all(len(d) == 1 for d in dims), dims
    idx = names.index("velocity_commands")
    assert tuple(dims[idx]) == (3,)
    cfg = u.cfg.observations.policy.velocity_commands
    for key in ("noise", "clip", "scale", "modifiers"):
        assert getattr(cfg, key, None) is None, (key, getattr(cfg, key))
    assert getattr(cfg, "history_length", 0) == 0
    assert getattr(u.cfg.observations.policy, "history_length", 0) in (None, 0)
    start = sum(d[0] for d in dims[:idx])
    return slice(start, start+3)


class SyncPolicy:
    def __init__(self, policy, term, sl):
        self.policy, self.term, self.sl = policy, term, sl
        self.changed_steps = 0

    def __call__(self, obs):
        import torch
        assert obs.ndim == 2 and obs.shape[1] >= self.sl.stop
        if not torch.equal(obs[:, self.sl], self.term.vel_command_b):
            self.changed_steps += 1
        obs = obs.clone()
        obs[:, self.sl] = self.term.vel_command_b
        act = self.policy(obs)
        if not torch.isfinite(act).all():
            raise RuntimeError("Nonfinite actions: invalid candidate")
        return act


class MeasuredEnv:
    def __init__(self, env, term, seg_steps, transient_steps, trace_envs=8):
        self.env, self.unwrapped, self.term = env, env.unwrapped, term
        self.seg_steps, self.transient_steps = seg_steps, transient_steps
        self.trace_envs = trace_envs
        self.errors, self.valid, self.vels, self.trace, self.terms = [], [], [], [], []
        self.calls = 0

    def reset(self):
        import torch
        result = self.env.reset()
        self.alive = torch.ones(self.unwrapped.num_envs, dtype=torch.bool, device=self.unwrapped.device)
        return result

    def step(self, action):
        import torch
        # Command at the beginning of this physical step is the reference.
        command = self.term.vel_command_b.clone()
        result = self.env.step(action)
        _, _, terminated, truncated, _ = result
        if truncated.any():
            raise RuntimeError("Unexpected timeout inside measurement; result invalid")
        self.alive &= ~terminated.bool()
        d = self.unwrapped.scene["robot"].data
        six = torch.cat((d.root_lin_vel_b, d.root_ang_vel_b), dim=1)
        err = six[:, [0,1,5]]-command
        if not torch.isfinite(six[self.alive]).all():
            raise RuntimeError("Nonfinite physical state")
        self.errors.append(err.detach().clone())
        self.valid.append(self.alive.clone())
        self.vels.append(six.detach().clone())
        n = min(self.trace_envs, action.shape[0])
        self.trace.append(torch.cat((command[:n], six[:n], action[:n],
                                    d.joint_vel[:n], d.joint_acc[:n],
                                    d.applied_torque[:n], d.root_pos_w[:n]), dim=1).detach().clone())
        self.calls += 1
        return result

    def summarize(self, trace_path):
        import numpy as np
        import torch
        e, valid, six = torch.stack(self.errors), torch.stack(self.valid), torch.stack(self.vels)
        ticks = torch.arange(e.shape[0],device=e.device)
        steady = valid & ((ticks % self.seg_steps) >= self.transient_steps)[:,None]
        all_axes = ("vx", "vy", "yaw")
        result = {"steps": self.calls, "survived": int(self.alive.sum()), "episodes": len(self.alive)}
        for label, mask in (("full", valid), ("steady", steady)):
            z = e[mask].double()
            assert len(z), "No scored samples"
            squared = z.square()
            mse = squared.mean(0)
            count = max(1, math.ceil(len(z)*.05))
            top = squared.topk(count,dim=0).values
            result[label] = {
                "count": len(z), "sum_squared": squared.sum(0).tolist(),
                "sum_absolute": z.abs().sum(0).tolist(), "sum_error": z.sum(0).tolist(),
                "rmse": dict(zip(all_axes,mse.sqrt().tolist())),
                "mae": dict(zip(all_axes,z.abs().mean(0).tolist())),
                "top5_mse_share": dict(zip(all_axes,(top.sum(0)/squared.sum(0).clamp(min=1e-20)).tolist())),
                "extra_rmse": dict(zip(("vz","wx","wy"),six[mask][:,[2,3,4]].double().square().mean(0).sqrt().tolist())),
            }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, error=e.cpu().numpy(), valid=valid.cpu().numpy(),
                            trace_first_envs=torch.stack(self.trace).cpu().numpy(),
                            dt=np.array(self.unwrapped.step_dt),
                            trace_columns=np.array("command3,body_lin3,body_ang3,action12,joint_vel12,joint_acc12,torque12,position3"))
        return result


def main():
    import t_eval_go2_core as go2
    base = go2.install()
    parser = go2.build_parser()
    parser.add_argument("--models_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--startup_seed", type=int, default=42)
    parser.add_argument("--terrain_seed", type=int, default=1234)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    import torch
    try:
        base.import_task_module(args)
        torch.manual_seed(args.startup_seed)
        ids = args.scenarios.split(",")
        assert len({base.SCENARIOS[s]["terrain"] for s in ids}) == 1
        # At dt=.02 this leaves five ticks; actual max_episode_length checked below.
        horizon = max(len(base.SCENARIOS[s]["cmds"])*base.SCENARIOS[s].get("seg_s",args.seg_s) for s in ids)
        args._horizon_s = horizon+.1
        env = base.make_env(args,base.SCENARIOS[ids[0]]["terrain"])
        try:
            assert env.unwrapped.max_episode_length*env.unwrapped.step_dt >= horizon+2*env.unwrapped.step_dt
            term = base.pin_command_term(env)
            sl = command_slice(env)
            print("DIAGNOSTIC_CONTRACT",json.dumps(dict(version=VERSION,command_slice=[sl.start,sl.stop],
                    horizon=horizon,timeout=env.unwrapped.max_episode_length*env.unwrapped.step_dt)),flush=True)
            root = Path(args.models_json).resolve().parent
            models = json.loads(Path(args.models_json).read_text())
            seeds = args.seeds.split(",")
            out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
            for model in models:
                path = root/model["checkpoint"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                assert not model.get("sha256") or sha == model["sha256"]
                policy = base.load_checkpoint_actor(str(path),args.device)
                result = dict(model=model["name"],checkpoint_sha256=sha,version=VERSION,
                              scenarios={},startup_seed=args.startup_seed,terrain_seed=args.terrain_seed,num_envs=args.num_envs,
                              horizon=horizon,command_slice=[sl.start,sl.stop],
                              warnings=["Internal evaluation, not an official score","Reset seeds are not independent training runs"])
                began=time.time()
                for sid in ids:
                    rows=[]
                    for seed in seeds:
                        args.seeds = seed
                        spec = base.SCENARIOS[sid]
                        measured=MeasuredEnv(env,term,round(spec.get("seg_s",args.seg_s)/env.unwrapped.step_dt),
                                             round(args.transient_s/env.unwrapped.step_dt))
                        sync=SyncPolicy(policy,term,sl)
                        legacy=base.run_scenario(measured,term,sync,sid,args,args.device)
                        row=measured.summarize(out/"traces"/f'{model["name"]}_{sid}_{seed}.npz')
                        row.update(seed=int(seed),command_resync_steps=sync.changed_steps,legacy_summary=legacy)
                        rows.append(row)
                    result["scenarios"][sid]=rows
                    dst=out/f'{model["name"]}.json'
                    tmp=dst.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(dst)
                    print("SCENARIO_COMPLETE",model["name"],sid,"seconds",round(time.time()-began,1),flush=True)
                print("MODEL_COMPLETE",model["name"],"seconds",round(time.time()-began,1),flush=True)
                del policy
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
