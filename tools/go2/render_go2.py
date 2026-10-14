#!/usr/bin/env python3
"""render_go2.py — 후보 비교용 **동일 조건** 영상 렌더러.

왜 필요한가 (2026-09-16):
    play.py 로 찍은 영상은 env 마다 명령을 무작위로 뽑는다. 후보 A는 우연히 제자리 회전을,
    후보 B는 최대 전진을 받은 상태로 나란히 놓고 "A가 추종이 좋다"고 하면 후보가 아니라
    주사위를 비교한 것이다. 이 스크립트는 t_eval_go2 의 시나리오 명령 스케줄(G1~G7)을
    그대로 넣고 찍으므로 **모든 후보가 같은 시각에 같은 명령을 받는다.** 화면에서 달라지는 건
    로봇뿐이고, 그 조건이 곧 공식 시나리오와 같다.

    또 한 가지: set_command 는 전 env 에 같은 명령을 박는다. num_envs 32 로 찍으면
    같은 명령을 받은 32마리의 **분포**가 한 화면에 보인다 — 한 마리 클립보다 훨씬 많이 말해준다.

    채점기(t_eval.py)는 건드리지 않는다. 환경 생성·명령 고정·정책 로드를 t_eval 것에서
    그대로 빌려 쓰므로, 영상 속 조건과 점수 속 조건이 같은 코드에서 나온다.

사용:
    cd /workspace/training/quadruped
    /workspace/IsaacLab/isaaclab.sh -p render_go2.py \
        --task Quadruped-v0 --checkpoint ckpt_keep/<run>/model_2000.pt \
        --scenario G1 --label R4A --num_envs 32 --headless --enable_cameras

    # 정책 파일(JIT)로 찍을 때는 --policy exported/policy.pt

출력:
    <video_dir>/<label>_<scenario>.mp4     같은 카메라·같은 명령·같은 seed
    <video_dir>/<label>_<scenario>.json    구간별 명령 vs 실제 속도, 낙상, 생존, 추종(참고치)

주의:
    · --enable_cameras 없으면 IsaacLab 이 rgb_array 렌더를 못 한다(검은 화면/에러).
    · 지형 그룹은 한 프로세스에 하나다(t_eval 과 같은 제약). 시나리오를 여러 개 찍으려면
      프로세스를 나눠 돌린다 — sweep_go2.sh 가 그렇게 한다.
    · 여기 숫자는 seed 1개 × num_envs 개다. **순위 판정용이 아니라 영상에 붙는 설명이다.**
      후보를 줄 세우는 건 여전히 t_eval_go2.py 의 일이다.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import t_eval  # noqa: E402
import t_eval_go2  # noqa: E402


def build_parser():
    t_eval_go2.install()
    p = t_eval_go2.build_parser()
    p.description = "T-EVAL-GO2 동일조건 영상 렌더러"
    p.add_argument("--scenario", default="G1",
                   help="찍을 시나리오 하나 (G1~G7, GX1~GX5). 지형 그룹 제약 때문에 하나씩")
    p.add_argument("--label", default="CAND", help="후보 이름 — 파일명과 표에 그대로 쓰인다")
    p.add_argument("--video_dir", default="films", help="mp4/json 출력 폴더")
    p.add_argument("--render_seed", type=int, default=0,
                   help="초기 상태 seed. 후보 전체가 같은 값을 써야 비교가 된다")
    p.set_defaults(num_envs=32, seg_s=5.0)
    return p


def main():
    parser = build_parser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    sid = args.scenario.strip().upper()
    if sid not in t_eval_go2.GO2_SCENARIOS:
        raise SystemExit(f"[render] 알 수 없는 시나리오: {sid}")
    spec = t_eval_go2.GO2_SCENARIOS[sid]

    # rgb_array 렌더는 카메라가 켜져 있어야 한다. 사용자가 빼먹어도 여기서 켠다.
    if not getattr(args, "enable_cameras", False):
        args.enable_cameras = True
        print("[render] --enable_cameras 를 자동으로 켰다 (rgb_array 렌더 필수)")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    device = args.device if getattr(args, "device", None) else "cuda:0"
    t_eval.import_task_module(args)

    n_seg = len(spec["cmds"])
    seg_s = float(spec.get("seg_s", args.seg_s))
    args._horizon_s = n_seg * seg_s + 2.0   # 에피소드가 시나리오 중간에 time_out 되지 않게

    # ---- 환경 (t_eval.make_env 그대로. render_mode 만 주입한다) ----------------------------
    # make_env 안에서 gym.make(..., render_mode=None) 을 부르므로 모듈 속성을 잠깐 바꿔치기한다.
    # 지형·DR·커리큘럼 처리를 복제하지 않는 게 핵심이다 — 복제하면 영상과 점수의 조건이 갈린다.
    import gymnasium as gym
    _orig_make = gym.make

    def _make_rgb(task, **kw):
        kw.pop("render_mode", None)
        return _orig_make(task, render_mode="rgb_array", **kw)

    gym.make = _make_rgb
    try:
        env = t_eval.make_env(args, spec["terrain"])
    finally:
        gym.make = _orig_make

    dt = env.unwrapped.step_dt
    steps_per_seg = int(round(seg_s / dt))
    total_steps = steps_per_seg * n_seg
    transient = int(round(args.transient_s / dt))

    os.makedirs(args.video_dir, exist_ok=True)
    prefix = f"{args.label}_{sid}"
    from gymnasium.wrappers import RecordVideo
    env = RecordVideo(env, video_folder=args.video_dir, name_prefix=prefix,
                      step_trigger=lambda s: s == 0, video_length=total_steps,
                      disable_logger=True)

    term = t_eval.pin_command_term(env)
    asset = env.unwrapped.scene["robot"]

    # ---- 정책 ----------------------------------------------------------------------------
    if getattr(args, "checkpoint", None):
        policy = t_eval.load_checkpoint_actor(args.checkpoint, device)
        src = args.checkpoint
    else:
        if not os.path.exists(args.policy):
            raise SystemExit(f"[render] 정책 파일이 없습니다: {args.policy}")
        policy = torch.jit.load(args.policy, map_location=device)
        policy.eval()
        src = args.policy
    print(f"[render] {args.label} / {sid} ({spec['desc']}) / {src}")

    # ---- 굴린다 --------------------------------------------------------------------------
    push_steps, pushes = [], None
    if spec.get("push"):
        period = max(1, int(round(float(spec.get("push_period_s", args.push_period_s)) / dt)))
        push_steps = list(range(period, total_steps, period))
        g = torch.Generator(device="cpu").manual_seed(args.render_seed * 7919 + 13)
        mag = float(spec.get("push_mag", args.push_mag))
        pushes = ((torch.rand(len(push_steps), args.num_envs, 2, generator=g) * 2 - 1)
                  * mag).to(device)

    env.unwrapped.seed(args.render_seed)
    torch.manual_seed(args.render_seed)
    t_eval.set_command(term, spec["cmds"][0])
    obs, _ = env.reset()
    obs = obs["policy"] if isinstance(obs, dict) else obs
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False

    alive = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    rows, step = [], 0
    for seg in range(n_seg):
        cmd = spec["cmds"][seg]
        cmd_t = torch.tensor(cmd, dtype=torch.float32, device=device)
        t_eval.set_command(term, cmd)
        vsum = torch.zeros(3, dtype=torch.float64, device=device)
        sq = torch.zeros(3, dtype=torch.float64, device=device)
        cnt = 0.0
        falls = 0
        for k in range(steps_per_seg):
            with torch.inference_mode():
                act = policy(obs)
            act = torch.nan_to_num(act)
            obs, _, terminated, truncated, _ = env.step(act)
            obs = obs["policy"] if isinstance(obs, dict) else obs
            t_eval.set_command(term, cmd)          # 리셋된 env 에서 명령이 되살아나지 않게

            newly = alive & terminated.bool()
            falls += int(newly.sum().item())
            alive &= ~terminated.bool()

            if k >= transient:
                lin = asset.data.root_lin_vel_b
                ang = asset.data.root_ang_vel_b
                actual = torch.stack([lin[:, 0], lin[:, 1], ang[:, 2]], dim=-1)
                m = alive.double()
                cnt += float(m.sum().item())
                vsum += (actual.double() * m[:, None]).sum(dim=0)
                sq += (((actual - cmd_t) ** 2).double() * m[:, None]).sum(dim=0)

            if push_steps and step in push_steps:
                pi = push_steps.index(step)
                vel = t_eval.root_velocity(asset).clone()
                vel[:, 0:2] += pushes[pi]
                asset.write_root_velocity_to_sim(vel)
            step += 1

        c = max(cnt, 1.0)
        got = (vsum / c).tolist()
        mse = (sq / c).tolist()
        rows.append(dict(seg=seg, cmd=list(cmd), got=[round(v, 3) for v in got],
                         mse={"vx": mse[0], "vy": mse[1], "yaw": mse[2]},
                         falls=falls, alive=int(alive.sum().item())))

    env.close()

    # ---- 표 -----------------------------------------------------------------------------
    mse_all = {a: sum(r["mse"][a] for r in rows) / len(rows) for a in ("vx", "vy", "yaw")}
    tracking = t_eval.tracking_from_mse(mse_all, spec, args.axis_mode)
    survival = float(rows[-1]["alive"]) / max(1, args.num_envs)

    print(f"\n[render] === {args.label} / {sid} ===")
    print("  구간 | 명령 vx,vy,yaw        | 실제 vx,vy,yaw        | 낙상 | 생존")
    for r in rows:
        c, g2 = r["cmd"], r["got"]
        print(f"   {r['seg']}   | {c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}       "
              f"| {g2[0]:+.2f},{g2[1]:+.2f},{g2[2]:+.2f}       "
              f"| {r['falls']:3d}  | {r['alive']}/{args.num_envs}")
    print(f"  추종(참고치) {tracking:.4f}   생존 {survival:.3f}   "
          f"seed {args.render_seed}  envs {args.num_envs}")
    if abs(rows[0]["cmd"][0]) > 0.2 and abs(rows[0]["got"][0]) < 0.15:
        print("  ⚠️ 전진 명령에 실제 vx 가 0 근처다 — '전진 거부' 패턴 (GB 와 같은 실패)")

    out = dict(label=args.label, scenario=sid, desc=spec["desc"], source=src,
               task=args.task, num_envs=args.num_envs, seed=args.render_seed,
               seg_s=seg_s, dt=dt, tracking_ref=tracking, survival=survival, segments=rows)
    jp = os.path.join(args.video_dir, prefix + ".json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # RecordVideo 는 <prefix>-step-0.mp4 로 쓴다. 후보명만 남게 정리한다.
    want = os.path.join(args.video_dir, prefix + ".mp4")
    for name in sorted(os.listdir(args.video_dir)):
        if name.startswith(prefix) and name.endswith(".mp4") and os.path.join(args.video_dir, name) != want:
            os.replace(os.path.join(args.video_dir, name), want)
            break
    if os.path.exists(want):
        print(f"[render] 영상 {want} ({os.path.getsize(want)} bytes)")
        print(f"[render] 수치 {jp}")
    else:
        print(f"[render] !! mp4 가 없다 — --enable_cameras 와 video_length({total_steps}) 확인")

    simulation_app.close()


if __name__ == "__main__":
    main()
