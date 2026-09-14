#!/usr/bin/env python3
"""G-0 정찰 — Go2 워크스페이스의 미지수를 한 번에 전부 캔다.

    cd /workspace/training/quadruped
    /workspace/IsaacLab/isaaclab.sh -p g0_probe.py --headless [--task Go2-v0]

이 스크립트 하나의 출력이 go2_plan.md 의 §4·§5 공백을 메운다. 캐는 것:

  1. gym 레지스트리에서 사족 태스크 이름 후보
  2. 보상 항 전체 + weight + std + threshold   ← §5.1 대조표의 'NCRC baseline' 열
  3. 명령 범위와 커리큘럼 유무                  ← §5.3, 전역 편향 0.909 용의자
  4. 지형 서브지형 키 이름                      ← t_eval_go2 의 'stairs' 탐색이 걸릴지
  5. 이벤트 항 이름 (push_robot / 마찰 / 질량)  ← G7 도메인 랜덤화, 밀침 함정 확인
  6. 종료 조건, 에피소드 길이, 관측 차원
  7. **보상 로더가 weight 말고 std/params 도 받는지**  ← RMSE 직격 손잡이의 존재 여부

학습을 시작하지 않는다. train.py 를 부르지 않는다(인자 없이 부르면 체크포인트가 지워진다).
"""

import argparse
import glob
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def hr(t):
    print("\n" + "=" * 72)
    print("== " + t)
    print("=" * 72)


def probe_reward_file():
    """참가자 수정 영역 파일과 그 로더를 찾는다."""
    hr("2. 참가자 보상 파일과 REWARD_WEIGHTS")
    cands = sorted(glob.glob("*_rewards.py") + glob.glob("*rewards*.py"))
    print("후보 파일:", cands or "없음")
    target = None
    for c in cands:
        if "baseline" in c or "_G" in c:
            continue
        target = c
        break
    if not target:
        print("!! 보상 파일을 못 찾았다. ls 결과를 보고 수동 확인하라.")
        return None
    print(f"\n--- {target} 의 REWARD_WEIGHTS ---")
    ns = {}
    try:
        exec(open(target, encoding="utf-8").read(), ns)
        w = ns.get("REWARD_WEIGHTS", {})
        for k, v in sorted(w.items()):
            print(f"  {k:<28} {v}")
        print(f"\n항목 수: {len(w)}   ← 이게 우리가 만질 수 있는 전부다")
    except Exception as e:
        print("!! exec 실패:", e)

    # --- 7. 로더가 std/params 를 받는지 ---
    hr("7. 보상 로더 — weight 말고 std/params 도 세팅 가능한가")
    print("REWARD_WEIGHTS 를 소비하는 코드를 찾는다 (읽기만 한다, 수정 금지).\n")
    hits = []
    for root, _dirs, files in os.walk("."):
        if ".git" in root or "logs" in root:
            continue
        for f in files:
            if not f.endswith(".py") or f == target:
                continue
            p = os.path.join(root, f)
            try:
                src = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if "REWARD_WEIGHTS" in src:
                hits.append((p, src))
    if not hits:
        print("!! REWARD_WEIGHTS 소비처를 못 찾았다. 태스크 패키지 안(site-packages)일 수 있다.")
    for p, src in hits:
        print(f"--- {p} ---")
        for m in re.finditer(r"REWARD_WEIGHTS", src):
            a = max(0, src.rfind("\n", 0, m.start() - 400))
            b = src.find("\n", m.end() + 600)
            print(src[a:b if b > 0 else len(src)])
            print("  " + "-" * 60)
            break
    print("\n>>> 판정 기준: '.weight = ' 만 하면 **가중치 전용**이다 → RMSE 손잡이는 추종 가중치뿐.")
    print(">>> 'params' 나 'std' 를 setattr/update 하는 경로가 있으면 std 하향이 가능하다.")
    return target


def probe_env(task):
    from isaaclab_tasks.utils import parse_env_cfg

    hr(f"3~6. 환경 설정 덤프  (task={task})")
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)

    print("--- 보상 항 (이름 / weight / 그 외 params) ---")
    rew = cfg.rewards
    for name in sorted(dir(rew)):
        if name.startswith("_"):
            continue
        t = getattr(rew, name)
        if t is None:
            print(f"  {name:<30} None  (꺼져 있음)")
            continue
        w = getattr(t, "weight", None)
        if w is None:
            continue
        prm = dict(getattr(t, "params", {}) or {})
        prm.pop("asset_cfg", None)
        prm.pop("sensor_cfg", None)
        extra = f"   params={prm}" if prm else ""
        print(f"  {name:<30} {w}{extra}")

    print("\n--- 명령 (범위·커리큘럼) ---")
    try:
        bv = cfg.commands.base_velocity
        for a in ("resampling_time_range", "heading_command", "rel_standing_envs",
                  "rel_heading_envs", "debug_vis"):
            if hasattr(bv, a):
                print(f"  {a:<26} {getattr(bv, a)}")
        for attr in ("ranges", "limit_ranges"):
            r = getattr(bv, attr, None)
            if r is None:
                continue
            print(f"  {attr}:")
            for a in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
                if hasattr(r, a):
                    print(f"      {a:<12} {getattr(r, a)}")
        print(f"  class_type                 {type(bv).__name__}")
        print("  >>> limit_ranges 가 있으면 명령 커리큘럼이 있는 것. 없으면 처음부터 최대 범위다.")
    except Exception as e:
        print("  !! 명령 덤프 실패:", e)

    print("\n--- 지형 서브지형 키 (t_eval_go2 의 stairs 탐색이 여기 걸린다) ---")
    try:
        tg = cfg.scene.terrain.terrain_generator
        if tg is None:
            print("  terrain_generator None (terrain_type=%s)" % cfg.scene.terrain.terrain_type)
        else:
            for k, v in tg.sub_terrains.items():
                bits = [f"proportion={v.proportion}"]
                for a in ("noise_range", "slope_range", "step_height_range", "step_width",
                          "grid_height_range", "platform_width"):
                    if hasattr(v, a):
                        bits.append(f"{a}={getattr(v, a)}")
                print(f"  {k:<26} " + "  ".join(bits))
            print(f"  curriculum={tg.curriculum}  size={getattr(tg, 'size', None)} "
                  f"rows={tg.num_rows} cols={tg.num_cols}")
            stairs = [k for k in tg.sub_terrains if "stair" in k]
            print(f"  >>> 계단 키: {stairs or '없음 — G5 를 미측정으로 두거나 다른 키를 써야 한다'}")
    except Exception as e:
        print("  !! 지형 덤프 실패:", e)

    print("\n--- 이벤트 항 (밀침 함정 / G7 도메인 랜덤화) ---")
    ev = cfg.events
    for name in sorted(dir(ev)):
        if name.startswith("_"):
            continue
        t = getattr(ev, name)
        if t is None:
            print(f"  {name:<30} None   <<< 꺼져 있음")
            continue
        if not hasattr(t, "params"):
            continue
        prm = {k: v for k, v in (t.params or {}).items()
               if not hasattr(v, "joint_names")}
        print(f"  {name:<30} mode={getattr(t, 'mode', '?')}  {prm}")
    print("  >>> push_robot 이 None 이면 H1과 같은 함정이다: 학습에 밀침이 없는데 공식은 채점한다.")

    print("\n--- 종료 조건 / 에피소드 / 관측 ---")
    for name in sorted(dir(cfg.terminations)):
        if name.startswith("_"):
            continue
        t = getattr(cfg.terminations, name)
        print(f"  terminations.{name:<24} {'None' if t is None else getattr(t, 'func', t)}")
    print(f"  episode_length_s           {cfg.episode_length_s}")
    print(f"  decimation                 {getattr(cfg, 'decimation', '?')}")
    print(f"  sim.dt                     {getattr(cfg.sim, 'dt', '?')}")
    try:
        pol = cfg.observations.policy
        terms = [n for n in dir(pol) if not n.startswith("_")
                 and hasattr(getattr(pol, n), "func")]
        print(f"  관측 항 ({len(terms)}): {terms}")
        print(f"  enable_corruption          {getattr(pol, 'enable_corruption', '?')}")
    except Exception as e:
        print("  !! 관측 덤프 실패:", e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="비우면 레지스트리에서 사족 후보를 찾는다")
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app

    hr("1. 태스크 레지스트리")
    import gymnasium as gym
    try:
        import isaaclab_tasks  # noqa: F401
    except Exception as e:
        print("isaaclab_tasks import 실패:", e)
    # train.py 의 등록 import 를 재현한다 (train.py 를 **실행하지는 않는다**)
    if os.path.exists("train.py"):
        src = open("train.py", encoding="utf-8").read()
        for line in re.findall(r"^\s*(?:import|from)\s+\S+.*$", src, re.M):
            if "isaaclab.app" in line or "argparse" in line:
                continue
            try:
                exec(line.strip(), {})
            except Exception:
                pass
    names = sorted(k for k in gym.registry
                   if re.search(r"go2|quad|anymal|dog|spot", k, re.I))
    print("사족 후보:", names or "없음")
    print("전체 등록 수:", len(gym.registry))
    if not args.task:
        args.task = names[0] if names else None
        print("→ 자동 선택:", args.task)
    if not args.task:
        print("!! 태스크를 못 정했다. --task 로 직접 넘겨라.")
        app.close()
        return

    probe_reward_file()
    try:
        probe_env(args.task)
    except Exception as e:
        import traceback
        print("!! 환경 덤프 중 예외:")
        traceback.print_exc()
        del e

    hr("끝 — 이 출력을 통째로 회수해서 go2_plan.md §5.1 대조표를 채운다")
    app.close()


if __name__ == "__main__":
    main()
