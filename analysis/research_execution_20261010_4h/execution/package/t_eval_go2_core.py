#!/usr/bin/env python3
"""T-EVAL-GO2 — NCRC 사족보행(Go2) 7시나리오 내부 평가기.

`t_eval.py` 를 **복제하지 않는다.** 시나리오 테이블과 지형 순서만 갈아끼우고
채점·통계·환경생성은 전부 t_eval 의 것을 그대로 쓴다. 그래야 H1에서 얻은
앵커 2개와 같은 계기로 잰 값이 되고, 평가기 개선이 양쪽에 동시에 적용된다.

    cd /workspace/training/quadruped
    python t_eval_go2.py --task <Go2태스크> --policy exported/policy.pt \
        --run_id GB_9000 --out eval_results/GB_9000.json --headless

⚠️ G-0 정찰 전까지 확정 못 하는 것 (서버에서 반드시 확인):
    1. 태스크 이름 (--task). H1은 Humanoid-v0 였다.
    2. 계단 서브지형 키 이름 — make_terrain_generator 가 'stair' 포함 키를 찾는다.
    3. 도메인 랜덤화 이벤트 항 이름 — make_env 가 후보 이름을 훑고 뭘 적용했는지 찍는다.
    4. 공식 Go2 시나리오 배점. 아래 w 는 **미확인 가정**이다 (§2 주석 참고).

⚠️ 배율(CALIB_RATIO)은 H1 앵커에서 나온 값이다. Go2 는 앵커가 아직 없다.
   3라운드 공식 점수가 나오면 calibration.json 에 적재하고 Go2 전용 배율을 뽑는다.
   그전까지 예측 총점은 **참고치**지 게이트가 아니다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import t_eval  # noqa: E402


# ---------------------------------------------------------------------------
# Go2 채점 시나리오 G1~G7
# ---------------------------------------------------------------------------
# ⚠️ 가중치 w 는 **아직 공식 확인 전이다.** H1은 공식 배점이 10.5/14/7/7/10.5/10.5/10.5
#    로 확인됐고(official_score_breakdown.md), Go2도 같은 구조일 것으로 가정했다.
#    G-0에서 예선 규정/README 를 보고 다르면 여기만 고치면 된다.
#    험지 계열(G3+G4+G7)=0.45 로 H1(평지 0.55)과 목적함수가 반대쪽이다.
GO2_SCENARIOS = {
    "G1": dict(w=0.15, terrain="flat", axes=("vx",),
               cmds=[(0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0)],
               desc="전진 속도 추종 (4구간 후진)"),
    "G2": dict(w=0.15, terrain="flat", axes=("vx", "vy", "yaw"),
               cmds=[(0.0, 0.5, 0.0), (0.0, -0.5, 0.0), (0.5, 0.0, 0.5), (0.5, 0.0, -0.5)],
               desc="전방위 속도 추종"),
    "G3": dict(w=0.20, terrain="rough", axes=("vx", "yaw"),
               cmds=[(0.5, 0.0, 0.0), (0.5, 0.0, 0.3), (0.8, 0.0, 0.0), (0.5, 0.0, -0.3)],
               desc="거친 지형"),
    "G4": dict(w=0.15, terrain="slope", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               desc="경사 +-20도 (오르막/내리막 50:50)"),
    "G5": dict(w=0.15, terrain="stairs", axes=("vx",),
               cmds=[(0.4, 0.0, 0.0)] * 4,
               desc="계단 10~15cm"),
    "G6": dict(w=0.10, terrain="flat", axes=("vx",),
               cmds=[(0.5, 0.0, 0.0)] * 4,
               push=True,
               desc="밀침 회복"),
    "G7": dict(w=0.10, terrain="dr", axes=("vx", "yaw"),
               cmds=[(0.5, 0.0, 0.0), (0.5, 0.0, 0.3), (0.5, 0.0, 0.0), (0.5, 0.0, -0.3)],
               desc="도메인 랜덤화 (험지 + 마찰/질량 랜덤)"),

    # ---- Stress 묶음 GX — w=0 이라 SimScore 에 안 들어간다.
    #      **자체 생존이 1.0으로 포화하면 95와 88을 구분할 수 없다.** 이게 그 해독제다.
    #      95 아닌 88을 가르는 자리는 십중팔구 계단(G5)과 내리막(G4)이다.
    "GX1": dict(w=0.0, terrain="flat", axes=("vx",),
                cmds=[(0.5, 0.0, 0.0)] * 4,
                push=True, push_mag=2.0, push_period_s=1.5,
                desc="STRESS 강한 밀침 2.0 m/s 주기 1.5 s"),
    "GX2": dict(w=0.0, terrain="rough", axes=("vx", "yaw"),
                cmds=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.5), (1.0, 0.0, 0.0), (1.0, 0.0, -0.5)],
                desc="STRESS 거친 요철 고속 (--rough_min 0.08 --rough_max 0.14 로 실행)"),
    "GX3": dict(w=0.0, terrain="slope", axes=("vx",),
                cmds=[(0.8, 0.0, 0.0), (0.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.0, 0.0)],
                desc="STRESS 경사 25도 + 내리막 정지 (--slope_deg 25 로 실행)"),
    "GX4": dict(w=0.0, terrain="stairs", axes=("vx",),
                cmds=[(0.6, 0.0, 0.0), (0.6, 0.0, 0.3), (0.6, 0.0, 0.0), (-0.4, 0.0, 0.0)],
                desc="STRESS 계단 15cm 고속 + 후진 (--stair_height 0.15 로 실행)"),
    "GX5": dict(w=0.0, terrain="flat", axes=("vx", "vy", "yaw"), seg_s=2.0,
                cmds=[(1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, -1.0),
                      (1.0, 0.0, 1.0), (0.0, -0.5, 0.0), (0.0, 0.0, 1.0), (-0.5, 0.0, -1.0),
                      (1.0, 0.5, 0.0), (0.0, 0.0, 0.0)],
                desc="STRESS 2초 급전환 x10"),
}

GO2_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
GO2_STRESS = ["GX1", "GX2", "GX3", "GX4", "GX5"]
# 지형 그룹은 한 프로세스에 하나씩만 돈다(두 번째 환경 생성 시 cfg.validate 무한 재귀).
# 순회 순서만 정의해 두고, 실제로는 --scenarios 로 그룹을 나눠 돌린 뒤 --merge 한다.
GO2_TERRAINS = ["flat", "rough", "slope", "stairs", "dr"]


def install():
    """t_eval 의 모듈 전역을 Go2 것으로 교체한다.

    t_eval 의 함수들은 전부 모듈 전역 SCENARIOS/SCENARIO_ORDER 를 이름으로 읽으므로,
    여기서 재바인딩하면 모든 함수가 Go2 테이블을 보게 된다.
    """
    t_eval.SCENARIOS = GO2_SCENARIOS
    t_eval.SCENARIO_ORDER = GO2_ORDER
    t_eval.STRESS_ORDER = GO2_STRESS
    t_eval.TERRAIN_ORDER = GO2_TERRAINS
    # Go2 는 공식 앵커가 없다. 오프셋 표를 비워 잘못된 비교를 막는다.
    t_eval.OFFICIAL_REFS = {"none": ({}, None)}
    # 예측기는 H1 앵커를 빌려 쓴다. 절대값 금지, 순위만 — 출력이 그렇게 경고한다.
    t_eval.CALIB_BORROWED = True
    return t_eval


def build_parser():
    install()
    p = t_eval.build_parser()
    p.description = "NCRC Go2 7시나리오 내부 평가기 (T-EVAL-GO2)"
    # Go2 기본값 — H1과 다른 것만 덮어쓴다.
    p.set_defaults(task="Go2-v0",            # G-0에서 실제 이름 확인 후 --task 로 지정
                   official_ref="none",
                   slope_deg=20.0,           # G4 공식은 +-20도
                   rough_min=0.04,           # H1 기본 0.02는 공식보다 쉬웠다(비율 0.874)
                   rough_max=0.10)
    return p


def main():
    install()
    parser = build_parser()

    if "--recompute" in sys.argv:
        args, _ = parser.parse_known_args()
        r = t_eval.do_recompute(args.recompute, args.out,
                                args.run_id if args.run_id != "UNSET" else None,
                                args.axis_mode)
        t_eval.print_offset_table(r, args.official_ref)
        return

    if "--merge" in sys.argv:
        args, _ = parser.parse_known_args()
        r = t_eval.do_merge(args.merge, args.out,
                            args.run_id if args.run_id != "UNSET" else None)
        t_eval.print_offset_table(r, args.official_ref)
        return

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    t_eval.run(args)

    simulation_app.close()


if __name__ == "__main__":
    main()
