"""CPU-only evidence audit for the 20h plan. No simulator, training or submission."""
import json
import math
import statistics
import struct
from pathlib import Path

import torch
import score_model as parser
from analyze_training_efficiency import sha
from deep_training_research import summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/round20_deeper_research_20261001"


def checkpoint_audit(checkpoint, jit):
    ck = torch.load(ROOT/checkpoint, map_location="cpu", weights_only=True)
    actor = ck["actor_state_dict"]
    width = actor["mlp.0.weight"].shape[1]
    sizes = [width] + [actor[f"mlp.{i}.weight"].shape[0] for i in (0, 2, 4, 6)]
    layers = []
    for i in range(4):
        layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
        if i < 3:
            layers.append(torch.nn.ELU())
    model = torch.nn.Sequential(*layers).eval()
    model.load_state_dict({k.removeprefix("mlp."): v for k, v in actor.items() if k.startswith("mlp.")})
    exported = torch.jit.load(str(ROOT/jit), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(241)
    x = torch.randn((128, width), generator=generator)*.2
    with torch.inference_mode():
        maxdiff = float((model(x)-exported(x)).abs().max())
    assert maxdiff < 1e-5
    lrs = [g["lr"] for g in ck["optimizer_state_dict"]["param_groups"]]
    return dict(checkpoint=checkpoint, checkpoint_sha256=sha(ROOT/checkpoint),
                jit=jit, jit_sha256=sha(ROOT/jit), actual_iteration=ck["iter"],
                optimizer_learning_rates=lrs, initial_config_lr=.001,
                initial_over_saved_lr=[.001/lr for lr in lrs],
                actor_jit_max_difference=maxdiff,
                probe_is_mathematical_parity_not_simulator_quality=True)


def event_wall_times(path):
    data = (ROOT/path).read_bytes()
    i, out = 0, {}
    while i + 12 <= len(data):
        length = struct.unpack("<Q", data[i:i+8])[0]
        i += 12
        record = data[i:i+length]
        i += length + 4
        if len(record) != length:
            break
        step, wall, iteration_metric = None, None, False
        for f, wt, v in parser._fields(record):
            if f == 1 and wt == 1:
                wall = struct.unpack("<d", v)[0]
            elif f == 2 and wt == 0:
                step = v
            elif f == 5 and wt == 2:
                for sf, swt, sv in parser._fields(v):
                    if sf == 1 and swt == 2:
                        for vf, vwt, vv in parser._fields(sv):
                            if vf == 1 and vwt == 2 and vv.decode("utf-8", "replace") == "Perf/total_fps":
                                iteration_metric = True
        # Train/*/time summaries use seconds as global_step: do not mix those
        # with iteration-indexed metrics when calculating seconds/iteration.
        if iteration_metric and step is not None and wall is not None:
            out[step] = max(wall, out.get(step, wall))
    return out


def log_audit(path, late_start):
    tags = parser.read_tfevents(ROOT/path)
    wall = event_wall_times(path)
    indices = sorted(s for s in wall if s >= late_start)
    steps = [100, 300, 600, 1000, 1500, 2000, 2500, 4000, 9000, 9900]
    windows = {}
    wanted = ["Curriculum/terrain_levels", "Metrics/base_velocity/error_vel_xy",
              "Metrics/base_velocity/error_vel_yaw", "Episode_Termination/base_contact",
              "Loss/value", "Loss/learning_rate", "Policy/mean_std"]
    for step in steps:
        window = {}
        for tag in wanted:
            values = [v for s, v in tags.get(tag, {}).items() if step-100 <= s < step and math.isfinite(v)]
            if values:
                window[tag] = statistics.median(values)
        if window:
            windows[str(step)] = window
    return dict(path=path, sha256=sha(ROOT/path), median_windows_100_iterations=windows,
                wall_seconds_per_iteration=(wall[indices[-1]]-wall[indices[0]])/(indices[-1]-indices[0]),
                wall_measurement_range=[indices[0], indices[-1]], wall_index_tag="Perf/total_fps",
                train_termination_metric_is_not_official_survival=True,
                scratch_curve_is_not_resume_recovery_evidence=True)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for label in ("R3A_9000", "R3A_10000", "R3C_9000", "R3C_10000"):
        checkpoints[label] = checkpoint_audit(f"R3_bundle_extracted/keep_R3/{label}_model.pt",
                                              f"R3_bundle_extracted/keep_R3/{label}.pt")
    checkpoints["Go2_C0_5500"] = checkpoint_audit("longrun_C0/model_5500.pt", "longrun_C0/C0_5500_policy.pt")
    logs = {}
    for label, path, late in (
        ("Go2_D1", "longrun_D1/logs/events.out.tfevents.1789946121.da-perfect13.11143.0", 500),
        ("H1_R3A", "R3_bundle_extracted/logs/rsl_rl/humanoid/2026-09-08_13-15-40/events.out.tfevents.1788840956.da-perfect28.16598.0", 500),
        ("H1_R3C", "R3_bundle_extracted/logs/rsl_rl/humanoid/2026-09-08_13-16-18/events.out.tfevents.1788840996.da-perfect28.17131.0", 500)):
        logs[label] = log_audit(path, late)
    evals = {name: json.loads((ROOT/f"R3_bundle_extracted/eval_results/R3_{name}.json").read_text(encoding="utf-8"))
             for name in ("R3A_9000", "R3A_10000", "R3C_9000", "R3C_10000")}
    baseline = summarize(evals["R3A_9000"])
    differences = {}
    for label, data in evals.items():
        summary = summarize(data)
        differences[label] = dict(actual_iteration=checkpoints[label]["actual_iteration"],
                                   proxy_delta={f: summary["scores"][f]-baseline["scores"][f]
                                                for f in ("raw3", "floor", "ratio")},
                                   main_survival=summary["weighted_survival"],
                                   strong_push_survival=summary["stress"]["X1"]["survival"],
                                   strong_push_official_weight=data["scenarios"]["X1"]["w"],
                                   scenario_delta={s: summary["scenario_scores"][s]-baseline["scenario_scores"][s]
                                                   for s in summary["scenario_scores"]})
    # Demonstration only: correlated timesteps/episodes cannot be inferred to be iid.
    selection_demo = []
    for paired_seed_sd in (.1, .2, .3):
        se = paired_seed_sd/math.sqrt(4)
        p_single = .5*math.erfc(.15/se/math.sqrt(2))
        selection_demo.append(dict(assumed_sd_of_paired_seed_delta=paired_seed_sd, seed_clusters=4,
                                    threshold=.15, normal_iid_demo_only=True,
                                    probability_one_null_candidate_exceeds_threshold=p_single,
                                    probability_either_of_two_independent_null_candidates_exceeds=1-(1-p_single)**2))
    budget = dict(diagnostics=1., h1_pair_conditional=2.25, go2_matched_pair=4.75,
                  adaptive_finalist=6.5, final_verification=3., contingency_including_shutdown=2.5)
    assert sum(budget.values()) == 20
    result = dict(date="2026-10-01", status="OFFLINE_RESEARCH_NO_TRAINING_OR_SUBMISSION",
                  checkpoint_audit=checkpoints, event_log_audit=logs, existing_h1_choices=differences,
                  h1_late_comparison_actual_iterations=900,
                  lr_resume_risk=dict(
                      upstream_reference="https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v5.0.0/rsl_rl/algorithms/ppo.py",
                      finding="v5.0.0 restores optimizer state but does not restore PPO.learning_rate scalar; adaptive update subsequently writes the scalar into optimizer groups.",
                      installed_server_version_not_verified=True,
                      consequence_is_hypothesis_not_crash_cause="If installed load has this behavior, resume initial scheduler scalar differs from saved optimizer LR; verify both before expensive branching.",
                      no_lr_algorithm_or_checkpoint_mutation_authorized_or_performed=True),
                  selection_threshold_demo=selection_demo,
                  survival_sensitivity=dict(common_one_percentage_point_drop_cost_at_sim58_14=.5814,
                                            common_survival_drop_that_erases_proxy_gain_point15=.15/58.14,
                                            assumption="Uniform survival reduction with unchanged tracking; sensitivity, not measured outcome."),
                  timing_findings=dict(
                      h1_main_plus_all_stress_eval_minutes_from_status=[23+52/60,23+54/60,23+52/60],
                      this_is_not_seven_main_scenarios_only=True,
                      go2_main_eval_minutes_from_status=[8+46/60,8+32/60,8+39/60],
                      old_h1_export_wait_minutes=(19*60+28+20/60)-(19*60+5+28/60),
                      old_pipeline_has_post_training_keepalive_and_dummy_training=True,
                      old_pipeline_not_safe_to_reuse_unchanged=True),
                  h1_concurrency=dict(
                      historical_runs_were_concurrent=True,
                      historical_gpu_memory_mib=13423, historical_gpu_total_mib=16303,
                      single_job_timing_under_same_conditions_not_available=True,
                      no_proven_parallel_speedup=True,
                      optional_test="Only H1: measure solo and paired steady-state throughput within planned first100-200 iterations, no additional training jobs. Require safe memory and >15% net throughput gain to retain concurrency.",
                      go2_parallel_not_recommended_under_recorded_configuration=True),
                  revised_budget_hours=budget,
                  revised_go2_screen=dict(branches=2, iterations_per_branch=1500, training_hours=4,
                                          replaced_three_branches_1000_same_training_cost=True,
                                          narrower_but_longer_is_not_proven_optimal=True),
                  sources={"statistics_paper":"https://arxiv.org/abs/2108.13264",
                           "runner":"https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v5.0.0/rsl_rl/runners/on_policy_runner.py"})
    original_hashes = json.loads((ROOT/"analysis/deep_research_20261001/analysis.json").read_text(encoding="utf-8"))["input_sha256"]
    result["prior_evaluation_inputs_unchanged"] = all(sha(ROOT/p) == expected for p, expected in original_hashes.items())
    assert result["prior_evaluation_inputs_unchanged"]
    (OUT/"analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = f"""20시간 계획 추가 연구와 수정안 — 2026-10-01
상태: 기존 자료와 공개 소스를 분석한 오프라인 연구. 서버 가동·새 학습·새 공식 제출 없음.
본 수정안은 이전20시간 계획의 위험 점검과 자원 배분을 보강한다. 예산 상한20시간은 유지한다.

1. 저장본 정체를 바로잡음
H1의R3A_10000_model.pt와R3C_10000_model.pt 내부 iteration은 모두9900이다.
상태 기록에도model_10000이 없어model_9900을 사용했다고 적혀 있다.
따라서 이전의9000→10000 추가학습 비교는 실제900회 차이다. 순위와작은이득이라는결론은변하지 않는다.
두 H1 설정의9000/9900 네 파일과Go2 C05500, 총다섯 checkpoint의actor/JIT출력 최대차이는 모두0이었다.
이는 임의입력128개에서의수학적일치검사이며시뮬레이션성능이나재개안정성을검증한것은아니다.
R3A9900의추종근사상승은0.003849 또는0.021636. 강한밀침생존은0.9102→0.8809.
R3C9900은밀침0.9512로좋지만7개본시나리오모두R3A9000보다낮다. 3축-1.31665, 두근사-0.64566/-1.19157.
기존R3순위표는배점0인stress생존하한으로정렬했다. 공식배점시나리오상승을우선하고stress는회귀검사로사용한다.
현재확인된저장본에서재학습없이확실하게교체할H1승자는없다. 근소한9900차이만확인하려고평가를대량반복하지않는다.

2. 이어 학습의 학습률 상태를 별도 확인해야 함
C05500 optimizer에저장된LR={checkpoints['Go2_C0_5500']['optimizer_learning_rates'][0]:.9f}.
R3A9000은{checkpoints['R3A_9000']['optimizer_learning_rates'][0]:.9f}, R3C9000/9900은0.000010000.
기본알고리즘설정LR은0.001이므로각각약13.17배/5.85배/100배차이가있다.
확인한RSL-RLv5.0.0코드는optimizer상태를복원하지만PPO.learning_rate별도스칼라를복원하지않는다.
adaptive update가이값을optimizerparam_groups에다시쓰므로참조버전에서는재개초기LR이불연속일수있다.
실제서버의설치버전/대회wrapper가이동작을갖는지는미확인이다. 우리C0발산원인으로단정하지않는다.
실행전설치된load/update소스와재개첫업데이트의실제LR을읽기전용으로확인한다.
불일치시LR/optimizer/std/알고리즘을고쳐서진행하지않는다. 허용된scratch경로로바꾸거나해당실험을생략한다.
같은부모checkpoint라는조건만으로학습의연속성을보장할수없다. 지형커리큘럼도별도로초기화된다.

3. 시간을 다시 실측한 기준
TensorBoard 이벤트의실제벽시계시간을iteration기준Perf/total_fps태그에대응시켰다.
Go2 D1 500~2499: {logs['Go2_D1']['wall_seconds_per_iteration']:.3f}초/iteration.
H1 R3A 500~9999: {logs['H1_R3A']['wall_seconds_per_iteration']:.3f}초, R3C: {logs['H1_R3C']['wall_seconds_per_iteration']:.3f}초.
Train/*/time태그는global_step이초단위여서iteration태그와섞으면잘못된속도가나온다. 본계산에서는제외했다.
기존계획의Go24.8초/H12.5초는예산가정으로유지가능하다. 실행초기속도와시작비용은다시측정한다.
H1본평가+stress전체묶음은과거정책당약24분이었다. Go2본평가만은약8.5~8.8분.
H1의24분은본평가7개만의비용이아니다. 모든후보에전체stress를반복하지않고주요밀침진단을같은flat그룹에합친다.
기존H1내보내기첫대기는약22분52초였고, 학습완료후dummytrain/keepalive가서버를계속썼다.
이수치는과거기록이며같은낭비가현재발생했다고주장하지않는다. 오래된pipeline을그대로재사용하지않는다.
중간checkpoint직접평가,최종후보만내보내기/영상,회수완료즉시서버종료를실행조건으로둔다.

4. Go2 후보 수보다 비교의 깊이를 보강
Go2 D1의scratch후반terrain수준중앙값은약5.11이다.
마지막100회구간의수준은1000회약4.17,1500회약4.89,2000회약5.12였다.
1000회에서는지형적응의차이가후보순위를흔들수있다. 기존C0기록도초기600~800회의순위불안정을보였다.
다만이것은scratch곡선이다. C05500에서재개한정책의회복시간을직접측정한자료는없다.
같은순수학습4시간을세후보×1000회대신대조/후보두개×1500회에쓰는더보수적인선별안을권한다.
1차진단에서회전추종문제가명확하면F10,동작변동/급변과회복의교환이명확하면F01을단일후보로선택한다.
둘을가르는증거가없으면F10을미리정한1차후보로두고이선택을가설로기록한다. 최적보상이라고부르지않는다.
짧은재개에서지형분포가회복되지않으면평균reward나근사점수만으로조기탈락시키지않는다.
다른후보와F11조합은이번필수실험에서제외하며, 남은예산과새근거가있을때만검토한다.

5. +0.15점 게이트는 통계적 검증이 아님
현재평가JSON은여러resetseed를합친요약이다. 시드별paired차이분산을계산할수없다.
두근사식은같은RMSE에서나오므로동시에좋아졌다고독립검증두개를통과한것이아니다.
합성정규예시에서pairedseed차이SD0.2/4seed라고가정하면효과0인두독립후보중하나가+0.15를넘을확률은약12.9%다.
실제분산이나후보독립성이측정되지않았으므로우리실험의오탐확률이라고보고하지않는다.
이예시는고정threshold만으로정확도를보장할수없음을보여준다. 단계별반복과최종보류seed가필요하다.
현재evaluator는seed별원자료를출력하지않는다. 필요시한seed씩실행하거나별도진단출력지원을구현/검증해야한다.
평가출력개선은자체진단도구에한정하고대회훈련환경은수정하지않는다. 이연구에서는evaluator를수정하지않았다.
step/episode3584개를독립학습seed3584개로간주하거나유효하지않은신뢰구간을만들지않는다.
전체생존이공통으로1%p줄면시뮬58.14기준약0.5814점을잃는다. +0.15추종이득은약0.258%p생존손실로상쇄된다.
이계산은동일추종/공통생존변화가정의민감도다. 실제공식생존손실추정은아니다.

6. 병렬 실행 판단 수정
Go2두작업병렬은기존기록에서총처리량이줄어이번계획에서계속제외한다.
H1의이번에분석한두run은동시에8192env씩돌았다. 당시GPU메모리13423/16303MiB였다.
동일조건의H1단독속도자료가없어H1병렬도느리다고단정할수는없다. 반대로병렬이빠르다는증거도없다.
실제장비/허용CLI/메모리를확인하고,예정된첫100~200회안에서단독/동시유효처리량을비교할수있다.
추가학습job을만들지않으며,기존진단예산내에서만실시한다. 순수총처리량15%이상상승과안정성이확인될때만유지한다.
초기워밍업과장비부하가다르면확정벤치마크로부르지않는다. 이연구에서현재서버의병렬시험을실행하지않았다.
안전한시작/출력경로분리가불가능하면순차기본안을유지한다.

7. 수정된20시간 배분
공통진단/재개LR·기준재현:1시간
H1대조/회전후보각1000회:최대2시간15분,건강하고허용된재개일때만
Go2대조/선택후보각1500회:최대4시간45분
승자집중학습:최대6시간30분,필수소비아님
최종보류평가/경계지형/내보내기/영상/회수:3시간보호
복구/변동/서버종료:2시간30분,이중마지막30분보호
합계20시간. 정상경로의예산합은17시간30분. 이미이번라운드사용한시간은먼저차감한다.
누적16시간30분이후에는새학습을시작하지않는다. 다음작업비용+검증3시간+종료30분이남은시간을넘으면시작하지않는다.
승자집중학습예:Go2새scratch최대4500회(순수6시간)또는H1새scratch최대8500회(순수약5.90시간).
두로봇의긴재학습을모두의무로하지않는다. 더유망한기회에먼저배분하고다른로봇은기존확정정책을보존한다.
후보의장기학습량은실제속도와시작/검증비용에따라줄인다. 처음부터한번의연속run한도를설정해커리큘럼재초기화를줄인다.
재개위험이확인되면Go2대조/후보scratch각4000회로대체할수있다(순수약10.67시간).
이실패경로는기존Go2탐색/승자예산11.25시간을대체한다. 시작비용까지안맞으면H1탐색을생략해재배분하거나4000회한도를줄인다.
개선이확인되지않으면추가6.5시간을쓰지않고필요한검증후종료한다.
짧은평가로확인못한가능성과실제실패를분리한다. 최종정책이현재정책을이기지못하면교체하지않는다.

한계와 실행 전 확인
공식합산176.14와마감/잔여시간은이전사이트확인기준이다. 이번연구에서는서버현재상태를새로조회하지않았다.
공식축/명령/지형/생존정의는미확인이고역산근사식은공식점수가아니다. 최종상승은공식채점으로판정한다.
제공REWARD_WEIGHTS값만변경한다. 지원CLI검증전에는재개/seed/env수변경을가정하지않는다.
혼합보상재개이력과scratch이력을정확히보고하고설계/보고서최대점수를자동보장하지않는다.
원본평가11개의SHA-256이이전연구와일치함을확인했다. 정책·보상·checkpoint·evaluator·옛계획은변경하지않았다.

재현
python -X utf8 tools/research_round20_efficiency.py
본폴더analysis.json에원자료SHA-256,벽시계계산,checkpoint/JIT대조,시나리오별차이,예산합계가있다.
참조LR복원소스: https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v5.0.0/rsl_rl/algorithms/ppo.py
Runner참조: https://raw.githubusercontent.com/leggedrobotics/rsl_rl/v5.0.0/rsl_rl/runners/on_policy_runner.py
적은run에서의RL평가불확실성: https://arxiv.org/abs/2108.13264
"""
    (OUT/"research.txt").write_text(report, encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), actual_iterations={k:v['actual_iteration'] for k,v in checkpoints.items()},
                         jit_parity={k:v['actor_jit_max_difference'] for k,v in checkpoints.items()},
                         wall_seconds_per_iteration={k:v['wall_seconds_per_iteration'] for k,v in logs.items()},
                         h1_deltas=differences, budget=budget), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
