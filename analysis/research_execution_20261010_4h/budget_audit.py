"""Independent local-only timing/contract audit. Does not launch training or Isaac."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'analysis/baseline_redesign_20261009_4h'
OLD=BASE/'recovered_full_20261009_1427/b1_baseline_20261009_4h'
RECENT=ROOT/'analysis/high_score_search_20261009'
SWEEP=ROOT/'analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    state=json.loads((OLD/'state.json').read_text(encoding='utf-8-sig'))
    recent=json.loads((RECENT/'execution.json').read_text(encoding='utf-8-sig'))
    trainlog=(OLD/'logs/train_B1.log').read_text(encoding='utf-8-sig')
    iterations=np.asarray([float(v) for v in re.findall(r'Iteration time:\s*([0-9.]+)s',trainlog)])
    assert len(iterations)==1500
    actual_training_seconds=float(state['training']['seconds'])
    assert state['training']['configuration']['actor_observation_normalization'] is False
    timing_sources=[];per_rollout=[]
    for terrain in ('flat','rough','slope','stairs','dr'):
        path=SWEEP/'logs'/f'heldout_{terrain}.log'
        previous={};examples=[]
        for model,sid,elapsed in re.findall(r'SCENARIO_COMPLETE\s+(\S+)\s+(G\d)\s+seconds\s+([0-9.]+)',path.read_text(encoding='utf-8-sig')):
            elapsed=float(elapsed);duration=elapsed-previous.get(model,0.);previous[model]=elapsed
            # Each legacy SCENARIO_COMPLETE accumulated exactly two reset seeds.
            rate=duration/2;per_rollout.append(rate)
            examples.append(dict(model=model,scenario=sid,two_reset_seconds=duration,seconds_per_reset_rollout=rate))
        timing_sources.append(dict(path=str(path),sha256=sha(path),timings=examples))
    shared=recent['evaluations']; common_seconds=sum(float(v['seconds']) for v in shared.values())
    assert len(shared)==5
    maximum_rollout=float(max(per_rollout))
    cases={}
    for name,screen_rollouts in (('deduplicated_refs_one_reset',28),('references_repeated_between_stages',36),('two_reset_screen_not_current_scope',56)):
        total=screen_rollouts+42
        per_rollout_planning=maximum_rollout*1.5
        eval_seconds=total*per_rollout_planning+11*75
        finish=15+130+eval_seconds/60
        cases[name]=dict(screen_rollouts=screen_rollouts,validation_rollouts=42,total_rollouts=total,
            terrain_processes=11,setup_minutes=15,training_planning_minutes=130,
            seconds_per_rollout_planning=per_rollout_planning,seconds_per_terrain_start_planning=75,
            evaluation_planning_minutes=eval_seconds/60,compute_finish_elapsed_minutes=finish,
            margin_before_compute195_minutes=195-finish,
            fits195_arithmetically=finish<195,has5min_compute_margin=finish<=190,
            assumption_status='Conservative planning assumptions, not measured new-evaluator or new-server timings')
    old_screen=(20*maximum_rollout*1.5+3*75)/60
    new_screen=(8*maximum_rollout*1.5+3*75)/60
    validation=(42*maximum_rollout*1.5+5*75)/60
    phases=dict(setup15=15,prior_screen20_rollouts_minutes=old_screen,
                latest_planned_training_start_minute=15+old_screen,
                training130=130,planned_training_finish_minute=15+old_screen+130,
                new_screen8_rollouts_minutes=new_screen,
                planned_selection_finish_minute=15+old_screen+130+new_screen,
                validation42_rollouts_minutes=validation,
                planned_compute_finish_minute=15+old_screen+130+new_screen+validation)
    final_cases={}
    for boots in (8,9):
        for factor in (1.,1.5):
            evaluation_minutes=(58*maximum_rollout*factor+boots*75)/60
            finish=15+130+evaluation_minutes
            final_cases[f'boots{boots}_oldmax_factor{factor}']=dict(selection_rollouts=16,validation_rollouts=42,total_rollouts=58,
                actual_selection_terrain_processes=3,actual_validation_terrain_processes=5,planned_boot_allowance=boots,
                seconds_per_rollout_planning=maximum_rollout*factor,diagnostic_buffer_factor=factor,
                evaluation_planning_minutes=evaluation_minutes,compute_finish_elapsed_minutes=finish,
                margin_before_compute195_minutes=195-finish,has5min_compute_margin=finish<=190,
                max_training_start_for_full130min_case_and5min_margin=195-130-evaluation_minutes-5,
                assumption_status='Factor1 uses old observedmax only; factor1.5 retains explicit extra diagnosticbuffer. Boots9 adds one extra75s pad beyond8evaluationterrainstarts.')
    test=subprocess.run([sys.executable,str(OUT/'execution/evaluation_tests.py')],capture_output=True,text=True,encoding='utf-8',errors='replace')
    output=test.stdout+test.stderr
    test_count=re.search(r'Ran (\d+) tests',output)
    (OUT/'evaluation_cpu_test_output.txt').write_text(output,encoding='utf-8')
    old_eval=BASE/'execution/evaluation.py';new_eval=OUT/'execution/evaluation.py'
    unchanged={name:sha(OUT/'execution'/name)==sha(BASE/'execution/package'/name) for name in ('t_eval.py','t_eval_go2_core.py')}
    report=dict(status='LOCAL_PREALLOCATION_BUDGET_AND_EVALUATOR_AUDIT',no_gpu=True,no_remote=True,no_browser=True,no_training=True,
        measured_training=dict(B1_process_seconds=actual_training_seconds,B1_process_minutes=actual_training_seconds/60,
            S1_process_seconds=recent['s1_training_seconds'],R1_process_seconds=recent['r1_training_seconds'],
            B1_iteration_samples=len(iterations),B1_iteration_mean_seconds=float(iterations.mean()),
            B1_iteration_median_seconds=float(np.median(iterations)),B1_iteration_p90_seconds=float(np.quantile(iterations,.9)),
            B1_recent50_mean_seconds=float(iterations[-50:].mean()),
            B1_training_log_sha256=sha(OLD/'logs/train_B1.log'),B1_state_sha256=sha(OLD/'state.json')),
        measured_evaluation=dict(recent_common56_rollout_seconds=common_seconds,recent_common56_minutes=common_seconds/60,
            common56_definition='4policies x7scenarios x2resetseeds,env128,5terrainprocesses; older batch evaluator',
            recent_per_terrain_seconds={k:v['seconds'] for k,v in shared.items()},
            latest_timing_evidence_path=str(RECENT/'execution.json'),latest_timing_evidence_sha256=sha(RECENT/'execution.json'),
            old_raw_log_per_rollout_max_seconds=maximum_rollout,old_raw_log_per_rollout_median_seconds=float(np.median(per_rollout)),
            old_timing_sources=timing_sources,
            new_evaluator_runtime_measured=False,warning='Old evaluator timing includes different diagnostics; env64 is not assumed to be twice as fast.'),
        rollout_count=dict(prior_screen=5*4*1,new_screen_new_policies_only=2*4*1,
            total_deduplicated_screen=7*4*1,with_repeated_refs=20+4*4*1,two_reset_screen=7*4*2,final_validation=3*7*2,
            screen_seed2029_only=True),
        caps_elapsed_minutes=dict(training_start_latest=30,training_complete_latest=160,selection_complete_latest=175,compute_stop=195,archive_retrieval_stop_and_shutdown_request_latest=215,confirmed_server_off=240),
        initial_plan_conservative_cases=cases,initial_plan_deduplicated_phase_arithmetic=phases,
        final_pipeline=dict(candidate='S5_torque_half',scratch_iterations=1500,num_training_envs=6144,training_seed=42,
            prior_B1_screen_enabled=False,prior_B1_screen_rollouts=0,
            selection_policies=['S5_model_1000','S5_model_1499','C0_5500','F10_3000'],selection_rollouts=16,
            validation_policies=['candidate','C0_5500','F10_3000'],validation_rollouts=42,total_eval_rollouts=58,
            selection_terrain_processes=3,validation_terrain_processes=5,reference_reuse_dependency=False,
            final_case_arithmetic=final_cases),
        measured_training_plus_conservative_eval_finish_minutes=15+actual_training_seconds/60+cases['deduplicated_refs_one_reset']['evaluation_planning_minutes'],
        assessment='Final no-prescreen58rollout pipeline is conditionally feasible: same conservative130mintraining/1.5diagnosticbuffer/9bootpad planning reaches185.83min and leaves9.17min before195, exceeding5min compute margin. Actual new-runtime speeds remain unmeasured; live timing/admission remains mandatory. The withdrawn initial28screen plan left only33sec, justifying disabling oldB1screen.',
        live_admission_rules=['Do not start training after elapsed30min. Require observed1500-iteration ETA to fit elapsed160min with preservation margin.',
            'Elapsed30min is an outer cap, not sufficient admission by itself: the conservative130mintraining+1.5diagnosticbuffer plan needs training start by elapsed19.17min with9bootpad (20.42min with8actualevaluationboots) to preserve5min margin.',
            'Reserve full42 validation rollouts and all5terrainstarts before admitting remaining selection work. Require projected compute finish plus5min <=195.',
            'After compute cutoff195min, archive/retrieval/verification may use only the first20min (195 to215). At215min or whenever remaining allocation time is<=25min, stop retrieval and immediately request dashboard shutdown; the final25min are reserved for actual OFF confirmation by240. Retrieval failure never extends these bounds.',
            'If a stage is incomplete at its bound, preserve partial metrics/checkpoints, stop process group/restore sources, do not silently retry or continue training.'],
        evaluator_audit=dict(cpu_test_count=int(test_count.group(1)) if test_count else None,cpu_test_returncode=test.returncode,
            cpu_test_all_passed=test.returncode==0,test_output=str(OUT/'evaluation_cpu_test_output.txt'),
            old_evaluation_sha256=sha(old_eval),new_evaluation_sha256=sha(new_eval),frozen_legacy_copies_unchanged=unchanged,
            new_diagnostics='Full-env exact error/mask recording and arithmetic samplep99/signedlag/ordinaryyaw only; no observation recompute/environment/score change',
            actual_rsl5_loader_previous_live_success=state['preflight']['reference_loader_verified'],
            previous_runtime=state['preflight']['runtime_versions'],new_runtime_not_yet_verified=True,
            dimensions=[235,512,256,128,12],normalization='Unsupported normalization is rejected in both checkpoint and manifest; effective actor config must be reverified live',
            source_audit='Immutable per-job receipt plus independent provided/PPO hash checks before/after evaluation required'),
        comparison_cautions=['Reset2029screen and2039/2063validation are different reset/terrain/startup conditions, but they are not independent training seeds.',
            'Validation seeds2039/2063 were predeclared in the previous B1 design; no previous B1 validation actually ran, but they should not be advertised as newly generated hidden holdout conditions.',
            'B1 intermediate checkpoints share one seed42 training trajectory; selecting600/1000/1499 is correlated checkpoint selection, not3trainingreplicates.',
            'All3baseline durations differ: C0_5500,F10_3000,new1500. A favorable fresh shared-protocol comparison does not identify a reward-only causal effect.',
            'HistoricalOct2diagnostics synchronized command observations; the reused evaluator preserves legacy cached observation and may have one old-command step. Compare fresh references only, not mix historicalquality rows.',
            'First4 mechanical traces do not represent full-env effort/tails; new full-error arrays allow full-env speederror diagnostics.',
            'Zero falls in limited reset samples is evidence within this protocol, not a guarantee of official survival or93points.'],
        official_prediction=None)
    (OUT/'budget_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    lines=['4시간 실행 예산 독립 점검 — 서버 시작 전 로컬 검증','',
           f"실측 B1 1500회 프로세스 {actual_training_seconds/60:.3f}분. 실측 기존56rollout 평가 {common_seconds/60:.3f}분.",
           f"이전 원시 로그의1reset rollout 최대 {maximum_rollout:.3f}초. 새 evaluator 런타임은 실측되지 않았다.",
           '최종 확정: 초기B1screen enabled=false. S5_torque_half 하나만scratch1500회,6144환경,seed42 학습.',
           '최종selection16 = 새1000/1499 + C0/F10, 각4시나리오×reset2029한개.',
           '최종검증42 = 후보+C0/F10 ×7시나리오×reset2039/2063. 전체58rollout, 실제eval지형부팅8회(3+5).',
           '철회된초기안: screen28 = 기존B1세 checkpoint + 새후보두 checkpoint + C0/F10, 각4시나리오×reset2029한개.',
           '기준 모델을 두 단계에서 재평가하면36개다. 56개는 두reset을 사용해야 하므로 현재 명세와 다르다.',
           '최종검증42 = 후보+C0/F10 ×7시나리오×reset2039/2063.','',
           '보수적 가정: setup15분 + 학습130분 + old최대rollout시간×1.5 + 지형프로세스당75초. 새runtime실측값이 아니다.',
           '최종안 | 계산종료 추정분 | 195분까지여유 | 5분여유확보']
    for name,row in final_cases.items():lines.append(f"{name} | {row['compute_finish_elapsed_minutes']:.3f} | {row['margin_before_compute195_minutes']:+.3f} | {row['has5min_compute_margin']}")
    lines+=['','철회된초기안(포함screen20+추가screen8)의 비용:']
    for name,row in cases.items():lines.append(f"{name} | {row['compute_finish_elapsed_minutes']:.3f} | {row['margin_before_compute195_minutes']:+.3f} | {row['has5min_compute_margin']}")
    lines+=['','중복기준회수 없는28screen일 때 단계별 보수적 추정:']
    lines += [f'{k}: {v:.3f}분' for k,v in phases.items()]
    lines += ['',report['assessment'],'','라이브 입장 조건:']+report['live_admission_rules']
    lines += ['',f"CPUtests {report['evaluator_audit']['cpu_test_count']}개; exit{test.returncode}. t_eval/core 원본 copy hash동일: {all(unchanged.values())}",
              '기존참고와 seed의 의미:']+report['comparison_cautions']
    (OUT/'budget_audit.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(training_minutes=actual_training_seconds/60,evaluation56_minutes=common_seconds/60,
                         max_old_rollout_seconds=maximum_rollout,final_cases=final_cases,cpu_tests=report['evaluator_audit']['cpu_test_count'],
                         cpu_test_returncode=test.returncode),indent=2))


if __name__=='__main__':main()
