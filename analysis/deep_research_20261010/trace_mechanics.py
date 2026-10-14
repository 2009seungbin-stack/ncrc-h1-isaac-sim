"""Recorded first8env mechanical quantities only; no simulator or inference."""
import json
from pathlib import Path
import numpy as np
from heldout_analysis import DATA, OUT, ROOT, MODELS, SEEDS, AXES, sha


def load(model, sid):
    errors=[]; traces=[]; evidence=[]
    for seed in SEEDS:
        path=DATA/'results/heldout_flat/traces'/f'{model}_{sid}_{seed}.npz'
        with np.load(path,allow_pickle=False) as f:
            error=f['error'].copy(); valid=f['valid'].copy(); trace=f['trace_first_envs'].copy()
            assert float(f['dt'])==.02
            assert str(f['trace_columns'])=='command3,body_lin3,body_ang3,action12,joint_vel12,joint_acc12,torque12,position3'
        assert trace.shape==(1000,8,60) and error.shape==(1000,128,3) and valid.all()
        assert np.isfinite(trace).all() and np.isfinite(error).all()
        assert np.allclose(trace[:,:,[3,4,8]]-trace[:,:,:3],error[:,:8],atol=1e-7,rtol=0)
        errors.append(error.astype(np.float64)); traces.append(trace.astype(np.float64))
        evidence.append(dict(path=str(path),sha256=sha(path),reset_seed=seed,trace_environment_indices=list(range(8))))
    return np.concatenate(errors,axis=1),np.concatenate(traces,axis=1),evidence


def rms(z): return float(np.sqrt((z**2).mean()))


def window_stats(error,trace,starts,ticks):
    full_error=np.concatenate([error[s:s+ticks] for s in starts],axis=0)
    subset_error=np.concatenate([np.concatenate((error[s:s+ticks,:8],error[s:s+ticks,128:136]),axis=1) for s in starts],axis=0)
    current=np.concatenate([trace[s:s+ticks] for s in starts],axis=0)
    previous=np.concatenate([trace[s-1:s+ticks-1] for s in starts],axis=0)
    action=current[:,:,9:21]; difference=action-previous[:,:,9:21]
    full_RMS={a:rms(full_error[:,:,i]) for i,a in enumerate(AXES)}
    subset_RMS={a:rms(subset_error[:,:,i]) for i,a in enumerate(AXES)}
    torque=current[:,:,45:57];acc=current[:,:,33:45];vel=current[:,:,21:33]
    return dict(recorded_samples_control_steps_times_environments=current.shape[0]*16,
        recorded_environments_per_reset=8,combined_recorded_environments=16,full_environments_per_reset=128,combined_full_environments=256,
        first8_subset_speed_error_RMS=subset_RMS,full128_speed_error_RMS=full_RMS,
        subset_to_full_speed_error_RMS_ratio={a:subset_RMS[a]/full_RMS[a] for a in AXES},
        applied_torque=dict(RMS_Nm=rms(torque),recorded_abs_peak_Nm=float(np.abs(torque).max()),
                            per_joint_RMS_Nm=np.sqrt((torque**2).mean((0,1))).tolist()),
        raw_actor_action=dict(RMS=rms(action),recorded_abs_peak=float(np.abs(action).max())),
        raw_action_difference=dict(RMS_per_control_step=rms(difference),RMS_per_second=rms(difference/.02),
                                  definition='Current raw actor action minus previous control-step raw actor action. Per-second version divides by dt=.02; this is not the reward summed-squared action-rate value.'),
        joint_acceleration=dict(RMS_rad_s2=rms(acc),recorded_abs_peak_rad_s2=float(np.abs(acc).max())),
        joint_velocity=dict(RMS_rad_s=rms(vel),recorded_abs_peak_rad_s=float(np.abs(vel).max())),
        actual_effort_saturation_fraction=None,actual_velocity_saturation_fraction=None)


def main():
    params=ROOT/'analysis/round20_execution_20261001/results/extracted/round20_20261001/preserved/F10/2026-10-01_18-55-04_round20_F10/params/env.yaml'
    text=params.read_text(encoding='utf-8-sig')
    # Evidence lookup only; do not deserialize Python-tagged YAML or invent runtime clipping.
    lines=text.splitlines(); matches=[]
    for i,line in enumerate(lines):
        if any(key in line for key in ('effort_limit:', 'velocity_limit:', 'effort_limit_sim:', 'velocity_limit_sim:')):
            matches.append(dict(line=i+1,text=line.strip()))
    report=dict(status='RECORDED_FIRST8ENV_MECHANICAL_DIAGNOSTIC',no_simulator=True,no_policy_inference=True,
        sample_limit='Exactly environment indices0..7 per reset809/1201;16 combined episodes per model/scenario. They are not assumed to represent all256 episodes or tail environments.',
        sampling_limit='50Hz control-step snapshots of applied torque/joint acceleration. These are not all200Hz physics samples or guaranteed physical peak values.',
        saturation_status='Deferred: historical F10 params contains effort_limit23.5/velocity_limit30 with *_sim null, but evaluated runtime resolved per-joint limits and actual clipping logic were not recorded. No actual saturation rate inferred.',
        historical_limit_reference=dict(path=str(params),sha256=sha(params),matching_yaml_lines=matches,verified_as_evaluation_runtime_clipping=False),
        ranking_definition_unchanged='Full1000-step valid-error and survival definition unchanged. Listed response-window values are diagnostics only.',
        columns=dict(action=[9,21],joint_velocity=[21,33],joint_acceleration=[33,45],applied_torque=[45,57]),
        models={},change_cause_or_optimal_coefficient_conclusion=None)
    for model in MODELS:
        report['models'][model]={}
        for sid in ('G1','G6'):
            error,trace,evidence=load(model,sid);windows={}
            if sid=='G1':
                starts=[250,500,750]
                for start in starts:
                    for seconds in (.5,1.):
                        label=f'{start}'+f'_first{seconds}s'
                        windows[label]=dict(boundary_step_index=start,old_command=trace[start-1,0,:3].tolist(),new_command=trace[start,0,:3].tolist(),
                                            metrics=window_stats(error,trace,[start],round(seconds/.02)))
            else:
                starts=list(range(126,1000,125))
                for seconds in (.5,1.):
                    windows[f'post_push_first{seconds}s']=dict(push_response_start_indices=starts,
                                                            metrics=window_stats(error,trace,starts,round(seconds/.02)))
            report['models'][model][sid]=dict(evidence=evidence,windows=windows)
    ratios=[m['subset_to_full_speed_error_RMS_ratio'][a]
            for scenarios in report['models'].values() for scenario in scenarios.values()
            for w in scenario['windows'].values() for m in [w['metrics']] for a in AXES]
    report['subset_representation_diagnostic']=dict(min_RMS_ratio=min(ratios),max_RMS_ratio=max(ratios),
        interpretation='These windows show subset/full RMS ratios only. Matching RMS is not proof that first8envs represent the full distribution, worst tails, effort or saturation.')
    report['recorded_torque_ceiling_note']='Every listed window has a recorded abs peak23.5Nm, equal to the historical parameter reference. This observation is not a verified runtime saturation fraction.'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'trace_mechanics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    out=['기록된 첫8환경의 기계적 부담 진단 — 새 실행 없음','',report['sample_limit'],report['sampling_limit'],report['saturation_status'],'',
         '모델 | 시나리오/구간 | applied torque RMS Nm | 기록 abs peak Nm | action Δ RMS/step | joint acc RMS rad/s² | vx RMS 첫8/전체128 | vy RMS 첫8/전체128 | yaw RMS 첫8/전체128']
    for model, scenarios in report['models'].items():
        for sid,row in scenarios.items():
            for label,window in row['windows'].items():
                m=window['metrics']
                out.append(f"{model} | {sid}/{label} | {m['applied_torque']['RMS_Nm']:.6f} | {m['applied_torque']['recorded_abs_peak_Nm']:.6f} | {m['raw_action_difference']['RMS_per_control_step']:.6f} | {m['joint_acceleration']['RMS_rad_s2']:.6f} | "+' | '.join(f"{m['first8_subset_speed_error_RMS'][a]:.6f}/{m['full128_speed_error_RMS'][a]:.6f}" for a in AXES))
    out+=['','torque/액션/관절 가속도 자료가 없다는 표현은 부정확하다. 자료는 첫8환경에만 있다.',
          report['recorded_torque_ceiling_note'],
          f"창별 축 RMS의 첫8/전체128 비율 범위는 {min(ratios):.6f}..{max(ratios):.6f}. 평균 수준이 비슷해도 tail 대표성을 증명하지 않는다.",
          '전체128환경의 effort 포화·tail 환경 대표성·인과적 변경 효과·최적 보상계수는 이 자료로 결론낼 수 없다.',
          '환경별 반복샘플과12관절을 합친 RMS다. 시간 분모를 줄인 공식 점수나 정책 성능 예측이 아니다.']
    (OUT/'trace_mechanics.txt').write_text('\n'.join(out)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],outputs=['trace_mechanics.json','trace_mechanics.txt'],rows=16)))


if __name__=='__main__': main()
