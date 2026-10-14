"""Reanalyse recorded Go2 trajectories; no model, simulator, server or training.

Full-window ranking stays intact. Timing windows and zero-error sensitivity are
diagnostics only. B1 has no measured rollout and receives no quality estimate.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT/'analysis/short_sweep_20261002/recovered_20261003/short_sweep_20261002'
RECENT = ROOT/'analysis/high_score_search_20261009'
AXES = ('vx','vy','yaw')
MODELS = ('C0_5500','F10_3000')
SEEDS = (809,1201)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(node):
    if isinstance(node,ast.Constant): return node.value
    if isinstance(node,(ast.List,ast.Tuple)):
        values=[literal(v) for v in node.elts]
        return tuple(values) if isinstance(node,ast.Tuple) else values
    if isinstance(node,ast.Dict): return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        return {k.arg:literal(k.value) for k in node.keywords}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub): return -literal(node.operand)
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult): return literal(node.left)*literal(node.right)
    raise ValueError('Unexpected scenario AST node '+type(node).__name__)


def scenarios_from_source(path):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n.value for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='GO2_SCENARIOS' for t in n.targets))
    values=literal(node)
    return {sid:values[sid] for sid in ('G1','G2','G3','G4','G5','G6','G7')}


def metrics(error,valid):
    z=error[valid].astype(np.float64)
    if not len(z) or not np.isfinite(z).all(): raise ValueError('Missing/nonfinite valid error')
    counts=valid.sum(0)
    sse=np.where(valid[:,:,None],error.astype(np.float64)**2,0).sum(0)
    ep_rmse=np.sqrt(sse/np.maximum(counts[:,None],1))
    result={}
    for i,a in enumerate(AXES):
        absolute=np.abs(z[:,i]); squared=z[:,i]**2
        shares={}
        for f in (.01,.05,.10):
            count=max(1,math.ceil(len(squared)*f))
            shares[str(f)]=float(np.partition(squared,-count)[-count:].sum()/max(squared.sum(),1e-20))
        result[a]=dict(RMSE=float(np.sqrt(squared.mean())),MAE=float(absolute.mean()),signed_bias=float(z[:,i].mean()),
            sample_abs_error_quantiles={str(q):float(np.quantile(absolute,q)) for q in (.5,.9,.95,.99,.999)},
            top_sample_MSE_share=shares,
            episode_RMSE_quantiles={str(q):float(np.quantile(ep_rmse[counts>0,i],q)) for q in (.5,.9,.95,.99,1)},
            episode_max_abs_error_quantiles={str(q):float(np.quantile(np.where(valid,np.abs(error[:,:,i]),0).max(0),q)) for q in (.5,.9,.95,.99,1)},
            SSE=float(squared.sum()),SAE=float(absolute.sum()))
    return dict(samples=int(valid.sum()),episodes=len(counts),per_episode_valid_counts=counts.tolist(),axes=result)


def phase_masks(spec,window_seconds=.5):
    window=int(round(window_seconds/.02)); masks={name:np.zeros(1000,bool) for name in ('initial','true_command_transition','post_push','ordinary')}
    masks['initial'][:window]=True
    for segment in range(1,4):
        if tuple(spec['cmds'][segment]) != tuple(spec['cmds'][segment-1]):
            start=segment*250; masks['true_command_transition'][start:start+window]=True
    if spec.get('push'):
        # Legacy push is written after physical step125,250,...; first affected step126.
        for push_after in range(125,1000,125):
            start=push_after+1; masks['post_push'][start:min(start+window,1000)]=True
    union=masks['initial']|masks['true_command_transition']|masks['post_push']
    masks['ordinary']=~union
    if sum(int(v.sum()) for v in masks.values())!=1000: raise ValueError('Timing masks overlap')
    return masks


def phase_metrics(error,valid,spec,window=.5):
    masks=phase_masks(spec,window)
    full_sse=np.where(valid[:,:,None],error.astype(np.float64)**2,0).sum((0,1))
    out={}
    for name,mask in masks.items():
        m=valid&mask[:,None]
        sse=np.where(m[:,:,None],error.astype(np.float64)**2,0).sum((0,1))
        count=int(m.sum())
        out[name]=dict(samples=count,full_window_sample_share=count/int(valid.sum()),
            SSE=dict(zip(AXES,sse.tolist())),full_window_MSE_share=dict(zip(AXES,(sse/np.maximum(full_sse,1e-20)).tolist())),
            diagnostic_only_RMSE={a:math.sqrt(float(sse[i])/count) if count else None for i,a in enumerate(AXES)})
    return out


def large_error_diagnostics(error, valid):
    """Measured threshold tails, with full live-sample denominators unchanged."""
    z = error[valid].astype(np.float64)
    full_sse = (z**2).sum(0)
    out = {}
    for label, amplitude in (('XY_error_norm_mps', np.linalg.norm(z[:, :2], axis=1)),
                             ('absolute_yaw_error_radps', np.abs(z[:, 2]))):
        out[label] = {}
        for threshold in (.5, 1., 1.5):
            selected = amplitude > threshold
            sse = (z[selected]**2).sum(0)
            out[label][str(threshold)] = dict(
                strict_greater_than=threshold, samples=int(selected.sum()),
                full_valid_sample_fraction=float(selected.mean()),
                SSE=dict(zip(AXES, sse.tolist())),
                full_axis_SSE_fraction=dict(zip(AXES, (sse / np.maximum(full_sse, 1e-20)).tolist())))
    return out


def q3_changed_error_counterfactuals(rows, weights):
    """Algebraic sensitivity only: no assumed feasible dynamics or official score."""
    mse = {s: {a: r['full_window']['axes'][a]['SSE'] / r['full_window']['samples'] for a in AXES}
           for s, r in rows.items()}
    survival = {s: r['survived'] / r['episodes'] for s, r in rows.items()}

    def quality(values):
        return sum(weights[s] * survival[s] * math.exp(-math.sqrt(sum(v.values()) / 3))
                   for s, v in values.items())

    baseline = quality(mse)
    results = {}
    for label, axes in (('vx_only', ('vx',)), ('yaw_only', ('yaw',)), ('XY_both', ('vx', 'vy')),
                        ('all_three_axes', AXES)):
        for reduction in (.1, .2, .3):
            changed = {s: {a: value * ((1 - reduction)**2 if a in axes else 1)
                           for a, value in values.items()} for s, values in mse.items()}
            q = quality(changed)
            results[f'{label}_RMSE_minus_{round(reduction * 100)}pct'] = dict(
                RMSE_reduction_fraction=reduction, affected_axes=list(axes),
                counterfactual_Q3=q, delta_Q3=q-baseline)
    prefixes = ('initial', 'true_command_transition', 'post_push')
    for label, phases in (('initial_and_true_changes', prefixes[:2]),
                          ('post_push', ('post_push',)), ('all_short_response_windows', prefixes),
                          ('ordinary_yaw_only', ('ordinary',))):
        axes = ('yaw',) if label == 'ordinary_yaw_only' else ('vx',)
        for reduction in (.25, .5, 1.):
            changed = {}
            for sid, values in mse.items():
                n = rows[sid]['full_window']['samples']
                change = {a: sum(rows[sid]['phases_0p5s'][p]['SSE'][a] for p in phases) / n
                          for a in axes}
                changed[sid] = {a: max(0., value-reduction * change.get(a, 0.)) for a, value in values.items()}
            q = quality(changed)
            results[f'{label}_SSE_minus_{round(reduction * 100)}pct'] = dict(
                SSE_reduction_fraction=reduction, diagnostic_window_seconds=.5,
                phases=list(phases), affected_axes=list(axes), counterfactual_Q3=q, delta_Q3=q-baseline,
                full_denominator_and_survival_preserved=True)
    return dict(baseline_Q3=baseline, cases=results,
                interpretation='Changed-error algebra only. No feasible policy forecast, no official score conversion; zeroing a phase is an unattainability-agnostic sensitivity bound.')


def transient_details(error,valid,spec):
    result=[]
    for segment in range(4):
        start=segment*250; previous=spec['cmds'][segment-1] if segment else None
        actual_change=segment==0 or tuple(previous)!=tuple(spec['cmds'][segment])
        if not actual_change: continue
        e=error[start:start+250].astype(np.float64)
        v=valid[start:start+250]
        block=metrics(e,v)
        prefixes={}
        for seconds in (.25,.5,1.):
            ticks=round(seconds/.02); mask=v[:ticks]
            z=e[:ticks][mask]
            prefixes[str(seconds)]={a:dict(RMSE=float(np.sqrt((z[:,i]**2).mean())),signed_bias=float(z[:,i].mean())) for i,a in enumerate(AXES)}
        # Exploratory response diagnostic; not a control law or an official scoring threshold.
        settled=[]
        for ep in range(e.shape[1]):
            okay=(np.abs(e[:,ep,0])<=.1)&v[:,ep]
            starts=np.flatnonzero(np.convolve(okay.astype(np.int64),np.ones(10,np.int64),mode='valid')==10)
            settled.append(float(starts[0]*.02) if len(starts) else None)
        finite=[x for x in settled if x is not None]
        result.append(dict(segment=segment,start_seconds=start*.02,previous_command=list(previous) if previous else None,
            command=list(spec['cmds'][segment]),kind='initial' if segment==0 else 'true_command_change',
            full_5s_metrics=block['axes'],diagnostic_prefixes=prefixes,
            vx_settle_diagnostic=dict(threshold_mps=.1,consecutive_steps=10,not_an_official_threshold=True,
                 measured_episodes=len(settled),settled_within_5s=len(finite),
                 median_seconds=float(np.median(finite)) if finite else None,
                 p90_seconds=float(np.quantile(finite,.9)) if finite else None)))
    return result


def main():
    core=DATA/'t_eval_go2_core.py'; specs=scenarios_from_source(core)
    weights={sid:float(spec['w']) for sid,spec in specs.items()}
    portal=json.loads((RECENT/'portal_evidence.json').read_text(encoding='utf-8-sig'))
    assert list(weights.values())==portal['rules']['scenario_weights']
    summary=json.loads((RECENT/'screen_summary.json').read_text(encoding='utf-8-sig'))['models']
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
        (core,DATA/'t_eval.py',DATA/'batch_eval.py',DATA/'sweep_runner.py',DATA/'finalist_runner.py',RECENT/'screen_summary.json',RECENT/'portal_evidence.json')}
    raw={}; totals={}
    for model in MODELS:
        raw[model]={}
        for sid,spec in specs.items():
            errors=[];valids=[];evidence=[]
            for seed in SEEDS:
                path=DATA/'results'/('heldout_'+spec['terrain'])/'traces'/f'{model}_{sid}_{seed}.npz'
                with np.load(path,allow_pickle=False) as f:
                    e=f['error'].copy(); v=f['valid'].copy(); trace=f['trace_first_envs'].copy(); step_dt=float(f['dt'])
                assert e.shape==(1000,128,3) and v.shape==(1000,128) and trace.shape==(1000,8,60)
                assert step_dt==.02 and not (np.diff(v.astype(np.int64),axis=0)>0).any()
                commands=np.repeat(np.asarray(spec['cmds']),250,axis=0)
                assert np.allclose(trace[:,:,:3],commands[:,None,:],atol=1e-7,rtol=0)
                evidence.append(dict(path=str(path.resolve()),sha256=sha(path),seed=seed,steps=1000,dt=step_dt,valid_samples=int(v.sum())))
                errors.append(e);valids.append(v)
            # Concatenate environments, not time:256initial episodes, each1000steps.
            error=np.concatenate(errors,axis=1); valid=np.concatenate(valids,axis=1)
            m=metrics(error,valid); phases=phase_metrics(error,valid,spec)
            episodes=valid.shape[1]; survived=int(valid[-1].sum())
            tracking=math.exp(-math.sqrt(sum(m['axes'][a]['RMSE']**2 for a in AXES)/3))
            row=dict(raw_evidence=evidence,full_window=m,phases_0p5s=phases,
                large_error_diagnostics=large_error_diagnostics(error, valid),
                phase_window_sensitivity={str(s):phase_metrics(error,valid,spec,s) for s in (.25,1.)},
                actual_transitions=transient_details(error,valid,spec),episodes=episodes,survived=survived,falls=episodes-survived,
                tracking=tracking,Q3_scenario=(survived/episodes)*tracking,
                max_RMSE_difference_from_latest_reference_summary=max(abs(m['axes'][a]['RMSE']-summary[model]['scenarios'][sid]['rmse'][a]) for a in AXES))
            assert row['max_RMSE_difference_from_latest_reference_summary'] < 1e-12
            assert all(n == 1000 for n in m['per_episode_valid_counts']), 'Unexpected early termination in known all-survival raw references'
            for a in AXES:
                assert math.isclose(sum(p['SSE'][a] for p in phases.values()), m['axes'][a]['SSE'], rel_tol=1e-12, abs_tol=1e-10)
            raw[model][sid]=row
        axis_mse={a:sum(weights[s]*raw[model][s]['full_window']['axes'][a]['RMSE']**2 for s in specs) for a in AXES}
        mixed=sum(axis_mse.values())/3
        global_phases={}
        for name in ('initial','true_command_transition','post_push','ordinary'):
            contributed={a:sum(weights[s]*raw[model][s]['phases_0p5s'][name]['SSE'][a]/raw[model][s]['full_window']['samples'] for s in specs) for a in AXES}
            global_phases[name]=dict(weighted_sample_share=sum(weights[s]*raw[model][s]['phases_0p5s'][name]['full_window_sample_share'] for s in specs),
                weighted_MSE_contribution=contributed,axis_MSE_fraction={a:contributed[a]/axis_mse[a] for a in AXES},
                three_axis_MSE_fraction=sum(contributed.values())/(3*mixed))
        threshold_totals = {}
        for label in ('XY_error_norm_mps', 'absolute_yaw_error_radps'):
            threshold_totals[label] = {}
            for threshold in ('.5', '1.', '1.5'):
                key = str(float(threshold))
                contributed = {a: sum(weights[s] * raw[model][s]['large_error_diagnostics'][label][key]['SSE'][a]
                                      / raw[model][s]['full_window']['samples'] for s in specs) for a in AXES}
                threshold_totals[label][key] = dict(
                    weighted_valid_sample_fraction=sum(weights[s] * raw[model][s]['large_error_diagnostics'][label][key]['full_valid_sample_fraction'] for s in specs),
                    weighted_axis_MSE_fraction={a: contributed[a] / axis_mse[a] for a in AXES},
                    weighted_three_axis_MSE_fraction=sum(contributed.values()) / (3 * mixed))
        totals[model]=dict(weighted_Q3=sum(weights[s]*raw[model][s]['Q3_scenario'] for s in specs),
            weighted_axis_RMSE={a:math.sqrt(axis_mse[a]) for a in AXES},weighted_three_axis_RMSE=math.sqrt(mixed),
            mixed_unit_axis_MSE_share={a:v/(3*mixed) for a,v in axis_mse.items()},weighted_phases=global_phases,
            falls=sum(r['falls'] for r in raw[model].values()),episodes=7*256,
            weighted_proxy_loss_by_scenario={s:weights[s]*(1-raw[model][s]['Q3_scenario']) for s in specs},
            large_error_diagnostics=threshold_totals)
    comparison={}
    for sid in specs:
        comparison[sid]={a:dict(C0_RMSE=raw['C0_5500'][sid]['full_window']['axes'][a]['RMSE'],
            F10_RMSE=raw['F10_3000'][sid]['full_window']['axes'][a]['RMSE'],
            F10_RMSE_change_percent=100*(raw['F10_3000'][sid]['full_window']['axes'][a]['RMSE']/raw['C0_5500'][sid]['full_window']['axes'][a]['RMSE']-1),
            C0_abs_error_p99=raw['C0_5500'][sid]['full_window']['axes'][a]['sample_abs_error_quantiles']['0.99'],
            F10_abs_error_p99=raw['F10_3000'][sid]['full_window']['axes'][a]['sample_abs_error_quantiles']['0.99']) for a in AXES}
    tracking_gap={}
    f10_quality=summary['F10_3000']['weighted_quality']
    for name,model in summary.items():
        q=sum(weights[s]*r['survival']*r['tracking'] for s,r in model['scenarios'].items())
        no_fall=sum(weights[s]*r['tracking'] for s,r in model['scenarios'].items())
        gap=f10_quality-q; fall_penalty=no_fall-q
        tracking_gap[name]=dict(Q3=q,no_fall_counterfactual_Q3=no_fall,falls=sum(r['episodes']-r['survived'] for r in model['scenarios'].values()),
            gap_to_F10=gap,fall_component=fall_penalty,tracking_component=f10_quality-no_fall,
            fall_fraction_of_gap=fall_penalty/gap if gap>1e-15 else None,
            tail_information_available=name in MODELS,
            no_fall_counterfactual_definition='Only change survival multiplier; recorded tracking held fixed, not a new policy result')
    maxima=[70*w for w in weights.values()]; points=portal['board']['scenario_points']
    official_loss={s:maxima[i]-points[i] for i,s in enumerate(specs)}
    report=dict(status='LOCAL_RECORDED_TRAJECTORY_REANALYSIS',raw_dataset='2026-10-02 C0/F10 heldout seeds809/1201,env128,startup303,terrain9049',
        later_summary_dataset='2026-10-09 four-policy summary; S1/R1 raw trajectories unavailable locally',
        source_SHA256=sources,scenario_definitions=specs,scenario_weights=weights,
        source_score_formula='Internal Q3 = sum_s weight*survival*exp(-sqrt((MSEvx+MSEvy+MSEyaw)/3)); not official scenario formula',
        official_formula_known='Simulation score is sum of seven scenario points/max70; official mapping from physical errors is not known',
        protocol=dict(full_window=True,command_observation_sync=True,new_noise_sampling_on_sync=False,dt=.02,steps=1000,
            command_segments_seconds=5,reset_seeds=list(SEEDS),envs_per_seed=128,horizon_with_timeout_padding_seconds=20.1,
            validity='aliveaftercurrentstep; firstterminationtick and later resettrajectory excluded; no timeout allowed inside horizon',
            diagnostic_window_seconds=.5,push_physical_timing='writeafterstep125,250,...875; responsewindow startsat126,251,...876',
            ordinary_definition='outsideinitial,truecommandchangeprefixesandpostpushprefixes; G4/G5/G6 identical5scommands are not command transitions'),
        models_raw=raw,models_aggregate=totals,F10_vs_C0_by_scenario_axis=comparison,later_summary_tracking_vs_falls=tracking_gap,
        F10_changed_error_counterfactuals=q3_changed_error_counterfactuals(raw['F10_3000'], weights),
        historical_official_headroom=dict(source_snapshot_date='2026-10-09',anchor=88.14,simulation=58.14,qualitative30_ifheldfixed=30,
            scenario_point_deficits=official_loss,target93_requires=4.86,
            perfect_G3_and_G5_only_arithmetic_upper_total=88.14+official_loss['G3']+official_loss['G5'],
            interpretation='Historical score-deficit arithmetic upperbound only; not a candidate score forecast'),
        B1_measured_quality=None,B1_forecast=None,no_new_rollout=True,
        design_interpretation=dict(
            AX_linear2p25_yaw1p25='Information-seeking axis tradeoff experiment, not a supported score-leading candidate. F10 yaw improves all seven scenarios and is still 41.49% of proxy mixed MSE; lowering its reward could surrender the observed gain.',
            response_and_tail='Prioritize full-window response/tail metrics alongside means; first/true-change/post-push windows are distinct and not repeated identical command boundaries.',
            ordinary_yaw='Preserve the demonstrated yaw gain while investigating ordinary-window bias/dispersion. Ordinary samples contribute 86.73% of remaining F10 yaw SSE.',
            B2='Preserving F10 tracking while changing B1 cost terms avoids an unvalidated axis tradeoff, but there is no B1/B2 closed-loop rollout here and no measured advantage is claimed.'),
        limitations=['One training seed and unequal training lengths; no causal weight-effect estimate',
                    'Tail/phase analyses describe C0/F10 recordedbehavior only; no S1/R1tail or B1 performance invented',
                    'Initial/transition/pushMSE is not wholly removable; no dynamic lowerbound estimated',
                    'Mixed-axis proxy combinesm/sandrad/s; axisfractionsareproxy-specific, not official weights',
                    'All diagnostic windows retain full-window score; no transient deletion is used for ranking',
                    'No official score forecast or93guarantee'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'heldout_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    write_text(report)
    print(json.dumps(dict(aggregates=totals,F10_vs_C0=comparison,tracking_vs_falls=tracking_gap),indent=2))


def write_text(report):
    totals=report['models_aggregate']; raw=report['models_raw']; comparison=report['F10_vs_C0_by_scenario_axis']
    lines=['기록된 Go2 검증 결과 재분석 — 새 학습/서버/시뮬레이션 없음','',
           '원본28개 NPZ의1000스텝×128환경×3축을 다시 집계. 시드809/1201로256에피소드/시나리오.',
           '현재순위는전체구간오차이며, 시작·진짜명령전환·밀침직후 분해는진단전용.',
           'G4/G5/G6의동일5초명령 반복을 새로운전환으로오인하지않았다.','',
           '모델 | fullQ3 | vxRMSE | vyRMSE | yawRMSE | 낙상']
    for name,m in totals.items():
        lines.append(f"{name} | {m['weighted_Q3']:.9f} | "+' | '.join(f"{m['weighted_axis_RMSE'][a]:.9f}" for a in AXES)+f" | {m['falls']}/{m['episodes']}")
    lines+=['','F10의내부3축혼합MSE중각축기여 (물리단위혼합이므로공식비중아님):']
    lines+= [f"{a}: {totals['F10_3000']['mixed_unit_axis_MSE_share'][a]*100:.3f}%" for a in AXES]
    lines+=['','F10 시작/전환/외란 분해 — 전체구간분모유지:','구간 | 가중표본비중 | vxMSE기여 | yawMSE기여 | 3축MSE기여']
    for name,p in totals['F10_3000']['weighted_phases'].items():
        lines.append(f"{name} | {p['weighted_sample_share']*100:.3f}% | {p['axis_MSE_fraction']['vx']*100:.3f}% | {p['axis_MSE_fraction']['yaw']*100:.3f}% | {p['three_axis_MSE_fraction']*100:.3f}%")
    lines += ['', 'F10 큰 오차 표본 — 전체 유효 표본 분모 유지:',
              '기준 | 임계값 | 가중 표본 비중 | vx MSE 몫 | vy MSE 몫 | yaw MSE 몫 | 3축 MSE 몫']
    for label, thresholds in totals['F10_3000']['large_error_diagnostics'].items():
        for threshold, row in thresholds.items():
            lines.append(f"{label} | > {threshold} | {row['weighted_valid_sample_fraction']*100:.5f}% | "
                         + ' | '.join(f"{row['weighted_axis_MSE_fraction'][a]*100:.3f}%" for a in AXES)
                         + f" | {row['weighted_three_axis_MSE_fraction']*100:.3f}%")
    lines += ['', 'F10 내부 Q3의 오차 변화 산술 민감도 — 실행 가능한 정책이나 공식 점수 예측이 아님:',
              '변경 | 바뀐 오차로 계산한 Q3 | Q3 증가']
    for label, row in report['F10_changed_error_counterfactuals']['cases'].items():
        lines.append(f"{label} | {row['counterfactual_Q3']:.9f} | {row['delta_Q3']:+.9f}")
    lines += ['각 축 RMSE 변경은 해당 축 SSE에 (1-감소율)^2를 곱한다. 구간 SSE 변경은 해당 구간 SSE만 줄인다.',
              '전체 구간 분모와 생존율은 유지한다. 구간 SSE 100% 제거는 동역학적 실현 가능성을 평가하지 않은 상한 진단이다.']
    lines+=['','F10의C0대비시나리오/축RMSE변화(음수개선,양수악화):','시나리오 | vx | vy | yaw']
    for sid,rows in comparison.items():
        lines.append(sid+' | '+' | '.join(f"{rows[a]['F10_RMSE_change_percent']:+.3f}%" for a in AXES))
    lines+=['','F10 전체구간 꼬리:','시나리오 | 축 | 샘플절대오차p90 | p99 | 상위1%MSE몫 | 에피소드RMSEp90']
    for sid,row in raw['F10_3000'].items():
        for a,m in row['full_window']['axes'].items():
            lines.append(f"{sid} | {a} | {m['sample_abs_error_quantiles']['0.9']:.6f} | {m['sample_abs_error_quantiles']['0.99']:.6f} | {m['top_sample_MSE_share']['0.01']*100:.3f}% | {m['episode_RMSE_quantiles']['0.9']:.6f}")
    lines+=['','추종실패와희소낙상분해:']
    for name in ('S1_tracking_pair','R1_tracking_double'):
        row=report['later_summary_tracking_vs_falls'][name]
        lines.append(f"{name}: F10Q3차이{row['gap_to_F10']:.9f}, 낙상곱항기여{row['fall_fraction_of_gap']*100:.3f}%, 추종잔여{(1-row['fall_fraction_of_gap'])*100:.3f}%.")
    lines+=['곱항을제거하는산술진단이며, 낙상만제거해도같은추종성능이유지된다는예측이아니다.','',
        '역사적공식여지: G3+G5의손실합은4.25점. 이둘만완벽히회복하고다른공식점수가같으면상한92.39.',
        '이는점수예측이아니며93목표에는광범위한개선이필요하다는산술경계다.','',
        '설계선택에쓰는3개근거:',
        '1. 진짜전환및초기응답의전진오차꼬리. 반복된동일명령까지전환으로묶지말고추종대역폭과꼬리를함께검증.',
        '2. F10은yaw를줄이지만계단/밀침/전환전진오차를악화시킨다. yaw개선만으로승격하지않고vx비퇴행조건유지.',
        '3. S1/R1격차의대부분은희소낙상이아닌추종. 낙상추가가중·추종보상일괄증대에연구예산을몰지않는다.','',
        '새보상계수의효과나B1성능은이기록에서계산할수없다. 진단중큰MSE가전부제거가능한오차라고보지않는다.',
        '시나리오별원시합·전체구간오차·위상민감도·전환상세·원본SHA는heldout_analysis.json에보존.']
    (OUT/'heldout_analysis.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
