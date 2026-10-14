"""Small diagnostic of saved G1/G6 responses; no policy inference or simulator."""
import json
import math
from pathlib import Path
import numpy as np
from heldout_analysis import DATA, OUT, MODELS, SEEDS, AXES, sha, scenarios_from_source


def load(model, sid, spec):
    errors = []; valids = []; evidence = []
    for seed in SEEDS:
        path = DATA/'results'/('heldout_'+spec['terrain'])/'traces'/f'{model}_{sid}_{seed}.npz'
        with np.load(path, allow_pickle=False) as f:
            e = f['error'].copy(); v = f['valid'].copy(); trace = f['trace_first_envs'].copy()
            assert float(f['dt']) == .02
            columns = str(f['trace_columns'])
        assert e.shape == (1000, 128, 3) and v.shape == (1000, 128)
        assert trace.shape == (1000, 8, 60) and bool(v.all()) and np.isfinite(e).all()
        commands = np.repeat(np.asarray(spec['cmds'], dtype=np.float32), 250, axis=0)
        assert np.array_equal(trace[:, :, :3], np.broadcast_to(commands[:, None, :], (1000, 8, 3)))
        # MeasuredEnv stores body six = [vx,vy,vz,wx,wy,wz], after the command3.
        reconstructed = trace[:, :, [3, 4, 8]] - trace[:, :, :3]
        max_signed_error_difference = float(np.abs(reconstructed - e[:, :8]).max())
        assert max_signed_error_difference <= 1e-7
        evidence.append(dict(path=str(path), sha256=sha(path), reset_seed=seed,
                             all_1000x128_samples_valid=True,
                             signed_error_trace_max_difference=max_signed_error_difference,
                             trace_columns=columns))
        errors.append(e); valids.append(v)
    return np.concatenate(errors, axis=1).astype(np.float64), np.concatenate(valids, axis=1), evidence


def directional_metrics(error, direction):
    z = error.reshape(-1)
    signed = direction*z
    sse = z*z
    total = float(sse.sum())
    n = max(1, math.ceil(len(z)*.01))
    top = np.argpartition(sse, -n)[-n:]
    top_sse = float(sse[top].sum())
    return dict(samples=len(z), mean_actual_minus_command_mps=float(z.mean()),
                mean_direction_aligned_error_mps=float(signed.mean()), RMSE_mps=float(np.sqrt(sse.mean())),
                mean_lag_magnitude_mps=float(np.maximum(-signed, 0).mean()),
                mean_past_target_magnitude_mps=float(np.maximum(signed, 0).mean()),
                lag_SSE_fraction=float(sse[signed < 0].sum()/total),
                past_target_SSE_fraction=float(sse[signed > 0].sum()/total),
                p99_absolute_error_mps=float(np.quantile(np.abs(z), .99)),
                top1pct_error_SSE_fraction=top_sse/total,
                lag_fraction_of_top1pct_SSE=float(sse[top][signed[top] < 0].sum()/top_sse),
                past_target_fraction_of_top1pct_SSE=float(sse[top][signed[top] > 0].sum()/top_sse))


def main():
    specs = scenarios_from_source(DATA/'t_eval_go2_core.py')
    result = dict(status='SAVED_RESPONSE_DIAGNOSTIC_ONLY', dt=.02,
                  time_definition='Offset k=0 is the first physical step after setting the new command; its post-step sample is t=.02 s. Prefix .02..0.5 uses offsets0..24 and .02..1 uses offsets0..49.',
                  sign_definition='Raw e = actual body vx minus current vx command. Direction d=sign(new-old). d*e<0 is lag-side error; d*e>0 is past-target/overshoot-side error. These labels describe instantaneous target side, not identified delay, causal overshoot, or bandwidth.',
                  ranking_unchanged=True,
                  denominator_definition='Full valid 1000-step denominator remains the ranking definition. Prefix and tail statistics are diagnostic only; all recorded references have 256/256 survival per scenario.',
                  source_SHA256={p.name: sha(p) for p in (DATA/'batch_eval.py', DATA/'t_eval.py', DATA/'t_eval_go2_core.py')},
                  available_data='Signed errors/valid masks for all128envs per seed; commands/bodyvel/actions/jointvel/acc/torque/position for first8envs. No full235D actor observation, no B1 rollout, no new action inference.',
                  G1={}, G6={})
    for model in MODELS:
        error, valid, evidence = load(model, 'G1', specs['G1'])
        transitions = []
        for seg in (1, 2, 3):
            start = 250*seg; old = float(specs['G1']['cmds'][seg-1][0]); new = float(specs['G1']['cmds'][seg][0])
            direction = float(np.sign(new-old))
            row = dict(old_vx_command_mps=old, new_vx_command_mps=new, command_delta_mps=new-old,
                       direction=direction, boundary_zero_based_step=start,
                       last_old_command_sample_step=start-1, first_new_command_poststep_sample=start,
                       last_old_command_actual_mean_vx_mps=float((error[start-1,:,0]+old).mean()),
                       windows={})
            for seconds in (.5, 1., 5.):
                ticks = round(seconds/.02)
                row['windows'][str(seconds)] = directional_metrics(error[start:start+ticks,:,0], direction)
            row['mean_response_first_1s'] = [dict(time_after_command_s=(k+1)*.02,
                actual_mean_vx_mps=float((error[start+k,:,0]+new).mean()),
                mean_raw_error_mps=float(error[start+k,:,0].mean()),
                mean_direction_aligned_error_mps=float(direction*error[start+k,:,0].mean()),
                RMS_error_mps=float(np.sqrt((error[start+k,:,0]**2).mean()))) for k in range(50)]
            transitions.append(row)
        result['G1'][model] = dict(episodes=256, full_valid_samples=int(valid.sum()), evidence=evidence, transitions=transitions)
        error, valid, evidence = load(model, 'G6', specs['G6'])
        windows = {}
        for seconds in (.5, 1.):
            ticks = round(seconds/.02)
            per_push = []
            for applied_after_index in range(125, 1000, 125):
                start = applied_after_index+1
                z = error[start:min(start+ticks,1000)]
                per_push.append(dict(push_written_after_step_index=applied_after_index,
                                     first_affected_step_index=start, samples=z.shape[0]*z.shape[1],
                                     RMS_by_axis={a: float(np.sqrt((z[:,:,i]**2).mean())) for i,a in enumerate(AXES)}))
            all_windows = np.concatenate([error[p['first_affected_step_index']:min(p['first_affected_step_index']+ticks,1000)] for p in per_push],axis=0)
            windows[str(seconds)] = dict(per_push=per_push, pooled_samples=all_windows.shape[0]*256,
                                        RMS_by_axis={a:float(np.sqrt((all_windows[:,:,i]**2).mean())) for i,a in enumerate(AXES)},
                                        mean_sign_not_interpreted=True)
        result['G6'][model] = dict(episodes=256, full_valid_samples=int(valid.sum()), evidence=evidence,
                                  post_push_windows=windows,
                                  push_timing='Legacy writes push after zero-based control indices125,250,...875; response begins index126,251,...876. No actual command changes in G6.')
    OUT.mkdir(parents=True,exist_ok=True)
    result['five_line_summary'] = [
        'G1의 큰 전환 오차는 새 목표를 지나친 초과보다 지연 쪽이다: 세 전환 첫0.5초 상위1% SSE는 C0/F10 모두100% 지연 쪽이다.',
        '0.3→0.6 첫0.5초 RMSE는 F10 0.097704/C0 0.102123으로 개선되지만, 0.6→1.0은0.155954/0.135731로 악화된다.',
        '1.0→−0.5 첫0.5초 평균 actual-command는 F10 +0.570345/C0 +0.491261이고 RMSE는0.713511/0.656193이다: 후진 목표 도달 전 오차가 커진다.',
        'G6 밀침 첫0.5초 vx RMS는 F10 0.288251/C0 0.247024, vy는0.382090/0.356463으로 악화되지만 yaw는0.139607/0.158793으로 개선된다.',
        '경계250/500/750의 새 명령과 post-step 오차 부호를 원본 trace에서 검증했다. 이는 저장 응답 진단이며 B1 추론·새 점수·동역학적 지연 식별이 아니다.'
    ]
    (OUT/'response_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['G1 저장 응답 진단 — 실제 정책 추론/점수 재평가 없음','',result['sign_definition'],result['time_definition'],
             'd*e가 음수면 지연 쪽, 양수면 새 목표를 넘어선 쪽이다. 시간 지연이나 제어 대역폭을 식별한 값이 아니다.',
             '전체 유효 1000스텝과 생존 기준은 바꾸지 않았다. 아래 구간 통계는 진단 전용.','',
             '모델 | 전환 | 구간 | 평균 actual-command | 방향 정렬 평균 | RMSE | 지연 쪽 SSE | 초과 쪽 SSE | p99 | 상위1% 지연 쪽']
    for model in MODELS:
        for row in result['G1'][model]['transitions']:
            transition=f"{row['old_vx_command_mps']}→{row['new_vx_command_mps']}"
            for seconds, m in row['windows'].items():
                lines.append(f"{model} | {transition} | .02..{seconds}s | {m['mean_actual_minus_command_mps']:+.6f} | {m['mean_direction_aligned_error_mps']:+.6f} | {m['RMSE_mps']:.6f} | {m['lag_SSE_fraction']*100:.3f}% | {m['past_target_SSE_fraction']*100:.3f}% | {m['p99_absolute_error_mps']:.6f} | {m['lag_fraction_of_top1pct_SSE']*100:.3f}%")
    lines += ['','G6는 밀침 방향이 섞여 평균 부호를 해석하지 않는다. 밀침 이후 RMS만:', '모델 | 구간 | vx RMS | vy RMS | yaw RMS']
    for model in MODELS:
        for seconds, row in result['G6'][model]['post_push_windows'].items():
            lines.append(f"{model} | 첫{seconds}s | "+' | '.join(f"{row['RMS_by_axis'][a]:.6f}" for a in AXES))
    lines += ['',result['available_data'], '새 B1/B2/AX 응답이나 공식 점수를 이 자료로 예측하지 않는다.', '', '5줄 요약:']
    lines += result['five_line_summary']
    (OUT/'response_analysis.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Keep ordinary Windows Python CP949 stdout usable; artifacts themselves are UTF-8.
    print(json.dumps(dict(status=result['status'], files=['response_analysis.json','response_analysis.txt'],
                          signed_error_trace_difference=0., models=list(MODELS)), ensure_ascii=True))


if __name__ == '__main__':
    main()
