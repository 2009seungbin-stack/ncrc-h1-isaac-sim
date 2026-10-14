"""Competition-host runner: reward values only, bounded jobs, preserved outputs.

No training/algorithm implementation is changed. Run inside tmux on assigned host.
"""
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
TRAIN = Path('/workspace/training/quadruped')
ISAAC = '/workspace/IsaacLab/isaaclab.sh'
PYTHON = '/workspace/IsaacLab/_isaac_sim/python.sh'
END = dt.datetime.fromisoformat('2026-10-02T00:01:37+00:00').timestamp()
TRAIN_END = dt.datetime.fromisoformat('2026-10-01T21:01:37+00:00').timestamp()
START = dt.datetime.fromisoformat('2026-10-01T04:31:37+00:00').timestamp()
STATE = ROOT / 'state.json'
GROUPS = [('flat', 'G1,G2,G6'), ('rough', 'G3'), ('slope', 'G4'),
          ('stairs', 'G5'), ('dr', 'G7')]
WEIGHTS = dict(zip(['G1','G2','G3','G4','G5','G6','G7'], [.15,.15,.20,.15,.15,.10,.10]))

def save_state(stage, **kw):
    data = dict(stage=stage, utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_hours=(time.time()-START)/3600, **kw)
    p = STATE.with_suffix('.tmp')
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
    p.replace(STATE)
    print(json.dumps(data), flush=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reward_node(text):
    for n in ast.parse(text).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='REWARD_WEIGHTS' for t in n.targets):
            if not isinstance(n.value, ast.Dict):
                raise RuntimeError('REWARD_WEIGHTS must be literal dict')
            return n.value
    raise RuntimeError('Missing REWARD_WEIGHTS')

def parts(text, n):
    # AST column offsets are UTF-8 bytes, so preserve the supplied file as bytes.
    lines = text.encode('utf-8').splitlines(keepends=True)
    a = sum(map(len, lines[:n.lineno-1])) + n.col_offset
    b = sum(map(len, lines[:n.end_lineno-1])) + n.end_col_offset
    raw = text.encode('utf-8')
    return raw[:a], raw[b:]

def apply_rewards(name):
    original = (ROOT/'provided_rewards_original.py').read_text(encoding='utf-8')
    candidate = (ROOT/f'{name}_rewards.py').read_text(encoding='utf-8')
    n = reward_node(candidate)
    weights = ast.literal_eval(n)
    assert all(isinstance(k,str) and isinstance(v,(float,int)) and math.isfinite(v) for k,v in weights.items())
    prefix, suffix = parts(original, reward_node(original))
    changed = prefix + repr(weights).encode() + suffix
    assert parts(changed.decode(), reward_node(changed.decode())) == (prefix, suffix)
    (TRAIN/'quadruped_rewards.py').write_bytes(changed)
    (ROOT/f'{name}_applied_rewards.py').write_bytes(changed)
    (ROOT/f'{name}_weights.json').write_text(json.dumps(weights,indent=2))

def terminate(p):
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()

def backup_models(name):
    for run in (TRAIN/'logs/rsl_rl/quadruped').glob(f'*_round20_{name}'):
        dest = ROOT/'preserved'/name/run.name
        dest.mkdir(parents=True, exist_ok=True)
        for src in run.glob('model_*.pt'):
            dst = dest/src.name
            try:
                sig = (src.stat().st_size, src.stat().st_mtime_ns)
                if dst.exists() and dst.stat().st_size == sig[0]:
                    continue
                tmp = dst.with_suffix('.copying')
                shutil.copy2(src, tmp)
                if (src.stat().st_size,src.stat().st_mtime_ns)==sig:
                    tmp.replace(dst)
            except FileNotFoundError:
                pass
        params = run/'params'
        if params.exists():
            shutil.copytree(params,dest/'params',dirs_exist_ok=True)
        for src in run.glob('events.out.tfevents.*'):
            shutil.copy2(src, dest/src.name)

def job(cmd, label, timeout, backup=None):
    if time.time()+timeout > END:
        raise RuntimeError(f'Insufficient time for {label}')
    save_state('RUNNING', job=label, command=cmd, timeout_seconds=timeout)
    log = ROOT/'logs'/f'{label}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:
        env=os.environ.copy()
        env['PYTHONPATH']=str(TRAIN)+os.pathsep+env.get('PYTHONPATH','')
        p=subprocess.Popen(cmd,cwd=TRAIN,stdout=f,stderr=subprocess.STDOUT,
                           start_new_session=True,env=env)
        begun=time.time()
        try:
            while p.poll() is None:
                if backup:
                    backup_models(backup)
                if time.time()-begun>timeout or time.time()>END:
                    terminate(p)
                    raise RuntimeError(f'Time limit reached in {label}')
                time.sleep(1)
        finally:
            terminate(p)
            if backup:
                backup_models(backup)
    if p.returncode:
        raise RuntimeError(f'{label} exit={p.returncode}; inspect {log}')
    save_state('JOB_COMPLETE',job=label,seconds=time.time()-begun)

def train(name):
    # Two scratch jobs replace resume screening + finalist; no further training queued.
    cap=6*3600
    if time.time()+cap+3.5*3600>END or time.time()>TRAIN_END:
        raise RuntimeError('Training would consume protected validation budget')
    apply_rewards(name)
    job([ISAAC,'-p','train.py','--task','Quadruped-v0','--num_envs','6144',
         '--seed','42','--max_iterations','4000','--run_name',f'round20_{name}',
         '--headless'],f'train_{name}',cap,backup=name)
    out=ROOT/'exports'/name
    shutil.copytree(TRAIN/'exported',out,dirs_exist_ok=True)

def checkpoints(name):
    paths=list((ROOT/'preserved'/name).glob('*/model_*.pt'))
    paths=[p for p in paths if p.stem.split('_')[-1].isdigit()]
    if not paths:
        raise RuntimeError(f'No preserved models for {name}')
    paths.sort(key=lambda p:int(p.stem.split('_')[-1]))
    chosen={paths[-1], min(paths,key=lambda p:abs(int(p.stem.split('_')[-1])-3000))}
    return sorted(chosen, key=lambda p:int(p.stem.split('_')[-1]))

def evaluate(label, checkpoint, seeds, groups=GROUPS, extra=()):
    paths=[]
    folder=ROOT/'eval'/label
    folder.mkdir(parents=True,exist_ok=True)
    for kind, ids in groups:
        out=folder/f'{kind}.json'
        cmd=[ISAAC,'-p',str(ROOT/'t_eval_go2.py'),'--task','Quadruped-v0',
             '--task_module','go2_task','--train_script',str(TRAIN/'train.py'),
             '--checkpoint',str(checkpoint),'--num_envs','128','--seeds',seeds,
             '--scenarios',ids,'--run_id',label,'--out',str(out),'--headless',*extra]
        job(cmd,f'eval_{label}_{kind}',12*60)
        if not out.exists():
            raise RuntimeError(f'Evaluator did not create {out}')
        paths.append(out)
    # Merge ourselves to avoid booting Isaac Sim; preserve literal scenario metrics.
    scenarios={}
    for path in paths:
        data=json.loads(path.read_text())
        scenarios.update(data['scenarios'])
    merged=dict(run_id=label,checkpoint=str(checkpoint),seeds=seeds,
                scenarios=scenarios,proxy_warning='Not an official score or calibrated confidence interval')
    (folder/'merged.json').write_text(json.dumps(merged,indent=2))
    return merged

def proxy(data):
    assert all(k in data['scenarios'] for k in WEIGHTS)
    floor=ratio=raw=survival=0.0
    for sid,w in WEIGHTS.items():
        s=data['scenarios'][sid]
        assert s.get('nan_actions',0)==0 and s.get('crashes',0)==0
        mse=sum(float(s['rmse'][a])**2 for a in ('vx','vy','yaw'))/3
        sv=float(s['survival'])
        floor+=70*w*sv*math.exp(-math.sqrt(mse+.02684614570005177))
        ratio+=70*w*sv*.905*math.exp(-math.sqrt(mse))
        raw+=70*w*sv*math.exp(-math.sqrt(mse))
        survival+=w*sv
    return dict(floor=floor,ratio=ratio,raw=raw,weighted_survival=survival)

def passes(candidate, original):
    c,b=proxy(candidate),proxy(original)
    return (c['floor']>=b['floor']+.15 and c['ratio']>=b['ratio']+.15
            and all(candidate['scenarios'][k]['survival']>=original['scenarios'][k]['survival'] for k in WEIGHTS))

def bundle():
    # Results, original policy, intermediate checkpoints, training logs and configs.
    target=ROOT.parent/'round20_results_20261001.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=1) as tf:
        for p in ROOT.iterdir():
            if p.name != 'round20_inputs.tar.gz':
                tf.add(p,arcname=f'round20_20261001/{p.name}')
    print(f'RETRIEVE {target} SHA256={sha(target)}',flush=True)

def main():
    save_state('STARTED',mode='scratch_pair',training_seed=42,
               reason='Installed PPO.load does not restore learning_rate scalar',
               server_budget_hours=20,official_submission=False)
    protected={p:sha(TRAIN/p) for p in ['train.py','play.py','go2_task/agent_cfg.py','go2_task/env_cfg.py']}
    (ROOT/'protected_source_hashes.json').write_text(json.dumps(protected,indent=2))
    assert sha(ROOT/'original_C0_model_5500.pt')=='8d553e697ed4c2856ab002431af96d626eb8b0d23e56b7c754d652193026b1d3'
    try:
        train('F00')
        train('F10')
        apply_rewards('F00')  # Matching evaluation environment reward config for all policies.
        base=ROOT/'original_C0_model_5500.pt'
        scores={}
        results={}
        results['original_A']=evaluate('original_A',base,'103,280,512,991')
        scores['original_A']=proxy(results['original_A'])
        candidates={}
        for name in ['F00','F10']:
            for ckpt in checkpoints(name):
                label=f'{name}_{ckpt.stem}_A'
                candidates[label]=ckpt
                results[label]=evaluate(label,ckpt,'103,280,512,991')
                scores[label]=proxy(results[label])
        winner=max(candidates,key=lambda k:scores[k]['floor'])
        chosen=candidates[winner]
        # Held-out seeds are evaluated once after selection; never tune using them.
        b0=evaluate('original_B',base,'137,389,743,1201')
        b1=evaluate('selected_B',chosen,'137,389,743,1201')
        accepted=passes(results[winner],results['original_A']) and passes(b1,b0)
        detail=dict(selected_screening_candidate=winner,selected_checkpoint=str(chosen),
                    accepted_for_review=accepted,screening=scores,
                    heldout_original=proxy(b0),heldout_selected=proxy(b1),
                    warning='Proxy gate is heuristic. No official score guarantee; no automatic submission.')
        (ROOT/'selection.json').write_text(json.dumps(detail,indent=2))
        # Boundary and push/command-switch regression checks, original and selected.
        for name,ckpt in [('original',base),('selected',chosen)]:
            evaluate(f'{name}_stress',ckpt,'137,389,743,1201',
                     [('flat','GX1,GX5')],['--transient_s','0'])
            for height in ['0.10','0.15']:
                evaluate(f'{name}_stairs_{height}',ckpt,'137,389,743,1201',
                         [('stairs','G5')],['--stair_height',height])
        review=ROOT/'review_candidate'
        review.mkdir(exist_ok=True)
        shutil.copy2(chosen,review/'model_candidate.pt')
        shutil.copy2(chosen.parent/'params/env.yaml',review/'env.yaml')
        # Provided play.py exports policy and records video; do not rewrite it.
        job([ISAAC,'-p','play.py','--task','Quadruped-v0','--num_envs','4',
             '--checkpoint',str(chosen),'--headless','--video','--video_length','1000',
             '--enable_cameras'],'candidate_video',20*60)
        shutil.copytree(TRAIN/'exported',review/'provided_play_output',dirs_exist_ok=True)
        save_state('ARTIFACTS_READY',selection=detail)
    finally:
        shutil.copy2(ROOT/'provided_rewards_original.py',TRAIN/'quadruped_rewards.py')
        assert all(sha(TRAIN/p)==v for p,v in protected.items()), 'Protected source changed'
        bundle()

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        save_state('STOPPED_REQUIRES_REVIEW',error=str(e))
        # Rebundle so the terminal error state is included. No dummy training or keepalive.
        bundle()
        raise
