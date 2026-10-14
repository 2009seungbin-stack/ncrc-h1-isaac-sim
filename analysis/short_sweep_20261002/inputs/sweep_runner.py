"""Bounded analysis jobs on the provided competition server. No submissions."""
import argparse
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

ROOT=Path(__file__).resolve().parent
TRAIN=Path('/workspace/training/quadruped')
ISAAC='/workspace/IsaacLab/isaaclab.sh'
WEIGHTS=dict(zip(['G1','G2','G3','G4','G5','G6','G7'],[.15,.15,.2,.15,.15,.1,.1]))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(stage,**kw):
    d=dict(stage=stage,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw)
    t=ROOT/'state.tmp';t.write_text(json.dumps(d,indent=2));t.replace(ROOT/'state.json')
    print(json.dumps(d),flush=True)

def protected():
    return {k:sha(TRAIN/k) for k in ('train.py','play.py','go2_task/agent_cfg.py','go2_task/env_cfg.py')}

def apply_rewards(name):
    original=(ROOT/'fresh_rewards_original.py').read_bytes()
    text=original.decode('utf-8')
    node=next(n.value for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REWARD_WEIGHTS' for t in n.targets))
    lines=original.splitlines(keepends=True)
    a=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    b=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    weights=json.loads((ROOT/'candidate_weights.json').read_text())[name]
    changed=original[:a]+repr(weights).encode()+original[b:]
    (TRAIN/'quadruped_rewards.py').write_bytes(changed)
    (ROOT/f'{name}_applied_rewards.py').write_bytes(changed)

def stop(p):
    if p.poll() is not None:return
    os.killpg(p.pid,signal.SIGINT)
    try:p.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def preserve(run_name):
    for run in (TRAIN/'logs/rsl_rl/quadruped').glob('*_'+run_name):
        dest=ROOT/'preserved'/run_name/run.name;dest.mkdir(parents=True,exist_ok=True)
        for p in run.glob('model_*.pt'):
            target=dest/p.name
            try:
                sig=(p.stat().st_size,p.stat().st_mtime_ns)
                if target.exists() and target.stat().st_size==sig[0]:continue
                tmp=target.with_suffix('.copying');shutil.copy2(p,tmp)
                if (p.stat().st_size,p.stat().st_mtime_ns)==sig:tmp.replace(target)
            except FileNotFoundError:pass
        if (run/'params').exists():shutil.copytree(run/'params',dest/'params',dirs_exist_ok=True)
        for p in run.glob('events.out.tfevents.*'):shutil.copy2(p,dest/p.name)

def job(cmd,label,cap,end,run_name=None):
    if time.time()+cap>end:raise RuntimeError('Insufficient bounded time: '+label)
    state('RUNNING',job=label,command=cmd,deadline=end)
    env=os.environ.copy();env['PYTHONPATH']=str(TRAIN)+os.pathsep+env.get('PYTHONPATH','')
    begun=time.time()
    with (ROOT/'logs'/f'{label}.log').open('w') as f:
        p=subprocess.Popen(cmd,cwd=TRAIN,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if run_name:preserve(run_name)
                if time.time()>end or time.time()-begun>cap:raise RuntimeError('Deadline '+label)
                time.sleep(2)
        finally:
            stop(p)
            if run_name:preserve(run_name)
    if p.returncode:raise RuntimeError(f'{label} failed: exit {p.returncode}')
    state('JOB_COMPLETE',job=label,seconds=time.time()-begun)

def evaluate(models,label,ids,end,n=32,seeds='211,607',terrain_seed=1234,startup_seed=42,cap=1800):
    args=[ISAAC,'-p',str(ROOT/'batch_eval.py'),'--task','Quadruped-v0','--task_module','go2_task',
          '--train_script',str(TRAIN/'train.py'),'--models_json',str(ROOT/models),
          '--output_dir',str(ROOT/'results'/label),'--num_envs',str(n),'--seeds',seeds,
          '--scenarios',ids,'--startup_seed',str(startup_seed),'--terrain_seed',str(terrain_seed),
          '--transient_s','0.5','--headless']
    job(args,label,cap,end)

def summarize(labels,out):
    all_models={}
    for label in labels:
        for p in (ROOT/'results'/label).glob('*.json'):
            d=json.loads(p.read_text());model=all_models.setdefault(d['model'],{'scenarios':{},'checkpoint_sha256':d['checkpoint_sha256']})
            for sid,rows in d['scenarios'].items():
                assert sid not in model['scenarios'],'Duplicate scenario'
                count=sum(x['full']['count'] for x in rows)
                mse=[sum(x['full']['sum_squared'][i] for x in rows)/count for i in range(3)]
                episodes=sum(x['episodes'] for x in rows);survived=sum(x['survived'] for x in rows)
                sv=survived/episodes;tracking=math.exp(-math.sqrt(sum(mse)/3))
                model['scenarios'][sid]=dict(episodes=episodes,survived=survived,survival=sv,
                     rmse=dict(zip(('vx','vy','yaw'),map(math.sqrt,mse))),tracking=tracking,score=sv*tracking)
    for model in all_models.values():
        norm=sum(WEIGHTS.get(k,0) for k in model['scenarios'])
        model['partial_weighted_quality']=sum(WEIGHTS.get(k,0)*s['score'] for k,s in model['scenarios'].items())/norm if norm else None
        model['weighted_scenario_coverage']=norm
        model['falls']=sum(s['episodes']-s['survived'] for s in model['scenarios'].values())
    result=dict(models=all_models,warning='Partial scenario quality is NOT a full score; no statistical significance claim')
    (ROOT/out).write_text(json.dumps(result,indent=2));return result

def bundle():
    archive=ROOT.parent/'short_sweep_20261002_results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        for p in ROOT.iterdir():
            if p.name not in ('models','__pycache__'):tf.add(p,arcname=ROOT.name+'/'+p.name)
    print('RETRIEVE',str(archive),'BYTES',archive.stat().st_size,'SHA256',sha(archive),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['screen','short_train'])
    parser.add_argument('--deadline',required=True,type=float)
    args=parser.parse_args();(ROOT/'logs').mkdir(exist_ok=True)
    snapshot=protected()
    previous=json.loads((ROOT/'previous_source_hashes.json').read_text())
    if snapshot!=previous:raise RuntimeError('Provided training sources changed; review before execution')
    path=ROOT/'protected_sources.json'
    if not path.exists():path.write_text(json.dumps(snapshot,indent=2))
    original=TRAIN/'quadruped_rewards.py'
    backup=ROOT/'fresh_rewards_original.py'
    if not backup.exists():backup.write_bytes(original.read_bytes())
    try:
        if args.stage=='screen':
            apply_rewards('control')
            # Seven models share identical startup material/mass and terrain in each process.
            evaluate('saved_models.json','saved_flat','G1,G2,G6',args.deadline,cap=3600)
            evaluate('saved_models.json','saved_stairs','G5',args.deadline,cap=2400)
            summary=summarize(['saved_flat','saved_stairs'],'saved_ranking.json')
            state('SAVED_SCREEN_COMPLETE',ranking=summary)
        else:
            manifests=[]
            for name in ['control','action','acc','linear']:
                run_name='short_20261002_'+name
                apply_rewards(name)
                job([ISAAC,'-p','train.py','--task','Quadruped-v0','--num_envs','6144','--seed','42',
                     '--max_iterations','800','--run_name',run_name,'--headless'],
                    'train_'+name,5100,args.deadline,run_name)
                files=list((ROOT/'preserved'/run_name).glob('*/model_*.pt'))
                numeric=[p for p in files if p.stem.split('_')[-1].isdigit()]
                ckpt=max(numeric,key=lambda p:int(p.stem.split('_')[-1]))
                assert int(ckpt.stem.split('_')[-1])>=700
                manifests.append(dict(name=name,checkpoint=str(ckpt.relative_to(ROOT)),sha256=sha(ckpt)))
                (ROOT/'short_models.json').write_text(json.dumps(manifests,indent=2))
            apply_rewards('control')
            evaluate('short_models.json','short_flat','G1,G2,G6',args.deadline,cap=2400)
            evaluate('short_models.json','short_stairs','G5',args.deadline,cap=1800)
            summary=summarize(['short_flat','short_stairs'],'short_ranking.json')
            state('SHORT_SCREEN_COMPLETE',ranking=summary)
    except Exception as e:
        traceback.print_exc();state('STOPPED_REQUIRES_REVIEW',error=str(e));raise
    finally:
        original.write_bytes(backup.read_bytes())
        assert protected()==snapshot and sha(original)==sha(backup)
        (ROOT/'reward_restored.json').write_text(json.dumps({'restored':True,'sha256':sha(original)}))
        bundle()

if __name__=='__main__':main()
