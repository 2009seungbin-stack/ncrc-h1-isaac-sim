"""Validation only: preserved Go2 checkpoints; no training or official submission."""
import ast, datetime as dt, hashlib, json, math, os
from pathlib import Path
import shutil, signal, subprocess, tarfile, time, traceback

ROOT = Path(__file__).resolve().parent
TRAIN = Path('/workspace/training/quadruped')
ISAAC = '/workspace/IsaacLab/isaaclab.sh'
END = dt.datetime.fromisoformat('2026-10-02T00:01:37+00:00').timestamp()
GROUPS = [('flat','G1,G2,G6'),('rough','G3'),('slope','G4'),('stairs','G5'),('dr','G7')]
WEIGHTS = dict(zip(['G1','G2','G3','G4','G5','G6','G7'], [.15,.15,.20,.15,.15,.10,.10]))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state(stage, **kw):
    data=dict(stage=stage,utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw)
    tmp=ROOT/'state.tmp'; tmp.write_text(json.dumps(data,indent=2)); tmp.replace(ROOT/'state.json')
    print(json.dumps(data),flush=True)
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGINT)
        try: p.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()
def job(cmd,label,cap=720):
    if time.time()+cap > END: raise RuntimeError('Insufficient protected time: '+label)
    state('RUNNING',job=label,command=cmd)
    env=os.environ.copy(); env['PYTHONPATH']=str(TRAIN)+os.pathsep+env.get('PYTHONPATH','')
    begun=time.time()
    with (ROOT/'logs'/f'{label}.log').open('w') as f:
        p=subprocess.Popen(cmd,cwd=TRAIN,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if time.time()-begun>cap or time.time()>END: raise RuntimeError('Job deadline '+label)
                time.sleep(1)
        finally: stop(p)
    if p.returncode: raise RuntimeError(f'{label} exit={p.returncode}')
    state('JOB_COMPLETE',job=label,seconds=time.time()-begun)
def finite(x):
    if isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)
    elif isinstance(x,float): assert math.isfinite(x), 'Nonfinite evaluation output'
def evaluate(label,ckpt,groups=GROUPS,extra=()):
    folder=ROOT/'eval'/label; folder.mkdir(parents=True,exist_ok=True)
    scenarios={}
    for kind,ids in groups:
        out=folder/f'{kind}.json'
        cmd=[ISAAC,'-p',str(ROOT/'t_eval_go2.py'),'--task','Quadruped-v0',
             '--task_module','go2_task','--train_script',str(TRAIN/'train.py'),
             '--checkpoint',str(ckpt),'--num_envs','128','--seeds','137,389,743,1201',
             '--scenarios',ids,'--run_id',label,'--out',str(out),'--headless',*extra]
        job(cmd,'eval_'+label+'_'+kind)
        data=json.loads(out.read_text()); finite(data)
        assert set(ids.split(',')) == set(data['scenarios']), 'Incomplete scenario group'
        for s in data['scenarios'].values():
            assert s.get('nan_actions',0)==0 and s.get('crashes',0)==0, 'Invalid actions/crashes'
        scenarios.update(data['scenarios'])
    merged=dict(run_id=label,checkpoint=str(ckpt),scenarios=scenarios,seeds='137,389,743,1201')
    (folder/'merged.json').write_text(json.dumps(merged,indent=2)); return merged
def proxy(d):
    floor=ratio=0
    for k,w in WEIGHTS.items():
        s=d['scenarios'][k]; mse=sum(s['rmse'][a]**2 for a in ('vx','vy','yaw'))/3
        floor+=70*w*s['survival']*math.exp(-math.sqrt(mse+.02684614570005177))
        ratio+=70*w*s['survival']*.905*math.exp(-math.sqrt(mse))
    return dict(floor=floor,ratio=ratio)
def passes(c,b):
    pc,pb=proxy(c),proxy(b)
    return pc['floor']>=pb['floor']+.15 and pc['ratio']>=pb['ratio']+.15 and all(c['scenarios'][k]['survival']>=b['scenarios'][k]['survival'] for k in WEIGHTS)
def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    expected=json.loads((ROOT/'protected_source_hashes.json').read_text())
    assert all(sha(TRAIN/k)==v for k,v in expected.items()), 'Provided source mismatch'
    base=ROOT/'original_C0_model_5500.pt'; chosen=ROOT/'candidate/model_3000.pt'
    assert sha(base)=='8d553e697ed4c2856ab002431af96d626eb8b0d23e56b7c754d652193026b1d3'
    assert sha(chosen)=='4684a58838229b729feb2f2534ab9423a4993377e75179a0d55a95904ecf23d2'
    reward=TRAIN/'quadruped_rewards.py'; original=reward.read_bytes()
    (ROOT/'fresh_server_rewards_original.py').write_bytes(original)
    try:
        # Same F00 evaluation reward values used for all A/B policies; no other source edits.
        text=original.decode(); node=next(n.value for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REWARD_WEIGHTS' for t in n.targets))
        lines=original.splitlines(keepends=True)
        a=sum(map(len,lines[:node.lineno-1]))+node.col_offset
        b=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
        weights=json.loads((ROOT/'F00_weights.json').read_text())
        assert all(math.isfinite(v) for v in weights.values())
        reward.write_bytes(original[:a]+repr(weights).encode()+original[b:])
        state('STARTED',mode='validation_only',training=False,official_submission=False)
        detail=json.loads((ROOT/'selection.json').read_text())
        assert detail['heldout_gate_passed'] is False
        state('ADDITIONAL_STRESS',selection=detail,B_rerun=False)
        for name,ck in [('original',base),('selected',chosen)]:
            evaluate(name+'_stress',ck,[('flat','GX1,GX5')],['--transient_s','0'])
            for height in ['0.10','0.15']:
                evaluate(name+'_stairs_'+height,ck,[('stairs','G5')],['--stair_height',height])
        exported=TRAIN/'exported'; exported.mkdir(exist_ok=True)
        shutil.copy2(chosen,exported/'model_best.pt')
        shutil.copy2(ROOT/'candidate/params/env.yaml',exported/'env.yaml')
        (exported/'params').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'candidate/params/env.yaml',exported/'params/env.yaml')
        assert sha(exported/'model_best.pt') == sha(chosen)
        job([ISAAC,'-p','play.py','--task','Quadruped-v0','--num_envs','4',
             '--checkpoint',str(chosen),'--headless','--video','--video_length','1000','--enable_cameras'], 'candidate_video',1200)
        if (TRAIN/'exported').exists(): shutil.copytree(TRAIN/'exported',ROOT/'provided_play_output',dirs_exist_ok=True)
        for folder in [TRAIN/'exported']:
            for video in list(folder.rglob('*.mp4')):
                dest=ROOT/'videos'/video.name; dest.parent.mkdir(exist_ok=True)
                if video.resolve()!=dest.resolve(): shutil.copy2(video,dest)
        state('ARTIFACTS_READY',selection=detail)
    except Exception as e:
        traceback.print_exc(); state('STOPPED_REQUIRES_REVIEW',error=str(e)); raise
    finally:
        reward.write_bytes(original)
        assert all(sha(TRAIN/k)==v for k,v in expected.items()), 'Protected source changed'
        assert sha(reward)==hashlib.sha256(original).hexdigest()
        (ROOT/'reward_restored.json').write_text(json.dumps(dict(restored=True,sha256=sha(reward))))
        archive=ROOT.parent/'go2_stress_20261002_results.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
            for p in ROOT.iterdir():
                if p.name not in ('__pycache__',): tf.add(p,arcname=ROOT.name+'/'+p.name)
        print('RETRIEVE',archive,'BYTES',archive.stat().st_size,'SHA256',sha(archive),flush=True)
if __name__=='__main__': main()
