"""CPU review of retrieved validation-only artifacts; no remote actions."""
import datetime as dt, hashlib, json, math
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE/'validation_results/extracted/go2_validation_20261002'
sys.path.insert(0,str(BASE))
from validation_continue import proxy, passes

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x):
    if isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)
    elif isinstance(x,float): assert math.isfinite(x)

checks={}
for p in (ROOT/'eval').glob('*/*.json'):
    d=json.loads(p.read_text()); finite(d)
    assert d['scenarios'] and all(s.get('nan_actions',0)==0 and s.get('crashes',0)==0 for s in d['scenarios'].values())
    checks[str(p.relative_to(ROOT))]=sha(p)

b={}
for name in ['original_B','selected_B']:
    p=ROOT/'eval'/name/'merged.json'
    if p.exists(): b[name]=json.loads(p.read_text())
complete=len(b)==2
gate=passes(b['selected_B'],b['original_B']) if complete else None
stress={}; stress_complete=True
for suffix in ['stress','stairs_0.10','stairs_0.15']:
    pair={}
    for name in ['original','selected']:
        p=ROOT/'eval'/f'{name}_{suffix}'/'merged.json'
        if p.exists(): pair[name]=json.loads(p.read_text())['scenarios']
    if len(pair)<2: stress_complete=False
    stress[suffix]=pair
stress_gate=(all(pair['selected'][k]['survival']>=pair['original'][k]['survival']
                 for pair in stress.values() for k in pair['original']) if stress_complete else None)
videos=list((ROOT/'videos').glob('*.mp4')) if (ROOT/'videos').exists() else []
result=dict(reviewed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            B_complete=complete,B_gate_passed=gate,B_scores={k:proxy(v) for k,v in b.items()},
            stress_complete=stress_complete,stress_survival_nonregression=stress_gate,
            stress_metrics=stress,videos=[str(v) for v in videos],video_visual_review_complete=False,
            official_submission=False,official_score_prediction=None,evaluation_hashes=checks,
            reward_restored=json.loads((ROOT/'reward_restored.json').read_text()) if (ROOT/'reward_restored.json').exists() else None)
(BASE/'validation_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['evaluation_hashes','stress_metrics']},ensure_ascii=False,indent=2))
