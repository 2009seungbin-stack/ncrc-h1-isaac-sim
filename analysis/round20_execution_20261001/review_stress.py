from pathlib import Path
import json,hashlib,math,datetime
B=Path(__file__).resolve().parent; R=B/'stress_results/extracted/go2_stress_20261002'
def finite(x):
    if isinstance(x,dict):
        for v in x.values(): finite(v)
    elif isinstance(x,list):
        for v in x: finite(v)
    elif isinstance(x,float): assert math.isfinite(x)
comparisons=[]; checks={}
assert json.loads((R/'reward_restored.json').read_text())['restored']
for suffix in ['stress','stairs_0.10','stairs_0.15']:
    pair={}
    for name in ['original','selected']:
        p=R/'eval'/f'{name}_{suffix}'/'merged.json';d=json.loads(p.read_text());finite(d)
        pair[name]=d['scenarios'];checks[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert pair['original'].keys()==pair['selected'].keys()
    for sid in pair['original']:
        a,b=(pair[n][sid] for n in ['original','selected'])
        assert a['episodes']==b['episodes']==512
        assert all(s.get('nan_actions',0)==s.get('crashes',0)==0 for s in [a,b])
        comparisons.append(dict(test=suffix,scenario=sid,original=a,selected=b,
            survival_delta=b['survival']-a['survival'],rmse_delta={k:b['rmse'][k]-a['rmse'][k] for k in a['rmse']}))
result=dict(reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=True,
    comparisons=comparisons,evaluation_hashes=checks,
    survival_nonregression=all(x['selected']['survived']>=x['original']['survived'] for x in comparisons),
    B_gate_remains_failed=True,official_submission=False,video_visual_review_complete=False)
(B/'stress_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for x in comparisons:
    a,b=x['original'],x['selected']
    print(x['test'],x['scenario'],f"survived {a['survived']}->{b['survived']}",a['rmse'],'->',b['rmse'])
