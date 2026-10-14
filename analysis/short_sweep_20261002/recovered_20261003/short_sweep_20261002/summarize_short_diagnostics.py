import json, re, math
from pathlib import Path
R=Path('/workspace/student_workspace/short_sweep_20261002')
out={'measurement_note':'Short screen internal diagnostics; no official score; one training seed; top5 share is pooled from per-reset top5 selections.', 'models':{}, 'learning_curves':{}}
for p in sorted((R/'results').glob('short_*/*.json')):
 d=json.loads(p.read_text())
 m=out['models'].setdefault(d['model'],{})
 for sid,rows in d['scenarios'].items():
  s={'episodes':sum(x['episodes'] for x in rows),'survived':sum(x['survived'] for x in rows)}
  for period in ['full','steady']:
   n=sum(x[period]['count'] for x in rows)
   squared=[sum(x[period]['sum_squared'][i] for x in rows) for i in range(3)]
   s[period]={'count':n,'rmse':[round(math.sqrt(v/n),8) for v in squared],'mae':[round(sum(x[period]['sum_absolute'][i] for x in rows)/n,8) for i in range(3)],'top5_share':[round(sum(x[period]['top5_mse_share'][a]*x[period]['sum_squared'][i] for x in rows)/max(1e-20,squared[i]),8) for i,a in enumerate(['vx','vy','yaw'])],'body_rmse':[round(math.sqrt(sum(x[period]['extra_rmse'][a]**2*x[period]['count'] for x in rows)/n),8) for a in ['vz','wx','wy']]}
  m[sid]=s
for arm in ['control','action','acc','linear']:
 txt=re.sub(r'\x1b\[[0-9;]*m','',(R/'logs'/('train_'+arm+'.log')).read_text(errors='replace'))
 matches=list(re.finditer(r'Learning iteration\s+(\d+)/800',txt))
 bins={}
 for j,v in enumerate(matches):
  k=int(v.group(1)); block=txt[v.end():matches[j+1].start() if j+1<len(matches) else len(txt)]
  bucket=bins.setdefault(k//100,{})
  for name,value in re.findall(r'^\s*([^:\n]+):\s*([-+0-9.eE]+)\s*$',block,re.M):
   if name.startswith(('Metrics/','Episode_Termination/')) or name=='Mean episode length':
    bucket.setdefault(name,[]).append(float(value))
 out['learning_curves'][arm]={'iterations_seen':len(matches),'windows':{str(b*100)+'-'+str(min(b*100+99,799)):{k:round(sum(v)/len(v),6) for k,v in metrics.items()} for b,metrics in bins.items()}}
(R/'short_diagnostics.json').write_text(json.dumps(out,indent=2))
print('SHORT_DIAGNOSTICS',len(out['models']),{a:d['iterations_seen'] for a,d in out['learning_curves'].items()})
