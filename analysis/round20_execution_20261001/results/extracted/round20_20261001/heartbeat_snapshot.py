"""Small read-only host diagnostic; saves a summary in the experiment directory."""
import datetime as dt
import json
import math
from pathlib import Path
import re
import subprocess
import time

root=Path(__file__).resolve().parent
state=json.loads((root/'state.json').read_text())
job=state.get('job','')
log=root/'logs'/f'{job}.log'
report=dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),state=state,
            elapsed_server_hours=(time.time()-dt.datetime.fromisoformat('2026-10-01T04:31:37+00:00').timestamp())/3600)
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-250000))
        text=f.read().decode('utf-8',errors='replace')
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    blocks=re.split(r'Learning iteration\s+(\d+)/(\d+)',text)
    report['log_modified_seconds_ago']=time.time()-log.stat().st_mtime
    report['recent_error_lines']=[s for s in text.splitlines() if any(x in s for x in ('Traceback','RuntimeError','expects all elements','contact buffer','capacity exceeded','overflow'))][-10:]
    if len(blocks)>=4:
        report['iteration']=int(blocks[-3]); report['iteration_limit']=int(blocks[-2])
        latest=blocks[-1]
        for label,key in [('Iteration time','seconds_per_iteration'),('Mean action std','mean_action_std'),('Mean reward','mean_reward'),('Mean value loss','mean_value_loss'),('Curriculum/terrain_levels','terrain_level')]:
            matches=re.findall(re.escape(label)+r':\s*([-+\deE.infNa]+)',latest)
            if matches:
                report[key]=float(matches[-1])
        report['eta_display']=re.findall(r'ETA:\s*(\d+:\d+:\d+)',latest)[-1:] or None
if job.startswith('train_'):
    name=job.removeprefix('train_')
    models=[p for p in (root/'preserved'/name).glob('*/model_*.pt') if p.stem.split('_')[-1].isdigit()]
    report['preserved_models']=[dict(name=p.name,bytes=p.stat().st_size) for p in sorted(models,key=lambda p:int(p.stem.split('_')[-1]))]
report['driver_tail']=(root/'driver.log').read_text(errors='replace')[-2500:]
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
report['gpu']=gpu.stdout.strip()
report['nonfinite_metrics']=[k for k,v in report.items() if isinstance(v,float) and not math.isfinite(v)]
for key in report['nonfinite_metrics']:
    report[key]=str(report[key])
(root/'heartbeat_latest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
