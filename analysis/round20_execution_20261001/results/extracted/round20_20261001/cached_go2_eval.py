"""Reuse only identical, verified internal evaluations; competition code untouched."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
args = sys.argv[1:]
def flag(name):
    return args[args.index(name)+1] if name in args else None
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out = flag('--out')
checkpoint = flag('--checkpoint')
if out and checkpoint and '--merge' not in args:
    out = Path(out)
    marker = out.with_suffix('.cache.json')
    files = [root/'t_eval_go2_core.py', root/'t_eval.py', Path(checkpoint),
             Path('/workspace/training/quadruped/quadruped_rewards.py'),
             Path('/workspace/training/quadruped/go2_task/env_cfg.py')]
    fingerprint = dict(argv=args, source_hashes={str(p):sha(p) for p in files})
    if out.exists() and marker.exists():
        cached = json.loads(marker.read_text())
        result = json.loads(out.read_text())
        if (cached.get('inputs')==fingerprint and cached.get('output_sha256')==sha(out)
                and result.get('nan_actions',0)==0 and result.get('crashes',0)==0
                and set(flag('--scenarios').split(',')) <= set(result.get('scenarios',{}))):
            print('[verified evaluation cache] Identical policy, CLI, evaluation source and task configuration.',flush=True)
            raise SystemExit(0)
    runpy.run_path(str(root/'t_eval_go2_core.py'),run_name='__main__')
    if out.exists():
        marker.write_text(json.dumps(dict(inputs=fingerprint,output_sha256=sha(out)),indent=2))
else:
    runpy.run_path(str(root/'t_eval_go2_core.py'),run_name='__main__')
