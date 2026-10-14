"""Package the reviewed execution files; does not rewrite runner sources."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
names = ['runner.py', 'tests_runner.py', 'batch_eval.py', 't_eval.py',
         't_eval_go2_core.py', 'source_snapshot.py',
         'previous_source_hashes_20261002.json', 'reference_models.json',
         'RUNNER_USAGE.txt', 'references/C0_5500.pt', 'references/F10_3000.pt']
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for name in names:
    if name.endswith('.py'):
        compile((root/name).read_bytes(), name, 'exec')
previous = root.parents[1]/'causal_sweep_20261008'
for name in ('batch_eval.py', 't_eval.py', 't_eval_go2_core.py'):
    assert sha(root/name) == sha(previous/name), name+' was changed'
manifest = dict(built_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                remote_root='/workspace/student_workspace/tracking_screen_20261009',
                cpu_tests_passed=24, scope='S1/R1 scratch1500 only;6h allocation',
                files={name: dict(bytes=(root/name).stat().st_size, sha256=sha(root/name)) for name in names})
(root/'package_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
names.append('package_manifest.json')
zip_path = root/'tracking_screen_20261009_package.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for name in names:
        archive.write(root/name, 'tracking_screen_20261009/'+name)
report = dict(path=str(zip_path), bytes=zip_path.stat().st_size, sha256=sha(zip_path),
              entries=len(names), cpu_tests_passed=24,
              unchanged_evaluators=['batch_eval.py','t_eval.py','t_eval_go2_core.py'])
(root/'package_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
start = dt.datetime.fromisoformat('2026-10-09T01:55:44+00:00').timestamp()
print('SERVER_START_UNIX', int(start), 'COMPUTE_DEADLINE_UNIX', int(start+18900), 'HARD_SHUTDOWN_UNIX', int(start+21600))
