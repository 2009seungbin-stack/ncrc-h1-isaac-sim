"""Verify the recovered experiment archive without loading executable checkpoints."""
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'recovered_20261003' / 'short_sweep_20261002'
ARCHIVE = BASE.parents[1] / 'short_sweep_20261002_results.tar.gz'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

execution = json.loads((BASE / 'execution.json').read_text(encoding='utf-8-sig'))
assert ARCHIVE.stat().st_size == 503786122
assert sha(ARCHIVE) == execution['final_archive']['sha256']
assert read('state.json')['stage'] == 'HELDOUT_COMPLETE'
models = read('short_models.json')
assert {m['name'] for m in models} == {'control', 'action', 'acc', 'linear'}
checkpoints = {}
for model in models:
    name = model['name']
    path = ROOT / model['checkpoint']
    digest = sha(path)
    assert digest == model['sha256'] == execution['last_heartbeat'][name + '_checkpoint_sha256']
    iterations = sorted(int(p.stem.split('_')[-1]) for p in path.parent.glob('model_*.pt'))
    assert iterations == [0, 100, 200, 300, 400, 500, 600, 700, 799]
    events = list(path.parent.glob('events.out.tfevents.*'))
    assert len(events) == 1 and events[0].stat().st_size > 0
    assert all((path.parent / 'params' / p).stat().st_size > 0 for p in ['agent.yaml', 'env.yaml'])
    log = (ROOT / 'logs' / ('train_' + name + '.log')).read_text(errors='replace')
    steps = [int(x) for x in re.findall(r'Learning iteration\s+(\d+)/800', log)]
    assert steps == list(range(800)), (name, len(steps))
    checkpoints[name] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                         'preserved_iterations': iterations, 'log_iterations': len(steps),
                         'event_file': str(events[0].relative_to(ROOT))}

stage_checks = {}
for stage, expected_models, expected_scenarios, seeds, nenv, startup, terrain in [
    ('saved', {'C0_5500', *['F10_' + str(n) for n in range(2000, 3001, 200)]}, {'G1', 'G2', 'G6', 'G5'}, {211, 607}, 32, 42, 1234),
    ('short', {'control', 'action', 'acc', 'linear'}, {'G1', 'G2', 'G6', 'G5'}, {211, 607}, 32, 42, 1234),
    ('heldout', {'C0_5500', 'F10_3000'}, {'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'}, {809, 1201}, 128, 303, 9049),
]:
    ranking = read(stage + '_ranking.json')['models']
    assert set(ranking) == expected_models
    raw = {name: {} for name in expected_models}
    traces = 0
    for path in sorted((ROOT / 'results').glob(stage + '_*/*.json')):
        data = json.loads(path.read_text())
        name = data['model']
        assert data['checkpoint_sha256'] == ranking[name]['checkpoint_sha256']
        assert (data['num_envs'], data['startup_seed'], data['terrain_seed']) == (nenv, startup, terrain)
        assert data['horizon'] == 20.0 and data['command_slice'] == [9, 12]
        for scenario, rows in data['scenarios'].items():
            assert scenario not in raw[name]
            assert {row['seed'] for row in rows} == seeds and len(rows) == 2
            assert all(row['episodes'] == nenv and row['steps'] == 1000 for row in rows)
            summary = ranking[name]['scenarios'][scenario]
            assert summary['episodes'] == sum(row['episodes'] for row in rows)
            assert summary['survived'] == sum(row['survived'] for row in rows)
            for axis, index in [('vx', 0), ('vy', 1), ('yaw', 2)]:
                rmse = math.sqrt(sum(row['full']['sum_squared'][index] for row in rows) / sum(row['full']['count'] for row in rows))
                assert math.isclose(rmse, summary['rmse'][axis], abs_tol=1e-12)
            for row in rows:
                trace = path.parent / 'traces' / f'{name}_{scenario}_{row["seed"]}.npz'
                with zipfile.ZipFile(trace) as z:
                    assert z.testzip() is None and z.namelist()
                traces += 1
            raw[name][scenario] = summary
    for name, scenarios in raw.items():
        assert set(scenarios) == expected_scenarios
        assert set(ranking[name]['scenarios']) == expected_scenarios
        assert ranking[name]['falls'] == sum(row['episodes'] - row['survived'] for row in scenarios.values())
    stage_checks[stage] = {'models': len(expected_models), 'scenarios_per_model': len(expected_scenarios),
                          'trace_archives_crc_verified': traces,
                          'episodes_per_model': nenv * 2 * len(expected_scenarios),
                          'aggregate_rmse_matches_raw': True}

restore = read('reward_restored.json')
assert restore['restored'] and restore['sha256'] == sha(ROOT / 'fresh_rewards_original.py')
assert restore['sha256'] == execution['final_rewards_restore_verified']['sha256']
expected_protected = {
    'train.py': 'c09ac99bcd93bb367f0eec69bdb05f4063cb8d60f0258d4667f07addf509de22',
    'play.py': 'ce703acf54eac40073e3f6d585d6002e3b4a2d9062581b3c0ed45db73ac5f9dc',
    'go2_task/agent_cfg.py': 'a46dbb1c5812745744b63744d96a8557b5831b228562ba3317ef57b21eb94a23',
    'go2_task/env_cfg.py': 'df2224900f6dbd49678b986b39b902d313fbadac8ac48f2c65fbd53d5df104b1',
}
assert read('protected_sources.json') == expected_protected
json_count = 0
for path in ROOT.rglob('*.json'):
    json.loads(path.read_text())
    json_count += 1
manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(ROOT.rglob('*')) if p.is_file()}
(BASE / 'recovered_file_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
report = {'status': 'VERIFIED', 'verified_utc': datetime.now(timezone.utc).isoformat(),
          'archive': str(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
          'archive_sha256': sha(ARCHIVE), 'extracted_root': str(ROOT),
          'file_count': len(manifest), 'json_files_parsed': json_count,
          'checkpoints': checkpoints, 'evaluation': stage_checks,
          'reward_restoration': restore, 'protected_source_manifest_matches': True,
          'remote_rechecked_utc': '2026-10-02T16:49:54Z',
          'remote_recheck': 'Archive size and SHA256 match; original reward SHA256 matches; tmux has no sessions.',
          'checkpoints_loaded_as_code': False, 'official_submission': False}
(BASE / 'recovery_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
