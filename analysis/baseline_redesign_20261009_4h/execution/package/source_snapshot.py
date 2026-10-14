"""Read protected source hashes; write a new analysis manifest only on match."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

PROTECTED = ('train.py', 'play.py', 'go2_task/agent_cfg.py', 'go2_task/env_cfg.py')


def snapshot_runtime_sources(package_root):
    """Hash the official installed package without importing its implementation."""
    package_root = Path(package_root).resolve()
    required = ['algorithms/ppo.py', 'runners/on_policy_runner.py']
    if (package_root/'modules/actor_critic.py').is_file():
        policy_layout = 'legacy_actor_critic'
        required.append('modules/actor_critic.py')
    elif all((package_root/name).is_file() for name in ('models/mlp_model.py', 'modules/mlp.py')):
        policy_layout = 'modular_mlp_model'
        required.extend(('models/mlp_model.py', 'modules/mlp.py'))
    else:
        raise ValueError('Missing recognized actor source layout: require modules/actor_critic.py '
                         'or models/mlp_model.py plus modules/mlp.py')
    for name in required:
        if not (package_root/name).is_file():
            raise ValueError('Missing required protected runtime source: '+str(package_root/name))
    return dict(captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                algorithm_package='rsl_rl', package_root=str(package_root),
                policy_layout=policy_layout, required_relative_paths=required,
                files={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(package_root.rglob('*.py')) if p.is_file()},
                note='Capture current installed source hashes; no package or training module imported; '
                     'file layout recognition does not prove provided configuration compatibility')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=Path('/workspace/training/quadruped'))
    parser.add_argument('--reference', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root/'source_hashes_current.json'
    if output.exists():
        parser.error('Current manifest already exists; preserve and review it instead of overwriting')
    expected = json.loads(args.reference.read_text(encoding='utf-8'))
    if set(expected) != set(PROTECTED):
        parser.error('Reference must contain exactly the four protected source paths')
    current = {name: hashlib.sha256((args.train_dir/name).read_bytes()).hexdigest() for name in PROTECTED}
    matched = current == expected
    report = dict(captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(), matched=matched,
                  train_dir=str(args.train_dir.resolve()), reference=str(args.reference.resolve()),
                  actual=current, expected=expected,
                  note='Only source hashes read; no training modules imported or files modified')
    report_path = root/'source_snapshot_report.json'
    if report_path.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        preserved = root/('source_snapshot_report.previous_'+stamp+'.json')
        if preserved.exists():
            parser.error('Snapshot report preservation filename collision; review before retry')
        report_path.rename(preserved)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    if not matched:
        raise SystemExit('SOURCE MISMATCH: inspect source_snapshot_report.json; execution not approved')
    runtime_path = root/'protected_runtime_sources.json'
    if runtime_path.exists():
        parser.error('Runtime manifest exists; refuse to overwrite earlier source evidence')
    spec = importlib.util.find_spec('rsl_rl')
    if spec is None or not spec.origin:
        parser.error('Cannot locate installed rsl_rl package without importing training code')
    package_root = Path(spec.origin).resolve().parent
    try:
        runtime = snapshot_runtime_sources(package_root)
    except ValueError as exc:
        parser.error(str(exc))
    with runtime_path.open('x', encoding='utf-8') as stream:
        json.dump(runtime, stream, indent=2)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(current, stream, indent=2)
    print('Verified current source manifest:', output)


if __name__ == '__main__':
    main()
