"""Read protected source hashes; write a new analysis manifest only on match."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

PROTECTED = ('train.py', 'play.py', 'go2_task/agent_cfg.py', 'go2_task/env_cfg.py')


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
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    if not matched:
        raise SystemExit('SOURCE MISMATCH: inspect source_snapshot_report.json; execution not approved')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(current, stream, indent=2)
    print('Verified current source manifest:', output)


if __name__ == '__main__':
    main()
