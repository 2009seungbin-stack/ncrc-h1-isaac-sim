"""Build the allow-listed upload package; never includes local logs or browser data."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
FILES = ('runner.py', 'source_snapshot.py', 'batch_eval.py', 't_eval.py', 't_eval_go2_core.py',
         'RUNNER_USAGE.txt', 'protocol.txt', 'previous_source_hashes_20261002.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = {}
    for name in FILES:
        data = (ROOT / name).read_bytes()
        if name.endswith('.py'):
            compile(data, name, 'exec')
        manifest[name] = {'bytes': len(data), 'sha256': sha(data)}
    source = ROOT.parent / 'short_sweep_20261002/recovered_20261003/short_sweep_20261002'
    for name in ('batch_eval.py', 't_eval.py', 't_eval_go2_core.py'):
        assert manifest[name]['sha256'] == sha((source / name).read_bytes()), name
    manifest_path = ROOT / 'package_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive = ROOT.parent / (ROOT.name + '_inputs.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        for name in (*FILES, 'package_manifest.json'):
            tf.add(ROOT / name, arcname=ROOT.name + '/' + name)
    with tarfile.open(archive, 'r:gz') as tf:
        assert len(tf.getmembers()) == len(FILES) + 1
        for name in FILES:
            member = tf.getmember(ROOT.name + '/' + name)
            assert member.isfile()
            assert sha(tf.extractfile(member).read()) == manifest[name]['sha256']
    report = {'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': sha(archive.read_bytes()), 'verified_members': len(FILES) + 1,
              'evaluators_unchanged': True, 'python_sources_compile': True,
              'gpu_validation': False, 'remote_launch': False}
    (ROOT / 'package_verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
