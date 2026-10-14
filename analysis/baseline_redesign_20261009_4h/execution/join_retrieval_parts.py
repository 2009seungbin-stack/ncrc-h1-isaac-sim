import hashlib
import json
from pathlib import Path
import shutil
import sys

downloads, target = map(Path, sys.argv[1:3])
manifest = json.loads((downloads / 'parts_manifest.json').read_text(encoding='utf-8-sig'))
receipt = json.loads((downloads / 'b1_baseline_20261009_4h_results.tar.gz.sha256.json').read_text(encoding='utf-8-sig'))
if target.exists():
    raise RuntimeError('Refusing to overwrite joined archive')
for part in manifest['parts']:
    path = downloads / part['name']
    if path.parent != downloads or path.stat().st_size != part['bytes']:
        raise RuntimeError('Part path or size mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
        raise RuntimeError('Part SHA mismatch: ' + part['name'])
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('xb') as output:
    for part in manifest['parts']:
        with (downloads / part['name']).open('rb') as stream:
            shutil.copyfileobj(stream, output)
actual = {'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
if actual['bytes'] != receipt['bytes'] or actual['sha256'] != receipt['sha256']:
    raise RuntimeError('Joined archive differs from final remote receipt')
print(json.dumps({'joined_archive': str(target), 'verified': True, **actual}))
