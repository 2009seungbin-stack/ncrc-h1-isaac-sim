"""Verify and safely unpack a local final archive without executing its contents."""
import argparse
import hashlib
import json
import pathlib
import shutil
import tarfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=pathlib.Path)
    parser.add_argument('destination', type=pathlib.Path)
    parser.add_argument('receipt', type=pathlib.Path)
    parser.add_argument('proof', type=pathlib.Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf-8-sig'))
    actual = {'bytes': args.archive.stat().st_size, 'sha256': digest(args.archive)}
    if actual['bytes'] != receipt['bytes'] or actual['sha256'] != receipt['sha256']:
        raise RuntimeError('Local archive does not match final remote receipt')
    destination = args.destination.resolve()
    if destination.exists():
        raise RuntimeError('Destination must be a fresh directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'r:gz') as tf:
        members = tf.getmembers()
        paths = set()
        for member in members:
            pure = pathlib.PurePosixPath(member.name)
            if pure.is_absolute() or '..' in pure.parts or '\\' in member.name or ':' in member.name:
                raise RuntimeError('Unsafe archive path: ' + member.name)
            if not (member.isdir() or member.isfile()):
                raise RuntimeError('Archive links and special files are prohibited: ' + member.name)
            target = destination.joinpath(*pure.parts).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError('Archive target escapes destination')
            folded = str(target).casefold()
            if folded in paths:
                raise RuntimeError('Duplicate archive target: ' + member.name)
            paths.add(folded)
        destination.mkdir()
        for member in members:
            target = destination.joinpath(*pathlib.PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tf.extractfile(member)
                with source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                if target.stat().st_size != member.size:
                    raise RuntimeError('Extracted size differs: ' + member.name)
    inventory = []
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            inventory.append({'path': path.relative_to(destination).as_posix(),
                              'bytes': path.stat().st_size, 'sha256': digest(path)})
    proof = {'observed_epoch': time.time(), 'source': str(args.archive.resolve()),
             'destination': str(destination), 'archive': actual,
             'receipt_matches': True, 'safe_paths_checked': True,
             'files': inventory, 'contains_complete_validation': receipt.get('contains_complete_validation')}
    args.proof.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps({'archive': actual, 'destination': str(destination), 'files': len(inventory),
                      'checkpoints': [entry for entry in inventory if entry['path'].endswith('.pt')]}))


if __name__ == '__main__':
    main()
