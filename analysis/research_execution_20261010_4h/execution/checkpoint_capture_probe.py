"""Linux CPU-only capture compatibility check; no Isaac, torch, policy or GPU.

Run once during setup and record its printed receipt before the training launch.
The temporary test tree is inside --root; no provided file is read or changed.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from runner import CheckpointCapture, sha


def probe(root):
    if os.name!='posix':
        raise RuntimeError('Actual Linux inotify probe required; Windows mock tests are separate')
    root=Path(root).resolve()
    with tempfile.TemporaryDirectory(prefix='checkpoint_capture_probe_',dir=root) as tmp:
        parent=Path(tmp)/'logs';parent.mkdir()
        run_name='CPU_PROBE'
        capture=CheckpointCapture(parent,run_name)
        capture.start()
        try:
            run=parent/('date_'+run_name);run.mkdir()
            deadline=time.monotonic()+2
            while run not in capture.watches.values() and time.monotonic()<deadline:
                capture.healthy();time.sleep(.01)
            if run not in capture.watches.values():
                raise RuntimeError('inotify did not discover the new owned run')
            original=run/'model_1499.pt'
            payload=b'CPU capture probe, not a policy checkpoint\n'*100
            with original.open('wb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            linked=run/capture.SPOOL/'model_1499.pt.capture'
            deadline=time.monotonic()+2
            while not linked.exists() and time.monotonic()<deadline:
                capture.healthy();time.sleep(.01)
            if not linked.exists() or linked.stat().st_ino!=original.stat().st_ino:
                raise RuntimeError('inotify/hardlink inode capture did not complete')
            expected=sha(original)
            original.unlink()
            if sha(linked)!=expected:
                raise RuntimeError('Captured bytes were not preserved after original unlink')
            return dict(ok=True,cpu_only=True,inotify_new_run_discovery=True,
                hardlink_inode_verified=True,original_unlink_preservation_verified=True,
                bytes=linked.stat().st_size,sha256=expected,verified_epoch=time.time(),
                scope='Compatibility probe; exact final policy still requires iteration/hash/finite verification')
        finally:
            capture.stop()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,
        help='Default ROOT/checkpoint_capture_probe.json; one-shot exclusive receipt')
    args=parser.parse_args()
    receipt=probe(args.root)
    output=args.out or args.root/'checkpoint_capture_probe.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(receipt,sort_keys=True),flush=True)
