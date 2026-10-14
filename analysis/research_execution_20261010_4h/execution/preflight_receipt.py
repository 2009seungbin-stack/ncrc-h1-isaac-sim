"""Seal live preflight plus operator-verified binary retrieval evidence."""
import argparse
import json
import re
import time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation-start-epoch',type=int,required=True)
    parser.add_argument('--verified-probe-local-sha256',required=True)
    parser.add_argument('--authorized-four-hour-execution',action='store_true',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    observation=json.loads((root/'live_preflight_observation.json').read_text(encoding='utf-8'))
    now=time.time()
    if not (args.allocation_start_epoch <= observation['observed_epoch'] <= now and
            now < args.allocation_start_epoch+1200):
        raise RuntimeError('Preflight is not within this actual allocation setup window')
    fields=('package_members_verified','provided_sources_match','original_reward_matches',
            'runtime_verified','reference_loader_verified','no_competing_gpu_jobs_verified')
    if any(observation.get(k) is not True for k in fields):
        raise RuntimeError('Live preflight evidence is incomplete or failed')
    capture=json.loads((root/'checkpoint_capture_probe.json').read_text(encoding='utf-8'))
    if not (args.allocation_start_epoch <= capture.get('verified_epoch',0) <= now):
        raise RuntimeError('Checkpoint capture probe is not from this allocation')
    if any(capture.get(k) is not True for k in ('ok','cpu_only','inotify_new_run_discovery',
            'hardlink_inode_verified','original_unlink_preservation_verified')):
        raise RuntimeError('Actual Linux checkpoint preservation probe failed or incomplete')
    probe=observation['probe_sha256']
    if not re.fullmatch('[0-9a-f]{64}',args.verified_probe_local_sha256) or args.verified_probe_local_sha256!=probe:
        raise RuntimeError('Verified local downloaded probe SHA differs from remote package SHA')
    receipt=dict(actual_allocation_start_epoch=args.allocation_start_epoch, verified_epoch=now,
                 user_authorized_execution=True, retrieval_smoke_verified=True,
                 no_competing_gpu_jobs_verified=True,runtime_verified=True,reference_loader_verified=True,
                 probe_sha256=probe,probe_local_sha256=args.verified_probe_local_sha256,
                 checkpoint_capture_probe_verified=True,checkpoint_capture_probe=capture,
                 runtime_versions=observation['runtime_versions'],live_preflight=observation,
                 note='Local hash supplied only after operator normal-UI binary download and actual local SHA verification')
    with (root/'preflight_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    main()
