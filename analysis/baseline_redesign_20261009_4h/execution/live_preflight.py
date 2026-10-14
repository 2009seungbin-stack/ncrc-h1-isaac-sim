"""Read-only allocation preflight. Never starts simulation or changes training sources."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'package_manifest.json').read_text(encoding='utf-8'))
    checked = {}
    for name, spec in manifest['files'].items():
        p = root / name
        if p.stat().st_size != spec['bytes'] or sha(p) != spec['sha256']:
            raise RuntimeError('Uploaded package hash mismatch: ' + name)
        checked[name] = spec
    import torch
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                               capture_output=True, text=True, check=True).stdout.strip()
    training_dir = Path('/workspace/training/quadruped')
    expected = json.loads((root / 'previous_source_hashes_20261002.json').read_text(encoding='utf-8'))
    current = {n: sha(training_dir/n) for n in expected}
    original = sha(training_dir / 'quadruped_rewards.py')
    import evaluation
    import t_eval
    references = json.loads((root / 'reference_models.json').read_text(encoding='utf-8'))
    loader_evidence = {}
    for item in references:
        resolved = dict(item, checkpoint=str(root / item['checkpoint']),
                        actor_observation_normalization=False, policy_clip_actions=None)
        actor, metadata = evaluation.load_actor(resolved, t_eval, 'cpu')
        loader_evidence[item['name']] = metadata
        del actor
    result = dict(observed_epoch=time.time(), package_members_verified=True,
                  provided_sources_match=current == expected, provided_source_hashes=current,
                  original_reward_matches=original == checked['fresh_rewards_original.py']['sha256'],
                  original_reward_sha256=original, probe_sha256=checked['recovery_probe.zip']['sha256'],
                  runtime_versions=dict(rsl_rl_lib=importlib.metadata.version('rsl-rl-lib'), torch=torch.__version__),
                  reference_loader_verified=True, reference_loader_evidence=loader_evidence,
                  gpu_inventory=inventory, compute_processes=gpu, no_competing_gpu_jobs_verified=not gpu)
    output = root / 'live_preflight_observation.json'
    if output.exists():
        raise RuntimeError('Preserve earlier observation; refuse duplicate preflight')
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if not (result['provided_sources_match'] and result['original_reward_matches'] and not gpu):
        raise SystemExit('Preflight failed: do not train')


if __name__ == '__main__':
    main()
