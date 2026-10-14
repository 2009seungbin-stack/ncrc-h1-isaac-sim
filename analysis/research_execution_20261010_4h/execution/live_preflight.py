"""Read-only live preflight; no simulation, CUDA context, signal, or service control.

Expected platform streaming is classified by exact live process identities.
Run only inside the hash-verified uploaded package using the official Python.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import streaming_control_libc as streaming

REQUIRED_VERSIONS = {'rsl_rl_lib': '5.0.1', 'torch': '2.7.0+cu128'}
PROTECTED = ('train.py', 'play.py', 'go2_task/agent_cfg.py', 'go2_task/env_cfg.py')
RECEIPT = 'live_preflight_observation.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_member(root, name):
    if not isinstance(name, str):
        raise RuntimeError('Invalid package member type')
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise RuntimeError('Invalid package member: ' + str(name))
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise RuntimeError('Package member escapes root')
    return resolved


def verify_package(root, manifest):
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise RuntimeError('Package file manifest missing')
    required = {'live_preflight.py', 'streaming_control_libc.py', 'evaluation.py', 't_eval.py',
                'reference_models.json', 'previous_source_hashes_20261002.json',
                'fresh_rewards_original.py', 'recovery_probe.zip'}
    if not required.issubset(files):
        raise RuntimeError('Required package members missing: ' + str(sorted(required - set(files))))
    for name, spec in files.items():
        path = local_member(root, name)
        if (not isinstance(spec, dict) or
                not re.fullmatch(r'[0-9a-f]{64}', spec.get('sha256', '')) or
                type(spec.get('bytes')) is not int or spec['bytes'] < 0 or
                path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']):
            raise RuntimeError('Uploaded package hash/size mismatch: ' + name)
    return files


def verify_versions(observed):
    if observed != REQUIRED_VERSIONS:
        raise RuntimeError('Runtime mismatch: required ' + str(REQUIRED_VERSIONS) +
                           ', observed ' + str(observed))
    return True


def verify_provided_sources(train, expected, original_digest):
    if set(expected) != set(PROTECTED):
        raise RuntimeError('Require exactly four protected provided sources')
    current = {name: sha(train / name) for name in PROTECTED}
    if current != expected:
        raise RuntimeError('Protected provided source hash mismatch')
    original = sha(train / 'quadruped_rewards.py')
    if original != original_digest:
        raise RuntimeError('Active reward differs from supplied original')
    return dict(provided=current, original_reward_sha256=original)


def verify_runtime_sources(snapshot, actual_root):
    package = Path(actual_root).resolve()
    if (snapshot.get('algorithm_package') != 'rsl_rl' or
            Path(snapshot.get('package_root', '')).resolve() != package):
        raise RuntimeError('Runtime snapshot does not identify current installed rsl_rl')
    expected = snapshot.get('files')
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError('Runtime source snapshot missing files')
    inventory = {str(p.resolve()) for p in package.rglob('*.py') if p.is_file()}
    if inventory != set(expected):
        raise RuntimeError('Runtime Python source inventory changed')
    required = ('algorithms/ppo.py', 'runners/on_policy_runner.py')
    if not all((package / name).is_file() for name in required):
        raise RuntimeError('Required PPO/runner runtime source missing')
    actor_layout = ((package / 'modules/actor_critic.py').is_file() or
                    all((package / name).is_file() for name in
                        ('models/mlp_model.py', 'modules/mlp.py')))
    if not actor_layout:
        raise RuntimeError('Unrecognized runtime actor source layout')
    for name, digest in expected.items():
        path = Path(name)
        if (not path.is_absolute() or not path.resolve().is_relative_to(package) or
                not re.fullmatch(r'[0-9a-f]{64}', digest) or sha(path) != digest):
            raise RuntimeError('Runtime source mismatch: ' + name)
    return dict(package_root=str(package), file_count=len(expected), files=expected)


def competing_job(item):
    if not streaming.alive(item):
        return False
    if streaming.has_training([item]):
        return True
    scripts = {Path(arg).name for arg in item.argv[1:]}
    if scripts & {'train.py', 'evaluation.py', 't_eval.py', 'play.py', 'benchmark.py'}:
        return True
    return 'runner.py' in scripts and '--authorized-execution' in item.argv


def classify_processes(gpu_ids, items, reader):
    by_pid = {}
    for item in items:
        if item.pid in by_pid:
            raise RuntimeError('Ambiguous duplicate PID inventory')
        by_pid[item.pid] = item
    competing = [item for item in items if competing_job(item)]
    if competing:
        raise RuntimeError('Competing train/evaluation job: ' +
                           str([item.record() for item in competing]))
    launchers = [item for item in items if streaming.is_launcher(item)]
    if len(launchers) > 1:
        raise RuntimeError('Ambiguous official launcher inventory')
    if not gpu_ids:
        return dict(classification='no_gpu_jobs', gpu_pids=[],
                    process_inventory_count=len(items),
                    official_streamer_present=False, competing_jobs=[])
    if len(gpu_ids) != 1:
        raise RuntimeError('Multiple/ambiguous GPU jobs; no preflight permission')
    pid = next(iter(gpu_ids))
    child = by_pid.get(pid)
    if child is None or not streaming.alive(child) or child.pid <= 1:
        raise RuntimeError('GPU PID missing/dead/unsafe in process inventory')
    parent = by_pid.get(child.ppid)
    if (parent is None or child.start_ticks <= 0 or parent.start_ticks <= 0 or
            not streaming.is_launcher(parent) or
            not streaming.is_streamer(child, parent) or
            child.state in ('T', 't') or parent.state in ('T', 't')):
        raise RuntimeError('GPU process is not an active exact official streamer/launcher pair')
    # Re-read both identities after GPU enumeration. A recycled PID or reparented
    # child never inherits the earlier classification.
    current_parent, current_child = reader(parent.pid), reader(child.pid)
    if (not parent.same_process(current_parent) or not child.same_process(current_child) or
            not streaming.is_launcher(current_parent) or
            not streaming.is_streamer(current_child, current_parent) or
            current_parent.state in ('T', 't') or current_child.state in ('T', 't')):
        raise RuntimeError('Official service identity changed during classification')
    return dict(classification='only_exact_official_streamer', gpu_pids=[pid],
                process_inventory_count=len(items), official_streamer_present=True,
                launcher=current_parent.record(), streamer=current_child.record(), competing_jobs=[],
                note='Expected platform service; not a free-GPU claim. Platform training detection and evaluation handshakes manage it.')


def verify_controller_launcher(items, reader):
    """This execution's later hold handshake requires an existing official launcher."""
    launchers = [item for item in items if streaming.is_launcher(item)]
    if len(launchers) != 1:
        raise RuntimeError('Execution controller requires exactly one current official launcher')
    item = launchers[0]
    current = reader(item.pid)
    if (item.start_ticks <= 0 or not item.same_process(current) or
            not streaming.is_launcher(current) or current.state in ('T', 't')):
        raise RuntimeError('Execution controller launcher identity not current/active')
    children = [entry for entry in items if streaming.alive(entry) and entry.ppid == item.pid]
    if len(children) > 1 or any(not streaming.is_streamer(entry, current) for entry in children):
        raise RuntimeError('Ambiguous or unknown current official launcher child')
    return current.record()


def observe_processes():
    items = streaming.identities()
    observed = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True,
                              text=True, check=True, timeout=8)
    gpu_ids = streaming.gpu_pids_from_xml(observed.stdout)
    classified = classify_processes(gpu_ids, items, streaming.read_identity)
    classified['controller_launcher'] = verify_controller_launcher(items, streaming.read_identity)
    tree = ET.fromstring(observed.stdout)
    classified['gpu_inventory'] = [
        dict(name=g.findtext('product_name'), total_memory=g.findtext('fb_memory_usage/total'))
        for g in tree.findall('gpu')]
    classified['observed_epoch'] = time.time()
    return classified


def load_reference_models_cpu(root, checked):
    import evaluation
    import t_eval
    evidence = {}
    paths = [root / 'reference_models.json']
    existing = root / 'existing_b1_models.json'
    if existing.is_file():
        raw = json.loads(existing.read_text(encoding='utf-8'))
        if (raw.get('models') if isinstance(raw, dict) else raw):
            if existing.name not in checked:
                raise RuntimeError('Optional model manifest absent from verified package')
            paths.append(existing)
    for path in paths:
        for item in evaluation.read_manifest(path):
            checkpoint = Path(item['checkpoint']).resolve()
            if not checkpoint.is_relative_to(root.resolve()):
                raise RuntimeError('Checkpoint outside verified package')
            member = checkpoint.relative_to(root.resolve()).as_posix()
            if member not in checked or checked[member]['sha256'] != item['sha256']:
                raise RuntimeError('Checkpoint absent/mismatched in verified package manifest')
            if item['name'] in evidence:
                raise RuntimeError('Duplicate checkpoint name')
            if (item.get('actor_observation_normalization', False) is not False or
                    item.get('policy_clip_actions') is not None):
                raise RuntimeError('Reference normalization/clipping differs from frozen evaluator contract')
            resolved = dict(item)
            resolved.setdefault('actor_observation_normalization', False)
            resolved.setdefault('policy_clip_actions', None)
            actor, metadata = evaluation.load_actor(resolved, t_eval, 'cpu')
            evidence[item['name']] = metadata
            del actor
    if not {'C0_5500', 'F10_3000'}.issubset(evidence):
        raise RuntimeError('Both C0/F10 reference models required')
    return evidence


def run(root, train):
    root, train = Path(root).resolve(), Path(train).resolve()
    result = dict(status='FAILED', observed_epoch=time.time(), service_signals_sent=False,
                  gpu_context_created=False, checkpoint_load_device='cpu',
                  required_runtime_versions=REQUIRED_VERSIONS)
    # Reserve before any checks: failure receipts are retained, never overwritten.
    with (root / RECEIPT).open('x', encoding='utf-8') as output:
        try:
            manifest = json.loads((root / 'package_manifest.json').read_text(encoding='utf-8'))
            checked = verify_package(root, manifest)
            result['package_members_verified'] = True
            import torch
            versions = dict(rsl_rl_lib=importlib.metadata.version('rsl-rl-lib'),
                            torch=str(torch.__version__))
            result['runtime_versions'] = versions
            verify_versions(versions)
            result['runtime_verified'] = True
            expected = json.loads((root / 'previous_source_hashes_20261002.json').read_text(encoding='utf-8'))
            source_result = verify_provided_sources(train, expected, checked['fresh_rewards_original.py']['sha256'])
            result.update(provided_sources_match=True, provided_source_hashes=source_result['provided'],
                          original_reward_matches=True, original_reward_sha256=source_result['original_reward_sha256'])
            snapshot = json.loads((root / 'protected_runtime_sources.json').read_text(encoding='utf-8'))
            spec = importlib.util.find_spec('rsl_rl')
            if spec is None or not spec.origin:
                raise RuntimeError('Current rsl_rl package cannot be located')
            package = Path(spec.origin).resolve().parent
            result['runtime_source_verification'] = verify_runtime_sources(snapshot, package)
            result['process_observation_before_cpu_load'] = observe_processes()
            result['reference_loader_evidence'] = load_reference_models_cpu(root, checked)
            result['reference_loader_verified'] = True
            verify_package(root, manifest)
            verify_provided_sources(train, expected, checked['fresh_rewards_original.py']['sha256'])
            verify_runtime_sources(snapshot, package)
            result['process_observation_after_cpu_load'] = observe_processes()
            result['no_competing_gpu_jobs_verified'] = True
            result['official_streamer_present'] = result['process_observation_after_cpu_load']['official_streamer_present']
            result['probe_sha256'] = checked['recovery_probe.zip']['sha256']
            result['status'] = 'PASS'
            result['verified_epoch'] = time.time()
        except Exception as exc:
            result['error'] = type(exc).__name__ + ': ' + str(exc)
            result['verified_epoch'] = time.time()
        finally:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--train-dir', type=Path, default=Path('/workspace/training/quadruped'))
    args = parser.parse_args()
    result = run(args.root, args.train_dir)
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit('Preflight failed: do not launch training')


if __name__ == '__main__':
    main()
