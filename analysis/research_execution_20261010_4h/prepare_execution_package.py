"""Build one reviewed local package. No server, simulator or policy inference."""
import ast
import hashlib
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS/'analysis/baseline_redesign_20261009_4h/execution/package'
DEST = ROOT/'execution/package'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    design = json.loads((ROOT/'design.json').read_text(encoding='utf-8'))
    assert design['execution_authorized'] is True and design['candidate_name'] == 'S5_torque_half'
    assert design['evaluation']['screen']['enabled'] is False
    names = ['runner.py','evaluation.py','streaming_control_libc.py',
             'checkpoint_capture_probe.py','live_preflight.py','preflight_receipt.py','start_once.py',
             'quadruped_rewards_S5_torque_half.py','RUNNER_USAGE.txt']
    sources = {n: ROOT/'execution'/n for n in names}
    sources.update({'design.json': ROOT/'design.json','decision_rules.py': ROOT/'decision_rules.py'})
    inherited = ['fresh_rewards_original.py','previous_source_hashes_20261002.json',
                 'reference_models.json','source_snapshot.py','references/C0_5500.pt',
                 'references/F10_3000.pt']
    sources.update({n: OLD/n for n in inherited})
    sources.update({n: ROOT/'execution'/n for n in ['t_eval.py','t_eval_go2_core.py']})
    for name, src in sources.items():
        if not src.is_file():
            raise FileNotFoundError('Package input incomplete: '+name)
        if name.endswith('.py'):
            ast.parse(src.read_text(encoding='utf-8'))
    DEST.mkdir(parents=True,exist_ok=True)
    for name,src in sources.items():
        dst = DEST/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
    assert sha(DEST/'fresh_rewards_original.py') == '09630e6f548b16a2fa0a81eef73aa393cf5f12eadccd0bad8a144b924e8c1be7'
    spec=importlib.util.spec_from_file_location('frozen_runner_contract',DEST/'runner.py')
    contract=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert design['budget']['deadlines_relative_to_actual_allocation_start']==contract.OFFSETS
    contract.validate_candidate((DEST/'fresh_rewards_original.py').read_bytes(),
        (DEST/design['candidate_reward_file']).read_bytes(),design['candidates'][design['candidate_name']]['weights'])
    assert sha(DEST/design['candidate_reward_file'])==design['candidate_reward_sha256']
    for item in json.loads((DEST/'reference_models.json').read_text(encoding='utf-8')):
        assert sha(DEST/item['checkpoint'])==item['sha256']
    probe=DEST/'recovery_probe.zip'
    with zipfile.ZipFile(probe,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in [('probe.bin',bytes(range(256))*16),('purpose.txt',b'Fresh four-hour binary retrieval probe. Not a training result.\n')]:
            info=zipfile.ZipInfo(name,date_time=(2026,10,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    entries={n:dict(bytes=(DEST/n).stat().st_size,sha256=sha(DEST/n)) for n in [*sources,'recovery_probe.zip']}
    manifest=dict(scope='S5_only_scratch1500_env6144_seed42_actual_allocation4h',
                  files=entries, extra_gpu_experiments=False, server_started_by_builder=False)
    (DEST/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    output=ROOT/'execution/research_execution_20261010_4h_package.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in [*entries,'package_manifest.json']:
            archive.write(DEST/name,name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,spec in entries.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==spec['sha256']
    report=dict(package_path=str(output.resolve()),bytes=output.stat().st_size,sha256=sha(output),
                member_hashes_verified=True,checkpoint_deserialization_by_builder=False,
                probe=entries['recovery_probe.zip'],source_files=entries)
    (ROOT/'execution/package_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    ledger=json.loads((ROOT/'execution.json').read_text(encoding='utf-8'))
    assert ledger['server_started'] is False, 'Never rewrite execution readiness after server start'
    ledger.update(package=report,design_sha256=sha(ROOT/'design.json'),local_package_ready=True)
    (ROOT/'execution.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['package_path','bytes','sha256','probe']},indent=2))

if __name__=='__main__':
    main()
