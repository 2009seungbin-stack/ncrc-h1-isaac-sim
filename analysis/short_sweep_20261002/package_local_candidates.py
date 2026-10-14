"""Package existing local candidates and analysis, without claiming remote retrieval."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent
dst = root / 'go2_candidate_and_analysis.zip'
if dst.exists():
    raise SystemExit('Package already exists; preserve it and inspect before replacing.')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()
models=json.loads((root/'inputs'/'saved_models.json').read_text(encoding='utf-8-sig'))
entries={}
for name in ['F10_3000','C0_5500']:
    record=next(m for m in models if m['name']==name)
    path=root/'inputs'/record['checkpoint']
    assert sha(path)==record['sha256'], name+' hash mismatch'
    entries['models/'+name+'.pt']=path
for name in ['final_analysis.txt','short_screen_analysis.txt','protocol.txt',
             'saved_screen_summary.json','short_screen_summary.json',
             'short_diagnostics_summary.json','heldout_screen_summary.json',
             'short_learning_curves.png']:
    entries['analysis/'+name]=root/name
manifest={
    'purpose':'Existing local candidate checkpoints and measured comparison summaries',
    'preferred_candidate':'F10_3000',
    'candidate_is_existing_saved_model':True,
    'official_submission_performed':False,
    'official_90_score_confirmed':False,
    'remote_archive_recovered':False,
    'missing':'Four newly trained 800-iteration checkpoints and full raw experiment logs/traces remain in the remote archive.',
    'remote_archive':{'name':'short_sweep_20261002_results.tar.gz','bytes':503786122,'sha256':'1a8a5a284a6622e9a7cb60a5066e080adb15fa8061a7eb8a20f7edaaf34ca6d9'},
    'files':{name:{'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in entries.items()}
}
readme="""Go2 후보 모델 및 분석 로컬 묶음

models/F10_3000.pt: 별도 7개 시나리오에서 우세했던 기존 저장 모델.
models/C0_5500.pt: 비교 기준 원본. 두 파일 모두 실험에 사용된 SHA256과 일치함.

새 800회 학습 모델 4개와 전체 원시 로그/트레이스는 이 묶음에 포함되지 않음.
이 묶음은 원격 503,786,122바이트 결과 archive를 회수한 것이 아님.

내부 품질: F10_3000 0.9167106065, C0_5500 0.9090643912.
각각 1792 episodes에서 낙상 0회. 독립적인 학습 seed 반복은 없음.
공식 점수가 아니며 공식 90점 이상 달성은 확인되지 않음.
공식 제출은 수행하지 않았음.

상세 결과와 한계는 analysis/final_analysis.txt 및 summary JSON 참조.
manifest.json에는 모든 포함 자료의 크기와 SHA256이 기록됨.
"""
with zipfile.ZipFile(dst,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in entries.items():
        z.write(p,name)
    z.writestr('README.txt',readme)
    z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(dst) as z:
    assert z.testzip() is None
    saved=json.loads(z.read('manifest.json'))
    for name,info in saved['files'].items():
        data=z.read(name)
        assert len(data)==info['bytes']
        assert hashlib.sha256(data).hexdigest()==info['sha256'], name
    verification={'archive':str(dst),'bytes':dst.stat().st_size,'sha256':sha(dst),'entries':len(z.namelist()),'verified_file_hashes':len(saved['files']),'remote_archive_recovered':False}
(root/'local_candidate_bundle_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
