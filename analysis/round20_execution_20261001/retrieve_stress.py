from pathlib import Path
import hashlib,tarfile,shutil,sys,json
B=Path(__file__).resolve().parent
src=Path.home()/'Downloads/go2_stress_20261002_results.tar.gz'
sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha==sys.argv[1],sha
D=B/'stress_results';D.mkdir(exist_ok=True);dst=D/src.name;shutil.copy2(src,dst)
X=D/'extracted';X.mkdir(exist_ok=True)
with tarfile.open(dst) as t:
    ms=t.getmembers()
    for m in ms:
        assert (X/m.name).resolve().is_relative_to(X.resolve())
        assert not(m.issym() or m.islnk())
    t.extractall(X)
R=X/'go2_stress_20261002'
assert hashlib.sha256((R/'candidate/model_3000.pt').read_bytes()).hexdigest()=='4684a58838229b729feb2f2534ab9423a4993377e75179a0d55a95904ecf23d2'
assert hashlib.sha256((R/'original_C0_model_5500.pt').read_bytes()).hexdigest()=='8d553e697ed4c2856ab002431af96d626eb8b0d23e56b7c754d652193026b1d3'
assert (R/'candidate/params/env.yaml').read_bytes()==(B/'validation_results/extracted/go2_validation_20261002/candidate/params/env.yaml').read_bytes()
restore=json.loads((R/'reward_restored.json').read_text());assert restore['restored']
assert hashlib.sha256((R/'fresh_server_rewards_original.py').read_bytes()).hexdigest()==restore['sha256']
assert len(list((R/'eval').glob('*/merged.json')))==6
video=list(R.rglob('*.mp4')); assert video, 'Missing generated video'
print(json.dumps(dict(sha256=sha,bytes=dst.stat().st_size,members=len(ms),videos=[str(p) for p in video]),indent=2))
