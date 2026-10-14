from pathlib import Path
import shutil,tarfile,hashlib,ast,json
B=Path(__file__).resolve().parent
S=B/'validation_results/extracted/go2_validation_20261002'
R=B/'stress_inputs/go2_stress_20261002'; R.mkdir(parents=True,exist_ok=True)
for name in ['candidate','original_C0_model_5500.pt','original_C0_env.yaml','t_eval_go2.py','t_eval.py','F00_weights.json','protected_source_hashes.json','selection.json']:
    p=S/name
    if p.is_dir(): shutil.copytree(p,R/name,dirs_exist_ok=True)
    else: shutil.copy2(p,R/name)
t=(B/'validation_continue.py').read_text()
a=t.index("        b0=evaluate('original_B'")
b=t.index("        for name,ck in",a)
t=t[:a]+'''        detail=json.loads((ROOT/'selection.json').read_text())
        assert detail['heldout_gate_passed'] is False
        state('ADDITIONAL_STRESS',selection=detail,B_rerun=False)
'''+t[b:]
a=t.index("        job([ISAAC,'-p','play.py'")
t=t[:a]+'''        exported=TRAIN/'exported'; exported.mkdir(exist_ok=True)
        shutil.copy2(chosen,exported/'model_best.pt')
        shutil.copy2(ROOT/'candidate/params/env.yaml',exported/'env.yaml')
        (exported/'params').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'candidate/params/env.yaml',exported/'params/env.yaml')
        assert sha(exported/'model_best.pt') == sha(chosen)
'''+t[a:]
t=t.replace("[ROOT/'candidate',TRAIN/'logs']","[TRAIN/'exported']")
t=t.replace("go2_validation_20261002_results.tar.gz","go2_stress_20261002_results.tar.gz")
ast.parse(t)
(R/'stress_continue.py').write_text(t)
archive=B/'go2_stress_20261002_inputs.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tf: tf.add(R,arcname=R.name)
print(archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())
