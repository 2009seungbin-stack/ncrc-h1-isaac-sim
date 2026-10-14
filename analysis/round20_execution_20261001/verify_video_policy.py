from pathlib import Path
import json,hashlib,torch
B=Path(__file__).resolve().parent;R=B/'stress_results/extracted/go2_stress_20261002'
checkpoint=R/'candidate/model_3000.pt'
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
sd=ck['actor_state_dict']; ws=[k for k,v in sd.items() if k.endswith('weight') and v.ndim==2]
torch.manual_seed(103);x=torch.randn(4096,235)
with torch.no_grad():
    h=x
    for i,k in enumerate(ws):
        h=h @ sd[k].T+sd[k[:-6]+'bias']
        if i<len(ws)-1:h=torch.nn.functional.elu(h)
policies=list(R.rglob('policy.pt'));assert policies,'No fresh JIT policy'
checks=[]
for p in policies:
    jit=torch.jit.load(str(p),map_location='cpu').eval()
    with torch.no_grad():y=jit(x)
    assert y.shape==h.shape and torch.isfinite(y).all()
    delta=float((y-h).abs().max());assert delta<1e-4,(p,delta)
    checks.append(dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_absolute_error=delta))
d=dict(checkpoint_iter=ck['iter'],random_observations=4096,policies=checks,verified=True)
(B/'video_policy_verification.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d,indent=2))
