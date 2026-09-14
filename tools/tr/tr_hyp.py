"""TR2 결과(진단 축 포함)로 공식 추종 함수 후보를 대조한다.
    python tools/tr/tr_hyp.py eval_results/TR2_flat.json [TR2_rough.json TR2_slope.json]
공식 q (R3A_9000, 2R): 시나리오 점수 = 생존×추종 이고 자체 생존 1.0 이므로 추종 = 공식 q 로 본다.
"""
import io, json, math, statistics as st, sys
OFF2 = {"H1": 0.8419, "H2": 0.83429, "H3": 0.83714, "H4": 0.83429, "H5": 0.79333, "H6": 0.82857, "H7": 0.83333}
OFF1 = {"H1": 0.667, "H2": 0.598, "H3": 0.661, "H4": 0.5, "H5": 0.562, "H6": 0.612, "H7": 0.62}
OFF = OFF1 if "--r1" in sys.argv else OFF2
sys.argv = [a for a in sys.argv if a != "--r1"]
sc = {}
for p in sys.argv[1:]:
    sc.update(json.load(io.open(p, encoding="utf-8"))["scenarios"])
ids = [k for k in ("H1","H2","H3","H4","H5","H6","H7") if k in sc]
def cands(v):
    r = v["rmse"]; x = v.get("rmse_extra", {}); mae = v.get("mae", {})
    m3 = [r[a]**2 for a in ("vx","vy","yaw")]
    m6 = m3 + [x[a]**2 for a in ("vz","wx","wy")] if x else None
    out = {
        "mean3": math.exp(-math.sqrt(sum(m3)/3)),
        "sum3": math.exp(-math.sqrt(sum(m3))),
        "rmse_sum3": math.exp(-sum(math.sqrt(q) for q in m3)),
        "mae_mean3": math.exp(-sum(mae.values())/3) if mae else None,
        "mae_sum3": math.exp(-sum(mae.values())) if mae else None,
    }
    if m6:
        out["mean6"] = math.exp(-math.sqrt(sum(m6)/6))
        out["sum6"] = math.exp(-math.sqrt(sum(m6)))
        out["mean_lin3_ang3"] = math.exp(-math.sqrt((sum(m6[:2])+m6[3] + m6[2]+m6[4]+m6[5])/6))
    return out
rows = {k: {n: (None if q is None else q * sc[k]["survival"]) for n, q in cands(sc[k]).items()} for k in ids}
names = [n for n in rows[ids[0]] if rows[ids[0]][n] is not None]
print("sid   offic  " + "  ".join(f"{n:>13}" for n in names))
for k in ids:
    print(f"{k:4} {OFF[k]:6.3f}  " + "  ".join(f"{rows[k][n]:13.4f}" for n in names))
print("\n후보별 공식/후보 비율 - 평균, 표준편차, 변동계수 (작을수록 공식 변동을 잘 설명)")
for n in names:
    rt = [OFF[k]/rows[k][n] for k in ids]
    mu = st.mean(rt); sd = st.pstdev(rt)
    print(f"{n:>14}: mean {mu:.4f}  sd {sd:.4f}  cv {sd/mu:.4f}   " + " ".join(f"{q:.3f}" for q in rt))
print("\n축별 RMSE (vx vy yaw | vz wx wy) / MAE")
for k in ids:
    v = sc[k]; r=v["rmse"]; x=v.get("rmse_extra",{}); mae=v.get("mae",{})
    print(f"{k}: {r['vx']:.3f} {r['vy']:.3f} {r['yaw']:.3f} | {x.get('vz',0):.3f} {x.get('wx',0):.3f} {x.get('wy',0):.3f} | mae {mae.get('vx',0):.3f} {mae.get('vy',0):.3f} {mae.get('yaw',0):.3f}")
