"""과도구간 실험 결과(TR_*.json)를 R3A_9000 공식 q 와 대조한다.
    python tools/tr/tr_analyze.py eval_results/TR_T0_seg5.json eval_results/TR_T0_seg25.json
"""
import io, json, math, sys
OFFICIAL = {"H1": 0.8419, "H2": 0.83429, "H3": 0.83714, "H4": 0.83429, "H7": 0.83333}
BASE = {"H1": 0.9275, "H2": 0.9229, "H3": 0.9173, "H4": 0.9166, "H7": 0.8761}  # 09-08, transient 0.5
for path in sys.argv[1:]:
    d = json.load(io.open(path, encoding="utf-8"))
    sc = d["scenarios"]
    print(f"\n== {path}")
    print(f"{'sid':4} {'surv':>6} {'track':>7} {'base':>7} {'offic':>7} {'ratio':>6} {'c_mse':>7}  rmse_axis")
    for k in ("H1", "H2", "H3", "H4", "H7"):
        if k not in sc: continue
        v = sc[k]; tr = v["tracking"]; o = OFFICIAL[k]
        c = (-math.log(o))**2 - (-math.log(tr))**2
        print(f"{k:4} {v['survival']:6.3f} {tr:7.4f} {BASE[k]:7.4f} {o:7.4f} {o/tr:6.3f} {c:7.4f}  {v.get('rmse_axis')}")
