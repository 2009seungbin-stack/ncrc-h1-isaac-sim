#!/usr/bin/env python3
"""R3 번들 로컬 분석 — GPU 0.

    python tools/r3/analyze_r3.py <R3_bundle.tgz 또는 풀어둔 디렉터리> [--s2 eval_results/S2_axisall.json]
                                  [--s2-tfevents <S2 events.out.tfevents.*>]

출력: 후보 순위표(SimScore, H 낙상, Stress 생존/LCB), S2 대비 시나리오별 Δ, 축별 RMSE,
      학습 궤적(에피소드 길이 정규화) R3A/R3C vs S2, 게이트 판정.
"""
import argparse, glob, json, math, os, sys, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import score_model as sm  # noqa: E402  (tfevents 파서)

H = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
X = ["X1", "X2", "X3", "X4", "X5"]
GATE_SIM = 63.32  # S2 62.32 + 1.0


def load_bundle(p):
    if os.path.isdir(p):
        return p
    out = os.path.splitext(os.path.splitext(p)[0])[0] + "_extracted"
    os.makedirs(out, exist_ok=True)
    with tarfile.open(p) as tf:
        tf.extractall(out)
    return out


def merged_results(root):
    res = {}
    for p in sorted(glob.glob(os.path.join(root, "eval_results", "R3_*.json"))):
        b = os.path.basename(p)
        if any(b.endswith(f"_{t}.json") for t in ("flat", "rough", "slope", "xflat", "xrough", "xslope", "xpush")):
            continue
        d = json.load(open(p, encoding="utf-8"))
        res[d["run_id"]] = d
    return res


def traj(path, every=1000):
    d = sm.read_tfevents(path)
    T = sm.FEATURE_TAGS
    steps = sorted(d[T["terrain"]].keys())
    def avg(tag, s, k=25):
        ks = [x for x in d[tag] if s - k * 10 <= x <= s]
        return sum(d[tag][x] for x in ks) / len(ks) if ks else float("nan")
    rows = []
    marks = [s for s in steps if s % every == 0] + [steps[-1]]
    for s in marks:
        el = avg(T["ep_len"], s)
        rows.append((s, avg(T["terrain"], s), avg(T["err_yaw"], s) / el * 1e3, avg(T["err_xy"], s) / el * 1e3,
                     avg(T["base_contact"], s), el, avg(T["mean_reward"], s)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--s2", default=os.path.join(HERE, "..", "..", "eval_results", "S2_axisall.json"))
    ap.add_argument("--s2-tfevents", default=None)
    a = ap.parse_args()
    root = load_bundle(a.bundle)
    print(f"bundle root: {root}")

    res = merged_results(root)
    if not res:
        print("!! eval_results/R3_*.json (merged) 없음"); return
    s2 = json.load(open(a.s2, encoding="utf-8")) if os.path.exists(a.s2) else None

    # ---- 순위표
    rows = []
    for rid, d in res.items():
        st = d.get("stress") or {}
        rows.append((rid, d["sim_score"], d["falls"], st.get("survival"), st.get("survival_lcb"), st.get("falls")))
    rows.sort(key=lambda r: (-(r[4] or 0), -r[1]))
    print("\n| policy | SimScore | H낙상 | X생존 | X LCB | X낙상 |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.2f} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")

    # ---- S2 대비 시나리오 Δ + 축별 RMSE
    for rid, d in res.items():
        print(f"\n## {rid}  (SimScore {d['sim_score']:.2f}" + (f", S2 대비 {d['sim_score'] - s2['sim_score']:+.2f}" if s2 else "") + ")")
        print("| 시나리오 | 점수 | S2 | Δ | vx | vy | yaw | 낙상 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k in H + X:
            sc = d["scenarios"].get(k)
            if not sc:
                continue
            base = s2["scenarios"].get(k) if s2 else None
            r = sc["rmse"]
            print(f"| {k} | {sc['score']:.3f} | {base['score']:.3f} | {sc['score'] - base['score']:+.3f} | " if base else f"| {k} | {sc['score']:.3f} | - | - | ",
                  f"{r['vx']:.3f} | {r['vy']:.3f} | {r['yaw']:.3f} | {sc['falls']} |", sep="")
        if s2:
            # 축별 RMSE 평균 변화 (H만)
            for ax in ("vx", "vy", "yaw"):
                m_new = sum(d["scenarios"][k]["rmse"][ax] for k in H if k in d["scenarios"]) / 7
                m_old = sum(s2["scenarios"][k]["rmse"][ax] for k in H) / 7
                print(f"  {ax} RMSE 평균: S2 {m_old:.4f} -> {m_new:.4f} ({(m_new / m_old - 1) * 100:+.1f}%)")

    # ---- 학습 궤적
    for ev in sorted(glob.glob(os.path.join(root, "logs", "rsl_rl", "humanoid", "*", "events.out.tfevents.*"))):
        run = os.path.basename(os.path.dirname(ev))
        print(f"\n## 학습 궤적 {run}")
        print("| iter | 지형 | yaw/ep×1e3 | xy/ep×1e3 | 낙상 | ep_len | reward |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for s, te, ey, ex, fa, el, rw in traj(ev):
            print(f"| {s} | {te:.3f} | {ey:.3f} | {ex:.3f} | {fa:.3f} | {el:.1f} | {rw:.2f} |")
    if a.s2_tfevents and os.path.exists(a.s2_tfevents):
        print("\n## 학습 궤적 S2 (대조)")
        for s, te, ey, ex, fa, el, rw in traj(a.s2_tfevents):
            print(f"| {s} | {te:.3f} | {ey:.3f} | {ex:.3f} | {fa:.3f} | {el:.1f} | {rw:.2f} |")

    # ---- 게이트
    print("\n## 게이트")
    lcb_ref = None
    if "S2_2900" in res and res["S2_2900"].get("stress"):
        lcb_ref = res["S2_2900"]["stress"]["survival_lcb"]; print(f"S2 Stress LCB 기준선: {lcb_ref}")
    else:
        print("S2 Stress 기준선 없음 → 후보 중 최고 LCB를 기준으로 상대 비교")
    best = None
    for r in rows:
        if r[0] == "S2_2900":
            continue
        ok_sim = r[1] >= GATE_SIM
        ok_x = (r[4] or 0) >= (lcb_ref if lcb_ref is not None else max((q[4] or 0) for q in rows if q[0] != "S2_2900"))
        print(f"  {r[0]}: SimScore {r[1]:.2f} {'>=' if ok_sim else '<'} {GATE_SIM}  |  X LCB {r[4]} {'OK' if ok_x else 'below'}  -> {'PASS' if ok_sim and ok_x else 'fail'}")
        if ok_sim and ok_x and best is None:
            best = r[0]
    print(f"\n채택 후보: {best}" if best else "\n채택 후보 없음 → S2 유지")


if __name__ == "__main__":
    main()
