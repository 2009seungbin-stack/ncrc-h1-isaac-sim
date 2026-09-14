#!/usr/bin/env python3
"""공식 점수 추정기 — 학습 로그(tfevents)만으로 SimScore 를 추정한다. GPU 0.

    python score_model.py --tfevents run/events.out.tfevents.* --step 4600
    python score_model.py --predict --tfevents <새 run> --step 5227
    python score_model.py --predict --features base_contact=0.127,err_yaw=4.60
    python score_model.py --ledger
    python score_model.py --add-official --run-id E1 --sim 44.10 \
        --q 0.70,0.63,0.68,0.50,0.58,0.63,0.64

--------------------------------------------------------------------------
정직하게 먼저: 이건 "학습된 모델" 이 아니다
--------------------------------------------------------------------------
공식 채점 기록은 **1건**뿐이다 (2026-09-01 제출, SimScore 42.34).
1건으로는 파라미터를 추정할 수 없다. 그래서 이 도구는:

  1. 시나리오별 민감도 지수(s, x, y, l)를 **설계로 고정한다** (아래 표, 가정이다)
  2. 그 1건을 앵커로 삼아 시나리오별 상수를 흡수한다 (앵커를 넣으면 42.34가 그대로 나온다)
  3. 새 run 은 **앵커 대비 능력비(ratio)** 로만 움직인다

즉 절대값을 새로 만들어내는 게 아니라, **앵커에서 얼마나 움직였는지**를 옮긴다.
따라서 신뢰할 수 있는 출력은 **Δ(변화량)** 이고, 절대값은 앵커의 정확도에 묶여 있다.
지수 표가 틀리면 예측도 틀린다 — `--sensitivity` 로 그 폭을 항상 같이 본다.

T-EVAL(`t_eval.py`)을 대체하지 않는다. T-EVAL 은 정책을 실제로 굴려 재고,
이건 학습 로그만 본다. 둘의 용도가 다르다:

  score_model : GPU 0, 학습 중에도 즉시. 러프한 조기 신호.
  t_eval      : GPU 필요. 시나리오별 생존/추종을 실제로 측정.
"""

import argparse
import collections
import json
import math
import os
import struct
import sys

# 표가 한글·기호를 쓴다. 윈도우 콘솔 기본 코덱(cp949)에서 죽지 않게 UTF-8로 돌린다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "calibration.json")

W = {"H1": 0.15, "H2": 0.20, "H3": 0.10, "H4": 0.10,
     "H5": 0.15, "H6": 0.15, "H7": 0.15}
ORDER = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
NAME = {"H1": "제자리 균형", "H2": "전진 속도 추종", "H3": "좌우 속도 추종",
        "H4": "회전 복합 추종", "H5": "약한 요철", "H6": "완만한 경사 ±10°",
        "H7": "밀침 회복"}

# 시나리오별 민감도 지수 — **설계 가정이다. 데이터로 추정한 값이 아니다.**
#   s: 안정성(1-낙상률)   x: 선속도 추종   y: 각속도 추종   l: 지형 커리큘럼 도달도
# 근거: 규정 제7조의 시나리오 정의 + official_score_breakdown.md 의 배점 구조.
EXP = {
    "H1": dict(s=1.0, x=0.3, y=0.3, l=0.0),   # 명령 0 — 순수 균형. 추종은 표류 억제로만
    "H2": dict(s=0.6, x=1.0, y=0.0, l=0.0),   # 선속도 추종이 지배
    "H3": dict(s=0.6, x=1.0, y=0.0, l=0.0),   # error_vel_xy 가 y축 오차를 함께 담는다
    "H4": dict(s=0.6, x=0.3, y=1.0, l=0.0),   # yaw 지배 + 전진 보조 (회전 "복합")
    "H5": dict(s=1.0, x=0.6, y=0.2, l=0.5),   # 요철 — 안정성 + 지형 도달도
    "H6": dict(s=1.0, x=0.6, y=0.0, l=0.5),   # 경사 — 안정성 + 지형 도달도
    "H7": dict(s=1.4, x=0.4, y=0.0, l=0.2),   # 밀침 — 넘어지느냐가 거의 전부
}

K_YAW = 0.15   # err_vel_yaw(≈4) 를 exp() 에 넣기 위한 스케일. 비(ratio)에만 영향.
TERRAIN_MAX = 10.0   # num_rows=10 (env.yaml 의 terrain_generator)

FEATURE_TAGS = {
    "base_contact": "Episode_Termination/base_contact",
    "err_xy": "Metrics/base_velocity/error_vel_xy",
    "err_yaw": "Metrics/base_velocity/error_vel_yaw",
    "terrain": "Curriculum/terrain_levels",
    "mean_reward": "Train/mean_reward",
    "ep_len": "Train/mean_episode_length",
}


# ---------------------------------------------------------------- tfevents
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            return r, i
        s += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i); yield f, wt, v
        elif wt == 1:
            yield f, wt, buf[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(buf, i); yield f, wt, buf[i:i + ln]; i += ln
        elif wt == 5:
            yield f, wt, buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")


def read_tfevents(path):
    """{tag: {step: value}} — Summary.value[].simple_value 만 뽑는 최소 파서.

    tensorboard 없이 동작한다 (서버에도 없을 수 있다).
    """
    out = collections.defaultdict(dict)
    with open(path, "rb") as fh:
        data = fh.read()
    i, n = 0, len(data)
    while i + 12 <= n:
        ln = struct.unpack("<Q", data[i:i + 8])[0]
        i += 12
        rec = data[i:i + ln]; i += ln + 4
        if len(rec) < ln:
            break
        step, summ = None, None
        for f, wt, v in _fields(rec):
            if f == 2 and wt == 0:
                step = v
            elif f == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for f, wt, v in _fields(summ):
            if f != 1 or wt != 2:
                continue
            tag = val = None
            for g, gwt, gv in _fields(v):
                if g == 1 and gwt == 2:
                    tag = gv.decode("utf-8", "replace")
                elif g == 2 and gwt == 5:
                    val = struct.unpack("<f", gv)[0]
            if tag is not None and val is not None:
                out[tag][step or 0] = val
    return dict(out)


def features_from_tfevents(path, step=None, win=25):
    """선택 체크포인트 시점의 지표. win 만큼 앞뒤 평균해 iteration 노이즈를 죽인다."""
    d = read_tfevents(path)
    missing = [k for k, t in FEATURE_TAGS.items() if t not in d]
    if missing:
        print(f"[warn] tfevents 에 없는 태그: {missing}", file=sys.stderr)
    anytag = FEATURE_TAGS["base_contact"]
    if step is None:
        step = max(d[anytag]) if anytag in d else 0
    feats = {"step": step}
    for k, tag in FEATURE_TAGS.items():
        if tag not in d:
            continue
        pts = d[tag]
        vals = [pts[s] for s in range(step - win, step + win + 1) if s in pts]
        feats[k] = sum(vals) / len(vals) if vals else pts.get(step)
    return feats


# ---------------------------------------------------------------- 모델
def competences(f):
    """지표 -> 4개 능력치 (전부 클수록 좋고 0~1 근방)."""
    return {
        "S": max(1e-6, 1.0 - f["base_contact"]),         # 안정성
        "Txy": math.exp(-f["err_xy"]),                    # 선속도 추종
        "Tyaw": math.exp(-K_YAW * f["err_yaw"]),          # 각속도 추종
        "L": max(1e-6, f["terrain"] / TERRAIN_MAX),       # 지형 도달도
    }


def predict(anchor, new_feats, exp=EXP):
    """앵커 대비 능력비를 시나리오별 지수로 옮긴다."""
    ca, cn = competences(anchor["features"]), competences(new_feats)
    ratio = {k: cn[k] / ca[k] for k in ca}
    q_anchor = anchor["official_q"]
    out, sim = {}, 0.0
    for sid in ORDER:
        e = exp[sid]
        m = (ratio["S"] ** e["s"] * ratio["Txy"] ** e["x"]
             * ratio["Tyaw"] ** e["y"] * ratio["L"] ** e["l"])
        q = min(1.0, max(0.0, q_anchor[sid] * m))
        out[sid] = {"q_anchor": q_anchor[sid], "mult": m, "q": q,
                    "delta": q - q_anchor[sid]}
        sim += W[sid] * q
    return out, 70.0 * sim, ratio


def sensitivity(anchor, new_feats, spread=0.5):
    """지수를 ±spread 배 흔들어 SimScore 가 얼마나 움직이는지 본다.

    지수 표가 '가정'이라는 사실을 숫자로 보여주는 것이 목적이다.
    """
    lo = {s: {k: v * (1 - spread) for k, v in e.items()} for s, e in EXP.items()}
    hi = {s: {k: v * (1 + spread) for k, v in e.items()} for s, e in EXP.items()}
    _, s_lo, _ = predict(anchor, new_feats, lo)
    _, s_hi, _ = predict(anchor, new_feats, hi)
    return min(s_lo, s_hi), max(s_lo, s_hi)


# ---------------------------------------------------------------- 원장
def default_ledger():
    """2026-09-01 제출본 = 유일한 공식 앵커.

    features 는 training_C_final.tfevents 의 step 4600 (제출된 체크포인트) 실측이다.
    step 4600 에서 Train/mean_reward = -4.58413 이 training_status.md 의
    -4.584132 와 일치하는 것으로 스텝 정렬을 확인했다.
    """
    return {
        "records": [{
            "run_id": "H1_R1_model_4600",
            "date": "2026-09-01",
            "checkpoint": "model_4600.pt",
            "note": "termination_penalty -5 -> -200 단일 변경. 공식 70.34점.",
            "features": {"step": 4600, "base_contact": 0.18823, "err_xy": 0.53001,
                         "err_yaw": 3.83144, "terrain": 5.24955,
                         "mean_reward": -5.36449, "ep_len": 884.77},
            "official_sim": 42.34,
            "official_q": {"H1": 0.667, "H2": 0.598, "H3": 0.661, "H4": 0.500,
                           "H5": 0.562, "H6": 0.612, "H7": 0.620},
        }]
    }


def load_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER, encoding="utf-8") as f:
            return json.load(f)
    led = default_ledger()
    save_ledger(led)
    return led


def save_ledger(led):
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(led, f, indent=2, ensure_ascii=False)


def anchor_of(led):
    return led["records"][0]


# ---------------------------------------------------------------- 출력
def show_features(f):
    print("학습 로그 지표 (step %s, ±25 iter 평균)" % f.get("step"))
    for k in ("base_contact", "err_xy", "err_yaw", "terrain", "mean_reward", "ep_len"):
        if k in f and f[k] is not None:
            print(f"  {k:<14} {f[k]:+.5f}")
    c = competences(f)
    print("  ---- 능력치 ----")
    for k, v in c.items():
        print(f"  {k:<14} {v:.4f}")


def show_prediction(led, feats, spread):
    a = anchor_of(led)
    rows, sim, ratio = predict(a, feats)
    print(f"\n앵커: {a['run_id']}  (공식 SimScore {a['official_sim']})")
    print("능력비 (새 run / 앵커):  " +
          "  ".join(f"{k}={v:.3f}" for k, v in ratio.items()))
    print("\n| ID | 시나리오 | w | 앵커 q | 배율 | 추정 q | Δ | Δ점수 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for sid in ORDER:
        r = rows[sid]
        print(f"| {sid} | {NAME[sid]} | {W[sid]:.2f} | {r['q_anchor']:.3f} | "
              f"{r['mult']:.3f} | {r['q']:.3f} | {r['delta']:+.3f} | "
              f"{70*W[sid]*r['delta']:+.2f} |")
    d = sim - a["official_sim"]
    print(f"\n추정 SimScore  **{sim:.2f}** / 70   (앵커 42.34 대비 {d:+.2f})")
    lo, hi = sensitivity(a, feats, spread)
    print(f"지수 ±{int(spread*100)}% 흔들 때 범위: {lo:.2f} ~ {hi:.2f}"
          f"  (폭 {hi-lo:.2f}점)")
    print(f"추정 총점 = {sim:.2f} + 설계의도/리포트 28  = **{sim+28:.2f}** / 100")
    print("\n주의: 앵커 1건짜리 외삽이다. 부호(개선/악화)와 Δ의 크기 순서는 믿을 만하지만,")
    print("      절대값은 T-EVAL 이나 공식 제출로 확인하기 전까지 참고치다.")
    return sim


def show_ledger(led):
    print(f"보정 원장 ({LEDGER})")
    print(f"공식 채점 기록 {len(led['records'])}건"
          + ("  ← 1건이면 추정이 아니라 외삽이다" if len(led["records"]) < 2 else ""))
    for r in led["records"]:
        f = r["features"]
        print(f"\n  {r['run_id']}  ({r.get('date','?')})  공식 SimScore {r['official_sim']}")
        print(f"    {r.get('note','')}")
        print(f"    base_contact={f['base_contact']:.4f}  err_xy={f['err_xy']:.4f}  "
              f"err_yaw={f['err_yaw']:.4f}  terrain={f['terrain']:.3f}")
        print("    q = " + " ".join(f"{k}:{v:.3f}" for k, v in r["official_q"].items()))
    if len(led["records"]) >= 2:
        print("\n앵커가 2건 이상이다 — 지수 표를 실제로 검증할 수 있다:")
        a = led["records"][0]
        for r in led["records"][1:]:
            _, sim, _ = predict(a, r["features"])
            print(f"  {r['run_id']}: 예측 {sim:.2f} vs 공식 {r['official_sim']:.2f}"
                  f"  → 오차 {sim - r['official_sim']:+.2f}")


def show_exponents():
    print("\n시나리오별 민감도 지수 (설계 가정 — 데이터로 추정한 값이 아니다)")
    print("| ID | 시나리오 | s 안정성 | x 선속도 | y 각속도 | l 지형 |")
    print("|---|---|---:|---:|---:|---:|")
    for sid in ORDER:
        e = EXP[sid]
        print(f"| {sid} | {NAME[sid]} | {e['s']:.1f} | {e['x']:.1f} | "
              f"{e['y']:.1f} | {e['l']:.1f} |")


# ---------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="학습 로그 기반 공식 SimScore 추정기")
    p.add_argument("--tfevents", help="tfevents 파일 경로")
    p.add_argument("--step", type=int, default=None,
                   help="선택 체크포인트의 iteration (생략 = 마지막)")
    p.add_argument("--win", type=int, default=25, help="평균 낼 앞뒤 iteration 수")
    p.add_argument("--features", help="tfevents 대신 직접: base_contact=0.127,err_xy=0.42,...")
    p.add_argument("--predict", action="store_true", help="SimScore 추정")
    p.add_argument("--sensitivity", type=float, default=0.5,
                   help="지수를 ±이 비율만큼 흔들어 범위를 낸다 (기본 0.5)")
    p.add_argument("--ledger", action="store_true", help="보정 원장 출력")
    p.add_argument("--exponents", action="store_true", help="민감도 지수 표 출력")
    p.add_argument("--add-official", action="store_true",
                   help="공식 점수를 받은 뒤 원장에 추가 (앵커가 늘어야 검증이 된다)")
    p.add_argument("--run-id")
    p.add_argument("--sim", type=float, help="공식 SimScore")
    p.add_argument("--q", help="공식 시나리오 득점률 7개, 쉼표 구분 (H1..H7)")
    p.add_argument("--note", default="")
    args = p.parse_args()

    led = load_ledger()

    if args.exponents:
        show_exponents()
        return
    if args.ledger:
        show_ledger(led)
        show_exponents()
        return

    feats = None
    if args.tfevents:
        feats = features_from_tfevents(args.tfevents, args.step, args.win)
    elif args.features:
        feats = {"step": args.step}
        for kv in args.features.split(","):
            k, v = kv.split("=")
            feats[k.strip()] = float(v)
        a = anchor_of(led)["features"]
        for k in ("base_contact", "err_xy", "err_yaw", "terrain"):
            if k not in feats:
                feats[k] = a[k]
                print(f"[warn] {k} 미지정 → 앵커값 {a[k]} 사용 (그 축은 변화 없음으로 본다)")

    if args.add_official:
        if not (args.run_id and args.sim and args.q and feats):
            raise SystemExit("--add-official 에는 --run-id --sim --q 와 "
                             "--tfevents(또는 --features) 가 모두 필요하다")
        qs = [float(x) for x in args.q.split(",")]
        if len(qs) != 7:
            raise SystemExit("--q 는 7개여야 한다 (H1..H7)")
        led["records"].append({
            "run_id": args.run_id, "date": "", "checkpoint": "",
            "note": args.note, "features": feats, "official_sim": args.sim,
            "official_q": dict(zip(ORDER, qs)),
        })
        save_ledger(led)
        print(f"원장에 추가했다. 이제 기록 {len(led['records'])}건.")
        show_ledger(led)
        return

    if feats is None:
        p.print_help()
        return

    show_features(feats)
    if args.predict:
        show_prediction(led, feats, args.sensitivity)


if __name__ == "__main__":
    main()
