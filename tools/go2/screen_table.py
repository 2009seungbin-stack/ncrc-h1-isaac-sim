#!/usr/bin/env python3
"""screen_table.py — 학습 로그만으로 짧은 스크리닝 팔을 비교한다 (GPU·export·채점 불필요).

왜 로그로 판정하나: GB 는 **첫 250 iter 안에** 망가졌다(지형 3.5 -> 0, 선속도 오차 고정).
실패가 초반에 결정되므로 200 iter(8192 envs 약 23분)면 처방이 그 붕괴를 막는지 보인다.

    python screen_table.py S1=/workspace/g3logs/train_S1.log S0=... --out SCREEN_RESULT.md

error_vel_xy 는 에피소드 동안 norm/500 을 **누적**한 값이라 에피소드가 짧으면(낙상) 작게 찍힌다.
그래서 평균 에피소드 길이로 나눠 m/s 로 환산한 xy_mps 로 비교한다
(명령 재샘플 10 s / step_dt 0.02 s = 500 스텝).
참고: 전진 명령을 전부 무시하고 나머지를 완벽히 따르는 정책은 vx~U(-1,2) 에서 xy_mps ~= 0.67 이다.
GB(12288 envs) 최종값이 0.69 — 이 선을 뚫느냐가 판정의 핵심이다.
"""

import argparse
import re
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

KEYS = {
    "err_xy": "Metrics/base_velocity/error_vel_xy",
    "err_yaw": "Metrics/base_velocity/error_vel_yaw",
    "terrain": "Curriculum/terrain_levels",
    "contact": "Episode_Termination/base_contact",
    "ep_len": "Mean episode length",
    "reward": "Mean reward",
    "it_time": "Iteration time",
}
CMD_STEPS = 500
FORWARD_REFUSAL_MPS = 0.67
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# GB = NCRC 무수정, 12288 envs, seed 42, 2026-09-11 (PhysX 패치 오버플로가 매 스텝 발생한 run).
# 로컬 G3_pack/logs/train_GB.log 에서 이 스크립트로 뽑은 값. 서버에는 이 로그가 없어서 박아둔다.
# iter: (xy m/s, 지형 레벨, 낙상비)
REF_GB12K = {
    25: (0.968, 2.799, 0.182), 50: (0.952, 2.066, 0.436), 75: (0.942, 1.636, 0.336),
    100: (0.880, 1.134, 0.189), 125: (0.922, 0.825, 0.182), 150: (0.923, 0.519, 0.143),
    175: (0.889, 0.293, 0.187), 200: (0.915, 0.147, 0.155), 225: (0.902, 0.050, 0.137),
    250: (0.942, 0.013, 0.126), 275: (0.893, 0.005, 0.116), 300: (0.826, 0.029, 0.113),
    325: (0.861, 0.088, 0.113), 350: (0.885, 0.158, 0.101), 375: (0.834, 0.203, 0.109),
    400: (0.813, 0.237, 0.129), 500: (0.836, 0.329, 0.130), 1000: (0.727, 0.686, 0.061),
    1500: (0.690, 0.929, 0.068), 1975: (0.735, 1.099, 0.063),
}


def parse(path):
    rows, cur, over = {}, None, 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Patch buffer overflow" in line:
                over += 1
                continue
            line = ANSI.sub("", line)
            m = re.search(r"Learning iteration (\d+)/", line)
            if m:
                cur = int(m.group(1))
                rows[cur] = {}
                continue
            if cur is None or ":" not in line:
                continue
            k, _, v = line.rpartition(":")
            k = k.strip()
            for short, full in KEYS.items():
                if k == full:
                    try:
                        rows[cur][short] = float(v.strip().rstrip("s"))
                    except ValueError:
                        pass
    for r in rows.values():
        if r.get("ep_len"):
            if "err_xy" in r:
                r["xy_mps"] = r["err_xy"] * CMD_STEPS / r["ep_len"]
            if "err_yaw" in r:
                r["yaw_rps"] = r["err_yaw"] * CMD_STEPS / r["ep_len"]
    return rows, over


def window(rows, lo, hi, key):
    v = [r[key] for i, r in rows.items() if lo <= i <= hi and key in r]
    return sum(v) / len(v) if v else float("nan")


def summarize(name, rows, over, win):
    if not rows:
        return dict(name=name, last=-1)
    last = max(rows)
    lo = max(0, last - win)
    return dict(
        name=name, last=last, overflow=over,
        overflow_per_iter=over / max(1, last + 1),
        xy_mps=window(rows, lo, last, "xy_mps"),
        yaw_rps=window(rows, lo, last, "yaw_rps"),
        terrain=rows[last].get("terrain", float("nan")),
        terrain_min=min((r["terrain"] for r in rows.values() if "terrain" in r), default=float("nan")),
        contact=window(rows, lo, last, "contact"),
        ep_len=window(rows, lo, last, "ep_len"),
        it_time=window(rows, lo, last, "it_time"),
    )


def fmt(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", metavar="NAME=LOG")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-ref", action="store_true", help="REF_GB12K 용 파이썬 dict 출력")
    a = ap.parse_args()

    arms = []
    for spec in a.arms:
        n, _, p = spec.partition("=")
        rows, over = parse(p)
        arms.append((n, rows, over))

    if a.dump_ref:
        n, rows, _ = arms[0]
        print({i: (round(r.get("xy_mps", 0), 3), round(r.get("terrain", 0), 3), round(r.get("contact", 0), 3))
               for i, r in sorted(rows.items()) if i % a.every == 0 and "xy_mps" in r})
        return

    out = []
    horizon = max((max(r) for _, r, _ in arms if r), default=0)
    out.append(f"## 추이 (xy m/s / 지형 / 낙상비) — 전진 거부선 {FORWARD_REFUSAL_MPS}\n")
    hdr = "| iter | " + " | ".join(n for n, _, _ in arms) + (" | GB12k(ref) |" if REF_GB12K else " |")
    out.append(hdr)
    out.append("|" + "---|" * (len(arms) + 1 + (1 if REF_GB12K else 0)))
    for i in range(a.every, horizon + 1, a.every):
        cells = []
        for _, rows, _ in arms:
            r = rows.get(i)
            cells.append("-" if not r else f"{fmt(r.get('xy_mps'),2)} / {fmt(r.get('terrain'),2)} / {fmt(r.get('contact'),2)}")
        if REF_GB12K:
            g = REF_GB12K.get(i)
            cells.append("-" if not g else f"{g[0]:.2f} / {g[1]:.2f} / {g[2]:.2f}")
        out.append(f"| {i} | " + " | ".join(cells) + " |")

    sums = [summarize(n, r, o, a.window) for n, r, o in arms]
    out.append(f"\n## 요약 (마지막 {a.window} iter 평균)\n")
    out.append("| 팔 | 마지막 iter | xy m/s | yaw rad/s | 지형(끝) | 지형 최저 | 낙상비 | 에피소드 | s/iter | 오버플로/iter |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in sums:
        if s["last"] < 0:
            out.append(f"| {s['name']} | 로그 없음 | | | | | | | | |")
            continue
        out.append(f"| {s['name']} | {s['last']} | {fmt(s['xy_mps'])} | {fmt(s['yaw_rps'])} | {fmt(s['terrain'],2)} | "
                   f"{fmt(s['terrain_min'],2)} | {fmt(s['contact'])} | {fmt(s['ep_len'],0)} | {fmt(s['it_time'],2)} | {fmt(s['overflow_per_iter'],1)} |")

    ok = [s for s in sums if s["last"] >= 0 and s["xy_mps"] == s["xy_mps"]]
    if ok:
        best = min(ok, key=lambda s: (s["xy_mps"], -s["terrain"]))
        out.append("")
        out.append(f"**최저 xy 오차: {best['name']}** ({fmt(best['xy_mps'])} m/s, 지형 {fmt(best['terrain'],2)})")
        for s in ok:
            flags = []
            if s["overflow"] > 0:
                flags.append(f"PhysX 오버플로 {s['overflow']}줄 — 접촉 누락, envs 줄여라")
            if s["xy_mps"] >= FORWARD_REFUSAL_MPS * 0.9:
                flags.append("전진 거부선 근처 — 해금 실패")
            if s["terrain"] == s["terrain"] and s["terrain"] < 0.2:
                flags.append("지형 붕괴")
            out.append(f"- {s['name']}: " + ("; ".join(flags) if flags else "이상 없음"))
        print(f"WINNER={best['name']}", file=sys.stderr)

    text = "\n".join(out)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
