#!/usr/bin/env python3
"""humanoid_rewards.py 의 REWARD_WEIGHTS 를 안전하게 고친다.

서버에서 손으로 고치다 오타가 나면 **경고만 뜨고 무시된 채 학습이 돈다**
(가이드북 14강: "이름을 잘못 적으면 Isaac Lab 기본값을 쓰고 경고만 낸다").
2시간짜리 run이 통째로 날아가는 실패 방식이라, 항목명 검증을 도구로 강제한다.

    python set_reward.py --show
    python set_reward.py track_ang_vel_z_exp=1.0
    python set_reward.py track_ang_vel_z_exp=1.0 action_rate_l2=-0.005 --dry-run
    python set_reward.py --restore                # .baseline.bak 로 되돌린다

기본 대상 파일은 humanoid_rewards.py, 백업은 humanoid_rewards.baseline.bak.
"""

import argparse
import os
import re
import shutil
import sys

# 표가 한글·기호를 쓴다. 윈도우 콘솔 기본 코덱(cp949)에서 죽지 않게 UTF-8로 돌린다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# 파일 주석의 추천 범위 + NVIDIA 원본값 + 우리 검증값.
# (항목명, 추천 하한, 추천 상한, NVIDIA 원본, NCRC baseline)
KNOWN = {
    "track_lin_vel_xy_exp": (0.5, 2.5, 1.0, 0.5),
    "track_ang_vel_z_exp": (0.5, 2.0, 1.0, 0.5),
    "feet_air_time": (0.1, 0.5, 0.25, 0.2),
    "termination_penalty": (-400.0, -50.0, -200.0, -5.0),
    "flat_orientation_l2": (-3.0, -0.2, -1.0, -1.5),
    "ang_vel_xy_l2": (-0.3, -0.02, -0.05, -0.05),
    "joint_deviation_hip": (-0.5, -0.1, -0.2, -0.2),
    "joint_deviation_torso": (-0.3, -0.02, -0.1, -0.1),
    "action_rate_l2": (-0.008, -0.0005, -0.005, -0.0005),
}

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def find_block(text):
    """REWARD_WEIGHTS = { ... } 블록의 (시작, 끝) 오프셋을 돌려준다."""
    m = re.search(r"^\s*REWARD_WEIGHTS\s*(?::[^=\n]+)?=\s*\{", text, re.M)
    if not m:
        raise SystemExit("REWARD_WEIGHTS = { ... } 블록을 찾지 못했습니다.")
    depth, i = 0, m.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return m.start(), i + 1
        i += 1
    raise SystemExit("REWARD_WEIGHTS 블록의 닫는 중괄호를 찾지 못했습니다.")


def parse_block(block):
    """{항목명: (값 문자열, 줄 전체)} — 주석 줄은 건너뛴다."""
    out = {}
    for line in block.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.match(rf"""^\s*(['"])(?P<k>[A-Za-z_][A-Za-z0-9_]*)\1\s*:\s*(?P<v>{NUM})\s*,?""", line)
        if m:
            out[m.group("k")] = (m.group("v"), line)
    return out


def range_note(key, val):
    if key not in KNOWN:
        return "  (추천 범위 정보 없음)"
    lo, hi, nv, base = KNOWN[key]
    lo2, hi2 = min(lo, hi), max(lo, hi)
    flag = "" if lo2 <= val <= hi2 else "  ** 추천 범위 밖 **"
    marks = []
    if abs(val - nv) < 1e-12:
        marks.append("NVIDIA 원본값")
    if abs(val - base) < 1e-12:
        marks.append("NCRC baseline")
    tag = ("  [" + ", ".join(marks) + "]") if marks else ""
    return f"  추천 {lo}~{hi}{tag}{flag}"


def cmd_show(path):
    text = open(path, encoding="utf-8").read()
    s, e = find_block(text)
    items = parse_block(text[s:e])
    print(f"{path} - REWARD_WEIGHTS {len(items)}개\n")
    for k, (v, _) in items.items():
        print(f"  {k:<24} = {v:>10}{range_note(k, float(v))}")
    unknown = [k for k in items if k not in KNOWN]
    if unknown:
        print("\n  (목록에 없는 항목: " + ", ".join(unknown) + ")")
    return items


def main():
    ap = argparse.ArgumentParser(description="REWARD_WEIGHTS 안전 편집기")
    ap.add_argument("assignments", nargs="*", metavar="KEY=VALUE")
    ap.add_argument("--file", default="humanoid_rewards.py")
    ap.add_argument("--backup", default="humanoid_rewards.baseline.bak")
    ap.add_argument("--show", action="store_true", help="현재 값만 출력")
    ap.add_argument("--restore", action="store_true", help="백업으로 되돌린다")
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 diff만 본다")
    ap.add_argument("--allow-new", action="store_true",
                    help="REWARD_WEIGHTS에 없는 항목을 새로 추가한다 (미검증 — 신중히)")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"파일이 없습니다: {args.file}  (--file 로 경로를 지정하세요)")

    if args.restore:
        if not os.path.exists(args.backup):
            raise SystemExit(f"백업이 없습니다: {args.backup}")
        shutil.copyfile(args.backup, args.file)
        print(f"복원 완료: {args.backup} -> {args.file}\n")
        cmd_show(args.file)
        return

    if args.show or not args.assignments:
        cmd_show(args.file)
        if not args.assignments:
            return

    text = open(args.file, encoding="utf-8").read()
    s, e = find_block(text)
    block = text[s:e]
    items = parse_block(block)

    # 1) 항목명 검증 — 오타는 여기서 죽인다
    changes = []
    for a in args.assignments:
        if "=" not in a:
            raise SystemExit(f"형식이 KEY=VALUE 가 아닙니다: {a}")
        k, v = a.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            fv = float(v)
        except ValueError:
            raise SystemExit(f"값이 숫자가 아닙니다: {a}")
        if k not in items:
            near = [c for c in items if c.lower().startswith(k.lower()[:6])]
            if not args.allow_new:
                raise SystemExit(
                    f"'{k}' 는 REWARD_WEIGHTS 에 없습니다. 이대로 학습하면 "
                    f"**경고만 뜨고 무시**됩니다.\n"
                    + (f"        혹시 이것입니까: {', '.join(near)}\n" if near else "")
                    + f"        정말 새 항목을 추가하려면 --allow-new 를 붙이세요.")
        changes.append((k, v, fv))

    # 2) 백업
    if not args.dry_run and not os.path.exists(args.backup):
        shutil.copyfile(args.file, args.backup)
        print(f"백업 생성: {args.backup}")

    # 3) 치환
    new_block = block
    added = []
    for k, v, fv in changes:
        if k in items:
            old_line = items[k][1]
            new_line = re.sub(rf"(:\s*){NUM}", lambda m: m.group(1) + v, old_line, count=1)
            new_block = new_block.replace(old_line, new_line, 1)
        else:
            added.append((k, v))

    if added:
        # 닫는 중괄호 앞에 끼워 넣는다. 들여쓰기는 마지막 항목에서 가져온다.
        indent = "    "
        for _, ln in items.values():
            indent = ln[: len(ln) - len(ln.lstrip())]
        insert = "".join(f'{indent}"{k}": {v},\n' for k, v in added)
        idx = new_block.rfind("}")
        new_block = new_block[:idx] + insert + new_block[idx:]

    new_text = text[:s] + new_block + text[e:]

    # 4) 결과 표
    print("\n보상 항목                        전 ->            후")
    print("-" * 58)
    after = parse_block(new_block)
    for k, v, fv in changes:
        before = items[k][0] if k in items else "(없음)"
        print(f"{k:<24} {before:>10} -> {after.get(k, ('?',))[0]:>10}{range_note(k, fv)}")

    n_diff = sum(1 for k in after
                 if k in items and after[k][0].strip() != items[k][0].strip())
    print(f"\n변경된 항목 수: {n_diff + len(added)}건")
    if n_diff + len(added) != len(changes):
        print("!! 요청한 변경 수와 실제 변경 수가 다릅니다. 파일을 눈으로 확인하세요.")

    if args.dry_run:
        print("\n--dry-run 이라 파일을 쓰지 않았습니다.")
        return

    open(args.file, "w", encoding="utf-8").write(new_text)
    print(f"\n{args.file} 저장 완료.")

    # 5) 재파싱 검증
    verify = parse_block(open(args.file, encoding="utf-8").read()[s:])
    bad = [k for k, v, fv in changes
           if k in verify and abs(float(verify[k][0]) - fv) > 1e-12]
    if bad:
        raise SystemExit(f"!! 저장 후 값이 맞지 않습니다: {bad}")
    print("재파싱 검증 통과. 이제 학습을 시작해도 됩니다.")
    print("\n학습 시작 후에는 이 파일을 고쳐도 돌고 있는 run에 반영되지 않습니다 (규정 제12조).")


if __name__ == "__main__":
    main()
