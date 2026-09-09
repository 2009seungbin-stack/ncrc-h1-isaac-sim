"""NCRC 제출 게이트 — env.yaml 보상 diff 추출 및 리포트 항목 대조.

규정 제6조: 설계 의도 분석 20점 = 리포트(의도) <-> env.yaml(실제 값) 일치도.
제출 전 이 스크립트로 diff 항목 수와 리포트 설명 항목 수가 같은지 확인한다.

사용법:
    python reward_diff.py env_initial.yaml env.yaml
    python reward_diff.py env_initial.yaml env.yaml --report report.md
"""

import argparse
import re
import sys


def extract_rewards(path):
    """env.yaml의 rewards 블록에서 {항목명: {weight, params}} 를 뽑는다.

    isaaclab env.yaml은 !!python/object/apply 태그를 써서 yaml.safe_load가 실패한다.
    보상 블록은 들여쓰기가 규칙적이므로 텍스트로 파싱한다.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    start = text.find("\nrewards:\n")
    if start < 0:
        raise SystemExit(f"{path}: rewards 블록을 찾지 못했습니다")
    start += 1

    # 다음 최상위 키(들여쓰기 0)까지가 rewards 블록
    end = len(text)
    for m in re.finditer(r"\n(?=[a-zA-Z_])", text[start + len("rewards:") :]):
        end = start + len("rewards:") + m.start()
        break
    block = text[start:end]

    rewards = {}
    current = None
    for line in block.splitlines()[1:]:
        if not line.strip():
            continue
        # 2칸 들여쓰기 = 보상 항목 이름
        m = re.match(r"^  (\w+):\s*(.*)$", line)
        if m:
            current = m.group(1)
            rest = m.group(2).strip()
            # "term: null" 형태 = 비활성
            rewards[current] = {"weight": None, "std": None, "threshold": None}
            if rest == "null":
                rewards[current]["weight"] = "DISABLED"
            continue
        if current is None:
            continue
        for key in ("weight", "std", "threshold"):
            m = re.match(rf"^\s+{key}:\s*(\S+)\s*$", line)
            if m:
                rewards[current][key] = m.group(1)
    return rewards


def fmt(v):
    return "-" if v is None else str(v)


def diff_rewards(before, after):
    """변경된 항목만 [(name, field, before, after)] 로 반환."""
    changes = []
    for name in sorted(set(before) | set(after)):
        b = before.get(name, {})
        a = after.get(name, {})
        if name not in before:
            changes.append((name, "ADDED", "-", fmt(a.get("weight"))))
            continue
        if name not in after:
            changes.append((name, "REMOVED", fmt(b.get("weight")), "-"))
            continue
        for field in ("weight", "std", "threshold"):
            if b.get(field) != a.get(field):
                changes.append((name, field, fmt(b.get(field)), fmt(a.get(field))))
    return changes


# 리포트는 한국어 권장(규정 제4조)이고 1라운드도 "낙상 감점"처럼 한글 표현으로
# 설계 의도 18/20을 받았다. 키 이름을 강제하면 리포트 품질(10/10)을 해칠 수 있으므로
# 자연스러운 한글 표현도 언급으로 인정한다.
KOREAN_ALIASES = {
    "track_lin_vel_xy_exp": ["전진 속도", "속도 추종", "직진 속도", "선속도"],
    "track_ang_vel_z_exp": ["회전 속도", "회전 추종", "각속도", "요 속도"],
    "lin_vel_z_l2": ["수직 속도", "상하 속도"],
    "ang_vel_xy_l2": ["몸통 회전", "롤 피치 각속도", "자세 각속도"],
    "dof_torques_l2": ["토크"],
    "dof_acc_l2": ["관절 가속도"],
    "action_rate_l2": ["동작 변화", "동작 급변", "행동 변화율", "액션 변화"],
    "feet_air_time": ["발 들기", "체공", "발 공중"],
    "undesired_contacts": ["원치 않는 접촉", "불필요한 접촉"],
    "flat_orientation_l2": ["몸통 수평", "자세 수평", "평평"],
    "dof_pos_limits": ["관절 한계", "관절 범위"],
    "termination_penalty": ["낙상 감점", "종료 감점", "넘어짐 감점", "종료 보상"],
    "feet_slide": ["발 미끄러", "미끄러짐"],
    "joint_deviation_hip": ["힙 이탈", "고관절 이탈", "엉덩이 관절"],
    "joint_deviation_arms": ["팔 이탈", "팔 관절"],
    "joint_deviation_torso": ["몸통 이탈", "허리 관절", "토르소"],
}


def find_in_report(report_text, name):
    """리포트에 해당 보상이 키 이름 또는 한글 별칭으로 언급됐는지 확인."""
    low = report_text.lower()
    if name.lower() in low:
        return True
    return any(alias in report_text for alias in KOREAN_ALIASES.get(name, []))


def main():
    # Windows 콘솔 기본 코드페이지에서 한글이 깨지는 것을 막는다
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("before", help="env_initial.yaml")
    ap.add_argument("after", help="제출할 env.yaml")
    ap.add_argument("--report", help="기술 개선 리포트 파일 (대조 게이트 실행)")
    args = ap.parse_args()

    before = extract_rewards(args.before)
    after = extract_rewards(args.after)
    changes = diff_rewards(before, after)

    print(f"보상 항목 수: before {len(before)} / after {len(after)}")
    print(f"변경된 항목: {len(changes)}건\n")

    if not changes:
        print("변경 없음.")
        return 0

    width = max(len(c[0]) for c in changes)
    print(f"{'보상 항목'.ljust(width)}  {'필드':<10} {'전':>12} -> {'후':>12}")
    print("-" * (width + 42))
    for name, field, b, a in changes:
        print(f"{name.ljust(width)}  {field:<10} {b:>12} -> {a:>12}")

    if not args.report:
        print("\n리포트 대조를 하려면 --report 를 넘기세요.")
        return 0

    with open(args.report, encoding="utf-8") as f:
        report_text = f.read()

    changed_names = sorted({c[0] for c in changes})
    missing = [n for n in changed_names if not find_in_report(report_text, n)]

    print(f"\n=== 제출 게이트 ===")
    print(f"변경된 보상 항목 {len(changed_names)}개, 리포트에 언급된 것 "
          f"{len(changed_names) - len(missing)}개")

    if missing:
        print("\n[FAIL] 리포트에 설명이 없는 변경 항목:")
        for n in missing:
            print(f"  - {n}")
        print("\n규정 제6조 설계 의도 분석은 리포트와 env.yaml의 일치도를 본다.")
        print("설명되지 않은 변경은 감점 사유다. 제출하지 말 것.")
        return 1

    print("[PASS] 모든 변경 항목이 리포트에 언급되어 있습니다.")
    print("\n주의: 이 게이트는 '언급 여부'만 본다. 다음은 사람이 확인해야 한다.")
    print("  1. 리포트에 적힌 전/후 수치가 위 표와 일치하는가")
    print("  2. 각 변경에 표적 시나리오와 예측->실측이 적혔는가")
    print("  3. 리포트에만 있고 실제로는 바꾸지 않은 항목이 없는가")
    return 0


if __name__ == "__main__":
    sys.exit(main())
