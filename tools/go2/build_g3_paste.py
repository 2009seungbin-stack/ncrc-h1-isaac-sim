"""tools/go2/g3_paste.sh 생성 — Go2 3라운드 도구 일체를 한 블록으로 서버에 설치한다.

서버 파일 전송은 code-server 페이지에 file input 을 주입하고 JS paste 이벤트를 쏘는
방식이다 (메모리 ncrc-server-file-transfer-via-paste-event). 그래서 설치물은
**base64 한 덩어리 + 검증 명령**이어야 한다.

    python tools/go2/build_g3_paste.py
    → tools/go2/g3_paste.sh 를 code-server 터미널에 통째로 붙여넣는다.

붙여넣는 위치: /workspace/training/quadruped  (없으면 ROBOT_DIR 을 바꿔서 쓴다)
"""
import base64
import gzip
import hashlib
import io
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES = [
    ("tools/t_eval.py", "t_eval.py"),
    ("tools/go2/t_eval_go2.py", "t_eval_go2.py"),
    ("tools/go2/g0_probe.py", "g0_probe.py"),
    ("tools/go2/g3_pipeline.sh", "g3_pipeline.sh"),
    ("tools/go2/v1_crossval.sh", "v1_crossval.sh"),
    ("tools/set_reward.py", "set_reward.py"),
    ("tools/reward_diff.py", "reward_diff.py"),
]

PY_FILES = [dst for _s, dst in FILES if dst.endswith(".py")]
SH_FILES = [dst for _s, dst in FILES if dst.endswith(".sh")]


def main():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for src, dst in FILES:
            p = os.path.join(ROOT, src)
            data = open(p, "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(dst)
            ti.size, ti.mode, ti.mtime = len(data), 0o755, 0
            tf.addfile(ti, io.BytesIO(data))
    gz = gzip.compress(buf.getvalue(), 9)
    b = base64.b64encode(gz).decode()
    md5 = hashlib.md5(gz).hexdigest()

    pychk = ("import ast;[ast.parse(open(f).read()) for f in "
             + repr(PY_FILES) + "];print('py OK')")
    out = [
        "# G3 설치 — Go2 워크스페이스에서 이 블록 전체를 붙여넣어라.",
        "# 붙여넣기 전에 디렉터리를 확인할 것: ls /workspace/training/",
        "cd ${ROBOT_DIR:-/workspace/training/quadruped} || { echo '!! 디렉터리 확인 필요'; }",
        "cat > /tmp/g3.b64 <<'B64EOF'",
    ]
    out += [b[i:i + 200] for i in range(0, len(b), 200)]
    out += [
        "B64EOF",
        f"base64 -d /tmp/g3.b64 > /tmp/g3.tgz && md5sum /tmp/g3.tgz | grep -q {md5} "
        "&& echo 'MD5 OK' || echo '!! MD5 MISMATCH — 다시 붙여넣어라'",
        "tar xzf /tmp/g3.tgz",
        "for s in " + " ".join(SH_FILES) + "; do bash -n $s || echo \"!! $s 문법오류\"; done",
        f"/isaac-sim/python.sh -c \"{pychk}\"",
        "ls -l " + " ".join(dst for _s, dst in FILES),
        "",
        "# --- 다음 순서 (go2_plan.md §6 D1) ---",
        "# 1) G-0 정찰:  /workspace/IsaacLab/isaaclab.sh -p g0_probe.py --headless "
        "| tee /workspace/G0_RECON.txt",
        "#    → 태스크 이름, 보상 키, 계단 서브지형 키, 이벤트 항, 로더가 std 를 받는지 확인",
        "# 2) 팔 정의를 G-0 결과에 맞춰 확인 (g3_pipeline.sh 상단 ARM_GH / ARM_GN)",
        "# 3) 파이프라인:  tmux new -d -s g3 'TASK=<태스크> bash g3_pipeline.sh'",
        "#    상태: tail -30 /workspace/G3_STATUS.txt",
        "# 4) 끝나면:  touch /workspace/STOP_KEEPALIVE  → 보드에서 서버 종료",
    ]
    dst = os.path.join(ROOT, "tools/go2/g3_paste.sh")
    with open(dst, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(ROOT, "tools/go2/g3_paste.b64"), "w", newline="\n") as f:
        f.write(b)
    print(f"g3_paste.sh: {os.path.getsize(dst)} bytes, {len(out)} lines")
    print(f"payload: {len(gz)} bytes gz, b64 {len(b)} chars, md5 {md5}")
    print("files:", ", ".join(dst_ for _s, dst_ in FILES))


if __name__ == "__main__":
    main()
