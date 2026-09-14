"""tools/go2/screen_paste.sh 생성 — 전진 해금 스크리닝 도구 2개만 담은 작은 붙여넣기 블록.

g3_paste.sh(49KB)는 paste 이벤트 한 번에 안 들어가 둘로 갈라야 했다. 스크리닝은 학습 로그만 보므로
fast_screen.sh + screen_table.py 두 파일이면 된다. 채점 도구는 학습이 도는 동안 g3_paste.sh 로 따로 올린다.

    python tools/go2/build_screen_paste.py
    → tools/go2/screen_paste.sh 를 /workspace/training/quadruped 에서 붙여넣는다.
"""
import base64
import gzip
import hashlib
import io
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = [
    ("tools/go2/fast_screen.sh", "fast_screen.sh"),
    ("tools/go2/screen_table.py", "screen_table.py"),
]


def main():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for src, dst in FILES:
            data = open(os.path.join(ROOT, src), "rb").read().replace(b"\r\n", b"\n")
            ti = tarfile.TarInfo(dst)
            ti.size, ti.mode, ti.mtime = len(data), 0o755, 0
            tf.addfile(ti, io.BytesIO(data))
    gz = gzip.compress(buf.getvalue(), 9, mtime=0)
    b = base64.b64encode(gz).decode()
    md5 = hashlib.md5(gz).hexdigest()
    out = [
        "# 전진 해금 스크리닝 설치 — /workspace/training/quadruped 에서 통째로 붙여넣어라.",
        "cd ${ROBOT_DIR:-/workspace/training/quadruped} || echo '!! 디렉터리 확인 필요'",
        "cat > /tmp/scr.b64 <<'B64EOF'",
    ]
    out += [b[i:i + 200] for i in range(0, len(b), 200)]
    out += [
        "B64EOF",
        f"base64 -d /tmp/scr.b64 > /tmp/scr.tgz && md5sum /tmp/scr.tgz | grep -q {md5} "
        "&& echo 'MD5 OK' || echo '!! MD5 MISMATCH — 다시 붙여넣어라'",
        "tar xzf /tmp/scr.tgz && bash -n fast_screen.sh && echo 'sh OK'",
        "/isaac-sim/python.sh -c \"import ast;ast.parse(open('screen_table.py').read());print('py OK')\"",
        "DRY_RUN=1 bash fast_screen.sh && tail -4 /workspace/SCREEN_STATUS.txt",
        "",
        "# --- 실행 (검증 통과 후) ---",
        "# tmux new -d -s screen 'AUTO_LONG=1 LONG_ITERS=5000 bash fast_screen.sh'",
        "# 상태: tail -20 /workspace/SCREEN_STATUS.txt    표: cat /workspace/SCREEN_RESULT.md",
    ]
    dst = os.path.join(ROOT, "tools/go2/screen_paste.sh")
    with open(dst, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"screen_paste.sh: {os.path.getsize(dst)} bytes, {len(out)} lines, tgz md5 {md5}")


if __name__ == "__main__":
    main()
