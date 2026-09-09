"""tools/t_eval_paste.sh 재생성 - t_eval.py + set_reward.py 를 한 번에 설치한다."""
import io, tarfile, gzip, base64, os

files = ["tools/t_eval.py", "tools/set_reward.py", "tools/screen_queue.sh"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for p in files:
        data = open(p, "rb").read()
        ti = tarfile.TarInfo(os.path.basename(p))
        ti.size, ti.mode, ti.mtime = len(data), 0o755, 0
        tf.addfile(ti, io.BytesIO(data))
b = base64.b64encode(gzip.compress(buf.getvalue(), 9)).decode()

out = [
    "# NCRC 도구 설치 - 서버 /workspace/training/humanoid/ 에서 이 블록 전체를 붙여넣어라.",
    "# 설치: t_eval.py (7시나리오 평가기), set_reward.py (보상 안전 편집기),",
    "#       screen_queue.sh (resume 스크리닝 큐)",
    "cat > /tmp/ncrc_tools.b64 <<'B64EOF'",
]
out += [b[i:i + 200] for i in range(0, len(b), 200)]
out += [
    "B64EOF",
    "base64 -d /tmp/ncrc_tools.b64 | tar xzf -",
    "python -c \"import ast;[ast.parse(open(f,encoding='utf-8').read()) for f in ['t_eval.py','set_reward.py']];print('py OK')\"",
    "bash -n screen_queue.sh && echo 't_eval.py / set_reward.py / screen_queue.sh 설치 OK'",
]
with open("tools/t_eval_paste.sh", "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("tools/t_eval_paste.sh:", os.path.getsize("tools/t_eval_paste.sh"), "bytes,", len(out), "줄")
