"""tools/tr/tr2_paste.sh 생성 — t_eval.py + tr_run.sh 붙여넣기 번들 (과도구간 역산 실험)."""
import base64, gzip, hashlib, io, os, tarfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = [("tools/t_eval.py", "t_eval.py"), ("tools/tr/tr2_run.sh", "tr2_run.sh")]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for src, dst in FILES:
        data = open(os.path.join(ROOT, src), "rb").read().replace(b"\r\n", b"\n")
        ti = tarfile.TarInfo(dst); ti.size, ti.mode, ti.mtime = len(data), 0o755, 0
        tf.addfile(ti, io.BytesIO(data))
gz = gzip.compress(buf.getvalue(), 9, mtime=0)
b = base64.b64encode(gz).decode(); md5 = hashlib.md5(gz).hexdigest()
out = ["cd /workspace/training/humanoid || echo '!! 디렉터리 확인 필요'", "cat > /tmp/tr2.b64 <<'B64EOF'"]
out += [b[i:i+200] for i in range(0, len(b), 200)]
out += ["B64EOF",
        f"base64 -d /tmp/tr2.b64 > /tmp/tr2.tgz && md5sum /tmp/tr2.tgz | grep -q {md5} && echo 'MD5 OK' || echo '!! MD5 MISMATCH'",
        "tar xzf /tmp/tr2.tgz && bash -n tr2_run.sh && echo 'sh OK' > /workspace/TR2_INSTALL.txt && md5sum t_eval.py tr2_run.sh >> /workspace/TR2_INSTALL.txt",
        "tmux new -d -s tr2 'bash tr2_run.sh'; sleep 2; tmux ls >> /workspace/TR2_INSTALL.txt; cat /workspace/TR2_INSTALL.txt"]
dst = os.path.join(ROOT, "tools/tr/tr2_paste.sh")
open(dst, "w", newline="\n", encoding="utf-8").write("\n".join(out) + "\n")
print(f"tr_paste.sh: {os.path.getsize(dst)} bytes, tgz md5 {md5}")
