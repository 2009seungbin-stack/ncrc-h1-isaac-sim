"""tools/r3/r3_paste.sh 생성 — t_eval.py + set_reward.py + R3 보상 2종 + 파이프라인을 한 블록으로 설치."""
import io, tarfile, gzip, base64, os, hashlib
files = ["tools/t_eval.py", "tools/set_reward.py", "tools/r3/humanoid_rewards_R3A.py",
         "tools/r3/humanoid_rewards_R3C.py", "tools/r3/r3_pipeline.sh"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tf:
    for p in files:
        data = open(p, "rb").read().replace(b"\r\n", b"\n")
        ti = tarfile.TarInfo(os.path.basename(p)); ti.size, ti.mode, ti.mtime = len(data), 0o755, 0
        tf.addfile(ti, io.BytesIO(data))
gz = gzip.compress(buf.getvalue(), 9)
b = base64.b64encode(gz).decode()
md5 = hashlib.md5(gz).hexdigest()
out = ["# R3 설치 — 서버 /workspace/training/humanoid/ 에서 이 블록 전체를 붙여넣어라.",
       "cd /workspace/training/humanoid || exit 1",
       "cat > /tmp/r3.b64 <<'B64EOF'"]
out += [b[i:i + 200] for i in range(0, len(b), 200)]
out += ["B64EOF",
        f"base64 -d /tmp/r3.b64 > /tmp/r3.tgz && md5sum /tmp/r3.tgz | grep -q {md5} && echo 'MD5 OK' || echo '!! MD5 MISMATCH — 다시 붙여넣어라'",
        "tar xzf /tmp/r3.tgz && bash -n r3_pipeline.sh && /isaac-sim/python.sh -c \"import ast;[ast.parse(open(f).read()) for f in ['t_eval.py','set_reward.py','humanoid_rewards_R3A.py','humanoid_rewards_R3C.py']];print('py OK')\"",
        "ls -l t_eval.py set_reward.py humanoid_rewards_R3A.py humanoid_rewards_R3C.py r3_pipeline.sh"]
with open("tools/r3/r3_paste.sh", "w", newline="\n", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
open("tools/r3/r3_paste.b64", "w", newline="\n").write(b)
print("r3_paste.sh:", os.path.getsize("tools/r3/r3_paste.sh"), "bytes,", len(out), "lines; b64", len(b), "chars; md5", md5)
