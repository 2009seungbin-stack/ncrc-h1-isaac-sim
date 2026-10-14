"""Render already computed diagnostic evidence, not simulated improvements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
r = json.loads((OUT/"analysis.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor("#f8fafc")
fig.suptitle("추종 개선과 큰 오차를 나누어 봐야 하는 이유", fontsize=17, y=1.02)
blue, red, green = "#2563eb", "#d95f44", "#17806d"
axes = ("vx", "vy", "yaw")
g6 = r["F10_vs_C0_moment_changes"]["B"]["G6"]
x = np.arange(3)
axs[0].bar(x-.18, [g6[a]["mae_change_percent"] for a in axes], .35, label="평균 절대오차", color=blue)
axs[0].bar(x+.18, [g6[a]["rmse_change_percent"] for a in axes], .35, label="RMSE", color=red)
axs[0].set_xticks(x, ["전진", "좌우", "회전"])
axs[0].set_ylabel("C0 대비 변화 (%) · 아래쪽이 개선")
axs[0].set_title("밀침 G6 · 검증 B")
axs[0].legend(frameon=False, loc="upper right", fontsize=9)
axs[0].axhline(0, color="#64748b", lw=.8)

b, f = [r["evaluations"][name]["weighted_axis_mse"] for name in ("C0_B", "F10_B")]
keys = ("vx", "vy", "yaw", "vz", "wx", "wy")
values = [100*(f[k]/b[k]-1) for k in keys]
axs[1].bar(np.arange(6), values, color=[red if v>0 else green for v in values])
axs[1].set_xticks(np.arange(6), ["전진", "좌우", "회전", "상하", "롤", "피치"])
axs[1].set_ylabel("시나리오 배점 가중 MSE 변화 (%)")
axs[1].set_title("7개 본 시나리오 · 검증 B")
axs[1].axhline(0, color="#64748b", lw=.8)
for i,v in enumerate(values):
    axs[1].text(i, v+(1 if v>=0 else -1), f"{v:+.1f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
axs[1].set_ylim(-42,20)

m = r["training_terms"]["F10"][0]["medians"]
terms = ("dof_acc_l2", "ang_vel_xy_l2", "action_rate_l2", "dof_torques_l2", "lin_vel_z_l2")
vals = [-m["Episode_Reward/"+t] for t in terms]
axs[2].barh(np.arange(5), vals, color=[red, "#94a3b8", blue, "#94a3b8", "#94a3b8"])
axs[2].set_yticks(np.arange(5), ["관절 가속도", "롤·피치 각속도", "동작 변화", "관절 토크", "상하 속도"])
axs[2].invert_yaxis()
axs[2].set_xlabel("학습 로그의 항별 벌점 크기 중앙값")
axs[2].set_title("F10 · iteration 2000–2799")
for i,v in enumerate(vals):
    axs[2].text(v+.006, i, f"{v:.3f}", va="center", fontsize=9)
axs[2].set_xlim(0,.37)
for ax in axs:
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f8fafc")
fig.text(.02,-.05,"기존 평가·학습 로그의 재분석입니다. 새 학습 효과·공식 점수 예측·독립 학습 반복 검증을 뜻하지 않습니다.", fontsize=10, color="#475569")
fig.tight_layout(w_pad=3)
fig.savefig(OUT/"reward_tradeoffs.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(OUT/"reward_tradeoffs.png")
