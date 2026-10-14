"""Plot recorded training signals only; no simulator or policy rollout."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '_plotdeps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / 'training_metrics.json').read_text(encoding='utf-8-sig'))
rows = data['rows']
x = np.array([r['iteration'] for r in rows])

def smooth(name, multiplier=1):
    values = np.array([r[name] * multiplier for r in rows])
    return np.r_[np.full(49, np.nan), np.convolve(values, np.ones(50)/50, mode='valid')]

plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'figure.facecolor': '#f7f9fc', 'axes.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#24334b', 'text.color': '#24334b',
                     'font.size': 11})
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout='constrained')
fig.suptitle('B1 내부 평가: 저장된 학습 기록 · 50회 이동평균', fontsize=17, fontweight='bold')
axes[0].plot(x, smooth('linear_weight_removed_proxy', 100), color='#2357be', label='XY 추종 보상 ÷ 가중치')
axes[0].plot(x, smooth('yaw_weight_removed_proxy', 100), color='#9d47a4', label='회전 추종 보상 ÷ 가중치')
axes[0].set_ylabel('가중치 제거 보상 × 100')
axes[0].set_ylim(0, 100)
axes[0].legend(loc='lower right', frameon=False)
axes[1].plot(x, smooth('Metrics/base_velocity/error_vel_xy'), color='#2357be', label='XY error logger')
axes[1].plot(x, smooth('Metrics/base_velocity/error_vel_yaw'), color='#9d47a4', label='Yaw error logger')
axes[1].set_ylabel('학습 error logger\n검증 RMSE와 다름')
axes[1].legend(loc='upper right', frameon=False)
axes[2].plot(x, smooth('Episode_Termination/base_contact', 100), color='#cc4d48', label='몸통 접촉 종료 지표')
axes[2].set_ylabel('학습 접촉 종료 지표 (%)', color='#cc4d48')
terrain = axes[2].twinx()
terrain.plot(x, smooth('Curriculum/terrain_levels'), color='#287d65', label='학습 지형 레벨')
terrain.set_ylabel('지형 레벨', color='#287d65')
terrain.spines['right'].set_visible(True)
axes[2].legend(loc='upper left', frameon=False)
terrain.legend(loc='upper right', frameon=False)
axes[2].set_xlabel('학습 iteration')
for ax in axes:
    ax.grid(alpha=.16)
    for iteration in (600, 1000, 1499):
        ax.axvline(iteration, color='#64748b', alpha=.5, linestyle=':', linewidth=1)
    ax.set_xlim(0, 1499)
fig.supxlabel('학습 중 명령·지형 분포가 바뀝니다. 고정 G1–G7 검증 성능·생존율·공식 점수가 아닙니다.', fontsize=10)
fig.savefig(root / 'B1_training_assessment.png', dpi=150)
fig.savefig(root / 'B1_training_assessment.svg')
print(root / 'B1_training_assessment.png')
