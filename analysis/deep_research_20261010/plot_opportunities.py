from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/baseline_redesign_20261009_4h/offline_eval/_plotdeps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
h=json.loads((OUT/'heldout_analysis.json').read_text(encoding='utf-8'))
p=json.loads((OUT/'phase_comparison.json').read_text(encoding='utf-8'))
r=json.loads((OUT/'response_analysis.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.titleweight':'bold', 'figure.facecolor':'white'})
fig,ax=plt.subplots(2,2,figsize=(14.8,9.6),gridspec_kw={'hspace':.65,'wspace':.30})
fig.suptitle('Go2 offline research: response tails + ordinary yaw',fontsize=19,fontweight='bold',y=.98)
fig.text(.5,.94,'Recorded C0/F10 rollouts and B1 training logs | No new policy evaluation | No official-score forecast',ha='center',fontsize=11,color='#555')
ph=h['models_aggregate']['F10_3000']['weighted_phases']
keys=['initial','true_command_transition','post_push','ordinary']
labels=['Start','True command change','Post-push','Ordinary']
colors=['#e79a3b','#d76753','#9062a4','#5c90ae']
left=np.zeros(3)
for key,label,color in zip(keys,labels,colors):
    vals=np.array([ph[key]['weighted_sample_share'],ph[key]['axis_MSE_fraction']['vx'],ph[key]['axis_MSE_fraction']['yaw']])*100
    ax[0,0].barh([2,1,0],vals,left=left,color=color,label=label,height=.55)
    for y,v,l in zip([2,1,0],vals,left):
        if v>10: ax[0,0].text(l+v/2,y,f'{v:.1f}%',ha='center',va='center',color='white',fontweight='bold')
    left+=vals
ax[0,0].set_yticks([2,1,0],['Samples','VX squared error','Yaw squared error'])
ax[0,0].set_xlim(0,100)
ax[0,0].set_xlabel('Scenario-weighted share (%)')
ax[0,0].set_title('A. 8.75% of samples hold 70.56% of VX error',pad=15)
ax[0,0].legend(loc='upper center',bbox_to_anchor=(.5,-.26),ncol=2,frameon=False,fontsize=10)

cases=h['F10_changed_error_counterfactuals']['cases']
names=['vx_only_RMSE_minus_20pct','yaw_only_RMSE_minus_20pct','XY_both_RMSE_minus_20pct','all_three_axes_RMSE_minus_20pct']
vals=[cases[n]['delta_Q3'] for n in names]
ax[0,1].barh([3,2,1,0],vals,color=['#d76753','#5c90ae','#9062a4','#456c67'],height=.6)
for y,v in zip([3,2,1,0],vals): ax[0,1].text(v+.00025,y,f'+{v:.5f}',va='center')
ax[0,1].set_yticks([3,2,1,0],['VX only','Yaw only','Both XY axes','All three axes'])
ax[0,1].set_xlim(0,.0205)
ax[0,1].set_xlabel('Internal Q3 increase (unitless; not score points)')
ax[0,1].set_title('B. Assumed 20% RMSE reduction',pad=15)
ax[0,1].text(.0,-.42,'Changed-error arithmetic; feasibility not established.',transform=ax[0,1].transAxes,fontsize=10,color='#666')

x=np.arange(3)
for model,offset,color in [('C0_5500',-.19,'#aab9c2'),('F10_3000',.19,'#336a8c')]:
    vals=[t['windows']['0.5']['RMSE_mps'] for t in r['G1'][model]['transitions']]
    bars=ax[1,0].bar(x+offset,vals,width=.36,color=color,label=model)
    ax[1,0].bar_label(bars,fmt='%.3f',padding=3,fontsize=10)
ax[1,0].set_xticks(x,['0.3 to 0.6','0.6 to 1.0','1.0 to -0.5'])
ax[1,0].set_ylim(0,.83)
ax[1,0].set_ylabel('VX RMSE in first 0.5 s (m/s)')
ax[1,0].set_xlabel('G1 velocity command transition (m/s)')
ax[1,0].set_title('C. Large G1 transition tails are lag-side',pad=15)
ax[1,0].legend(frameon=False)
ax[1,0].text(.0,-.34,'Lag-side = before reaching new target; not an identified delay.',transform=ax[1,0].transAxes,fontsize=10,color='#666')

contrast=p['B1_vs_F10_terrain_group_descriptive_contrasts']
for metric,offset,color,label in [('Metrics/base_velocity/error_vel_xy',-.19,'#d76753','XY log error'),('Metrics/base_velocity/error_vel_yaw',.19,'#5c90ae','Yaw log error')]:
    vals=[contrast[t][metric] for t in ['early','middle','late']]
    bars=ax[1,1].bar(x+offset,vals,width=.36,color=color,label=label)
    ax[1,1].bar_label(bars,labels=[f'{v:+.1f}%' for v in vals],padding=3,fontsize=10)
ax[1,1].axhline(0,color='#999',linewidth=.8)
ax[1,1].set_xticks(x,['1.5-2.5','2.5-4.5','4.5-5.5'])
ax[1,1].set_ylim(-9,29)
ax[1,1].set_ylabel('B1 vs F10 change (%) | negative = lower')
ax[1,1].set_xlabel('Shared training terrain range')
ax[1,1].set_title('D. B1 early benefit does not persist',pad=15)
ax[1,1].legend(loc='upper left',frameon=False,fontsize=10)
ax[1,1].text(.0,-.34,'Descriptive 0.5-width bins; endogenous terrain, no causal claim.',transform=ax[1,1].transAxes,fontsize=10,color='#666')
fig.subplots_adjust(left=.105,right=.965,bottom=.15,top=.865)
for a in ax.flat:
    a.grid(axis='x' if a in [ax[0,0],ax[0,1]] else 'y',alpha=.13)
    a.set_axisbelow(True)
fig.savefig(OUT/'opportunity_map.png',dpi=180)
fig.savefig(OUT/'opportunity_map.svg')
print(OUT/'opportunity_map.png')
