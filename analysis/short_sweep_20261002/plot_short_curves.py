import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
d=json.loads((root/'short_diagnostics_summary.json').read_text(encoding='utf-8-sig'))
x=[49.5+100*i for i in range(8)]
colors={'control':'#183f69','action':'#bb7135','acc':'#864a9c','linear':'#43866b'}
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,index,title in zip(axes.flat,[1,2,4,0],['Training XY velocity error','Training yaw velocity error','Training base-contact termination rate','Mean episode length (steps)']):
 for name,rows in d['learning_curves'].items():
  ax.plot(x,[r[index] for r in rows],label=name,color=colors[name],marker='o',markersize=3,lw=2)
 ax.set_title(title)
 ax.set_xlabel('Training iteration (100-iteration window mean)')
 ax.grid(alpha=.2)
 if index==4: ax.set_ylim(0,.45)
axes[0,0].legend(ncol=2,frameon=False)
fig.suptitle('Go2 short sweep: four scratch runs, 800 iterations each',fontsize=15)
fig.supxlabel('One training seed (42). Training metrics are not evaluation survival, RMSE, or official scores.',fontsize=9)
fig.savefig(root/'short_learning_curves.png',dpi=160)
