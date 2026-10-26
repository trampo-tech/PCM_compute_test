"""Generate current summary charts directly from saved evaluation metrics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'current_findings_figures';OUT.mkdir(exist_ok=True)
results=ROOT/'results'
folders=['batchnorm_40_fixed_eval_area3','weighted_ce_epoch39_area3','weighted_ce_60_smoothing01_area3','classmix_epoch35_area3']
s=[json.loads((results/f/'summary.json').read_text()) for f in folders]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,6.4),gridspec_kw={'width_ratios':[1,1.2]});fig.patch.set_facecolor('#fafaf8')
for a in (ax,bx):a.set_facecolor('#fafaf8')
values=[x['final']['miou']*100 for x in s];labels=['Unweighted\n40 epochs','Weighted CE\n40 epochs','Weighted CE\n60 epochs, smoothing 0.1','Weighted + class-mix\n40 epochs'];colors=['#96a2ae','#18796e','#a5b5c0','#c48635']
ax.barh(range(4),values,color=colors,height=.58);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,82);ax.set_xlabel('Area 3 mIoU (%)');ax.set_title('Weighted loss delivers the clearest gain',loc='left',fontweight='bold',pad=18);ax.xaxis.grid(True,alpha=.15);ax.set_axisbelow(True)
for i,v in enumerate(values):ax.text(v+.8,i,f'{v:.2f}%',va='center',fontweight='bold',color='#243843')
ax.text(.02,-.17,'Weighted vs unweighted: +17.80 percentage points',transform=ax.transAxes,color='#18796e',fontweight='bold',fontsize=11)
y=np.arange(5);h=.24
for j,(idx,label,color) in enumerate([(0,'Unweighted','#96a2ae'),(1,'Weighted CE','#18796e'),(3,'Weighted + class-mix','#c48635')]):
 vals=[c['iou']*100 for c in s[idx]['final']['per_class']];bx.barh(y+(j-1)*h,vals,height=h,color=color,label=label)
bx.set_yticks(y,['I-beam','Pipe','Pump','Rectangular beam','Tank']);bx.invert_yaxis();bx.set_xlim(0,105);bx.set_xlabel('Area 3 class IoU (%)');bx.set_title('Class-mix trades tank/beam performance for gains elsewhere',loc='left',fontweight='bold',pad=18,fontsize=11);bx.xaxis.grid(True,alpha=.15);bx.set_axisbelow(True);bx.legend(loc='lower center',bbox_to_anchor=(.5,-.27),ncol=3,frameon=False,fontsize=9)
fig.suptitle('PSNet5: observed improvements and remaining tradeoffs',x=.03,ha='left',fontsize=17,fontweight='bold',color='#243843');fig.text(.03,.02,'One-vote Area 3 evaluations, 100% raw coverage. Single training seed; Area 3 has informed exploration.\n60-epoch experiment also changes label smoothing. Diagnoses have not established a specific cause.',fontsize=9,color='#586772')
fig.subplots_adjust(left=.15,right=.98,top=.84,bottom=.23,wspace=.65)
for suffix in ['png','svg']:fig.savefig(OUT/f'area3_comparison.{suffix}',dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
# Preserve the plotted numerical data and provenance for reproducibility.
(OUT/'chart_data.json').write_text(json.dumps([{'source':str((results/f/'summary.json').relative_to(ROOT)),'miou':x['final']['miou'],'per_class':x['final']['per_class']} for f,x in zip(folders,s)],indent=2)+'\n')
print('Generated summary charts')
