from pathlib import Path
import numpy as np,json,csv
r=Path('/home/blau/projects/pcm-psnet5-project/PointCloudMamba/analysis/psnet5/results');out=r/'classmix_error_analysis';out.mkdir(exist_ok=True)
names=['ibeam','pipe','pump','rectangularbeam','tank'];runs=['weighted_ce_epoch39_area3','weighted_ce_60_smoothing01_area3','classmix_epoch35_area3'];data={};ss={}
for run in runs:
 z=np.load(r/run/'Area_3_predictions.npz');data[run]={k:z[k] for k in z.files};ss[run]=json.loads((r/run/'summary.json').read_text())
b=data[runs[0]];c=data[runs[-1]];assert np.array_equal(b['label'],c['label']);assert all(x['covered'].all() for x in data.values())
y=c['label'];bp=b['prediction'];cp=c['prediction'];n=len(y)
lines=['# Area 3 checkpoint error analysis','','All comparisons use the same raw labels and 100% coverage. The best available Area 3 checkpoint is weighted CE epoch 39 (69.23% mIoU); class-mix epoch 35 scores 67.62%, a decrease of 1.60 percentage points.','','## Per-class comparison','','| Class | Best IoU | Latest IoU | Change pp | Best recall | Latest recall | Best precision | Latest precision |','|---|---:|---:|---:|---:|---:|---:|---:|']
for a,d in zip(ss[runs[0]]['final']['per_class'],ss[runs[-1]]['final']['per_class']):
 lines.append(f"| {a['class']} | {a['iou']*100:.2f} | {d['iou']*100:.2f} | {(d['iou']-a['iou'])*100:+.2f} | {a['recall']*100:.2f} | {d['recall']*100:.2f} | {a['precision']*100:.2f} | {d['precision']*100:.2f} |")
conf=np.bincount(y.astype('int64')*5+cp,minlength=25).reshape(5,5);base=np.bincount(y.astype('int64')*5+bp,minlength=25).reshape(5,5)
errors=sorted([(int(conf[i,j]),i,j) for i in range(5) for j in range(5) if i!=j],reverse=True)
lines+=['','## Largest latest-run mistakes','','| True class | Predicted class | Points | % of true class | Change in errors vs best |','|---|---|---:|---:|---:|']
for count,i,j in errors[:10]:lines.append(f'| {names[i]} | {names[j]} | {count:,} | {100*count/conf[i].sum():.2f}% | {int(conf[i,j]-base[i,j]):+,} |')
fixed=(bp!=y)&(cp==y);broken=(bp==y)&(cp!=y);both=(bp!=y)&(cp!=y)
lines+=['','## Paired point comparison','',f'Latest fixes {fixed.sum():,} previously wrong points, but breaks {broken.sum():,} previously correct points. {both.sum():,} points are wrong in both models. Net correct-point change: {int(fixed.sum())-int(broken.sum()):+,}.','','| True class | Fixed by latest | Broken by latest | Net correct change |','|---|---:|---:|---:|']
for i,name in enumerate(names):
 f=int(np.count_nonzero(fixed&(y==i)));d=int(np.count_nonzero(broken&(y==i)));lines.append(f'| {name} | {f:,} | {d:,} | {f-d:+,} |')
with (out/'confusion_comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['true_class','predicted_class','best_count','latest_count','delta'])
 for i in range(5):
  for j in range(5):w.writerow([names[i],names[j],base[i,j],conf[i,j],conf[i,j]-base[i,j]])
lines+=['','## Interpretation and next checks','','The paired counts identify actual changed point decisions; IoU also reflects false positives from other classes. Class sampling changes which training centers are visited, so class exposure does not isolate geometric coverage or loss effects. These single-seed comparisons cannot establish causality.','','Before another training run, inspect spatial clusters of i-beam→pipe/rectangularbeam, pump↔pipe, and tank→pump/pipe errors. Separate object interiors from semantic boundaries, and compare raw versus subsampled metrics to test projection effects. Use development validation to choose changes; repeated Area 3 tuning reduces its value as an independent test.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
