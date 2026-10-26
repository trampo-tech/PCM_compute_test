"""Compare saved Area 3 predictions without training or modifying checkpoints.
Run in the project Docker image; outputs are deterministic and CPU-only.
"""
from pathlib import Path
import csv,json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'analysis/psnet5/results'; OUT=R/'classmix_spatial_analysis';OUT.mkdir(exist_ok=True)
CACHE=ROOT/'data/PSNet5/processed_pcm/psnet5sphere_benchmark_benchmark-area-v1_test_v0.040'
NAMES=['ibeam','pipe','pump','rectangularbeam','tank']
RUNS={'best':'weighted_ce_epoch39_area3','latest':'classmix_epoch35_area3'}
COLORS=['#d55e00','#0072b2','#cc79a7','#009e73','#e6ab02'];CMAP=ListedColormap(COLORS)
def savecsv(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p):
 c=np.bincount(y.astype(np.int64)*5+p,minlength=25).reshape(5,5);tp=c.diagonal();union=c.sum(0)+c.sum(1)-tp
 return {'oa':float(tp.sum()/max(c.sum(),1)),'miou':float(np.divide(tp,union,out=np.zeros(5),where=union>0)[union>0].mean()),'iou':np.divide(tp,union,out=np.zeros(5),where=union>0).tolist()}
xyz=np.load(CACHE/'cloud_0_sub_points.npy');y=np.load(CACHE/'cloud_0_sub_labels.npy');rawy=np.load(CACHE/'cloud_0_raw_labels.npy',mmap_mode='r');proj=np.load(CACHE/'cloud_0_projection.npy',mmap_mode='r');n=len(y)
present,first,counts=np.unique(proj,return_index=True,return_counts=True)
original_n=n; remap=np.full(n,-1,dtype=np.int32);remap[present]=np.arange(len(present));original_proj=proj;proj=remap[proj];xyz=xyz[present];y=y[present];n=len(y)
pred={};votes={};rawpred={}
for name,run in RUNS.items():
 z=np.load(R/run/'Area_3_predictions.npz');assert np.array_equal(z['label'],rawy);assert np.array_equal(z['projection'],original_proj);assert z['covered'].all()
 rawpred[name]=z['prediction'];pred[name]=rawpred[name][first];votes[name]=z['subsampled_vote_count'][present]
 assert np.array_equal(pred[name][proj],rawpred[name])
print('Loaded and verified prediction alignment',flush=True)
# Distance to closest differently labelled subsampled point: a semantic proximity proxy,
# not an instance boundary or physical surface distance.
boundary=np.full(n,np.inf)
for k in range(5):
 sel=y==k;boundary[sel]=cKDTree(xyz[~sel]).query(xyz[sel],workers=4)[0]
 print('Boundary class',k,flush=True)
knn_dist=cKDTree(xyz).query(xyz,k=17,workers=4)[0][:,-1]
rows=[]
for name in RUNS:
 for cls in range(5):
  for low,high in [(0,.08),(.08,.2),(.2,.5),(.5,np.inf)]:
   mask=(y==cls)&(boundary>=low)&(boundary<high)
   if mask.any():rows.append({'model':name,'class':NAMES[cls],'distance_min_m':low,'distance_max_m':high,'sub_points':int(mask.sum()),'raw_weight':int(counts[mask].sum()),'recall_sub':float((pred[name][mask]==cls).mean()),'recall_raw_weighted':float(np.average(pred[name][mask]==cls,weights=counts[mask]))})
savecsv('semantic_proximity.csv',rows)
rows=[]
for name in RUNS:
 for cls in range(5):
  sel=y==cls
  for key,values,bins in [('votes',votes[name],[(0,2),(2,5),(5,10),(10,20),(20,np.inf)]),('neighbor17_distance_m',knn_dist,[(0,.08),(.08,.12),(.12,.2),(.2,np.inf)])]:
   for lo,hi in bins:
    m=sel&(values>=lo)&(values<hi)
    if m.any():rows.append({'model':name,'class':NAMES[cls],'measure':key,'min':lo,'max':hi,'sub_points':int(m.sum()),'recall':float((pred[name][m]==cls).mean())})
savecsv('sampling_density.csv',rows)
# Best possible class assignment for each projection cell, evaluated on raw labels.
hist=np.bincount(proj.astype(np.int64)*5+rawy,minlength=n*5).reshape(n,5)
oracle=hist.argmax(1).astype('uint8');oracle_raw=metrics(rawy,oracle[proj]);projection_error=1-float(hist.max(1).sum()/len(rawy))
summary={'raw_points':len(rawy),'sub_points':n,'original_sub_points':original_n,'excluded_sub_points_without_raw_projection':original_n-n,'metrics':{},'projection_oracle':oracle_raw,'projection_oracle_error_fraction':projection_error,'subsample_label_projection':metrics(rawy,y[proj]),'per_point_vote_counts_identical':bool(np.array_equal(votes['best'],votes['latest'])),'confusions':[]}
for name in RUNS:summary['metrics'][name]={'raw':metrics(rawy,rawpred[name]),'subsampled':metrics(y,pred[name])}
# Spatial bins identify contiguous error regions, not object instances.
clusters=[];offsets=[(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if (i,j,k)!=(0,0,0)]
for true,false in [(4,2),(4,1),(0,3),(0,1),(2,1),(1,2)]:
 mask=(y==true)&(pred['latest']==false);indices=np.flatnonzero(mask);cells=np.floor(xyz[indices]/.5).astype(int);unique,inv=np.unique(cells,axis=0,return_inverse=True);lookup={tuple(v):i for i,v in enumerate(unique)};pending=set(lookup);groups=[]
 while pending:
  seed=pending.pop();stack=[seed];group=[lookup[seed]]
  while stack:
   cur=stack.pop()
   for d in offsets:
    neighbor=tuple(cur[a]+d[a] for a in range(3))
    if neighbor in pending:pending.remove(neighbor);stack.append(neighbor);group.append(lookup[neighbor])
  groups.append(group)
 ranked=[]
 for group in groups:
  ix=indices[np.isin(inv,group)];ranked.append((int(counts[ix].sum()),ix))
 ranked.sort(key=lambda q:q[0],reverse=True)
 raw_error=int(np.count_nonzero((rawy==true)&(rawpred['latest']==false)))
 info={'true':NAMES[true],'predicted':NAMES[false],'raw_error_points':raw_error,'sub_error_points':int(mask.sum()),'median_other_class_distance_m':float(np.median(boundary[mask])),'fraction_farther_20cm':float((boundary[mask]>.2).mean()),'median_votes':float(np.median(votes['latest'][mask])),'median_neighbor17_distance_m':float(np.median(knn_dist[mask])),'coarse_components':len(groups),'top_component_sub_error_fraction':len(ranked[0][1])/len(indices)}
 summary['confusions'].append(info)
 for rank,(weight,ix) in enumerate(ranked[:3],1):
  lo=xyz[ix].min(0);hi=xyz[ix].max(0);clusters.append({'true':NAMES[true],'predicted':NAMES[false],'rank':rank,'sub_error_points':len(ix),'raw_cell_weight':weight,'xmin':float(lo[0]),'ymin':float(lo[1]),'zmin':float(lo[2]),'xmax':float(hi[0]),'ymax':float(hi[1]),'zmax':float(hi[2])})
 print('Confusion',info,flush=True)
savecsv('error_clusters.csv',clusters)
# Deterministic spatial overview, same point sample and framing for all panels.
rng=np.random.default_rng(20261021);sample=np.sort(rng.choice(n,min(n,100000),replace=False))
fig,axes=plt.subplots(2,3,figsize=(18,10))
for col,(title,labels) in enumerate([('Ground truth',y),('Best: weighted CE epoch 39',pred['best']),('Latest: class-mix epoch 35',pred['latest'])]):
 for row,axes_pair in enumerate([(0,1),(0,2)]):
  ax=axes[row,col];a,b=axes_pair;ax.scatter(xyz[sample,a],xyz[sample,b],c=labels[sample],cmap=CMAP,vmin=-.5,vmax=4.5,s=.3,rasterized=True);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('XYZ'[a]+' (m)');ax.set_ylabel('XYZ'[b]+' (m)')
fig.legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color=c,label=t) for c,t in zip(COLORS,NAMES)],loc='lower center',ncol=5);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
# Largest error-component crop for each main confusion, plotted in two views.
for cl in [v for v in clusters if v['rank']==1]:
 low=np.array([cl[k+'min'] for k in 'xyz'])-.25;high=np.array([cl[k+'max'] for k in 'xyz'])+.25;ix=np.flatnonzero(((xyz>=low)&(xyz<=high)).all(1));ix=np.sort(rng.choice(ix,min(len(ix),50000),replace=False))
 fig,axes=plt.subplots(2,3,figsize=(16,9))
 for col,(title,labels) in enumerate([('Ground truth',y),('Best checkpoint',pred['best']),('Latest checkpoint',pred['latest'])]):
  for row,(a,b) in enumerate([(0,1),(0,2)]):
   ax=axes[row,col];ax.scatter(xyz[ix,a],xyz[ix,b],c=labels[ix],cmap=CMAP,vmin=-.5,vmax=4.5,s=2);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('XYZ'[a]+' (m)');ax.set_ylabel('XYZ'[b]+' (m)')
 fig.suptitle(f"Largest coarse error region: {cl['true']} → {cl['predicted']} (0.5 m cells, not an instance)");fig.legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color=c,label=t) for c,t in zip(COLORS,NAMES)],loc='lower center',ncol=5);fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/f"crop_{cl['true']}_to_{cl['predicted']}.png",dpi=150);plt.close(fig)
# Aggregate the actually consumed class-mix schedule epochs (40 x 1000 spheres).
train=ROOT/'data/PSNet5/processed_pcm/psnet5sphere_development_psnet5-spatial-v1_train_v0.040'
p=train/'schedule_r2_n15000_e100_s2000_seed20260919_classmix-v1-0.25.sampling.json'
sampling=json.loads(p.read_text());summary['sampling_diagnostic_keys']=list(sampling)
summary['sampling_diagnostic_note']='First 40 scheduled epochs, 2000 spheres per epoch. Training batch size 2 yields 1000 optimizer steps per epoch.'
summary['sampling_diagnostic_first40']={k:np.asarray(v[:40]).sum(0).tolist() for k,v in sampling.items() if isinstance(v,list) and len(v)==100}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
