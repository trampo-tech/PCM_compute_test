"""Replay cached training neighborhoods exactly; compare spatial/context exposure."""
from pathlib import Path
import json,csv,time
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'data/PSNet5/processed_pcm/psnet5sphere_development_psnet5-spatial-v1_train_v0.040';OUT=ROOT/'analysis/psnet5/results/training_exposure_analysis';OUT.mkdir(exist_ok=True)
NAMES=['ibeam','pipe','pump','rectangularbeam','tank'];COLORS=['#d55e00','#0072b2','#cc79a7','#009e73','#e6ab02'];CMAP=ListedColormap(COLORS)
manifest=json.loads((CACHE/'manifest.json').read_text());clouds=[]
for i,record in enumerate(manifest['clouds']):
 p=np.load(CACHE/f'cloud_{i}_sub_points.npy');y=np.load(CACHE/f'cloud_{i}_sub_labels.npy');clouds.append({'xyz':p,'y':y,'tree':KDTree(p,leaf_size=50),'area':record['area']})
RUNS=[('spatial','',39),('classmix','_classmix-v1-0.25',35)];exposures={};sphere_rows=[];totals={};snapshots={};start=time.monotonic()
for name,suffix,best_epoch in RUNS:
 schedule=np.load(CACHE/f'schedule_r2_n15000_e100_s2000_seed20260919{suffix}.npz');ci=schedule['cloud_indices'][:80000];pi=schedule['point_indices'][:80000];noise=schedule['noise'][:80000]
 exposure=[np.zeros(len(c['y']),dtype=np.int64) for c in clouds];class_sum=np.zeros(5,dtype=np.int64);context_sum=np.zeros((5,5));centers=np.zeros((3,5),dtype=np.int64);cap_count=0
 # Group queries for speed while preserving each cached jitter and nearest-15000 cap.
 for begin in range(0,80000,200):
  end=min(begin+200,80000)
  for area,c in enumerate(clouds):
   ix=np.flatnonzero(ci[begin:end]==area)+begin
   if not len(ix):continue
   picks=c['xyz'][pi[ix]]+noise[ix];queries,distances=c['tree'].query_radius(picks,r=2.,return_distance=True,sort_results=True);joined=[]
   for step,q,d in zip(ix,queries,distances):
    clipped=len(q)>15000;cap_count+=int(clipped);q=q[:15000];d=d[:15000];joined.append(q);freq=np.bincount(c['y'][q],minlength=5);class_sum+=freq;label=int(c['y'][pi[step]]);centers[area,label]+=1;context_sum[label]+=freq/max(len(q),1)
    sphere_rows.append({'sampler':name,'index':int(step),'epoch':int(step//2000+1),'area':c['area'],'center_class':NAMES[label],'valid_points':len(q),'clipped':int(clipped),'furthest_retained_m':float(d[-1]),'center_x':float(picks[np.flatnonzero(ix==step)[0],0]),'center_y':float(picks[np.flatnonzero(ix==step)[0],1]),'center_z':float(picks[np.flatnonzero(ix==step)[0],2]),**{f'fraction_{NAMES[k]}':float(freq[k]/len(q)) for k in range(5)}})
   exposure[area]+=np.bincount(np.concatenate(joined),minlength=len(c['y']))
  if end==best_epoch*2000:snapshots[name]=[a.copy() for a in exposure]
  if end%10000==0:print(name,end,'spheres',round(time.monotonic()-start,1),'seconds',flush=True)
 exposures[name]=exposure;totals[name]={'class_point_exposure':class_sum.tolist(),'center_counts_by_area':centers.tolist(),'mean_context_fractions_by_center_class':(context_sum/centers.sum(0).clip(1)[:,None]).tolist(),'clipped_sphere_fraction':cap_count/80000,'best_epoch':best_epoch}
 if name=='classmix':
  diag=json.loads((CACHE/f'schedule_r2_n15000_e100_s2000_seed20260919{suffix}.sampling.json').read_text());assert np.array_equal(class_sum,np.asarray(diag['valid_point_counts_per_epoch'][:40]).sum(0))
 print('Totals',name,json.dumps(totals[name]),flush=True)
# Per-point geometry descriptors use same-class kNN; not object-instance shape labels.
rows=[];region_rows=[];geometry_rows=[]
for area,c in enumerate(clouds):
 p=c['xyz'];y=c['y'];linear=np.zeros(len(y));planar=np.zeros(len(y))
 for label in [0,3,4]:
  ix=np.flatnonzero(y==label)
  if not len(ix):continue
  tree=cKDTree(p[ix]);k=min(16,len(ix))
  for start_idx in range(0,len(ix),20000):
   chunk=ix[start_idx:start_idx+20000];nn=tree.query(p[chunk],k=k,workers=4)[1];local=p[ix[nn]];local=local-local.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',local,local)/k;eig=np.linalg.eigvalsh(cov);den=eig[:,2].clip(1e-12);linear[chunk]=(eig[:,2]-eig[:,1])/den;planar[chunk]=(eig[:,1]-eig[:,0])/den
  # Fixed occupied 0.5m cells provide spatial units; no claim of object identity.
  cells,inv=np.unique(np.floor(p[ix]/.5).astype(np.int64),axis=0,return_inverse=True)
  counts=np.bincount(inv);a=np.bincount(inv,weights=exposures['spatial'][area][ix]);b=np.bincount(inv,weights=exposures['classmix'][area][ix])
  for i in range(len(cells)):
   region_rows.append({'area':c['area'],'class':NAMES[label],'cell_x':int(cells[i,0]),'cell_y':int(cells[i,1]),'cell_z':int(cells[i,2]),'sub_points':int(counts[i]),'spatial_exposure':int(a[i]),'classmix_exposure':int(b[i]),'ratio':float((b[i]+1)/(a[i]+1))})
  for shape,mask in [('line_like',linear[ix]>.7),('surface_like',planar[ix]>.5),('other',~((linear[ix]>.7)|(planar[ix]>.5)))]:
   for name in exposures:
    e=exposures[name][area][ix[mask]]
    geometry_rows.append({'area':c['area'],'class':NAMES[label],'descriptor':shape,'sampler':name,'sub_points':len(e),'total_exposure':int(e.sum()),'mean_visits':float(e.mean()) if len(e) else 0})
 for label in range(5):
  ix=np.flatnonzero(y==label)
  if not len(ix):continue
  for name in exposures:
   e=exposures[name][area][ix];snap=snapshots[name][area][ix]
   rows.append({'area':c['area'],'class':NAMES[label],'sampler':name,'sub_points':len(ix),'total_exposure':int(e.sum()),'mean_visits':float(e.mean()),'p10_visits':float(np.quantile(e,.1)),'median_visits':float(np.median(e)),'p90_visits':float(np.quantile(e,.9)),'never_seen_fraction':float((e==0).mean()),'effective_point_fraction':float(e.sum()**2/(max(float((e.astype(float)**2).sum()),1)*len(e))),'best_epoch':totals[name]['best_epoch'],'best_checkpoint_total_exposure':int(snap.sum()),'best_checkpoint_never_seen_fraction':float((snap==0).mean())})
 # Exposure maps for tank and beam classes, keep geometry fixed between samplers.
 fig,axes=plt.subplots(3,3,figsize=(16,13))
 for row,label in enumerate([0,3,4]):
  ix=np.flatnonzero(y==label)
  if len(ix)>40000:ix=np.random.default_rng(42).choice(ix,40000,replace=False)
  for col,name in enumerate(['spatial','classmix','ratio']):
   ax=axes[row,col]
   if len(ix):
    v=np.log2((exposures['classmix'][area][ix]+1)/(exposures['spatial'][area][ix]+1)) if name=='ratio' else np.log10(exposures[name][area][ix]+1)
    sc=ax.scatter(p[ix,0],p[ix,1],c=v,s=1,cmap='coolwarm' if name=='ratio' else 'viridis',vmin=-2 if name=='ratio' else 0,vmax=2 if name=='ratio' else 4);fig.colorbar(sc,ax=ax,shrink=.6,label='log2 visit ratio' if name=='ratio' else 'log10(visits+1)')
   ax.set_title(f'{NAMES[label]} — {name}');ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
 fig.suptitle(f'{c["area"]}: 40 epochs of exact valid-point exposure');fig.tight_layout();fig.savefig(OUT/f'{c["area"]}_exposure.png',dpi=140);plt.close(fig)
 np.savez_compressed(OUT/f'{c["area"]}_exposure.npz',spatial=exposures['spatial'][area],classmix=exposures['classmix'][area],linearity=linear,planarity=planar)
for filename,data in [('point_exposure.csv',rows),('spatial_cells.csv',region_rows),('geometry_exposure.csv',geometry_rows),('sphere_context.csv',sphere_rows)]:
 with (OUT/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
# Training examples: median-purity and low-purity tank / beam-centered spheres.
examples=[]
for label in [0,3,4]:
 for name,suffix,_ in RUNS:
  subset=[r for r in sphere_rows if r['sampler']==name and r['center_class']==NAMES[label]]
  if not subset:continue
  subset.sort(key=lambda r:r[f'fraction_{NAMES[label]}'])
  schedule=np.load(CACHE/f'schedule_r2_n15000_e100_s2000_seed20260919{suffix}.npz')
  for quantile in [.1,.5,.9]:
   row=subset[int((len(subset)-1)*quantile)];step=row['index'];area=int(schedule['cloud_indices'][step]);c=clouds[area];pick=c['xyz'][schedule['point_indices'][step]]+schedule['noise'][step];q=c['tree'].query_radius(pick[None],r=2.,return_distance=True,sort_results=True)[0][0][:15000];local=c['xyz'][q]-pick
   fig,axes=plt.subplots(1,2,figsize=(10,4))
   for ax,(a,b) in zip(axes,[(0,1),(0,2)]):ax.scatter(local[:,a],local[:,b],c=c['y'][q],cmap=CMAP,vmin=-.5,vmax=4.5,s=2);ax.set_aspect('equal');ax.set_xlabel('XYZ'[a]+' relative m');ax.set_ylabel('XYZ'[b]+' relative m')
   title=f'{name}: {NAMES[label]} center, purity {row[f"fraction_{NAMES[label]}"]:.1%}, {c["area"]}, {len(q)} points';fig.suptitle(title);fig.legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color=co,label=na) for co,na in zip(COLORS,NAMES)],loc='lower center',ncol=5);fig.tight_layout(rect=(0,.08,1,.92));filename=f'{name}_{NAMES[label]}_purity_q{quantile:.1f}.png';fig.savefig(OUT/filename,dpi=140);plt.close(fig);examples.append({**row,'quantile':quantile,'plot':filename})
(OUT/'summary.json').write_text(json.dumps({'epochs':40,'spheres_per_sampler':80000,'radius_m':2,'max_valid_points':15000,'samplers':totals,'examples':examples,'note':'Exact cached sphere centers and jitter; sorted radius queries truncated to nearest 15000; padding excluded. Shape descriptors and spatial cells are not instance labels.'},indent=2)+'\n')
print('COMPLETE',time.monotonic()-start,flush=True)
