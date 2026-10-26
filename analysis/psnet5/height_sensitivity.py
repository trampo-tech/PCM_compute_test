"""Paired height-only perturbations on the fixed development validation schedule."""
from pathlib import Path
import sys,json,time,random,csv
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analysis.psnet5.evaluate_multivote import load_checkpoint,projected_results,metrics_from_confusion,write_confusion
from openpoints.dataset import build_dataloader_from_cfg,get_features_by_keys
from openpoints.dataset.psnet5.voting import add_sphere_votes
from openpoints.models import build_model_from_cfg
from openpoints.models.layers.norm import batch_stats_for_eval
from openpoints.utils import EasyConfig,set_random_seed
OUT=ROOT/'analysis/psnet5/results/height_sensitivity_dev';OUT.mkdir(exist_ok=True)
RUNS=[('weighted_best','PCM-ce-weighted.yaml','psnet5-train-PCM-ce-weighted-ngpus1-20261014-013255-RYxnShfJR67zVkme9fdFat'),('classmix_latest','PCM-ce-classmix.yaml','psnet5-train-PCM-ce-classmix-ngpus1-20261021-021447-h5cKTHgDmf88V578n3amNu')]
OFFSETS=[0.,-.5,-.25,.25,.5]
def state():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
def restore(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state(s[3])
all_results=[]
for name,config,run in RUNS:
 set_random_seed(20261004);torch.backends.cudnn.benchmark=False
 cfg=EasyConfig();cfg.load(str(ROOT/'cfgs/psnet5'/config),recursive=True)
 assert cfg.dataset.common.protocol=='development' and cfg.dataset.val.split=='val'
 assert cfg.feature_keys=='x,heights' and cfg.dataset.common.height_mode=='absolute'
 cfg.dataloader.num_workers=4;cfg.dataset.val.num_epochs=1
 loader=build_dataloader_from_cfg(cfg.val_batch_size,cfg.dataset,cfg.dataloader,datatransforms_cfg=cfg.datatransforms,split='val',distributed=False);dataset=loader.dataset
 assert [c['area'] for c in dataset.clouds]==['Area_1','Area_2','Area_4']
 model=build_model_from_cfg(cfg.model);ckpt=ROOT/'log/psnet5'/run/'checkpoint'/f'{run}_ckpt_best.pth';info=load_checkpoint(model,ckpt,'cuda:0')
 sums=[[torch.zeros((len(p),5),device='cuda') for p in dataset.sub_clouds_points] for _ in OFFSETS]
 counts=[[torch.zeros(len(p),device='cuda') for p in dataset.sub_clouds_points] for _ in OFFSETS]
 start=time.monotonic();baseline_repeat_max=0.;offset_effect_max={str(v):0. for v in OFFSETS};probe_checked=False
 print(json.dumps({'event':'start','model':name,'checkpoint':info,'spheres':len(dataset),'offsets':OFFSETS}),flush=True)
 with torch.no_grad(),batch_stats_for_eval(model):
  for step,data in enumerate(loader):
   for key in tuple(data):data[key]=data[key].cuda(non_blocking=True)
   height=data['heights'].clone();data['epoch']=info['epoch'];data['iter']=-1
   before=state();after=None;baseline=None
   for j,offset in enumerate(OFFSETS):
    restore(before);data['heights']=height+offset
    # Preserve the original RGB tensor; assembled features are a separate tensor.
    if j==0:rgb=data['x'].clone()
    data['x']=get_features_by_keys({'x':rgb,'heights':data['heights']},cfg.feature_keys)
    logits=model(data).float()
    if j==0:baseline=logits.clone();after=state()
    else:offset_effect_max[str(offset)]=max(offset_effect_max[str(offset)],float((logits-baseline).abs().max()))
    add_sphere_votes(sums[j],counts[j],logits,data['cloud_index'],data['input_inds'],data['mask'])
   if not probe_checked:
    restore(before);data['heights']=height;data['x']=get_features_by_keys({'x':rgb,'heights':height},cfg.feature_keys);repeat=model(data).float();baseline_repeat_max=float((repeat-baseline).abs().max());probe_checked=True
    print(json.dumps({'event':'repeat_control','model':name,'max_logit_difference':baseline_repeat_max}),flush=True)
   restore(after)
   if (step+1)%50==0:
    elapsed=time.monotonic()-start
    (OUT/'progress.json').write_text(json.dumps({'model':name,'spheres_complete':step+1,'total':len(dataset),'elapsed_seconds':elapsed,'eta_seconds':elapsed/(step+1)*(len(dataset)-step-1)},indent=2)+'\n')
    print(f'{name}: {step+1}/{len(dataset)} spheres, {elapsed:.1f}s',flush=True)
 for j in range(1,len(OFFSETS)):
  assert all(torch.equal(a,b) for a,b in zip(counts[0],counts[j]))
 rows=[];baseline_areas=None;bands=[]
 # Identical bands for every condition, using raw GT coordinates, not predictions.
 for cloud in dataset.clouds:
  labels=np.asarray(cloud['raw_labels']).reshape(-1);z=np.asarray(cloud['raw_points'])[:,2];area_bands=[]
  for cls in [0,4]:
   edges=np.quantile(z[labels==cls],[0,.25,.5,.75,1]) if (labels==cls).any() else []
   area_bands.append((cls,edges))
  bands.append(area_bands)
 for j,offset in enumerate(OFFSETS):
  conf,areas=projected_results(dataset,sums[j],counts[j]);metric=metrics_from_confusion(conf)
  if j==0:baseline_areas=[a['raw_prediction'].copy() for a in areas]
  result={'model':name,'checkpoint':str(ckpt.relative_to(ROOT)),'checkpoint_info':info,'height_offset_m':offset,'protocol':'development','split':'val','spheres':len(dataset),'seed':20261004,'bn_mode':'current_batch','metrics':metric,'areas':[],'height_bands':[],'repeat_control_max_logit_difference':baseline_repeat_max,'offset_max_logit_difference':offset_effect_max[str(offset)]}
  tag=f'{name}_offset_{offset:+.2f}';write_confusion(OUT/f'{tag}_confusion.csv',conf)
  for i,(cloud,area) in enumerate(zip(dataset.clouds,areas)):
   covered=area['raw_covered'];labels=area['raw_labels'];p=area['raw_prediction'];z=np.asarray(cloud['raw_points'])[:,2];b=baseline_areas[i]
   result['areas'].append({'name':area['name'],'raw_coverage':area['raw_coverage'],'subsampled_coverage':area['subsampled_coverage'],'metrics':metrics_from_confusion(area['confusion']),'prediction_flip_fraction':float((p[covered]!=b[covered]).mean()),'fixed_points':int(np.count_nonzero(covered&(b!=labels)&(p==labels))),'broken_points':int(np.count_nonzero(covered&(b==labels)&(p!=labels)))})
   for cls,edges in bands[i]:
    for k in range(max(0,len(edges)-1)):
     m=covered&(labels==cls)&(z>=edges[k])&((z<=edges[k+1]) if k==3 else (z<edges[k+1]))
     result['height_bands'].append({'area':area['name'],'class':int(cls),'quartile':k+1,'zmin':float(edges[k]),'zmax':float(edges[k+1]),'covered_points':int(m.sum()),'recall':float((p[m]==cls).mean()) if m.any() else None})
   averaged=sums[j][i]/counts[j][i].clamp_min(1)[:,None]
   np.savez_compressed(OUT/f'{tag}_{area["name"]}_sub_predictions.npz',prediction=averaged.argmax(1).cpu().numpy().astype('uint8'),covered=(counts[j][i]>0).cpu().numpy(),vote_count=counts[j][i].cpu().numpy().astype('uint16'))
  (OUT/f'{tag}.json').write_text(json.dumps(result,indent=2)+'\n');all_results.append(result)
  print(json.dumps({'event':'result','model':name,'offset':offset,'miou':metric['miou'],'tank_recall':metric['per_class'][4]['recall']}),flush=True)
 (OUT/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
 del model,sums,counts,areas,baseline_areas,loader,dataset;torch.cuda.empty_cache()
print('COMPLETE',flush=True)
