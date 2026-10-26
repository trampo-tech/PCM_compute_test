"""Summarize a completed paired height-offset experiment."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'analysis/psnet5/results/height_sensitivity_dev'
results=json.loads((OUT/'summary.json').read_text());assert len(results)==10,'Both checkpoints must finish all five conditions'
rows=[];lines=['# Height-feature sensitivity on development validation','','Two checkpoints, each evaluated on the same fixed 2,000 development-validation spheres in Areas 1, 2 and 4. Height-feature offsets only: −0.50, −0.25, 0, +0.25 and +0.50 m. XYZ geometry, RGB, point masks, sphere order and vote accumulation are unchanged. Random states are restored before each condition within a sphere. Current-sphere BatchNorm is used without updating running statistics. Area 3 is not accessed.','','## Results','','| Checkpoint | Offset m | mIoU % | Change pp | Tank recall % | Prediction flips % |','|---|---:|---:|---:|---:|---:|']
for name in ['weighted_best','classmix_latest']:
 subset=[r for r in results if r['model']==name];baseline=next(r for r in subset if r['height_offset_m']==0)
 for r in sorted(subset,key=lambda x:x['height_offset_m']):
  metric=r['metrics'];total=sum(sum(a['metrics']['per_class'][i]['support'] for i in range(5)) for a in r['areas']);flips=sum(a['prediction_flip_fraction']*sum(c['support'] for c in a['metrics']['per_class']) for a in r['areas'])/total
  row={'checkpoint':name,'offset_m':r['height_offset_m'],'miou_percent':100*metric['miou'],'miou_change_pp':100*(metric['miou']-baseline['metrics']['miou']),'oa_percent':100*metric['oa'],'tank_recall_percent':100*metric['per_class'][4]['recall'],'prediction_flips_percent':100*flips};rows.append(row)
  lines.append(f"| {name} | {row['offset_m']:+.2f} | {row['miou_percent']:.5f} | {row['miou_change_pp']:+.5f} | {row['tank_recall_percent']:.5f} | {row['prediction_flips_percent']:.5f} |")
class_rows=[];band_rows=[]
for r in results:
 baseline=next(b for b in results if b['model']==r['model'] and b['height_offset_m']==0)
 for a,b in zip(r['metrics']['per_class'],baseline['metrics']['per_class']):
  class_rows.append({'checkpoint':r['model'],'offset_m':r['height_offset_m'],'class':a['class'],'iou_percent':100*a['iou'],'iou_change_pp':100*(a['iou']-b['iou']),'recall_percent':100*a['recall'],'recall_change_pp':100*(a['recall']-b['recall'])})
 for a,b in zip(r['height_bands'],baseline['height_bands']):
  band_rows.append({'checkpoint':r['model'],'offset_m':r['height_offset_m'],**a,'recall_change_pp':100*(a['recall']-b['recall']) if a['recall'] is not None else None})
for filename,data in [('per_class_comparison.csv',class_rows),('height_band_comparison.csv',band_rows)]:
 with (OUT/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
with (OUT/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines+=['','## Measured effect','']
for name in ['weighted_best','classmix_latest']:
 subset=[r for r in rows if r['checkpoint']==name and r['offset_m']!=0]
 lines.append(f"- {name}: largest absolute mIoU change {max(abs(r['miou_change_pp']) for r in subset):.6f} percentage points; largest changed-prediction fraction {max(r['prediction_flips_percent'] for r in subset):.6f}%.")
lines+=['','## Coverage and controls','']
for name in ['weighted_best','classmix_latest']:
 subset=[r for r in results if r['model']==name];baseline=next(r for r in subset if r['height_offset_m']==0)
 lines.append(f"- {name}: unchanged-input repeat maximum logit difference {baseline['repeat_control_max_logit_difference']:.9g}.")
 for a in baseline['areas']:lines.append(f"  - {a['name']}: raw coverage {a['raw_coverage']*100:.3f}%; subsampled coverage {a['subsampled_coverage']*100:.3f}%.")
 assert all(a['subsampled_coverage']>=.9 for a in baseline['areas'])
 assert all(a['raw_coverage']==b['raw_coverage'] for r in subset for a,b in zip(r['areas'],baseline['areas']))
lines+=['','## Interpretation','','A constant offset tests sensitivity to the absolute origin of the extra height feature. It does not remove the height feature, alter within-sphere height variation, or move the XYZ geometry. The encoder starts with a linear convolution followed by BatchNorm; using current-sphere statistics tends to remove a constant activation shift. Small numerical changes can remain due to floating-point operations and singleton fallbacks. Consequently, stability here cannot prove that height is unused or explain away the observed height-dependent tank errors.','','To test reliance on height information itself, separately evaluate height removal (set the feature to zero) or controlled scaling while preserving XYZ. Those changes are outside this offset experiment and can move features away from the training distribution. Absolute height equals local Z plus a sphere-specific constant, and local Z is also an input in this configuration. Under current-sphere BatchNorm, switching only from absolute to local height is therefore expected to have little effect. Height removal or scaling is the more informative next diagnostic; XYZ still contains relative vertical geometry. Any retraining experiment should be selected on development validation.','','The primary conclusion should follow the measured magnitude: large prediction changes indicate offset sensitivity; negligible changes argue against an absolute height-origin explanation under the current evaluation normalization. These are deterministic paired measurements, not an estimate of between-training-seed uncertainty.','','## Validation tank coverage limitation','','Covered validation tank points occur only in Area 2. Its tank raw-height quartiles span approximately 85.30 to 93.14 m, whereas the Area 3 tank range inspected earlier was approximately -1.37 to 3.77 m at subsampled resolution. Small offset stability does not establish generalization across those coordinate ranges or different tank shapes.','','## Artifacts','','[Full metrics and per-area height bands](summary.json), [comparison table](comparison.csv). Each condition also has its own JSON, confusion matrix, and compact per-area subsampled prediction export. Metrics exclude uncovered raw points and use the same support in all conditions.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:22]))
