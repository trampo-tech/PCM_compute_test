# PSNet5 error analysis

Checkpoint: epoch 85 from `psnet5-train-PCM-ngpus1-20260921-015453-TaGtTm9SrUhiW7w5mT6bo3`.

## 1. Twenty-schedule voting

`evaluate_multivote.py` evaluated 2,000 deterministic spheres per schedule and accumulated only valid-point logits.

| Metric | Vote 1 | Vote 20 | Vote 20 on vote-1 support |
|---|---:|---:|---:|
| mIoU | 48.89% | 46.61% | 46.63% |
| mean accuracy | 57.74% | 55.70% | 55.70% |
| overall accuracy | 84.78% | 83.82% | 83.92% |

Coverage increased while accuracy did not:

| Area | Vote-1 raw coverage | Vote-20 raw coverage | Vote-20 mIoU |
|---|---:|---:|---:|
| Area 1 | 95.76% | 97.83% | 41.92% |
| Area 2 | 90.40% | 93.05% | 35.18% |
| Area 4 | 94.43% | 96.98% | 43.37% |

The fixed-support result is decisive: even when evaluated on exactly the points covered by vote 1, twenty-vote averaging reduces mIoU by 2.26 points. The decline is therefore not just newly covered hard points.

Final class IoUs were:

| I-beam | Pipe | Pump | Rectangular beam | Tank |
|---:|---:|---:|---:|---:|
| 8.88% | 90.67% | 0.00% | 66.42% | 67.09% |

The checkpoint never predicts the pump class. Of 842,880 covered pump points, 832,958 (98.82%) are classified as pipe. I-beam is also weak: it is predicted about 3.1 times as often as its ground-truth support, but reaches only 10.77% precision.

Boundary-distance exports do not show a single split-boundary failure. Area 2 is weak both near and away from the boundary; Areas 1 and 4 retain strong overall accuracy in multiple distance bands. The dominant error is semantic confusion, especially pump-to-pipe, rather than boundary leakage.

Artifacts are in `analysis/psnet5/results/multivote_epoch85_20votes/`. The small review files are `summary.json`, `vote_metrics.csv`, `per_class_metrics.csv`, the confusion CSVs, and `boundary_metrics.csv`. The three prediction NPZ files are optional local drill-down artifacts.
