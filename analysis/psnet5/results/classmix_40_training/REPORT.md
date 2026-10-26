# Latest class-mix training and Area 3 evaluation

Training completed 40 epochs. Best checkpoint: epoch 35, selected by development validation mIoU.

| Metric | Development validation | Area 3 |
|---|---:|---:|
| mIoU | 75.82% | 67.62% |
| Overall accuracy | 93.35% | 85.44% |
| Mean class accuracy | 89.51% | 76.45% |

Area 3: one vote, 2,000 spheres, seed 20261004, current-sphere BatchNorm. Raw and subsampled coverage: 100%; no uncovered points. Evaluator exited successfully.

| Class | Area 3 IoU |
|---|---:|
| ibeam | 40.27% |
| pipe | 75.40% |
| pump | 52.72% |
| rectangularbeam | 91.80% |
| tank | 77.93% |

Area 3 mIoU difference versus the preceding 60-epoch smoothing-0.1 run: -0.18 percentage points (previous 67.81%).
Difference versus the weighted 40-epoch epoch-39 checkpoint: -1.60 percentage points (baseline 69.23%).

Development training uses portions of Areas 1, 2, and 4. Area 3 has already informed prior experiments; these are exploratory comparisons.

Full evaluation: [summary.json](../classmix_epoch35_area3/summary.json). Training history: [epochs.csv](epochs.csv).
