# Area 3 checkpoint error analysis

All comparisons use the same raw labels and 100% coverage. The best available Area 3 checkpoint is weighted CE epoch 39 (69.23% mIoU); class-mix epoch 35 scores 67.62%, a decrease of 1.60 percentage points.

## Per-class comparison

| Class | Best IoU | Latest IoU | Change pp | Best recall | Latest recall | Best precision | Latest precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| ibeam | 37.76 | 40.27 | +2.51 | 40.94 | 42.71 | 82.91 | 87.55 |
| pipe | 69.63 | 75.40 | +5.77 | 92.14 | 91.73 | 74.03 | 80.90 |
| pump | 46.02 | 52.72 | +6.70 | 55.65 | 70.43 | 72.67 | 67.71 |
| rectangularbeam | 97.85 | 91.80 | -6.06 | 99.45 | 99.15 | 98.39 | 92.53 |
| tank | 94.87 | 77.93 | -16.94 | 95.78 | 78.24 | 99.01 | 99.49 |

## Largest latest-run mistakes

| True class | Predicted class | Points | % of true class | Change in errors vs best |
|---|---|---:|---:|---:|
| ibeam | pipe | 989,231 | 32.72% | -615,122 |
| pump | pipe | 725,961 | 29.57% | -362,986 |
| ibeam | rectangularbeam | 692,800 | 22.92% | +570,865 |
| pipe | pump | 553,719 | 6.58% | +142,694 |
| tank | pump | 220,991 | 14.51% | +178,010 |
| tank | pipe | 106,195 | 6.97% | +84,930 |
| pipe | ibeam | 103,704 | 1.23% | -106,935 |
| rectangularbeam | ibeam | 76,603 | 0.85% | +32,162 |
| ibeam | pump | 49,731 | 1.65% | -9,297 |
| pipe | rectangularbeam | 31,588 | 0.38% | +6,054 |

## Paired point comparison

Latest fixes 935,697 previously wrong points, but breaks 847,506 previously correct points. 2,713,718 points are wrong in both models. Net correct-point change: +88,191.

| True class | Fixed by latest | Broken by latest | Net correct change |
|---|---:|---:|---:|
| ibeam | 204,469 | 150,915 | +53,554 |
| pipe | 313,255 | 347,740 | -34,485 |
| pump | 402,303 | 39,317 | +362,986 |
| rectangularbeam | 15,221 | 42,004 | -26,783 |
| tank | 449 | 267,530 | -267,081 |

## Interpretation and next checks

The paired counts identify actual changed point decisions; IoU also reflects false positives from other classes. Class sampling changes which training centers are visited, so class exposure does not isolate geometric coverage or loss effects. These single-seed comparisons cannot establish causality.

Before another training run, inspect spatial clusters of i-beam→pipe/rectangularbeam, pump↔pipe, and tank→pump/pipe errors. Separate object interiors from semantic boundaries, and compare raw versus subsampled metrics to test projection effects. Use development validation to choose changes; repeated Area 3 tuning reduces its value as an independent test.
