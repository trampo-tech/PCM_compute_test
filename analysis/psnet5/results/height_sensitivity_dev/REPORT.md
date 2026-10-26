# Height-feature sensitivity on development validation

Two checkpoints, each evaluated on the same fixed 2,000 development-validation spheres in Areas 1, 2 and 4. Height-feature offsets only: −0.50, −0.25, 0, +0.25 and +0.50 m. XYZ geometry, RGB, point masks, sphere order and vote accumulation are unchanged. Random states are restored before each condition within a sphere. Current-sphere BatchNorm is used without updating running statistics. Area 3 is not accessed.

## Results

| Checkpoint | Offset m | mIoU % | Change pp | Tank recall % | Prediction flips % |
|---|---:|---:|---:|---:|---:|
| weighted_best | -0.50 | 86.07987 | -0.00056 | 100.00000 | 0.00010 |
| weighted_best | -0.25 | 86.08132 | +0.00088 | 100.00000 | 0.00029 |
| weighted_best | +0.00 | 86.08043 | +0.00000 | 100.00000 | 0.00000 |
| weighted_best | +0.25 | 86.08132 | +0.00088 | 100.00000 | 0.00052 |
| weighted_best | +0.50 | 86.08095 | +0.00052 | 100.00000 | 0.00014 |
| classmix_latest | -0.50 | 75.82465 | -0.00108 | 84.74736 | 0.00071 |
| classmix_latest | -0.25 | 75.82483 | -0.00090 | 84.75044 | 0.00028 |
| classmix_latest | +0.00 | 75.82574 | +0.00000 | 84.75044 | 0.00000 |
| classmix_latest | +0.25 | 75.82599 | +0.00025 | 84.75044 | 0.00044 |
| classmix_latest | +0.50 | 75.82528 | -0.00046 | 84.74736 | 0.00063 |

## Measured effect

- weighted_best: largest absolute mIoU change 0.000884 percentage points; largest changed-prediction fraction 0.000524%.
- classmix_latest: largest absolute mIoU change 0.001082 percentage points; largest changed-prediction fraction 0.000710%.

## Coverage and controls

- weighted_best: unchanged-input repeat maximum logit difference 0.
  - Area_1: raw coverage 95.758%; subsampled coverage 96.134%.
  - Area_2: raw coverage 90.401%; subsampled coverage 91.207%.
  - Area_4: raw coverage 94.430%; subsampled coverage 93.904%.
- classmix_latest: unchanged-input repeat maximum logit difference 0.
  - Area_1: raw coverage 95.758%; subsampled coverage 96.134%.
  - Area_2: raw coverage 90.401%; subsampled coverage 91.207%.
  - Area_4: raw coverage 94.430%; subsampled coverage 93.904%.

## Interpretation

A constant offset tests sensitivity to the absolute origin of the extra height feature. It does not remove the height feature, alter within-sphere height variation, or move the XYZ geometry. The encoder starts with a linear convolution followed by BatchNorm; using current-sphere statistics tends to remove a constant activation shift. Small numerical changes can remain due to floating-point operations and singleton fallbacks. Consequently, stability here cannot prove that height is unused or explain away the observed height-dependent tank errors.

To test reliance on height information itself, separately evaluate height removal (set the feature to zero) or controlled scaling while preserving XYZ. Those changes are outside this offset experiment and can move features away from the training distribution. Absolute height equals local Z plus a sphere-specific constant, and local Z is also an input in this configuration. Under current-sphere BatchNorm, switching only from absolute to local height is therefore expected to have little effect. Height removal or scaling is the more informative next diagnostic; XYZ still contains relative vertical geometry. Any retraining experiment should be selected on development validation.

The primary conclusion should follow the measured magnitude: large prediction changes indicate offset sensitivity; negligible changes argue against an absolute height-origin explanation under the current evaluation normalization. These are deterministic paired measurements, not an estimate of between-training-seed uncertainty.

## Validation tank coverage limitation

Covered validation tank points occur only in Area 2. Its tank raw-height quartiles span approximately 85.30 to 93.14 m, whereas the Area 3 tank range inspected earlier was approximately -1.37 to 3.77 m at subsampled resolution. Small offset stability does not establish generalization across those coordinate ranges or different tank shapes.

## Artifacts

[Full metrics and per-area height bands](summary.json), [comparison table](comparison.csv). Each condition also has its own JSON, confusion matrix, and compact per-area subsampled prediction export. Metrics exclude uncovered raw points and use the same support in all conditions.
