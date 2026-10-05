# PSNet5 evaluation audit

The original ResPointNet2 evaluator is `ResPointNet2/function/evaluate_psnet_dist.py`.
It discards padded entries using `mask`, accumulates logits by cloud and
subsampled point index, averages repeated votes, projects predictions to every
raw point, then computes a global five-class confusion matrix and mIoU.
The PCM `validate_sphere` path follows the same voting and projection steps.

## Protocol differences

| Aspect | Original ResPointNet2 | Current PCM development run |
| --- | --- | --- |
| Scored region | Entire Area_3 | Spatially held-out parts of Areas 1, 2, 4 |
| Point cap | 15,000 in both train and evaluation | 4,096 in train, 15,000 in validation for the 2026-10-01 run |
| Unvisited subsampled points | Zero logits, hence class 0 after argmax; every raw point scored | Raw points projected to unvisited subsampled points excluded |
| Voting | 1 vote in standalone evaluator; 2 during training; 20 at final training evaluation | 1 fixed 2,000-sphere schedule in the cited development run |

All five classes occur in the development validation regions, so the original
metric's special treatment of absent classes does not affect this comparison.
The PCM score is a covered-point development metric. It is not a directly
comparable Area_3 benchmark score.

## Same-checkpoint scoring comparison

`evaluate_multivote.py` now reports `reference_full_raw` alongside the existing
coverage-aware metric. The reference calculation gives unvisited subsampled
points zero logits, matching `seg_metrics` in the original evaluator. Both
scores use the same predictions and the same held-out development regions.
The normal `validate_sphere` path now logs raw coverage and this full-raw mIoU
while keeping the covered-point metric for checkpoint selection.

| Replay of epoch-84 checkpoint | Covered-point mIoU | Original-style full-raw mIoU | Uncovered raw points |
| --- | ---: | ---: | ---: |
| First process | 29.56% | 28.39% | 1,213,356 |
| Seeded second process | 28.29% | 27.21% | 1,213,356 |

These values are in `results/reference_eval_epoch84_1vote/summary.json` and
`results/reference_eval_epoch84_seeded/summary.json`. The original run logged
30.01% covered-point validation mIoU. The reference-style scoring difference
is about 1.1 percentage points for these replays; it does not account for the
large training-versus-validation gap.

## Checkpoint replay defect

PCM previously used `list(set(mamba_layers_orders))` to assign embedding rows
to serialization-order prompts. Python set iteration can change across
processes. The checkpoint stores prompt weights but not this mapping, so the
same checkpoint could associate learned prompt rows with different orders
after reload. A fixed sphere had identical input, checkpoint, and outputs on
repeated forwards within one process, but its predictions changed across fresh
processes. Tracing located the first difference between the first pre-Mamba
block and its first Mamba input; the serialized point order itself matched.

`PointMambaEncoder` now assigns rows in first-appearance configuration order.
In two fresh processes with `PYTHONHASHSEED=1` and `PYTHONHASHSEED=2`, the first
Mamba input matched and the fixed sphere predicted the same number of points
per class. This fixes prompt assignment for newly trained checkpoints. The
original mapping for an already-trained checkpoint was not saved, so changing
the mapping cannot reconstruct its exact training-time association.

## Implications

- Do not interpret the run's logged `test_miou` as an Area_3 result: the
  end-of-training sphere branch calls the validation loader. The runner now
  states this explicitly in its log.
- Keep the coverage-aware development score and the original-style full-raw
  score distinct. Full raw scoring is most meaningful when sphere coverage is
  complete; otherwise the original evaluator implicitly calls unvisited
  points class 0.
- Retrain with the deterministic prompt mapping before using a checkpoint for
  a reproducible benchmark comparison. Evaluate that model on Area_3 only once
  the development choices are settled, using the benchmark split and reporting
  coverage and vote count.
