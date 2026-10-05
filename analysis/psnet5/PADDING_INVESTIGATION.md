# PCM padding investigation

PSNet5 spheres have a fixed tensor size. When a radius query returns fewer than
`num_points`, the loader fills the remaining slots with randomly selected copies
of valid points and marks those slots false in `mask`.

Before this change, `BaseSeg` passed every slot to PCM. The mask was used only by
the loss and vote accumulator. Consequently, duplicated padding could affect:

- `LocalGrouper`: farthest-point sampling, KNN membership, anchor selection and
  neighborhood normalization;
- `PointMambaEncoder`: spatial serialization, window selection, positional
  embeddings and recurrent Mamba context;
- `PointMambaDecoder`: feature propagation/interpolation;
- `PointMambaDecoder` and `SegHead`: global max/average context.

## Change

`BaseSeg(mask_padding=true)` now gathers only valid points from each example,
runs the complete encoder/decoder/head on that compact cloud, and scatters the
result into the original tensor layout. Invalid output slots are zero. This is
enabled only in the PSNet5 PCM configurations; the default is false, so other
datasets and models without a mask retain their original behavior. All-valid
batches keep the original batched fast path.

The exact approach processes examples separately if any example in a batch is
padded. This changes BatchNorm statistics relative to the legacy batch-of-two
training path and can reduce throughput. It is intentional: padding cannot be
made inert merely by zeroing features because its coordinates and sequence
positions have already influenced PCM.

The decoder's final global-context projection receives a `[1, C, 1]` tensor on
these per-example passes. Its BatchNorm now falls back to stored running
statistics only for that singleton case; otherwise it is byte-for-byte the
normal BatchNorm path. Without this fallback, PyTorch correctly rejects a
single-value training batch.

### Sparse compact spheres

The fixed schedule includes valid spheres much smaller than PCM's configured
12-neighbor groups. Grouping now uses `min(configured_k, available_points)`,
each reducer retains at least one point, and feature propagation uses at most
the available support count. No arbitrary input padding is reintroduced.
Zero-extent serialization uses the minimum valid one-bit cube, and singleton
BatchNorm layers use stored statistics only for the otherwise-invalid
one-value case. The segmentation decoder also preserves `[B, C, 1]` instead of
squeezing away the point dimension.

Scan every sphere in the intended 10-epoch schedule with:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/blau/projects/pcm-psnet5-project:/workspace/pcm-psnet5-project \
  -w /workspace/pcm-psnet5-project/PointCloudMamba \
  pointcloudmamba:latest \
  python analysis/psnet5/check_compaction_schedule.py \
    --epochs 10 \
    --output analysis/psnet5/results/compaction_schedule_10epochs.json
```

The scan found a minimum of one valid point among 20,000 scheduled spheres;
101 spheres contained exactly one point. Such a sphere remains one point at all
four hierarchy levels and is covered by the full-model one-point forward and
backward regression.

## Fast checkpoint diagnostic

The diagnostic selects a padded validation sphere, creates alternate padding by
copying different valid points, and evaluates the same frozen checkpoint with
legacy full-input handling and mask compaction. Valid points are unchanged.

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/blau/projects/pcm-psnet5-project:/workspace/pcm-psnet5-project \
  -w /workspace/pcm-psnet5-project/PointCloudMamba \
  pointcloudmamba:latest \
  python analysis/psnet5/check_padding_sensitivity.py \
    --checkpoint log/psnet5/psnet5-train-PCM-ngpus1-20260921-015453-TaGtTm9SrUhiW7w5mT6bo3/checkpoint/psnet5-train-PCM-ngpus1-20260921-015453-TaGtTm9SrUhiW7w5mT6bo3_ckpt_best.pth \
    --variants 3 \
    --output analysis/psnet5/results/padding_sensitivity_epoch85.json
```

On a sphere with 10,622 valid points out of 15,000, three padding variants
changed roughly 2-5% of legacy valid-point predictions (the exact GPU result is
recorded in `results/padding_sensitivity_epoch85.json`). Mask compaction reduced
the mean change, maximum change, and prediction-flip fraction to exactly zero
for all three variants.

This proves padding isolation and shows that legacy PCM was materially sensitive
to arbitrary duplicated padding. It does **not** prove that retraining improves
mIoU: the checkpoint learned under legacy padding.

Exercise padded forward, backward, optimizer and validation paths before a full
run with:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/blau/projects/pcm-psnet5-project:/workspace/pcm-psnet5-project \
  -w /workspace/pcm-psnet5-project/PointCloudMamba \
  pointcloudmamba:latest \
  python examples/segmentation/main.py \
    --cfg cfgs/psnet5/PCM-smoke.yaml \
    --dataset.common.num_points 15000 \
    --model.test_crop 15000
```

Before a full 100-epoch run, compare a short fixed-seed run (for example 5-10
epochs) against an otherwise identical `--model.mask_padding false` control.
The checkpoint sensitivity test is the fast pass/fail test for padding
isolation; early loss and validation curves are evidence about optimization,
but a full benchmark remains necessary for a final accuracy claim.
