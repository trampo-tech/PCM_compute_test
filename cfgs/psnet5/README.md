# PSNet5 development runs

Both configs implement the finalized `psnet5-spatial-v1` development split.
They partition raw Areas 1, 2, and 4 before 0.04 voxelization, use radius-2
spheres, RGB plus absolute Z, mask padded points in loss and voting, and never
read Area 3.

PCM also compacts each sphere to valid points before encoding, so duplicated
padding cannot influence grouping, Mamba context, decoding, or global pooling.
See `analysis/psnet5/PADDING_INVESTIGATION.md` for the checkpoint diagnostic and
its limitations.

Build the CPU subsampling extension once in the same environment used for
training:

```bash
cd openpoints/cpp/subsampling
python setup.py build_ext --inplace
cd ../../..
```

Run the one-epoch, eight-step smoke job first:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
  --cfg cfgs/psnet5/PCM-smoke.yaml
```

The first invocation creates versioned train/validation caches below
`data/PSNet5/processed_pcm`. Cache generation verifies the finalized packed
masks against the SHA-256 digests of the audited raw area caches. The smoke
configuration uses 4,096 points and 16 validation spheres, so it reports but
does not require full validation coverage.

After the smoke job completes with finite loss, start the full 100-epoch run:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
  --cfg cfgs/psnet5/PCM.yaml
```

The full configuration restores 15,000 points and 2,000 training steps per
epoch. Validation reports coverage per area, excludes uncovered points from
metrics, and requires at least 90% coverage in every area. Its deterministic
potential schedule is created once and reused on subsequent launches.

`eval_bn_batch_stats: true` makes PSNet5 validation and standalone evaluation
use statistics from each compact sphere in BatchNorm layers. Dropout stays in
evaluation mode, and BatchNorm running buffers are not updated. One-point
spheres use the existing stored-statistics fallback. This option changes
validation and checkpoint selection; it does not change optimizer updates.
Set it to `false` to replay the older running-statistics evaluation.

The PCM configuration explicitly selects BatchNorm. The completed GroupNorm
run's best checkpoint scored 39.70% mIoU on Area 3, compared with 60.50% for
the earlier BatchNorm checkpoint using current-sphere statistics. The two runs
also differed in drop-path rate and training duration, so this is an operational
choice rather than an isolated estimate of the normalization effect. The earlier
BatchNorm checkpoint was selected using the old evaluation setting; a new run
with `eval_bn_batch_stats: true` will select checkpoints using the fixed mode.

To evaluate a completed PCM checkpoint on the original PSNet5 Area_3 split,
run the standalone evaluator after training:

```bash
CUDA_VISIBLE_DEVICES=0 python analysis/psnet5/evaluate_multivote.py \
  --cfg cfgs/psnet5/PCM.yaml \
  --checkpoint /absolute/path/to/ckpt_best.pth \
  --benchmark-area3 --votes 1 \
  --output analysis/psnet5/results/area3-evaluation
```

In `summary.json`, compare `reference_full_raw.miou` with the original
ResPointNet2 standalone one-vote evaluator. `final.miou` is the coverage-aware
score. The development training configuration uses only portions of Areas 1, 2,
and 4; a strict original-protocol benchmark requires retraining on all three
areas with model selection completed before evaluating Area_3.
