# PSNet5 development runs

Both configs implement the finalized `psnet5-spatial-v1` development split.
They partition raw Areas 1, 2, and 4 before 0.04 voxelization, use radius-2
spheres, RGB plus absolute Z, mask padded points in loss and voting, and never
read Area 3.

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
