# PSNet5 loader port for PCM

## Executive summary

The recommended port is a PCM-native `PSNet5Sphere` dataset modeled on
`openpoints/dataset/s3dis/s3dis_sphere.py`, with PSNet5 parsing and the fixed
`psnet5-spatial-v1` development split added explicitly. This is a better fit
than adapting PCM's whole-room `S3DIS` loader because the established PSNet5
protocol trains on radius-2 neighborhoods selected by coverage potentials and
padded to 15,000 points.

The model itself does not consume PSNet5 files. PCM's segmentation loop expects
each dataset item to be a dictionary containing point coordinates, raw feature
columns and labels. Immediately before the forward pass it builds a
channel-first feature tensor. With the proposed `feature_keys: x,heights`, the
effective model inputs are:

| Value | Dataset item shape | Model shape after collation/feature assembly |
|---|---:|---:|
| Local XYZ (`pos`) | `[N, 3]` | `[B, N, 3]` |
| RGB (`x`) | `[N, 3]` | contributes 3 channels |
| Height (`heights`) | `[N, 1]` | contributes 1 channel |
| Features passed to PCM | n/a | `[B, 4, N]` |
| Labels (`y`) | `[N]` | `[B, N]` |
| Padding validity (`mask`) | `[N]` | `[B, N]` |

`get_features_by_keys` initially produces four channels. The stock PCM encoder
then appends the three XYZ channels internally when `combine_pos: true`, before
its embedding layer. Consequently `PointMambaEncoder.in_channels` must be **7**
for `feature_keys: x,heights` with `combine_pos: true`, or **4** when
`combine_pos: false`. XYZ remains separately available in `data['pos']` for
grouping and positional encoding in either case.

## What the existing PSNet5 loader does

The reference implementation is `ResPointNet2/datasets/PSNet5.py`.

### Source layout and labels

- Reads `data/PSNet/PSNet5/Area_*/Room_*/Annotations/*.txt`.
- Every row is `x y z r g b`; the class comes from the filename prefix.
- The fixed label order is `ibeam=0`, `pipe=1`, `pump=2`,
  `rectangularbeam=3`, `tank=4`.
- The historical benchmark split trains on Areas 1, 2 and 4 and evaluates on
  Area 3.

### Preprocessing

- Concatenates annotation files into one raw cloud per area.
- Grid-subsamples each area at 0.04 source units. The reference C++ operation
  averages coordinates and colors in each voxel and assigns a majority label.
- Divides subsampled RGB by 255 and constructs a `sklearn.neighbors.KDTree`.
- Saves raw-area caches, subsampled caches, iteration schedules and raw-to-
  subsampled nearest-neighbor projections as pickle files.

The verified local caches contain 80,915,438 raw points and 1,539,230 points at
0.04 voxel size. Existing verification found the per-area caches consistent
with the source annotations and the reference grid-subsampling implementation.

### Neighborhood sampling

- Generates an offline schedule of `num_epochs * num_steps` centers using a
  potential/coverage heuristic.
- Adds Gaussian center jitter with standard deviation `in_radius / 10`.
- Radius-queries the KDTree and keeps at most the nearest `num_points` points.
- Shuffles retained points. If the sphere is smaller than `num_points`, repeats
  existing points and supplies a mask that marks the repeats invalid.
- Expresses `pos` relative to the jittered sphere center, but uses the original
  absolute Z coordinate as `height`.

At the reference settings (`in_radius=2`, `num_points=15000`), almost every
audited sphere requires padding. Correct mask handling is therefore mandatory,
not an edge case.

### Reference output contract

The ResPointNet2 loader returns a list:

1. local points `[N,3]`;
2. mask `[N]`;
3. already assembled features `[C,N]`;
4. labels `[N]`;
5. local cloud index;
6. indices into the subsampled cloud.

For `input_features_dim=4`, features are normalized RGB plus absolute Z. The
reference RGB statistics are S3DIS statistics, not statistics calculated from
PSNet5: mean `[0.5136457, 0.49523646, 0.44921124]`, standard deviation
`[0.18308958, 0.18415008, 0.19252081]`.

## What PCM expects

### Dataset registration and construction

PCM uses the OpenPoints registry. A new class must be decorated with
`@DATASETS.register_module()` and imported from
`openpoints/dataset/__init__.py`. YAML selects it through
`dataset.common.NAME`. `build_dataloader_from_cfg` injects `split` and a
transform pipeline into the class constructor.

### Per-item dictionary

The fixed-size segmentation path relies on default PyTorch collation and needs
at least:

```text
{
  'pos': float32 [N,3],
  'x': float32 [N,F],
  'y': int64 [N],
  'heights': float32 [N,1]       # when named by feature_keys
}
```

Sphere validation additionally needs:

```text
{
  'mask': integer/bool [N],
  'cloud_index': int64 scalar,
  'input_inds': int64 [N]
}
```

All dictionary values that reach the training loop must be tensors because the
loop calls `.cuda()` on every value. `PointsToTensor` converts only `pos`; the
loader or other transforms must ensure the remaining values are tensors or
collatable NumPy arrays that default collation converts.

`get_features_by_keys` concatenates the requested last-axis columns and
transposes them to `[B,C,N]`. The model receives the whole dictionary; the
segmentation wrapper uses `data['pos']` and the encoder consumes `data['x']`.

### Coordinates, color and transforms

- PCM's `PointCloudXYZAlign` recenters all axes by their mean and then shifts Z
  so its minimum is zero. If the loader has already made XYZ relative to the
  sphere center, this changes the reference geometry again.
- `ChromaticNormalize` converts RGB to `[0,1]` when necessary and applies the
  configured mean/std.
- The stock PCM training transforms include scale, rotation, jitter and color
  drop. They operate on the dictionary and preserve labels and index metadata.
- Height creation must happen at the right time. The S3DIS sphere loader stores
  height from the unaugmented absolute points after transforms, reproducing the
  old RGB+absolute-Z convention but also retaining Area 2's very different Z
  offset. The existing dataset audit already identifies this cross-scene shift.

### Training and evaluation behavior

- The main loop sets `dataset.epoch = epoch - 1`, matching an offline schedule
  indexed as `idx + epoch * num_steps`.
- A criterion whose registered name contains `mask` receives the padding mask.
  A plain cross-entropy criterion does not, so it incorrectly trains on repeated
  padding labels.
- Selection of `validate_sphere` is string-based: the registered dataset name
  must contain `sphere`, unless the dispatch logic is changed.
- `validate_sphere` aggregates logits by `input_inds`, then projects them to raw
  points for metrics.

## Gaps and hazards in the current PCM code

These must be handled as part of the port, not deferred to configuration.

1. **Sphere validation is single-cloud only.** It ignores `cloud_index` and
   evaluates only `clouds_points_labels[0]` and `projections[0]`. Development
   validation contains regions from Areas 1, 2 and 4, so aggregation must be
   per cloud.
2. **Padded votes are not filtered.** `validate_sphere` scatters every logit,
   including repeated padding entries. Although averaging identical point
   indices often looks harmless, repeated entries can receive different
   predictions in context and bias the vote. Apply `mask` before aggregation.
3. **Uncovered points are unsafe.** A finite validation schedule may not visit
   every subsampled point. Scatter output and projection need an explicit
   coverage count; metrics must either run only after complete coverage or use
   a documented fallback. Silent class-0 logits are unacceptable.
4. **Distributed sphere aggregation is not correct as written.** Calling
   `all_reduce` on concatenated logits and point-index arrays assumes identical
   shapes and semantically aligned rows on every rank. Validation should either
   be single-process initially or gather per-cloud `(index, logit)` records and
   reduce sums/counts correctly.
5. **The generic test path does not know PSNet5.** `generate_data_list` and
   `load_data` support only S3DIS, ScanNet and SemanticKITTI. Final Area 3
   evaluation needs a PSNet5 branch or, preferably, a dataset-owned evaluation
   interface.
6. **The finalized development split is stricter than the historical area
   split.** Raw points must be partitioned before voxelization; training and
   validation cannot reuse whole-area subsampled caches without leakage at the
   boundary.
7. **Cache keys need protocol identity.** Filenames must encode dataset class,
   split version, role, voxel size and sampling settings. Reusing the reference
   `train_0.040_data.pkl` would silently select the historical Area-level split.
8. **Potential schedules are large and expensive to build.** Cache generation
   should be deterministic, progress-reported and atomic. It should not happen
   independently in several distributed workers.
9. **Feature semantics require an explicit decision.** Absolute Z matches the
   reference, but Area 2 is offset by roughly 80 source units. Local/min-relative
   height is more robust but changes the benchmark input. Both should be named
   configurations, with reference-compatible behavior as the initial control.

## Proposed loader design

Create `openpoints/dataset/psnet5/` with a small separation of concerns:

```text
psnet5/
  __init__.py
  psnet5.py          # registered dataset and item contract
  preprocessing.py  # parsing, partitioning, voxelization, cache/projection IO
  sampling.py       # deterministic potential schedule and radius extraction
```

The public registered class should be `PSNet5Sphere`. Suggested constructor
fields are:

```text
data_root, split, split_version, protocol,
voxel_size, in_radius, num_points, num_steps, num_epochs,
transform, seed, height_mode, cache_dir
```

Use two protocol modes:

- `development`: apply `psnet5-spatial-v1` to raw Areas 1, 2 and 4, producing
  independent train and validation point sets before voxelization.
- `benchmark`: train on all raw points from Areas 1, 2 and 4 and test on Area 3.

For each role and area:

1. parse raw annotations in a stable sorted order;
2. validate row width, finite XYZ, RGB range and known filename class;
3. verify the saved source digest before applying a packed membership mask;
4. apply raw-point ownership for the selected protocol;
5. voxelize that role independently with the verified reference operation;
6. build its KDTree;
7. restrict candidate centers according to the split rule;
8. jitter a center, re-check it against the boundary, and query only that
   role's KDTree;
9. save raw-to-role-subsample projections for validation/test only;
10. return the PCM dictionary with local XYZ, RGB, label, valid mask, cloud id,
    subsampled indices and the selected height representation.

For the spatial development split, the exact rule in `split.json` is part of
the loader contract: validation owns coordinates at or above the threshold,
training owns those below it, jitter is applied before center eligibility is
checked, and centers must remain at least radius 2 from the boundary. The query
must only see points owned by the same role.

## Implementation plan

### Phase 1: lock contracts with tests

- Add unit fixtures with tiny synthetic annotation trees containing all five
  filename classes.
- Assert label mapping, stable source ordering, dtype and every returned key.
- Assert batched shapes, that `feature_keys: x,heights` produces `[B,4,N]`, and
  that `combine_pos: true` presents seven channels to the embedding layer.
- Assert every item has at least one valid point, padding mask cardinality is
  correct, and masked entries are excluded from loss/voting.
- Add split-boundary tests covering equality, both sides, the radius buffer and
  jitter-before-eligibility behavior.

### Phase 2: preprocessing and cache layer

- Reuse or wrap the verified reference grid-subsampling implementation rather
  than PCM's random representative `voxelize`; the two algorithms are not
  equivalent.
- Implement raw parsing and source-digest checks.
- Implement development/benchmark role construction and versioned cache
  manifests. Write temporary files and rename on success.
- Save arrays and lightweight metadata separately where practical; avoid one
  opaque pickle containing all 80.9 million raw points.
- Verify regenerated 0.04 whole-area outputs against the existing audit before
  trusting partitioned outputs.

### Phase 3: sampling and PCM registration

- Port the potential sampler with an explicit seed and candidate-center mask.
- Decide whether schedules remain fully offline or are generated epoch-wise.
  Start offline for baseline parity, then profile storage/startup costs.
- Implement `PSNet5Sphere.__getitem__`, import it in the package registry and
  add `cfgs/psnet5/default.yaml` plus `PCM.yaml`.
- Start with radius 2, cap 15,000, voxel 0.04, RGB+absolute-Z, five classes and
  the reference augmentations as the parity configuration.
- Use a mask-aware cross-entropy loss and set encoder `in_channels: 7` when
  retaining PCM's `combine_pos: true` (otherwise set it to 4).

### Phase 4: correct validation and test aggregation

- Replace the current one-cloud sphere validator with per-cloud logit sums and
  vote counts keyed by `(cloud_index, input_inds)`.
- Filter invalid padded entries before accumulation.
- Report subsampled coverage; require full coverage for official metrics.
- Project each area's averaged predictions back to its owned raw points and
  update a common five-class confusion matrix.
- Keep validation single-process first. Add a correct distributed gather/reduce
  only after single-process parity tests pass.
- Add PSNet5 final-test support for Area 3 and ensure it cannot be selected by a
  development validation configuration.

### Phase 5: verification and parity runs

- Smoke-test one CPU item, one collated batch and one GPU forward/backward pass.
- Compare loader samples against `PSNet5Seg` using a fixed schedule: selected
  indices, local XYZ, RGB normalization, absolute height, labels and mask.
- Run an overfit test on a tiny fixed subset.
- Run a short development training job and confirm finite loss, all five label
  IDs, expected memory use and nonzero validation coverage in every area.
- Retrain on all Areas 1, 2 and 4 only after model/config selection, then run a
  single Area 3 benchmark evaluation.

## Recommended first deliverable

The first implementation milestone should stop after a deterministic
development loader, a mask-aware one-batch training test and a per-cloud
validation aggregation test. Do not begin a full training run until these
invariants pass:

- no raw point belongs to both development roles;
- partitioning occurs before voxelization;
- every sampled neighborhood stays within its role;
- assembled RGB+height features are exactly four channels, and the encoder
  input is seven channels only when `combine_pos` appends XYZ;
- padded elements affect neither loss nor voting;
- area identity participates in every aggregation index;
- validation reports explicit coverage before metrics;
- Area 3 is absent from all development artifacts.

## Open decisions

1. **Height mode:** retain absolute Z for strict ResPointNet2 parity, or make
   per-neighborhood/per-area relative height the primary PCM setting. Recommend
   implementing both and treating absolute Z as the parity control.
2. **Validation schedule:** fixed step count versus sampling until every
   subsampled point has at least one valid vote. Recommend coverage-driven
   validation for reliable metrics.
3. **Cache backend:** NumPy/memmap-style arrays plus JSON manifests versus
   compatibility pickles. Recommend array files/manifests for raw data and small
   pickles only for KDTree/schedule compatibility.
4. **Baseline parity versus architecture tuning:** keep sampling and features
   fixed for the first run. Tune PCM point cap, window sizes and height mode only
   after the adapter passes parity and leakage tests.
