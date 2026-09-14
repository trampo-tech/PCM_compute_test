# PSNet5: Dataset Characterization

PSNet5 contains industrial point clouds from four scenes and five semantic classes: I-beams, pipes, pumps, rectangular beams and tanks. This report characterizes the complete local dataset, its preprocessing, spatial sampling and evaluation partitions.

## Integrity and inventory

- 18 annotation files; **80,915,438 raw labeled points** and **1,539,230 cached subsampled points**.
- Invalid numeric/RGB rows: 0.
- All four raw scene caches match every annotation row after the baseline float32 coordinate conversion.
- All four KDTrees match their cached coordinates.
- Full regeneration of XYZ, RGB and labels at voxel size 0.04: **PASS** for all four areas.
- Verification covers raw scene caches, per-scene voxel caches and KDTrees; aggregate split/projection caches are outside its scope.
- Physical units have no explicit confirmation in the local dataset README. Distances use source coordinate units; the baseline interprets these as meters.

## Class distribution

![Raw and subsampled class counts](results/class_distribution.png)

![Per-area proportions](results/class_heatmap.png)

| Class | Raw points | Subsampled points | Retained |
|---|---:|---:|---:|
| ibeam | 4,506,466 | 101,842 | 2.26% |
| pipe | 39,973,956 | 636,482 | 1.59% |
| pump | 5,049,832 | 108,568 | 2.15% |
| rectangularbeam | 24,406,028 | 543,703 | 2.23% |
| tank | 6,979,156 | 148,635 | 2.13% |

Area 2 is the only training area containing tanks; Area 3 also has tanks but is reserved for final evaluation. Area 2 pumps retain only 2,551 of 279,617 raw points (0.91%). This is a point-count reduction, not proof that whole pump objects disappear. Subsampling changes the class proportions because point density varies across surfaces and scenes.

## Density and coordinates

The density below is raw points divided by the XY bounding-box area. It is a rough comparison, not true scanned surface density; empty space and stacked structures affect it.

| Area | XYZ extent (source units) | Raw points / XY bbox unit² | Raw Z range |
|---|---|---:|---|
| Area_1 | 32.15 × 43.40 × 5.95 | 21,829 | -1.78 to 4.16 |
| Area_2 | 23.67 × 26.09 × 28.55 | 23,324 | 78.81 to 107.35 |
| Area_3 | 35.37 × 41.98 × 9.63 | 16,480 | -1.40 to 8.23 |
| Area_4 | 20.01 × 16.93 × 5.39 | 34,207 | -0.93 to 4.46 |

Area 2 uses XY coordinates around 2,700 and Z around 79–107, unlike the other areas. The baseline centers XYZ on the neighborhood center but uses absolute Z as its height feature. Absolute height therefore has different ranges across scenes even after local XYZ centering.

## Sampling inspection

200 recorded potential-sampling neighborhoods per area, radius 2, cap 15,000, seed 20260912. Nearest points are retained when over the cap; smaller samples require repeated padding points with a validity mask. The analysis counts unique points before padding and does not run model augmentation.

| Area | Neighbors: min / median / max | Need padding | Truncated |
|---|---|---:|---:|
| Area_1 | 1 / 5,053 / 13,296 | 100.0% | 0.0% |
| Area_2 | 235 / 4,914 / 16,313 | 99.5% | 0.5% |
| Area_3 | 354 / 2,980 / 12,521 | 100.0% | 0.0% |
| Area_4 | 281 / 3,296 / 9,741 | 100.0% | 0.0% |

![Neighborhood sizes](results/neighborhood_sizes.png)

799 of 800 examined neighborhoods require padding. One sampled Area 1 neighborhood contains only one unique point. Padding repeats existing points and the baseline marks those repetitions with a validity mask. The fixed tensor size is consequently much larger than the number of distinct measurements in most sampled neighborhoods.

## Industrial scenes

XY previews can hide vertical overlap and do not establish expert annotation correctness. The files named `sample_*` show class-centered examples with the same radius/cap; they are not randomly selected baseline samples.

### Area_1: Chiller House

![Area_1 RGB and labels](results/Area_1_overview.png)

![Area_1 pump neighborhood](results/Area_1_sample_pump.png)

### Area_2: On-Site Chlorine Generation

![Area_2 RGB and labels](results/Area_2_overview.png)

![Area_2 pump neighborhood](results/Area_2_sample_pump.png)

![Area_2 tank neighborhood](results/Area_2_sample_tank.png)

### Area_3: Sludge Press House

![Area_3 RGB and labels](results/Area_3_overview.png)

![Area_3 pump neighborhood](results/Area_3_sample_pump.png)

![Area_3 tank neighborhood](results/Area_3_sample_tank.png)

### Area_4: Wash-water Recovery Tank

![Area_4 RGB and labels](results/Area_4_overview.png)

![Area_4 pump neighborhood](results/Area_4_sample_pump.png)

## Training, validation and test partitions

The fixed partition **psnet5-spatial-v1** uses spatial training and validation regions in Areas 1, 2 and 4. Area 3 is the held-out test scene. Boundaries were selected using training-area labels only, requiring at least 100 eligible centers of each present class on both sides. The resulting split is not a uniform 80/20 split: retaining rare classes requires larger holdouts in Areas 2 and 4.

| Area | Boundary | Validation fraction (subsampled points) | Eligible train / validation centers |
|---|---|---:|---:|
| Area_1 | y ≥ 59.361000 | 20.0% | 332,100 / 56,734 |
| Area_2 | y ≥ 2736.472656 | 50.0% | 130,317 / 127,829 |
| Area_4 | x ≥ 8.553347 | 40.0% | 61,334 / 42,686 |

### Raw-point membership

| Class | Training | Validation | Test (Area 3) |
|---|---:|---:|---:|
| ibeam | 1,003,995 | 479,468 | 3,023,003 |
| pipe | 20,817,678 | 10,747,371 | 8,408,907 |
| pump | 1,727,168 | 867,452 | 2,455,212 |
| rectangularbeam | 10,903,152 | 4,447,237 | 9,055,639 |
| tank | 3,599,401 | 1,856,424 | 1,523,331 |
| **Total** | **38,051,394** | **18,397,952** | **24,466,092** |

Both partitions have eligible centers for every class present in each area. Area 2 still has only 369 eligible pump centers on the training side and 915 tank centers on the validation side. Counts describe correlated points, not independent instances.

The sampling protocol admits centers at least radius 2 from the split plane after center jitter. Raw points are assigned to their partition before voxelization; neighborhood queries are restricted to that partition. These rules prevent shared input points; they do not guarantee that a physical component cannot extend across both partitions. Normalization statistics and class weights use training points only. Boundary points may have limited prediction coverage under the center exclusion rule. The table summarizes the verified whole-scene voxel caches; voxelizing partitions independently can change boundary voxels.

![Area_1 split](results/Area_1_split.png)

![Area_2 split](results/Area_2_split.png)

![Area_4 split](results/Area_4_split.png)

## Reproducibility

The fixed split is recorded in `split.json`. Raw-point ownership masks are stored as packed bits in `results/Area_*_partition.npz`, with a SHA-256 digest identifying the source cache and its point order. Area 3 is assigned wholly to test. The masks define ownership; dataset loaders must apply the documented sampling and voxelization rules.

For model development, the spatial validation regions remain separate from training. After model selection, the benchmark protocol retrains on all of Areas 1, 2 and 4 and evaluates on Area 3.

Audit seed: 20260912. Sampling statistics use 200 recorded baseline neighborhoods per area; class-centered close-ups are illustrative. Scripts and execution instructions are documented in README.md.

Evidence: inventory.json, summary.json, class_counts.csv, sampling.json, subsampling_verification.json and split_validation.json in results/.
