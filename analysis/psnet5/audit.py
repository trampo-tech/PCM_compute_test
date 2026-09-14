"""Read-only PSNet5 audit. Run with the working ResPointNet conda environment.

Only load pickle caches from a trusted local dataset. Output is written separately.
"""
import argparse
import csv
import itertools
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/psnet5-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np

CLASSES = ['ibeam', 'pipe', 'pump', 'rectangularbeam', 'tank']
COLORS = np.array(['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377'])


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def load(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def counts(labels):
    return np.bincount(np.asarray(labels).reshape(-1), minlength=5).tolist()


def preview(points, labels, rgb, path, title, rng, centers=None, boundary=None):
    idx = rng.choice(len(points), min(45000, len(points)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, colors, name in zip(axes, [np.clip(rgb[idx], 0, 1), COLORS[labels[idx]]], ['RGB', 'Ground truth']):
        ax.scatter(points[idx, 0], points[idx, 1], c=colors, s=0.5, rasterized=True)
        if centers is not None:
            for center in centers:
                ax.add_patch(Circle(center[:2], 2, fill=False, color='black', linewidth=1))
        if boundary is not None:
            axis, threshold = boundary
            line = ax.axvline if axis == 0 else ax.axhline
            for offset, style in [(-2, ':'), (0, '--'), (2, ':')]:
                line(threshold + offset, color='black', linestyle=style, linewidth=1)
        ax.set(title=name, xlabel='X (source units)', ylabel='Y (source units)', aspect='equal')
    axes[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES)], fontsize=8)
    fig.suptitle(title + '\nXY projection; display sample only (vertical structures may overlap)')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--samples-per-area', type=int, default=200)
    args = parser.parse_args()
    if args.samples_per_area < 1:
        parser.error('--samples-per-area must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    root = args.data_root.resolve()
    summary = {'seed': args.seed, 'data_root': str(root), 'classes': CLASSES,
               'units': 'Not independently confirmed; baseline assumes meters.', 'areas': [], 'issues': []}
    manifest = json.loads((Path(__file__).parent / 'split.json').read_text())
    fixed_regions = {r['area']: r for r in manifest['regions']}
    manifest['regions'] = []
    sampling_rows = []
    inventory = []
    train_indices = load(root / 'processed/train_0.040_2_2000_iterinds.pkl')
    val_indices = load(root / 'processed/val_0.040_20_2000_iterinds.pkl')
    for area_no in range(1, 5):
        area = f'Area_{area_no}'
        print(f'{area}: checking raw annotations against scene cache', flush=True)
        xyz, rgb, labels = load(root / f'processed/{area}.pkl')
        labels = labels.reshape(-1)
        raw_counts = [0] * 5
        cache_matches = True
        for source in sorted((root / area).glob('*/Annotations/*.txt')):
            name = source.stem.split('_')[0]
            if name not in CLASSES:
                raise ValueError(f'Unknown class: {source}')
            cid = CLASSES.index(name)
            cache_idx = np.flatnonzero(labels == cid)
            offset = raw_counts[cid]
            n = invalid = rgb_invalid = 0
            minimum = np.full(3, np.inf)
            maximum = np.full(3, -np.inf)
            with source.open() as stream:
                while lines := list(itertools.islice(stream, 200000)):
                    arr = np.loadtxt(lines, dtype=np.float64, ndmin=2)
                    if arr.shape[1] != 6:
                        raise ValueError(f'Expected XYZRGB columns: {source}')
                    invalid += int((~np.isfinite(arr)).any(axis=1).sum())
                    rgb_invalid += int(((arr[:, 3:] < 0) | (arr[:, 3:] > 255) |
                                        (arr[:, 3:] != np.floor(arr[:, 3:]))).any(axis=1).sum())
                    minimum = np.minimum(minimum, arr[:, :3].min(axis=0))
                    maximum = np.maximum(maximum, arr[:, :3].max(axis=0))
                    ix = cache_idx[offset+n:offset+n+len(arr)]
                    cache_matches &= (len(ix) == len(arr) and
                                      np.array_equal(xyz[ix], arr[:, :3].astype(np.float32)) and
                                      np.array_equal(rgb[ix], arr[:, 3:]))
                    n += len(arr)
            raw_counts[cid] += n
            inventory.append({'file': str(source.relative_to(root)), 'bytes': source.stat().st_size,
                              'points': n, 'class': name, 'invalid_rows': invalid,
                              'invalid_rgb_rows': rgb_invalid, 'xyz_min': minimum.tolist(), 'xyz_max': maximum.tolist()})
            if invalid or rgb_invalid:
                summary['issues'].append(f'{source.name}: invalid values detected')
        cache_matches &= raw_counts == counts(labels)
        if not cache_matches:
            summary['issues'].append(f'{area}: raw/cache mismatch (including possible within-class order differences)')
        points, colors, y, tree = load(root / f'processed/{area}_0.040_sub.pkl')
        y = np.asarray(y).reshape(-1)
        tree_matches = np.array_equal(np.asarray(tree.data), points)
        if not tree_matches:
            summary['issues'].append(f'{area}: KDTree coordinates mismatch')
        print(f'{area}: {len(xyz):,} raw / {len(points):,} subsampled; sampling neighborhoods', flush=True)
        bounds = [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
        item = {'area': area, 'raw_counts': raw_counts, 'subsampled_counts': counts(y),
                'raw_cache_exact_match': bool(cache_matches), 'kdtree_exact_match': bool(tree_matches),
                'xyz_bounds': bounds, 'subsampled_rgb_range': [float(colors.min()), float(colors.max())]}
        # Use recorded baseline potential-sampling centers/noise, stratified by area.
        cloud_inds, point_inds, noise = train_indices if area_no != 3 else val_indices
        cloud_id = {1: 0, 2: 1, 4: 2, 3: 0}[area_no]
        eligible = np.flatnonzero(np.asarray(cloud_inds) == cloud_id)
        chosen = rng.choice(eligible, min(args.samples_per_area, len(eligible)), replace=False)
        area_rows = []
        for step in chosen:
            center = points[point_inds[step]].reshape(1, -1) + np.asarray(noise[step]).astype(points.dtype)
            ids = tree.query_radius(center, r=2, return_distance=True, sort_results=True)[0][0]
            selected = ids[:15000]
            row = {'area': area, 'recorded_step': int(step), 'neighbors': len(ids),
                   'unique_selected': len(selected), 'padding': max(0, 15000-len(ids)),
                   'discarded': max(0, len(ids)-15000), 'class_counts': counts(y[selected])}
            area_rows.append(row)
            sampling_rows.append(row)
        item['sampling'] = {'n': len(area_rows), 'neighbors_min_median_max': np.percentile([r['neighbors'] for r in area_rows], [0, 50, 100]).tolist(),
                            'padding_fraction': float(np.mean([r['padding'] > 0 for r in area_rows])),
                            'truncation_fraction': float(np.mean([r['discarded'] > 0 for r in area_rows])),
                            'class_presence_fraction': np.mean([np.array(r['class_counts']) > 0 for r in area_rows], axis=0).tolist()}
        # Apply the versioned fixed spatial split.
        boundary = None
        if area_no != 3:
            axis = ['x', 'y'].index(fixed_regions[area]['axis'])
            threshold = fixed_regions[area]['threshold']
            train = points[:, axis] < threshold
            val = ~train
            region = {'area': area, 'axis': ['x', 'y'][axis], 'threshold': threshold,
                      'buffer_radius': 2.0, 'training_point_counts': counts(y[train]),
                      'validation_point_counts': counts(y[val]),
                      'eligible_training_centers': int((points[:, axis] < threshold-2).sum()),
                      'eligible_validation_centers': int((points[:, axis] >= threshold+2).sum())}
            region['eligible_training_center_class_counts'] = counts(y[points[:, axis] < threshold-2])
            region['eligible_validation_center_class_counts'] = counts(y[points[:, axis] >= threshold+2])
            manifest['regions'].append(region)
            boundary = (axis, threshold)
        centers = []
        for cid in range(5):
            candidates = np.flatnonzero(y == cid)
            if not len(candidates):
                continue
            center = points[rng.choice(candidates)]
            centers.append(center)
            ids = tree.query_radius(center[None], r=2, return_distance=True, sort_results=True)[0][0][:15000]
            preview(points[ids], y[ids], colors[ids], out / f'{area}_sample_{CLASSES[cid]}.png',
                    f'{area}: neighborhood centered on {CLASSES[cid]} ({len(ids):,} unique points)', rng)
        preview(points, y, colors, out / f'{area}_overview.png', f'{area}: cached 0.040 subsampling; circles show example neighborhoods', rng, centers)
        if boundary is not None:
            preview(points, y, colors, out / f'{area}_split.png', f'{area}: spatial split; dashed boundary, dotted center exclusion limits', rng, boundary=boundary)
        summary['areas'].append(item)
        dump(out / 'summary.json', summary)
        del xyz, rgb, labels, points, colors, y, tree
    dump(out / 'inventory.json', inventory)
    dump(out / 'sampling.json', sampling_rows)
    dump(out / 'split_statistics.json', manifest)
    with (out / 'class_counts.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['area', 'class', 'raw_points', 'subsampled_points', 'retention_fraction'])
        for area in summary['areas']:
            for name, raw, sub in zip(CLASSES, area['raw_counts'], area['subsampled_counts']):
                writer.writerow([area['area'], name, raw, sub, sub/raw if raw else ''])
    raw = np.array([a['raw_counts'] for a in summary['areas']])
    sub = np.array([a['subsampled_counts'] for a in summary['areas']])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, matrix, title in zip(axes, [raw, sub], ['Raw annotations', 'Cached 4 cm subsampling (assumed meters)']):
        bottom = np.zeros(4)
        for cid, name in enumerate(CLASSES):
            ax.bar(np.arange(4), matrix[:, cid]/1e6, bottom=bottom, color=COLORS[cid], label=name)
            bottom += matrix[:, cid]/1e6
        ax.set(xticks=range(4), xticklabels=[f'Area {i}' for i in range(1, 5)], ylabel='Points (millions)', title=title)
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'class_distribution.png', dpi=160)
    plt.close(fig)
    fractions = raw / raw.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    im = ax.imshow(fractions, cmap='Blues', vmin=0, vmax=100)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f'{fractions[i,j]:.1f}%', ha='center', va='center')
    ax.set(xticks=range(5), xticklabels=CLASSES, yticks=range(4), yticklabels=[f'Area {i}' for i in range(1, 5)], title='Raw class composition within each area')
    fig.colorbar(im, ax=ax, label='% of area points')
    fig.savefig(out / 'class_heatmap.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.boxplot([[r['neighbors'] for r in sampling_rows if r['area'] == f'Area_{i}'] for i in range(1, 5)], tick_labels=[f'Area {i}' for i in range(1, 5)], showfliers=False)
    ax.axhline(15000, color='red', linestyle='--', label='15,000-point cap')
    ax.set(ylabel='Points within radius 2', title='Recorded baseline neighborhoods (outliers hidden)')
    ax.legend()
    fig.savefig(out / 'neighborhood_sizes.png', dpi=160)
    plt.close(fig)
    print('Audit complete:', out, flush=True)


if __name__ == '__main__':
    main()
