"""Compare the PCM PSNet5 port with ResPointNet2's historical loader contract."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.dataset.psnet5.preprocessing import load_cache, prepare_cache


COLOR_MEAN = np.array([0.5136457, 0.49523646, 0.44921124], dtype=np.float32)
COLOR_STD = np.array([0.18308958, 0.18415008, 0.19252081], dtype=np.float32)


def array_check(left, right, atol=0.0):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {'passed': False, 'left_shape': list(left.shape), 'right_shape': list(right.shape)}
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    maximum = float(difference.max(initial=0))
    return {'passed': bool(np.allclose(left, right, atol=atol, rtol=0)),
            'shape': list(left.shape), 'max_abs_error': maximum}


def padded_indices(query, num_points, rng):
    query = query[:num_points]
    query = query[rng.permutation(len(query))]
    valid = len(query)
    if valid < num_points:
        query = np.concatenate([query, rng.choice(query, num_points - valid, replace=True)])
    mask = np.zeros(num_points, dtype=np.int32)
    mask[:valid] = 1
    return query, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('../ResPointNet2/data/PSNet/PSNet5'))
    parser.add_argument('--split-manifest', type=Path, default=Path('analysis/psnet5/split.json'))
    parser.add_argument('--partition-root', type=Path, default=Path('analysis/psnet5/results'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/PSNet5/processed_pcm'))
    parser.add_argument('--reference-ops-root', type=Path, default=Path('openpoints/cpp/subsampling'))
    parser.add_argument('--samples-per-area', type=int, default=16)
    parser.add_argument('--roles', nargs='+', choices=('train', 'test'), default=('train', 'test'))
    parser.add_argument('--output', type=Path, default=Path('analysis/psnet5/results/loader_parity.json'))
    args = parser.parse_args()

    report = {'protocol': 'benchmark', 'voxel_size': 0.04, 'areas': [], 'passed': True}
    rng = np.random.default_rng(20260921)
    for role_index, role in enumerate(args.roles):
        cache_path, manifest = prepare_cache(
            data_root=args.data_root, cache_root=args.cache_dir,
            split_manifest=args.split_manifest, partition_root=args.partition_root,
            protocol='benchmark', role=role, voxel_size=0.04,
            reference_ops_root=args.reference_ops_root, verify_source_digest=True)
        pcm_clouds = load_cache(cache_path, manifest)
        for cloud_index, cloud in enumerate(pcm_clouds):
            area = cloud['area']
            with (args.data_root / 'processed' / f'{area}_0.040_sub.pkl').open('rb') as stream:
                ref_points, ref_colors, ref_labels, ref_tree = pickle.load(stream)
            ref_points = np.asarray(ref_points, dtype=np.float32)
            ref_colors = np.asarray(ref_colors, dtype=np.float32)
            ref_labels = np.asarray(ref_labels, dtype=np.int64).reshape(-1)
            checks = {
                'subsampled_points': array_check(cloud['sub_points'], ref_points, atol=1e-6),
                'subsampled_colors': array_check(cloud['sub_colors'], ref_colors, atol=1e-6),
                'subsampled_labels': array_check(cloud['sub_labels'], ref_labels),
            }
            sample_checks = []
            centers = rng.choice(len(ref_points), size=min(args.samples_per_area, len(ref_points)), replace=False)
            for sample_index, center_index in enumerate(centers):
                noise = rng.normal(scale=0.2, size=3).astype(np.float32)
                pick = ref_points[center_index] + noise
                ref_query = ref_tree.query_radius(pick.reshape(1, -1), r=2.0,
                                                  return_distance=True, sort_results=True)[0][0]
                pcm_query = cloud['tree'].query_radius(pick.reshape(1, -1), r=2.0,
                                                       return_distance=True, sort_results=True)[0][0]
                query_check = array_check(pcm_query, ref_query)
                sample_seed = 100000 * role_index + 10000 * cloud_index + sample_index
                indices, mask = padded_indices(ref_query, 15000, np.random.default_rng(sample_seed))
                absolute = ref_points[indices]
                ref_position = (absolute - pick).astype(np.float32)
                pcm_position = (np.asarray(cloud['sub_points'][indices]) - pick).astype(np.float32)
                ref_features = np.concatenate([
                    (ref_colors[indices] - COLOR_MEAN) / COLOR_STD,
                    absolute[:, 2:3]], axis=1)
                pcm_features = np.concatenate([
                    (np.asarray(cloud['sub_colors'][indices]) - COLOR_MEAN) / COLOR_STD,
                    np.asarray(cloud['sub_points'][indices])[:, 2:3]], axis=1)
                entry = {
                    'center_index': int(center_index), 'valid_points': int(mask.sum()),
                    'query_indices': query_check,
                    'local_position': array_check(pcm_position, ref_position, atol=1e-6),
                    'rgb_height_features': array_check(pcm_features, ref_features, atol=1e-6),
                    'labels': array_check(cloud['sub_labels'][indices], ref_labels[indices]),
                    'mask_cardinality': {'passed': int(mask.sum()) == min(len(ref_query), 15000),
                                         'value': int(mask.sum())},
                }
                entry['passed'] = all(value['passed'] for value in entry.values()
                                      if isinstance(value, dict) and 'passed' in value)
                sample_checks.append(entry)
            area_passed = all(value['passed'] for value in checks.values()) and all(
                value['passed'] for value in sample_checks)
            report['areas'].append({'role': role, 'area': area, 'checks': checks,
                                    'samples': sample_checks, 'passed': area_passed})
            report['passed'] &= area_passed
            print(role, area, 'PASS' if area_passed else 'FAIL')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('report:', args.output)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
