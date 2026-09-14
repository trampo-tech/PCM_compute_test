"""Recompute baseline voxel caches and compare XYZ, RGB and labels without overwriting."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-root', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/subsampling_verification.json')
args = parser.parse_args()
sys.path.insert(0, str(args.baseline_root.resolve() / 'ops'))
from cpp_wrappers.cpp_subsampling import grid_subsampling

results = []
for i in range(1, 5):
    area = f'Area_{i}'
    root = args.baseline_root / 'data/PSNet/PSNet5/processed'
    with (root / f'{area}.pkl').open('rb') as stream:
        xyz, rgb, labels = pickle.load(stream)
    with (root / f'{area}_0.040_sub.pkl').open('rb') as stream:
        points, colors, y, tree = pickle.load(stream)
    print(f'{area}: recomputing baseline subsampling', flush=True)
    p, c, target = grid_subsampling.compute(xyz, features=rgb, classes=labels, sampleDl=.04, verbose=0)
    c /= 255.0
    # Hash-map traversal can change output order; compare sorted coordinates.
    left = np.lexsort(p.T)
    right = np.lexsort(points.T)
    record = {'area': area, 'point_count_match': len(p) == len(points)}
    for name, a, b in [('xyz', p, points), ('rgb', c, colors), ('labels', target.reshape(-1), np.asarray(y).reshape(-1))]:
        record[name + '_exact_match'] = bool(np.array_equal(a[left], b[right]))
    results.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(record, flush=True)
    del xyz, rgb, labels, points, colors, y, tree, p, c, target
