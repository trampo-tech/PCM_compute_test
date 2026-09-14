"""Validate the fixed split and materialize raw-point ownership masks.

Packed validation masks index the verified Area_N.pkl raw point order.
False = training, True = validation. Area 3 is exclusively test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import load, counts, preview, dump

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).parent
out = root / 'results'
manifest = json.loads((root / 'split.json').read_text())
stats = {'version': manifest['version'], 'regions': []}
for r in manifest['regions']:
    area = r['area']
    axis = ['x', 'y'].index(r['axis'])
    threshold = r['threshold']
    source = args.data_root / 'processed' / f'{area}.pkl'
    xyz, rgb, labels = load(source)
    labels = labels.reshape(-1)
    val = xyz[:, axis] >= threshold
    record = {'area': area, 'raw_training_counts': counts(labels[~val]),
              'raw_validation_counts': counts(labels[val]), 'raw_points': len(xyz)}
    # Bind masks to exact cache bytes to detect changed data/order before use.
    h = hashlib.sha256()
    with source.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    np.savez_compressed(out / f'{area}_partition.npz', validation_bits=np.packbits(val),
                        point_count=len(val), bitorder='big', source_sha256=h.hexdigest(),
                        split_version=manifest['version'])
    with np.load(out / f'{area}_partition.npz') as saved:
        restored = np.unpackbits(saved['validation_bits'], count=int(saved['point_count'])).astype(bool)
        assert np.array_equal(restored, val)
    assert sum(record['raw_training_counts']) + sum(record['raw_validation_counts']) == len(xyz)
    del xyz, rgb, labels, val, restored
    p, c, y, tree = load(args.data_root / 'processed' / f'{area}_0.040_sub.pkl')
    y = np.asarray(y).reshape(-1)
    train_center = p[:, axis] < threshold-2
    val_center = p[:, axis] >= threshold+2
    assert not np.any(train_center & val_center)
    assert counts(y[train_center]) == r['eligible_training_center_class_counts']
    assert counts(y[val_center]) == r['eligible_validation_center_class_counts']
    present = np.array(counts(y)) > 0
    assert np.all(np.array(counts(y[train_center]))[present] >= 100)
    assert np.all(np.array(counts(y[val_center]))[present] >= 100)
    record['checks'] = 'raw mask round-trip, exhaustive ownership and eligible-center class coverage passed'
    stats['regions'].append(record)
    preview(p, y, c, out / f'{area}_split.png',
            f'{area}: spatial split; dashed boundary, dotted center exclusion limits',
            np.random.default_rng(20260912), boundary=(axis, threshold))
    print(area, record, flush=True)
dump(out / 'split_validation.json', stats)
