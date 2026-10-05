"""Scan PSNet5 schedules for compact-cloud hierarchy edge cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.dataset.psnet5.psnet5 import PSNet5Sphere
from openpoints.utils import EasyConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default='cfgs/psnet5/PCM.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', type=Path,
                        default=Path('analysis/psnet5/results/compaction_schedule_10epochs.json'))
    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    dataset_args = dict(cfg.dataset.common)
    dataset_args.update(dict(cfg.dataset.train))
    dataset_args['transform'] = None
    dataset = PSNet5Sphere(**dataset_args)
    sample_count = min(args.epochs * dataset.num_steps, len(dataset.cloud_inds))
    counts = np.empty(sample_count, dtype=np.int64)

    # Batch each area's radius queries. This checks the entire requested schedule
    # without materializing 15,000-point padded examples.
    for cloud_index, cloud in enumerate(dataset.clouds):
        schedule_indices = np.flatnonzero(dataset.cloud_inds[:sample_count] == cloud_index)
        picks = (cloud['sub_points'][dataset.point_inds[schedule_indices]] +
                 dataset.noise[schedule_indices])
        counts[schedule_indices] = cloud['tree'].query_radius(
            picks, r=dataset.in_radius, count_only=True)
    counts = np.minimum(counts, dataset.num_points)

    reducers = list(cfg.model.encoder_args.reducers)
    configured_k = list(cfg.model.encoder_args.k_neighbors)
    levels = [counts]
    for reducer in reducers:
        levels.append(np.maximum(1, levels[-1] // int(reducer)))

    order = np.argsort(counts)
    report = {
        'epochs': args.epochs,
        'samples': int(sample_count),
        'valid_points': {
            'min': int(counts.min()),
            'percentiles': {str(q): float(np.percentile(counts, q))
                            for q in (0, 0.1, 1, 5, 25, 50, 75, 100)},
        },
        'hierarchy': [
            {
                'stage': index + 1,
                'input_min': int(levels[index].min()),
                'output_min': int(levels[index + 1].min()),
                'configured_k': int(configured_k[index]),
                'samples_requiring_adaptive_k': int(
                    (levels[index] < int(configured_k[index])).sum()),
            }
            for index in range(len(reducers))
        ],
        'smallest_samples': [
            {
                'schedule_index': int(index),
                'epoch': int(index // dataset.num_steps),
                'step': int(index % dataset.num_steps),
                'valid_points': int(counts[index]),
                'levels': [int(level[index]) for level in levels],
                'cloud_index': int(dataset.cloud_inds[index]),
                'point_index': int(dataset.point_inds[index]),
            }
            for index in order[:20]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
