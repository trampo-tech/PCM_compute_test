from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from ..build import DATASETS
from .preprocessing import LABEL_TO_NAME, load_cache, prepare_cache
from .sampling import make_schedule


@DATASETS.register_module()
class PSNet5Sphere(data.Dataset):
    label_to_names = LABEL_TO_NAME
    name_to_label = {name: label for label, name in LABEL_TO_NAME.items()}
    classes = list(LABEL_TO_NAME.values())
    num_classes = 5
    gravity_dim = 2
    cmap = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25],
                     [0, 130, 200], [145, 30, 180]], dtype=np.uint8)

    def __init__(self, voxel_size, in_radius, num_points, num_steps, num_epochs,
                 data_root, split_manifest, partition_root, cache_dir,
                 protocol='development', split='train', transform=None, seed=20260919,
                 height_mode='absolute', reference_ops_root=None,
                 verify_source_digest=True, **kwargs):
        super().__init__()
        self.epoch = 0
        self.transform = transform
        self.voxel_size = float(voxel_size)
        self.in_radius = float(in_radius)
        self.num_points = int(num_points)
        self.num_steps = int(num_steps)
        self.num_epochs = int(num_epochs)
        self.seed = int(seed)
        self.height_mode = height_mode
        self.split = split
        self.protocol = protocol
        if height_mode not in {'absolute', 'local', 'area_relative'}:
            raise ValueError(f'Unknown height_mode: {height_mode}')
        cache_path, manifest = prepare_cache(
            data_root=data_root, cache_root=cache_dir, split_manifest=split_manifest,
            partition_root=partition_root, protocol=protocol, role=split,
            voxel_size=self.voxel_size, reference_ops_root=reference_ops_root,
            verify_source_digest=verify_source_digest)
        self.clouds = load_cache(cache_path, manifest)
        for cloud in self.clouds:
            cloud['role'] = split
        schedule = Path(cache_path) / (
            f'schedule_r{self.in_radius:g}_n{self.num_points}_e{self.num_epochs}_s{self.num_steps}_seed{self.seed}.npz')
        self.cloud_inds, self.point_inds, self.noise = make_schedule(
            self.clouds, self.num_epochs, self.num_steps, self.in_radius,
            self.num_points, self.seed, schedule)
        self.sub_clouds_points = [cloud['sub_points'] for cloud in self.clouds]
        self.sub_clouds_points_labels = [cloud['sub_labels'] for cloud in self.clouds]
        if split in {'val', 'test'}:
            self.clouds_points = [cloud['raw_points'] for cloud in self.clouds]
            self.clouds_points_colors = [cloud['raw_colors'] for cloud in self.clouds]
            self.clouds_points_labels = [cloud['raw_labels'] for cloud in self.clouds]
            self.projections = [cloud['projection'] for cloud in self.clouds]
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for cloud in self.clouds:
            counts += np.bincount(np.asarray(cloud['sub_labels']), minlength=self.num_classes)
        self.num_per_class = counts

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        schedule_index = idx + self.epoch * self.num_steps
        if schedule_index >= len(self.cloud_inds):
            raise IndexError(f'epoch {self.epoch} exceeds configured num_epochs={self.num_epochs}')
        cloud_index = int(self.cloud_inds[schedule_index])
        point_index = int(self.point_inds[schedule_index])
        cloud = self.clouds[cloud_index]
        points = cloud['sub_points']
        pick = np.asarray(points[point_index]) + self.noise[schedule_index]
        query = cloud['tree'].query_radius(pick.reshape(1, -1), r=self.in_radius,
                                           return_distance=True, sort_results=True)[0][0]
        query = query[:self.num_points]
        if not len(query):
            raise RuntimeError('Scheduled PSNet5 neighborhood is empty')
        rng = np.random.default_rng(self.seed + schedule_index)
        query = query[rng.permutation(len(query))]
        valid_count = len(query)
        if valid_count < self.num_points:
            padding = rng.choice(query, self.num_points - valid_count, replace=True)
            input_inds = np.concatenate([query, padding])
        else:
            input_inds = query
        mask = np.zeros(self.num_points, dtype=np.int32)
        mask[:valid_count] = 1
        absolute = np.asarray(points[input_inds], dtype=np.float32)
        local = absolute - pick.astype(np.float32)
        if self.height_mode == 'absolute':
            heights = absolute[:, 2:3]
        elif self.height_mode == 'local':
            heights = local[:, 2:3]
        else:
            heights = absolute[:, 2:3] - float(np.min(cloud['sub_points'][:, 2]))
        output = {
            'pos': local.astype(np.float32),
            'x': np.asarray(cloud['sub_colors'][input_inds], dtype=np.float32),
            'y': np.asarray(cloud['sub_labels'][input_inds], dtype=np.int64),
            'heights': heights.astype(np.float32),
            'mask': mask,
            'cloud_index': np.asarray(cloud_index, dtype=np.int64),
            'input_inds': input_inds.astype(np.int64),
        }
        if self.transform is not None:
            output = self.transform(output)
        return output
