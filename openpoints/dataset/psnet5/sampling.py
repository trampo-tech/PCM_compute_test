"""Deterministic coverage-potential schedules for PSNet5 spheres."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def _valid_jitter(rng, point, cloud, radius):
    for _ in range(100):
        noise = rng.normal(scale=radius / 10.0, size=(3,)).astype(np.float32)
        pick = point + noise
        axis = int(cloud['axis'])
        if axis < 0:
            return noise
        boundary = float(cloud['threshold'])
        buffer_radius = float(cloud['buffer_radius'])
        if cloud['role'] == 'train' and pick[axis] < boundary - buffer_radius:
            return noise
        if cloud['role'] != 'train' and pick[axis] >= boundary + buffer_radius:
            return noise
    raise RuntimeError('Could not draw a center jitter that respects the split buffer')


def make_schedule(clouds, num_epochs, num_steps, radius, num_points, seed, path):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data['cloud_indices'], data['point_indices'], data['noise']
    rng = np.random.default_rng(seed)
    potentials, eligible_indices = [], []
    for cloud in clouds:
        eligible = np.flatnonzero(np.asarray(cloud['eligible']))
        eligible_indices.append(eligible)
        potentials.append(rng.random(len(cloud['sub_points'])) * 1e-3)
    cloud_indices = np.empty(num_epochs * num_steps, dtype=np.int16)
    point_indices = np.empty(num_epochs * num_steps, dtype=np.int32)
    noise_values = np.empty((num_epochs * num_steps, 3), dtype=np.float32)
    for schedule_index in range(len(cloud_indices)):
        minimums = [potential[eligible].min() for potential, eligible in zip(potentials, eligible_indices)]
        cloud_index = int(np.argmin(minimums))
        eligible = eligible_indices[cloud_index]
        point_index = int(eligible[np.argmin(potentials[cloud_index][eligible])])
        cloud = clouds[cloud_index]
        point = np.asarray(cloud['sub_points'][point_index])
        noise = _valid_jitter(rng, point, cloud, radius)
        pick = point + noise
        query = cloud['tree'].query_radius(pick.reshape(1, -1), r=radius,
                                           return_distance=True, sort_results=True)[0][0]
        query = query[:num_points]
        if not len(query):
            raise RuntimeError('Eligible center produced an empty neighborhood')
        squared_distance = np.sum((np.asarray(cloud['sub_points'][query]) - pick) ** 2, axis=1)
        tukey = np.square(1.0 - squared_distance / radius ** 2)
        potentials[cloud_index][query] += tukey
        cloud_indices[schedule_index] = cloud_index
        point_indices[schedule_index] = point_index
        noise_values[schedule_index] = noise
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, cloud_indices=cloud_indices, point_indices=point_indices, noise=noise_values)
    os.replace(temporary, path)
    path.chmod(0o644)
    return cloud_indices, point_indices, noise_values
