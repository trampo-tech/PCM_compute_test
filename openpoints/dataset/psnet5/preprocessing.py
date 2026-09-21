"""Preprocessing and versioned cache IO for PSNet5.

The development protocol partitions raw points before voxelization.  Cache
directories therefore include the protocol, split version, role, and voxel
size; whole-area PSNet5 subsamples must never be reused for development data.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.neighbors import KDTree


LABEL_TO_NAME = {0: 'ibeam', 1: 'pipe', 2: 'pump', 3: 'rectangularbeam', 4: 'tank'}
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_raw_area(data_root: Path, area: str):
    """Load the audited raw cache, or parse a stable annotation tree."""
    cached = data_root / 'processed' / f'{area}.pkl'
    if cached.exists():
        with cached.open('rb') as stream:
            xyz, rgb, labels = pickle.load(stream)
        return (np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8),
                np.asarray(labels, dtype=np.int32).reshape(-1))

    area_dir = data_root / area
    if not area_dir.is_dir():
        raise FileNotFoundError(f'Neither {cached} nor annotation directory {area_dir} exists')
    xyzs, rgbs, labels = [], [], []
    for annotation in sorted(area_dir.glob('Room_*/Annotations/*.txt')):
        class_name = annotation.stem.split('_', 1)[0]
        if class_name not in NAME_TO_LABEL:
            raise ValueError(f'Unknown PSNet5 class in {annotation}: {class_name}')
        values = np.loadtxt(annotation, dtype=np.float32, ndmin=2)
        if values.shape[1] != 6 or not np.isfinite(values).all():
            raise ValueError(f'Expected finite XYZRGB rows in {annotation}, got {values.shape}')
        if np.any((values[:, 3:] < 0) | (values[:, 3:] > 255)):
            raise ValueError(f'RGB outside [0,255] in {annotation}')
        xyzs.append(values[:, :3])
        rgbs.append(values[:, 3:].astype(np.uint8))
        labels.append(np.full(len(values), NAME_TO_LABEL[class_name], dtype=np.int32))
    if not xyzs:
        raise ValueError(f'No annotations found below {area_dir}')
    return np.concatenate(xyzs), np.concatenate(rgbs), np.concatenate(labels)


def _numpy_grid_subsample(points, colors, labels, voxel_size):
    """Reference-compatible barycenters, mean colors, and majority labels.

    This portable implementation is intended for fixtures and installations
    without the reference extension. Large PSNet5 preprocessing should use the
    compiled backend through ``reference_ops_root``.
    """
    voxels = np.floor(points / voxel_size).astype(np.int64)
    order = np.lexsort((voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_voxels, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)
    out_points = np.add.reduceat(points[order], starts, axis=0) / counts[:, None]
    out_colors = np.add.reduceat(colors[order].astype(np.float32), starts, axis=0) / counts[:, None]
    out_labels = np.empty(len(starts), dtype=np.int32)
    for index, (start, stop) in enumerate(zip(starts, np.r_[starts[1:], len(order)])):
        out_labels[index] = np.bincount(labels[order[start:stop]], minlength=5).argmax()
    return out_points.astype(np.float32), (out_colors / 255.0).astype(np.float32), out_labels


def grid_subsample(points, colors, labels, voxel_size, reference_ops_root=None):
    if voxel_size <= 0:
        return points, colors.astype(np.float32) / 255.0, labels
    if reference_ops_root:
        ops_root = str(Path(reference_ops_root).resolve())
        if ops_root not in sys.path:
            sys.path.insert(0, ops_root)
        try:
            try:
                import grid_subsampling as cpp_subsampling
            except ImportError:
                import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
        except (ImportError, OSError) as error:
            raise ImportError(
                f'Could not import reference grid subsampling from {ops_root}. '
                'Compile the PCM/ResPointNet2 extension or omit reference_ops_root only for small fixtures.'
            ) from error
        sub_points, sub_colors, sub_labels = cpp_subsampling.compute(
            points, features=colors, classes=labels.reshape(-1, 1),
            sampleDl=voxel_size, verbose=0)
        return (np.asarray(sub_points, dtype=np.float32),
                np.asarray(sub_colors, dtype=np.float32) / 255.0,
                np.asarray(sub_labels, dtype=np.int32).reshape(-1))
    return _numpy_grid_subsample(points, colors, labels, voxel_size)


def _atomic_save(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, array)
    os.replace(temporary, path)
    path.chmod(0o644)


def _partition_mask(area, points, role, protocol, split, regions, partition_root, source_path,
                    split_version, verify_source_digest):
    if protocol == 'benchmark':
        expected = {'train': {'Area_1', 'Area_2', 'Area_4'}, 'test': {'Area_3'}}
        if area not in expected[split]:
            raise ValueError(f'{area} is forbidden in benchmark {split}')
        return np.ones(len(points), dtype=bool)
    region = regions[area]
    packed_path = partition_root / f'{area}_partition.npz'
    with np.load(packed_path, allow_pickle=False) as packed:
        if str(packed['split_version']) != split_version:
            raise ValueError(f'Split version mismatch in {packed_path}')
        if int(packed['point_count']) != len(points):
            raise ValueError(f'Point count mismatch in {packed_path}')
        if verify_source_digest and str(packed['source_sha256']) != sha256_file(source_path):
            raise ValueError(f'Source digest mismatch for {source_path}')
        validation = np.unpackbits(packed['validation_bits'], count=len(points), bitorder='big').astype(bool)
    return ~validation if role == 'train' else validation


def prepare_cache(data_root, cache_root, split_manifest, partition_root, protocol, role,
                  voxel_size, reference_ops_root=None, verify_source_digest=True):
    data_root, cache_root = Path(data_root), Path(cache_root)
    manifest_path = Path(split_manifest)
    split_spec = json.loads(manifest_path.read_text())
    split_version = split_spec['version'] if protocol == 'development' else 'benchmark-area-v1'
    cache_dir = cache_root / f'psnet5sphere_{protocol}_{split_version}_{role}_v{voxel_size:.3f}'
    ready = cache_dir / 'manifest.json'
    if ready.exists():
        return cache_dir, json.loads(ready.read_text())

    if protocol not in {'development', 'benchmark'}:
        raise ValueError(f'Unknown protocol: {protocol}')
    if protocol == 'development' and role not in {'train', 'val'}:
        raise ValueError('Development protocol only supports train/val')
    if protocol == 'benchmark' and role not in {'train', 'test'}:
        raise ValueError('Benchmark protocol only supports train/test')
    areas = split_spec['training_areas'] if protocol == 'development' or role == 'train' else split_spec['final_evaluation']
    regions = {entry['area']: entry for entry in split_spec['regions']}
    partition_root = Path(partition_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{cache_dir.name}.', dir=cache_root))
    temporary.chmod(0o755)
    records = []
    try:
        for cloud_index, area in enumerate(areas):
            source = data_root / 'processed' / f'{area}.pkl'
            points, colors, labels = load_raw_area(data_root, area)
            owned = _partition_mask(area, points, role, protocol, role, regions, partition_root,
                                    source, split_version, verify_source_digest)
            raw_points, raw_colors, raw_labels = points[owned], colors[owned], labels[owned]
            sub_points, sub_colors, sub_labels = grid_subsample(
                raw_points, raw_colors, raw_labels, voxel_size, reference_ops_root)
            region = regions.get(area)
            if protocol == 'development':
                axis = {'x': 0, 'y': 1, 'z': 2}[region['axis']]
                threshold, buffer_radius = region['threshold'], region['buffer_radius']
                eligible = (sub_points[:, axis] < threshold - buffer_radius) if role == 'train' \
                    else (sub_points[:, axis] >= threshold + buffer_radius)
            else:
                axis, threshold, buffer_radius = -1, 0.0, 0.0
                eligible = np.ones(len(sub_points), dtype=bool)
            if not eligible.any():
                raise ValueError(f'No eligible centers for {area}/{role}')
            prefix = temporary / f'cloud_{cloud_index}'
            _atomic_save(prefix.with_name(prefix.name + '_sub_points.npy'), sub_points)
            _atomic_save(prefix.with_name(prefix.name + '_sub_colors.npy'), sub_colors)
            _atomic_save(prefix.with_name(prefix.name + '_sub_labels.npy'), sub_labels)
            _atomic_save(prefix.with_name(prefix.name + '_eligible.npy'), eligible)
            record = {'area': area, 'sub_points': len(sub_points), 'owned_raw_points': len(raw_points),
                      'eligible_centers': int(eligible.sum()), 'axis': axis,
                      'threshold': threshold, 'buffer_radius': buffer_radius}
            if role in {'val', 'test'}:
                tree = KDTree(sub_points, leaf_size=50)
                projection = tree.query(raw_points, return_distance=False).reshape(-1).astype(np.int32)
                _atomic_save(prefix.with_name(prefix.name + '_raw_points.npy'), raw_points)
                _atomic_save(prefix.with_name(prefix.name + '_raw_colors.npy'), raw_colors)
                _atomic_save(prefix.with_name(prefix.name + '_raw_labels.npy'), raw_labels)
                _atomic_save(prefix.with_name(prefix.name + '_projection.npy'), projection)
            records.append(record)
            del points, colors, labels, owned, raw_points, raw_colors, raw_labels
        output = {'dataset': 'PSNet5Sphere', 'protocol': protocol, 'split_version': split_version,
                  'role': role, 'voxel_size': voxel_size, 'clouds': records}
        temporary_manifest = temporary / 'manifest.json'
        temporary_manifest.write_text(json.dumps(output, indent=2) + '\n')
        temporary_manifest.chmod(0o644)
        try:
            os.replace(temporary, cache_dir)
        except FileExistsError:
            shutil.rmtree(temporary)
        return cache_dir, json.loads(ready.read_text())
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_cache(cache_dir, manifest, mmap_mode='r'):
    cache_dir = Path(cache_dir)
    clouds = []
    for index, record in enumerate(manifest['clouds']):
        prefix = cache_dir / f'cloud_{index}'
        cloud = dict(record)
        for name in ('sub_points', 'sub_colors', 'sub_labels', 'eligible'):
            cloud[name] = np.load(prefix.with_name(prefix.name + f'_{name}.npy'), mmap_mode=mmap_mode)
        if manifest['role'] in {'val', 'test'}:
            for name in ('raw_points', 'raw_colors', 'raw_labels', 'projection'):
                cloud[name] = np.load(prefix.with_name(prefix.name + f'_{name}.npy'), mmap_mode=mmap_mode)
        cloud['tree'] = KDTree(cloud['sub_points'], leaf_size=50)
        clouds.append(cloud)
    return clouds
