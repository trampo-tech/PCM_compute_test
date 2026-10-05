import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openpoints.dataset.data_util import get_features_by_keys
from openpoints.dataset.psnet5.preprocessing import load_raw_area
from openpoints.dataset.psnet5.psnet5 import PSNet5Sphere
from openpoints.dataset.psnet5.voting import add_sphere_votes, averaged_cloud_logits
from openpoints.loss.build import MaskedCrossEntropy
from openpoints.models.PCM.PointMLP_layers import (
    LocalGrouper, PointNetFeaturePropagation, PreExtraction)
from openpoints.models.layers import SingletonSafeBatchNorm1d
from openpoints.models.layers.norm import batch_stats_for_eval
from openpoints.models.segmentation.base_seg import BaseSeg


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_eval_batch_stats_preserves_running_buffers_and_dropout_mode():
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(p=0.9)).eval()
    x = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5)
    bn = model[0]
    before = (bn.running_mean.clone(), bn.running_var.clone(),
              bn.num_batches_tracked.clone())
    with batch_stats_for_eval(model):
        actual = model(x)
        assert not model[1].training
    expected = torch.nn.functional.batch_norm(
        x, None, None, bn.weight, bn.bias, True, 0.0, bn.eps)
    torch.testing.assert_close(actual, expected)
    assert not bn.training and bn.track_running_stats
    for current, original in zip((bn.running_mean, bn.running_var,
                                  bn.num_batches_tracked), before):
        torch.testing.assert_close(current, original)


def _fixture(tmp_path):
    root = tmp_path / 'PSNet5'
    processed = root / 'processed'
    processed.mkdir(parents=True)
    rng = np.random.default_rng(7)
    left = rng.normal(loc=(0, 0, 1), scale=0.15, size=(30, 3)).astype(np.float32)
    right = rng.normal(loc=(10, 0, 2), scale=0.15, size=(30, 3)).astype(np.float32)
    points = np.concatenate([left, right])
    colors = rng.integers(0, 256, size=(60, 3), dtype=np.uint8)
    labels = (np.arange(60) % 5).astype(np.int32)[:, None]
    source = processed / 'Area_1.pkl'
    with source.open('wb') as stream:
        pickle.dump((points, colors, labels), stream)

    split = {'version': 'psnet5-test-v1', 'training_areas': ['Area_1'],
             'final_evaluation': [], 'regions': [{'area': 'Area_1', 'axis': 'x',
             'threshold': 5.0, 'buffer_radius': 2.0}]}
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(split))
    partitions = tmp_path / 'partitions'
    partitions.mkdir()
    validation = points[:, 0] >= 5.0
    np.savez_compressed(partitions / 'Area_1_partition.npz',
                        validation_bits=np.packbits(validation, bitorder='big'),
                        point_count=len(points), source_sha256=_sha256(source),
                        split_version='psnet5-test-v1')
    return root, split_path, partitions


def test_annotation_parser_has_stable_five_class_mapping(tmp_path):
    root = tmp_path / 'raw'
    annotation_dir = root / 'Area_1' / 'Room_1' / 'Annotations'
    annotation_dir.mkdir(parents=True)
    for index, name in enumerate(('tank', 'pump', 'pipe', 'rectangularbeam', 'ibeam')):
        np.savetxt(annotation_dir / f'{name}_1.txt', np.array([[index, 0, 0, 10, 20, 30]]))
    _, _, labels = load_raw_area(root, 'Area_1')
    # Files are sorted; labels come from the fixed semantic mapping, not file order.
    assert labels.tolist() == [0, 1, 2, 3, 4]


def test_development_loader_contract_and_partitioning(tmp_path):
    root, split, partitions = _fixture(tmp_path)
    common = dict(voxel_size=0.1, in_radius=1.0, num_points=16, num_steps=2,
                  num_epochs=1, data_root=root, split_manifest=split,
                  partition_root=partitions, cache_dir=tmp_path / 'cache', seed=11)
    train = PSNet5Sphere(split='train', protocol='development', **common)
    val = PSNet5Sphere(split='val', protocol='development', **common)
    assert all(np.max(cloud['sub_points'][:, 0]) < 5.0 for cloud in train.clouds)
    assert all(np.min(cloud['sub_points'][:, 0]) >= 5.0 for cloud in val.clouds)
    for dataset, predicate in ((train, lambda x: x < 3.0), (val, lambda x: x >= 7.0)):
        for cloud_index, point_index, noise in zip(dataset.cloud_inds, dataset.point_inds, dataset.noise):
            jittered_x = dataset.clouds[int(cloud_index)]['sub_points'][int(point_index), 0] + noise[0]
            assert predicate(jittered_x)
    item = train[0]
    assert set(item) == {'pos', 'x', 'y', 'heights', 'mask', 'cloud_index', 'input_inds'}
    assert item['pos'].shape == (16, 3)
    assert item['x'].shape == (16, 3)
    assert item['heights'].shape == (16, 1)
    assert item['mask'].sum() > 0
    batch = {key: torch.as_tensor(value)[None] for key, value in item.items()}
    assert get_features_by_keys(batch, 'x,heights').shape == (1, 4, 16)


def test_masked_loss_and_votes_ignore_padding():
    logits = torch.tensor([[[8., -8., 99.], [-8., 8., -99.]]])
    target = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[1, 1, 0]])
    loss = MaskedCrossEntropy(label_smoothing=0.0)(logits, target, mask)
    assert loss.item() < 1e-5

    sums = [torch.zeros(3, 2)]
    counts = [torch.zeros(3)]
    add_sphere_votes(sums, counts, logits, torch.tensor([0]),
                     torch.tensor([[0, 1, 2]]), mask)
    averaged, covered = averaged_cloud_logits(sums[0], counts[0])
    assert covered.tolist() == [True, True, False]
    assert torch.equal(averaged[:2], logits[0, :, :2].T)


class _IdentityEncoder(nn.Module):
    def forward_seg_feat(self, data):
        return [data['pos']], [data['x']]


class _GlobalContextDecoder(nn.Module):
    """Toy decoder whose valid outputs are deliberately padding-sensitive."""
    def forward(self, positions, features):
        feature = features[0]
        return feature + feature.mean(dim=-1, keepdim=True)


def _toy_segmenter(mask_padding):
    # Avoid the config registry here: this isolates BaseSeg's masking contract.
    model = BaseSeg.__new__(BaseSeg)
    nn.Module.__init__(model)
    model.encoder = _IdentityEncoder()
    model.decoder = _GlobalContextDecoder()
    model.head = nn.Identity()
    model.test_crop = 100
    model.mask_padding = mask_padding
    return model.eval()


def test_mask_aware_model_is_invariant_to_padding_content_and_layout():
    valid_pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    valid_x = torch.tensor([[1., 2., 4.]])
    variants = []
    for valid_slots, padded_x in (([0, 1, 2], [10., 20.]), ([0, 2, 4], [-30., 50.])):
        mask = torch.zeros(1, 5, dtype=torch.int64)
        mask[0, valid_slots] = 1
        pos = torch.full((1, 5, 3), 99.)
        x = torch.tensor(padded_x + [0., 0., 0.]).reshape(1, 1, 5)
        pos[0, valid_slots] = valid_pos
        x[0, 0, valid_slots] = valid_x
        variants.append({'pos': pos, 'x': x, 'mask': mask})

    masked = _toy_segmenter(mask_padding=True)
    first = masked(variants[0])[0, :, variants[0]['mask'][0].bool()]
    second = masked(variants[1])[0, :, variants[1]['mask'][0].bool()]
    assert torch.equal(first, second)
    assert torch.count_nonzero(masked(variants[1])[0, :, ~variants[1]['mask'][0].bool()]) == 0

    legacy = _toy_segmenter(mask_padding=False)
    legacy_first = legacy(variants[0])[0, :, variants[0]['mask'][0].bool()]
    legacy_second = legacy(variants[1])[0, :, variants[1]['mask'][0].bool()]
    assert not torch.allclose(legacy_first, legacy_second)


def test_singleton_safe_batch_norm_supports_compacted_training():
    norm = SingletonSafeBatchNorm1d(3).train()
    singleton = torch.randn(1, 3, 1, requires_grad=True)
    output = norm(singleton)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert singleton.grad is not None


def test_pcm_grouping_and_extraction_accept_one_valid_point():
    xyz = torch.tensor([[[1., 2., 3.]]])
    features = torch.randn(1, 1, 4, requires_grad=True)
    grouper = LocalGrouper(4, sample_ratio=4, kneighbors=12,
                           use_xyz=True, normalize='anchor').train()
    new_xyz, grouped, residual = grouper(xyz, features, None)
    assert new_xyz.shape == (1, 1, 3)
    assert grouped.shape == (1, 1, 1, 11)
    assert residual is None

    extracted = PreExtraction(4, 8, blocks=1, use_xyz=True).train()(grouped)
    assert extracted.shape == (1, 8, 1)
    assert torch.isfinite(extracted).all()
    extracted.sum().backward()
    assert features.grad is not None


def test_feature_propagation_adapts_when_support_has_two_points():
    propagation = PointNetFeaturePropagation(6, 5, blocks=0).eval()
    dense_xyz = torch.randn(1, 4, 3)
    sparse_xyz = torch.randn(1, 2, 3)
    dense_features = torch.randn(1, 2, 4)
    sparse_features = torch.randn(1, 4, 2)
    output = propagation(dense_xyz, sparse_xyz, dense_features, sparse_features)
    assert output.shape == (1, 5, 4)
    assert torch.isfinite(output).all()
