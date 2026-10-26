"""Small, reproducible PSNet5 checkpoint diagnostic; no optimizer steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from openpoints.dataset.psnet5.psnet5 import PSNet5Sphere
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig


def confusion(labels, predictions):
    return np.bincount(labels * 5 + predictions, minlength=25).reshape(5, 5)


def metrics(matrix):
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    union = support + predicted - np.diag(matrix)
    iou = np.divide(np.diag(matrix), union, out=np.zeros(5), where=union > 0)
    return {'miou': float(iou.mean()), 'oa': float(np.trace(matrix) / max(matrix.sum(), 1)),
            'iou': iou.tolist(), 'support': support.tolist(), 'predicted': predicted.tolist(),
            'confusion': matrix.tolist()}


def input_from_sample(sample, device):
    colors = torch.from_numpy(np.array(sample['x'], copy=True)).float()
    mean = torch.tensor([0.5136457, 0.49523646, 0.44921124])
    std = torch.tensor([0.18308958, 0.18415008, 0.19252081])
    colors = (colors - mean) / std
    height = torch.from_numpy(np.array(sample['heights'], copy=True)).float()
    features = torch.cat((colors, height), dim=-1).T.unsqueeze(0).contiguous()
    return {'pos': torch.from_numpy(np.array(sample['pos'], copy=True)).float().unsqueeze(0).to(device),
            'x': features.to(device),
            'mask': torch.from_numpy(np.array(sample['mask'], copy=True)).unsqueeze(0).to(device)}


def bn_buffers(model):
    return [(module, module.running_mean.clone(), module.running_var.clone(),
             module.num_batches_tracked.clone()) for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats]


def restore_bn(saved):
    for module, mean, var, count in saved:
        module.running_mean.copy_(mean)
        module.running_var.copy_(var)
        module.num_batches_tracked.copy_(count)


@torch.no_grad()
def predict(model, data, mode):
    saved = bn_buffers(model)
    model.eval()
    if mode == 'batch_stats':
        for module, *_ in saved:
            module.train()
    elif mode == 'train_mode':
        model.train()
    logits = model(data)
    restore_bn(saved)
    return logits.argmax(1)[0].cpu().numpy()


def area_stats(dataset):
    output = []
    for cloud in dataset.clouds:
        labels = np.asarray(cloud['sub_labels'])
        xyz = np.asarray(cloud['sub_points'])
        rgb = np.asarray(cloud['sub_colors'])
        output.append({'area': cloud['area'], 'subsampled_points': int(len(labels)),
                       'class_counts': np.bincount(labels, minlength=5).tolist(),
                       'z_min_max': [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
                       'rgb_mean': rgb.mean(0).tolist()})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/psnet5/PCM.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datasets = {split: PSNet5Sphere(**dict(cfg.dataset.common), **dict(cfg.dataset[split]))
                for split in ('train', 'val')}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = build_model_from_cfg(cfg.model)
    state = checkpoint['model']
    if next(iter(state)).startswith('module.'):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    report = {'checkpoint_epoch': int(checkpoint['epoch']), 'device': str(device),
              'area_stats': {split: area_stats(dataset) for split, dataset in datasets.items()},
              'spheres': []}
    for split, dataset in datasets.items():
        for area_index, cloud in enumerate(dataset.clouds):
            indices = np.flatnonzero(dataset.cloud_inds[:dataset.num_steps] == area_index)
            chosen = None
            for index in indices[:20]:
                sample = dataset[int(index)]
                valid_count = int(sample['mask'].sum())
                if valid_count >= 1000:
                    chosen = (int(index), sample, valid_count)
                    break
            if chosen is None:
                raise RuntimeError(f'No sphere with >=1000 valid points in {split}/{cloud["area"]}')
            index, sample, valid_count = chosen
            data = input_from_sample(sample, device)
            labels = np.asarray(sample['y'])[:valid_count]
            row = {'split': split, 'area': cloud['area'], 'sample_index': index,
                   'valid_points': valid_count, 'modes': {}}
            for mode in ('eval', 'batch_stats', 'train_mode'):
                predictions = predict(model, data, mode)[:valid_count]
                row['modes'][mode] = metrics(confusion(labels, predictions))
            report['spheres'].append(row)
            print(f'{split}/{cloud["area"]}: {valid_count} points; ' + ', '.join(
                f'{mode}={row["modes"][mode]["miou"]:.3f}' for mode in row['modes']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
