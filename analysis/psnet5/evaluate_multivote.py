"""Multi-schedule PSNet5 checkpoint evaluation with error-analysis exports."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.dataset.psnet5.voting import add_sphere_votes, averaged_cloud_logits
from openpoints.models import build_model_from_cfg
from openpoints.models.layers.norm import batch_stats_for_eval
from openpoints.utils import EasyConfig, set_random_seed


CLASS_NAMES = ('ibeam', 'pipe', 'pump', 'rectangularbeam', 'tank')


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = support + predicted - true_positive
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    return {
        'oa': float(true_positive.sum() / max(confusion.sum(), 1)),
        'macc': float(recall[support > 0].mean()),
        'miou': float(iou[union > 0].mean()),
        'per_class': [
            {'class': name, 'support': int(support[index]), 'predicted': int(predicted[index]),
             'precision': float(precision[index]), 'recall': float(recall[index]), 'iou': float(iou[index])}
            for index, name in enumerate(CLASS_NAMES)
        ],
    }


def confusion_from_predictions(labels, predictions, num_classes=5):
    encoded = labels.astype(np.int64) * num_classes + predictions.astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def write_confusion(path, confusion):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ground_truth\\prediction', *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, confusion):
            writer.writerow([name, *row.tolist()])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['model']
    if state and next(iter(state)).startswith('module.'):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return {'epoch': int(checkpoint.get('epoch', -1)), 'best_val': float(checkpoint.get('best_val', np.nan))}


def projected_results(dataset, logit_sums, vote_counts):
    areas, global_confusion = [], np.zeros((5, 5), dtype=np.int64)
    for cloud_index, cloud in enumerate(dataset.clouds):
        averaged, covered = averaged_cloud_logits(logit_sums[cloud_index], vote_counts[cloud_index])
        prediction = averaged.argmax(dim=1)
        projection = torch.as_tensor(np.asarray(dataset.projections[cloud_index]),
                                     device=prediction.device, dtype=torch.long)
        labels = np.asarray(dataset.clouds_points_labels[cloud_index], dtype=np.int64).reshape(-1)
        raw_covered = covered[projection].cpu().numpy()
        raw_prediction = prediction[projection].cpu().numpy().astype(np.uint8)
        confusion = confusion_from_predictions(labels[raw_covered], raw_prediction[raw_covered])
        global_confusion += confusion
        areas.append({
            'name': cloud['area'], 'confusion': confusion,
            'subsampled_coverage': float(covered.float().mean().item()),
            'raw_coverage': float(raw_covered.mean()),
            'raw_prediction': raw_prediction, 'raw_labels': labels.astype(np.uint8),
            'raw_covered': raw_covered,
        })
    return global_confusion, areas


def write_class_rows(path, vote, scope, metrics, mode='a'):
    exists = path.exists() and mode == 'a'
    with path.open(mode, newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'vote', 'scope', 'class', 'support', 'predicted', 'precision', 'recall', 'iou'])
        if not exists or mode == 'w':
            writer.writeheader()
        for row in metrics['per_class']:
            writer.writerow({'vote': vote, 'scope': scope, **row})


def write_vote_rows(path, rows):
    """Persist convergence data after every completed vote."""
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_boundary_analysis(path, dataset, area_results):
    bins = [(0, 2), (2, 5), (5, 10), (10, np.inf)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'area', 'distance_min', 'distance_max', 'covered_points', 'accuracy', 'miou'])
        writer.writeheader()
        for cloud, result in zip(dataset.clouds, area_results):
            axis = int(cloud['axis'])
            if axis < 0:
                continue
            coordinates = np.asarray(cloud['raw_points'])[:, axis]
            distance = np.abs(coordinates - float(cloud['threshold']))
            for lower, upper in bins:
                selected = result['raw_covered'] & (distance >= lower) & (distance < upper)
                if not selected.any():
                    continue
                confusion = confusion_from_predictions(
                    result['raw_labels'][selected], result['raw_prediction'][selected])
                metrics = metrics_from_confusion(confusion)
                writer.writerow({'area': result['name'], 'distance_min': lower,
                                 'distance_max': upper, 'covered_points': int(selected.sum()),
                                 'accuracy': metrics['oa'], 'miou': metrics['miou']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default='cfgs/psnet5/PCM.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--votes', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Match the training run seed and cuDNN settings for replay')
    parser.add_argument('--benchmark-area3', action='store_true',
                        help='Evaluate the original PSNet5 Area_3 test split')
    parser.add_argument('--bn-batch-stats', action='store_true',
                        help='Diagnostic: use current-sphere BatchNorm statistics in evaluation')
    parser.add_argument('--norm-mode', choices=('batch', 'group'), default=None,
                        help='Match the checkpoint model normalization mode')
    parser.add_argument('--norm-groups', type=int, default=8,
                        help='Maximum GroupNorm groups when --norm-mode group')
    parser.add_argument('--recalibrate-bn-steps', type=int, default=0,
                        help='Update BN running statistics using this many unaugmented train spheres')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    if args.norm_mode is not None:
        cfg.model.norm_mode = args.norm_mode
        if args.norm_mode == 'group':
            cfg.model.norm_groups = args.norm_groups
            cfg.eval_bn_batch_stats = False
    if args.benchmark_area3:
        cfg.dataset.common.protocol = 'benchmark'
        cfg.dataset.val.split = 'test'
    if args.seed is not None:
        set_random_seed(args.seed)
    cfg.dataset.val.num_epochs = args.votes
    if args.num_steps is not None:
        cfg.dataset.val.num_steps = args.num_steps
    cfg.dataloader.num_workers = min(int(cfg.dataloader.num_workers), 4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PCM multi-vote evaluation')
    device = torch.device('cuda:0')
    loader = build_dataloader_from_cfg(
        cfg.val_batch_size, cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms, split='val', distributed=False)
    model = build_model_from_cfg(cfg.model)
    checkpoint_info = load_checkpoint(model, args.checkpoint, device)
    if args.recalibrate_bn_steps:
        train_transforms = cfg.datatransforms.train
        cfg.datatransforms.train = cfg.datatransforms.val
        train_loader = build_dataloader_from_cfg(
            1, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms,
            split='train', distributed=False)
        cfg.datatransforms.train = train_transforms
        bn_modules = [module for module in model.modules()
                      if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        momenta = [module.momentum for module in bn_modules]
        model.eval()
        for module in bn_modules:
            module.momentum = 0.01
            module.train()
        with torch.no_grad():
            for step, data in enumerate(tqdm(train_loader, desc='BN recalibration')):
                if step >= args.recalibrate_bn_steps:
                    break
                for key in tuple(data):
                    data[key] = data[key].to(device, non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                model(data)
        for module, momentum in zip(bn_modules, momenta):
            module.momentum = momentum
        model.eval()
    use_batch_stats = args.bn_batch_stats or bool(cfg.get('eval_bn_batch_stats', False))
    dataset = loader.dataset
    if args.benchmark_area3 and [cloud['area'] for cloud in dataset.clouds] != ['Area_3']:
        raise RuntimeError('Benchmark evaluation must contain Area_3 only')
    logit_sums = [torch.zeros((len(points), cfg.num_classes), device=device, dtype=torch.float32)
                  for points in dataset.sub_clouds_points]
    vote_counts = [torch.zeros(len(points), device=device, dtype=torch.float32)
                   for points in dataset.sub_clouds_points]
    vote_rows, first_vote = [], None
    class_path = args.output / 'per_class_metrics.csv'
    vote_path = args.output / 'vote_metrics.csv'
    if class_path.exists():
        class_path.unlink()

    for vote in range(args.votes):
        dataset.epoch = vote
        for data in tqdm(loader, desc=f'Vote {vote + 1}/{args.votes}'):
            for key in tuple(data):
                data[key] = data[key].to(device, non_blocking=True)
            data['x'] = get_features_by_keys(data, cfg.feature_keys)
            data['epoch'] = checkpoint_info['epoch']
            data['iter'] = -1
            with torch.no_grad():
                with batch_stats_for_eval(model) if use_batch_stats else nullcontext():
                    logits = model(data).float()
            add_sphere_votes(logit_sums, vote_counts, logits, data['cloud_index'],
                             data['input_inds'], data['mask'])
        confusion, areas = projected_results(dataset, logit_sums, vote_counts)
        metrics = metrics_from_confusion(confusion)
        row = {'vote': vote + 1, 'miou': metrics['miou'], 'macc': metrics['macc'], 'oa': metrics['oa']}
        for area in areas:
            row[f"{area['name']}_subsampled_coverage"] = area['subsampled_coverage']
            row[f"{area['name']}_raw_coverage"] = area['raw_coverage']
            write_class_rows(class_path, vote + 1, area['name'], metrics_from_confusion(area['confusion']))
        vote_rows.append(row)
        write_vote_rows(vote_path, vote_rows)
        write_class_rows(class_path, vote + 1, 'global', metrics)
        write_confusion(args.output / f'confusion_vote_{vote + 1:02d}.csv', confusion)
        if vote == 0:
            first_vote = {
                'metrics': metrics,
                'areas': [{key: area[key] for key in ('name', 'raw_prediction', 'raw_covered')}
                          for area in areas],
            }
        logging.info('vote=%d mIoU=%.2f mAcc=%.2f OA=%.2f coverage=%s',
                     vote + 1, 100 * metrics['miou'], 100 * metrics['macc'], 100 * metrics['oa'],
                     ', '.join(f"{area['name']}:{100 * area['raw_coverage']:.2f}%" for area in areas))

    final_confusion, final_areas = projected_results(dataset, logit_sums, vote_counts)
    # ResPointNet2's seg_metrics scores every raw point after projection.
    # Unvisited subsampled points retain zero logits, whose argmax is class 0.
    # Keep this separate from our coverage-aware development metric so the
    # effect of incomplete sphere coverage is visible.
    reference_confusion = np.zeros((5, 5), dtype=np.int64)
    for area in final_areas:
        reference_confusion += confusion_from_predictions(
            area['raw_labels'], area['raw_prediction'])
    write_boundary_analysis(args.output / 'boundary_metrics.csv', dataset, final_areas)
    for cloud_index, area in enumerate(final_areas):
        np.savez_compressed(
            args.output / f"{area['name']}_predictions.npz",
            prediction=area['raw_prediction'], label=area['raw_labels'], covered=area['raw_covered'],
            projection=np.asarray(dataset.projections[cloud_index], dtype=np.int32),
            subsampled_vote_count=vote_counts[cloud_index].cpu().numpy().astype(np.uint16))

    common_confusion = np.zeros((5, 5), dtype=np.int64)
    for first, final in zip(first_vote['areas'], final_areas):
        support = first['raw_covered'] & final['raw_covered']
        common_confusion += confusion_from_predictions(
            final['raw_labels'][support], final['raw_prediction'][support])
    summary = {
        'checkpoint': str(args.checkpoint), 'checkpoint_info': checkpoint_info,
        'bn_mode': ('group_norm' if cfg.model.get('norm_mode') == 'group' else
                    'current_batch' if use_batch_stats else
                    f'recalibrated_{args.recalibrate_bn_steps}_train_spheres'
                    if args.recalibrate_bn_steps else 'checkpoint_running'),
        'evaluation_protocol': dataset.protocol, 'evaluation_split': dataset.split,
        'votes': args.votes, 'num_steps_per_vote': len(dataset),
        'one_vote': first_vote['metrics'], 'final': metrics_from_confusion(final_confusion),
        'reference_full_raw': metrics_from_confusion(reference_confusion),
        'uncovered_raw_points': int(sum((~area['raw_covered']).sum() for area in final_areas)),
        'final_on_one_vote_support': metrics_from_confusion(common_confusion),
        'areas': [{key: area[key] for key in ('name', 'subsampled_coverage', 'raw_coverage')}
                  for area in final_areas],
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    logging.info('wrote analysis to %s', args.output)


if __name__ == '__main__':
    main()
