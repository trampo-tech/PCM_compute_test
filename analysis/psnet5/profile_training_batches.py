"""Profile a few PSNet5 training batches without changing the training run."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.utils import EasyConfig, set_random_seed


def step(model, criterion, optimizer, batch, cfg, synchronize):
    timings = {}
    start = time.perf_counter()
    valid_counts = batch['mask'].sum(dim=1).tolist()
    with record_function('transfer_and_features'):
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        target = batch['y'].squeeze(-1)
        batch['x'] = get_features_by_keys(batch, cfg.feature_keys)
    if synchronize:
        torch.cuda.synchronize()
    timings['transfer'] = time.perf_counter() - start

    start = time.perf_counter()
    with record_function('model_forward'):
        logits = model(batch)
    if synchronize:
        torch.cuda.synchronize()
    timings['forward'] = time.perf_counter() - start

    start = time.perf_counter()
    with record_function('masked_loss'):
        loss = criterion(logits, target, batch['mask'])
    if synchronize:
        torch.cuda.synchronize()
    timings['loss'] = time.perf_counter() - start

    start = time.perf_counter()
    with record_function('backward'):
        loss.backward()
    if synchronize:
        torch.cuda.synchronize()
    timings['backward'] = time.perf_counter() - start

    start = time.perf_counter()
    with record_function('optimizer_step'):
        if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()
        optimizer.zero_grad()
    if synchronize:
        torch.cuda.synchronize()
    timings['optimizer'] = time.perf_counter() - start
    return timings, valid_counts


def summarize(rows):
    return {
        key: {
            'mean_ms': round(1000 * statistics.mean(row[key] for row in rows), 2),
            'min_ms': round(1000 * min(row[key] for row in rows), 2),
            'max_ms': round(1000 * max(row[key] for row in rows), 2),
        }
        for key in ('data_wait', 'transfer', 'forward', 'loss', 'backward', 'optimizer', 'total')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default='cfgs/psnet5/PCM.yaml')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--profiler-steps', type=int, default=2)
    parser.add_argument('--cudnn-benchmark', choices=('on', 'off'), default='on')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.profiler_steps < 0:
        parser.error('steps must be positive; warmup and profiler-steps must be nonnegative')

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    set_random_seed(20261004)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == 'on'
    loader = build_dataloader_from_cfg(
        cfg.batch_size, cfg.dataset, cfg.dataloader,
        datatransforms_cfg=cfg.datatransforms, split='train', distributed=False)
    model = build_model_from_cfg(cfg.model).cuda().train()
    cfg.criterion_args.weight = None
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    iterator = iter(loader)

    for _ in range(args.warmup):
        batch = next(iterator)
        step(model, criterion, optimizer, batch, cfg, synchronize=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for index in range(args.steps):
        start = time.perf_counter()
        batch = next(iterator)
        wait = time.perf_counter() - start
        timings, counts = step(model, criterion, optimizer, batch, cfg, synchronize=True)
        row = {'batch': index, 'valid_points': counts, 'data_wait': wait, **timings}
        row['total'] = wait + sum(timings.values())
        rows.append(row)
        print(f"batch={index} valid={counts} " +
              ' '.join(f'{key}={1000 * row[key]:.1f}ms' for key in
                       ('data_wait', 'forward', 'backward', 'total')), flush=True)

    operator_tables = {}
    if args.profiler_steps:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False) as prof:
            for _ in range(args.profiler_steps):
                batch = next(iterator)
                step(model, criterion, optimizer, batch, cfg, synchronize=False)
                torch.cuda.synchronize()
        averages = prof.key_averages()
        operator_tables = {
            'cuda': averages.table(sort_by='self_cuda_time_total', row_limit=20),
            'cpu': averages.table(sort_by='self_cpu_time_total', row_limit=20),
        }

    report = {
        'config': str(args.cfg), 'batch_size': int(cfg.batch_size),
        'point_cap': int(cfg.dataset.common.num_points),
        'cudnn_benchmark': args.cudnn_benchmark,
        'warmup_batches': args.warmup, 'timed_batches': args.steps,
        'profiler_batches': args.profiler_steps,
        'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
        'timings': summarize(rows), 'batches': rows,
        'operators': operator_tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('timings', 'peak_allocated_gib')}, indent=2))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
