"""Build a compact visual report from a completed PSNet5 Area 3 evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('I-beam', 'Pipe', 'Pump', 'Rectangular beam', 'Tank')
COLORS = ('#D55E00', '#0072B2', '#CC79A7', '#009E73', '#E6B800')
COLOR_MAP = ListedColormap(COLORS)
RAW_POINTS = ROOT / ('data/PSNet5/processed_pcm/'
    'psnet5sphere_benchmark_benchmark-area-v1_test_v0.040/cloud_0_raw_points.npy')
RAW_LABELS = RAW_POINTS.with_name('cloud_0_raw_labels.npy')


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def load_confusion(path: Path) -> np.ndarray:
    with path.open(newline='') as stream:
        rows = list(csv.reader(stream))
    return np.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)


def confusion_figure(confusion: np.ndarray, out: Path) -> None:
    fractions = confusion / confusion.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5), layout='constrained')
    image = ax.imshow(fractions * 100, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(5), CLASSES, rotation=35, ha='right')
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    ax.set_title('Area 3 · confusion by true class', fontsize=16, weight='bold', pad=18)
    for row in range(5):
        for col in range(5):
            value = fractions[row, col] * 100
            count = confusion[row, col]
            label = f'{value:.1f}%\n{count / 1e6:.2f}M' if count >= 100_000 else f'{value:.1f}%\n{count:,}'
            ax.text(col, row, label, ha='center', va='center', fontsize=10,
                    color='white' if value >= 55 else '#1d2733')
    fig.colorbar(image, ax=ax, shrink=.78, label='Share of each true class (%)')
    fig.savefig(out, dpi=170, facecolor='white')
    plt.close(fig)


def prediction_figure(predictions: Path, out: Path) -> None:
    points = np.load(RAW_POINTS, mmap_mode='r')
    cached_labels = np.load(RAW_LABELS, mmap_mode='r')
    with np.load(predictions, allow_pickle=False) as data:
        total = len(data['label'])
        if len(points) != total:
            raise ValueError(f'Coordinate/prediction length mismatch: {len(points)} vs {total}')
        rng = np.random.default_rng(20261005)
        chosen = np.sort(rng.choice(total, min(total, 160_000), replace=False))
        xyz = np.asarray(points[chosen], dtype=np.float32)
        truth = data['label'][chosen]
        predicted = data['prediction'][chosen]
        covered = data['covered'][chosen]
    if not np.array_equal(truth, cached_labels[chosen]):
        raise ValueError('Cached coordinates and prediction labels use different point order')
    if not covered.all():
        xyz, truth, predicted = xyz[covered], truth[covered], predicted[covered]
    wrong = truth != predicted
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True, layout='constrained')
    for ax, labels, title in zip(axes[:2], (truth, predicted), ('Ground truth', 'Prediction')):
        ax.scatter(xyz[:, 0], xyz[:, 1], c=labels, cmap=COLOR_MAP, vmin=-.5,
                   vmax=4.5, s=.32, alpha=.7, rasterized=True, linewidths=0)
        ax.set_title(title, fontsize=15, weight='bold')
    axes[2].scatter(xyz[~wrong, 0], xyz[~wrong, 1], c='#ced4da', s=.25,
                    alpha=.3, rasterized=True, linewidths=0)
    axes[2].scatter(xyz[wrong, 0], xyz[wrong, 1], c='#c62828', s=.45,
                    alpha=.7, rasterized=True, linewidths=0)
    axes[2].set_title('Errors in red', fontsize=15, weight='bold')
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.grid(alpha=.12)
    axes[0].set_ylabel('Y')
    handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                          markeredgecolor='none', markersize=9, label=name)
               for name, color in zip(CLASSES, COLORS)]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(.5, -.035), frameon=False)
    fig.suptitle('Area 3 · top view of the same 160k sampled raw points', fontsize=17, weight='bold')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def learning_figure(log_text: str, out: Path) -> tuple[int | None, float | None]:
    matches = re.findall(r'Epoch (\d+) LR [\d.]+ train_miou ([\d.]+), val_miou ([\d.]+)', log_text)
    if not matches:
        return None, None
    epochs = np.array([int(x[0]) for x in matches])
    train = np.array([float(x[1]) for x in matches])
    val = np.array([float(x[2]) for x in matches])
    best = int(np.argmax(val))
    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    ax.plot(epochs, train, color='#0072B2', label='Train', linewidth=2)
    ax.plot(epochs, val, color='#D55E00', label='Validation', linewidth=2)
    ax.scatter([epochs[best]], [val[best]], color='#D55E00', s=75, zorder=5)
    ax.annotate(f'Best E{epochs[best]} · {val[best]:.1f}%',
                (epochs[best], val[best]),
                xytext=(-7 if epochs[best] > max(epochs) * .75 else 7, -22),
                ha='right' if epochs[best] > max(epochs) * .75 else 'left',
                textcoords='offset points')
    ax.set(xlabel='Epoch', ylabel='mIoU (%)', title='Training and development validation')
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0, 100)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc='lower right')
    fig.savefig(out, dpi=170, facecolor='white')
    plt.close(fig)
    return int(epochs[best]), float(val[best])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--train-log', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--title', default='BatchNorm · 40 epochs')
    parser.add_argument('--fixed-dev-eval', type=Path,
                        help='Fixed-mode development evaluation for a legacy checkpoint')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.eval_dir / 'summary.json').read_text())
    if summary['evaluation_protocol'] != 'benchmark' or summary['evaluation_split'] != 'test':
        raise ValueError('Report requires an Area 3 benchmark evaluation')
    confusion = load_confusion(args.eval_dir / 'confusion_vote_01.csv')
    metrics = summary['final']
    confusion_figure(confusion, args.out_dir / 'confusion.png')
    prediction_figure(args.eval_dir / 'Area_3_predictions.npz', args.out_dir / 'prediction_comparison.png')
    train_log = args.train_log.read_text(errors='replace')
    best_epoch, best_val = learning_figure(train_log, args.out_dir / 'learning_curve.png')

    def pct(value: float) -> str:
        return f'{value * 100:.1f}%'

    ibeam_to_pipe = confusion[0, 1] / confusion[0].sum()
    ibeam_to_beam = confusion[0, 3] / confusion[0].sum()
    pump_to_pipe = confusion[2, 1] / confusion[2].sum()
    fixed_eval = 'eval_bn_batch_stats: True' in train_log
    if best_epoch is None:
        train_note = 'Training curve was unavailable.'
    elif fixed_eval:
        train_note = f'Best development validation: **{best_val:.1f}% mIoU at epoch {best_epoch}**.'
    else:
        train_note = f'Legacy development validation selected epoch {best_epoch} at **{best_val:.1f}% mIoU** using stored BatchNorm statistics.'
    if args.fixed_dev_eval:
        fixed_summary = json.loads(args.fixed_dev_eval.read_text())
        fixed_dev_note = f' Re-evaluating that checkpoint with current-sphere statistics gives **{pct(fixed_summary["final"]["miou"])} development mIoU**.'
    else:
        fixed_dev_note = ''
    bn_note = ('This run used current-sphere BatchNorm statistics for validation and checkpoint selection.'
               if fixed_eval else 'This earlier run selected its checkpoint using stored BatchNorm running statistics.'
               + fixed_dev_note)
    report = f'''# {args.title} · Area 3 visual report

| Area 3 mIoU | Overall accuracy | Mean class accuracy | Coverage |
|---:|---:|---:|---:|
| **{pct(metrics['miou'])}** | **{pct(metrics['oa'])}** | **{pct(metrics['macc'])}** | **{pct(summary['areas'][0]['raw_coverage'])}** |

Training and development validation used spatial partitions of **Areas 1, 2 and 4**. Area 3 was evaluated separately from the saved checkpoint with **{summary['votes']} vote** over all raw points. {train_note}

![Training and development validation](learning_curve.png)

## Area 3 predictions

![Area 3 ground truth, predictions and errors](prediction_comparison.png)

The top view shows the same sampled raw points in each panel. Vertical structures overlap in this projection; red marks incorrect predictions.

![Area 3 confusion matrix](confusion.png)

| Class | IoU | Recall | Main error |
|---|---:|---:|---|
| I-beam | {pct(metrics['per_class'][0]['iou'])} | {pct(metrics['per_class'][0]['recall'])} | {pct(ibeam_to_pipe)} → pipe; {pct(ibeam_to_beam)} → rectangular beam |
| Pipe | {pct(metrics['per_class'][1]['iou'])} | {pct(metrics['per_class'][1]['recall'])} | — |
| Pump | {pct(metrics['per_class'][2]['iou'])} | {pct(metrics['per_class'][2]['recall'])} | {pct(pump_to_pipe)} → pipe |
| Rectangular beam | {pct(metrics['per_class'][3]['iou'])} | {pct(metrics['per_class'][3]['recall'])} | — |
| Tank | {pct(metrics['per_class'][4]['iou'])} | {pct(metrics['per_class'][4]['recall'])} | — |

## BatchNorm evaluation issue

{bn_note} Stored statistics produced misleading development scores for compact, variable-size spheres. The fixed mode uses each evaluation sphere's BatchNorm statistics while leaving dropout in evaluation mode and keeping running buffers unchanged. **This report's Area 3 score uses `{summary.get('bn_mode', 'unknown')}`.** This evaluation option does not change optimizer updates.

## Sources

- [Area 3 evaluation summary]({relative(args.eval_dir / 'summary.json', args.out_dir)})
- [Confusion counts]({relative(args.eval_dir / 'confusion_vote_01.csv', args.out_dir)})
- [Training log]({relative(args.train_log, args.out_dir)})
'''
    (args.out_dir / 'REPORT.md').write_text(report)
    print(args.out_dir / 'REPORT.md')


if __name__ == '__main__':
    main()
