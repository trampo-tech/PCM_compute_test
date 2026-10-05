"""Measure checkpoint sensitivity to changes affecting only masked padding points."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint['model']
    if state and next(iter(state)).startswith('module.'):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def padding_variant(data, generator):
    """Replace every padded slot with a newly selected valid point."""
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
    for batch_index, row_mask in enumerate(data['mask'].bool()):
        valid = row_mask.nonzero(as_tuple=False).squeeze(1)
        padded = (~row_mask).nonzero(as_tuple=False).squeeze(1)
        source = valid[torch.randint(len(valid), (len(padded),), generator=generator,
                                     device=valid.device)]
        result['pos'][batch_index, padded] = data['pos'][batch_index, source]
        result['x'][batch_index, :, padded] = data['x'][batch_index, :, source]
    return result


def compare(reference, candidate, mask):
    valid_reference = reference.transpose(1, 2)[mask]
    valid_candidate = candidate.transpose(1, 2)[mask]
    difference = (valid_candidate - valid_reference).abs()
    return {
        'mean_abs_logit_change': float(difference.mean()),
        'max_abs_logit_change': float(difference.max()),
        'prediction_flip_fraction': float(
            (valid_candidate.argmax(1) != valid_reference.argmax(1)).float().mean()),
    }


@torch.no_grad()
def run_mode(model, data, variants, mask_padding):
    model.mask_padding = mask_padding
    reference = model(data)
    return [compare(reference, model(variant), data['mask'].bool()) for variant in variants]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', default='cfgs/psnet5/PCM.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--min-padding-fraction', type=float, default=0.25)
    parser.add_argument('--output', type=Path,
                        default=Path('analysis/psnet5/results/padding_sensitivity.json'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the PCM padding-sensitivity diagnostic')
    device = torch.device('cuda:0')
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.dataloader.num_workers = min(int(cfg.dataloader.num_workers), 2)
    loader = build_dataloader_from_cfg(
        1, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms,
        split='val', distributed=False)

    selected = None
    for data in loader:
        padding_fraction = 1.0 - float(data['mask'].float().mean())
        if padding_fraction >= args.min_padding_fraction:
            selected = data
            break
    if selected is None:
        raise RuntimeError(f'No sphere has at least {args.min_padding_fraction:.1%} padding')
    for key in tuple(selected):
        selected[key] = selected[key].to(device, non_blocking=True)
    selected['x'] = get_features_by_keys(selected, cfg.feature_keys)

    generator = torch.Generator(device=device).manual_seed(20260929)
    variants = [padding_variant(selected, generator) for _ in range(args.variants)]
    model = load_checkpoint(build_model_from_cfg(cfg.model), args.checkpoint, device)
    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'valid_points': int(selected['mask'].sum()),
        'total_points': int(selected['mask'].numel()),
        'padding_fraction': 1.0 - float(selected['mask'].float().mean()),
        'variants': args.variants,
        'legacy_full_padded_input': run_mode(model, selected, variants, False),
        'mask_compacted_input': run_mode(model, selected, variants, True),
        'interpretation': (
            'A successful compaction has zero (or numerical-zero) sensitivity because variants '
            'change only points whose mask is false. This diagnoses isolation, not post-retraining mIoU.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
