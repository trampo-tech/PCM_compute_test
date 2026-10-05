""" Normalization layers and wrappers
"""
from sklearn.decomposition import FactorAnalysis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import copy
from contextlib import contextmanager
from easydict import EasyDict as edict


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


class LayerNorm1d(nn.LayerNorm):
    """ LayerNorm for channels of '1D' spatial BCN tensors """

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 2, 1).contiguous()


class FastBatchNorm1d(nn.Module):
    """Fast BachNorm1d for input with shape [B, N, C], where the feature dimension is at last. 
    Borrowed from torch-points3d: https://github.com/torch-points3d/torch-points3d
    """
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def _forward_dense(self, x):
        return self.bn(x.transpose(1,2)).transpose(2, 1)

    def _forward_sparse(self, x):
        return self.bn(x)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use stored statistics only when training has one value per channel."""
    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


@contextmanager
def batch_stats_for_eval(model):
    """Use each input's BN statistics in eval without changing running buffers.

    Call after ``model.eval()`` so dropout and other layers remain in eval mode.
    The singleton-safe BN keeps its stored-statistics fallback for one-point
    inputs, which cannot supply a batch variance.
    """
    modules = [(module, module.training, module.track_running_stats)
               for module in model.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _, _ in modules:
            module.track_running_stats = False
            module.train()
        yield
    finally:
        for module, training, track_running_stats in modules:
            module.track_running_stats = track_running_stats
            module.train(training)


def replace_batch_norm_with_group_norm(model, max_groups=8):
    """Replace pointwise BatchNorm modules with per-example GroupNorm."""
    if max_groups < 1:
        raise ValueError('max_groups must be positive')
    for name, child in list(model.named_children()):
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            channels = child.num_features
            group_limit = min(max_groups, max(1, channels // 2))
            groups = next(count for count in range(group_limit, 0, -1)
                          if channels % count == 0)
            replacement = nn.GroupNorm(groups, channels, eps=child.eps,
                                       affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(model, name, replacement)
        else:
            replace_batch_norm_with_group_norm(child, max_groups)


_NORM_LAYER = dict(
    bn1d=nn.BatchNorm1d,
    safebn1d=SingletonSafeBatchNorm1d,
    bn2d=nn.BatchNorm2d,
    bn=nn.BatchNorm2d,
    in2d=nn.InstanceNorm2d, 
    in1d=nn.InstanceNorm1d, 
    gn=nn.GroupNorm,
    syncbn=nn.SyncBatchNorm,
    ln=nn.LayerNorm,    # for tokens
    ln1d=LayerNorm1d,   # for point cloud
    ln2d=LayerNorm2d,   # for point cloud
    fastbn1d=FastBatchNorm1d, 
    fastbn2d=FastBatchNorm1d, 
    fastbn=FastBatchNorm1d, 
)


def create_norm(norm_args, channels, dimension=None):
    """Build normalization layer.
    Returns:
        nn.Module: Created normalization layer.
    """
    if norm_args is None:
        return None
    if isinstance(norm_args, dict):    
        norm_args = edict(copy.deepcopy(norm_args))
        norm = norm_args.pop('norm', None)
    else:
        norm = norm_args
        norm_args = edict()
    if norm is None:
        return None
    if isinstance(norm, str):
        norm = norm.lower()
        if dimension is not None:
            dimension = str(dimension).lower()
            if dimension not in norm:
                norm += dimension
        assert norm in _NORM_LAYER.keys(), f"input {norm} is not supported"
        norm = _NORM_LAYER[norm]
    return norm(channels, **norm_args)


if __name__ == "__main__":
    norm_type = 'bn2d'
    from easydict import EasyDict as edict

    norm_args = {'norm': 'bn2d'}
    norm_layer = create_norm(norm_args, 64)
    print(norm_layer)
