"""Mask-aware, per-cloud sphere vote aggregation."""
import torch


def add_sphere_votes(logit_sums, vote_counts, logits, cloud_indices, point_indices, masks):
    """Accumulate ``[B,C,N]`` logits by cloud and subsampled point index."""
    for batch_index in range(logits.shape[0]):
        cloud = int(cloud_indices[batch_index])
        valid = masks[batch_index].bool()
        indices = point_indices[batch_index][valid].long()
        values = logits[batch_index, :, valid].transpose(0, 1)
        logit_sums[cloud].index_add_(0, indices, values)
        vote_counts[cloud].index_add_(0, indices, torch.ones_like(indices, dtype=vote_counts[cloud].dtype))


def averaged_cloud_logits(logit_sum, vote_count):
    covered = vote_count > 0
    averages = torch.zeros_like(logit_sum)
    averages[covered] = logit_sum[covered] / vote_count[covered, None]
    return averages, covered
