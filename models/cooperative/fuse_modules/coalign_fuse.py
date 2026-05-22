"""CoAlign fusion (affine-aligned attention).

forward(x, record_len, normalized_tfm): (sum_cav, C, H, W) -> (B, C, H, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from .self_attn import ScaledDotProductAttention


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio,
                            downsample_rate=1):
    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
        downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
        downsample_rate * discrete_ratio * H) * 2
    return pairwise_t_matrix


def warp_affine_simple(src, M, dsize, mode='bilinear', padding_mode='zeros',
                       align_corners=False):
    B, C, H, W = src.size()
    grid = F.affine_grid(M, [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


@MODELS.register_module()
class CoAlignFusion(nn.Module):
    def __init__(self, feature_dims):
        super(CoAlignFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0
            x = warp_affine_simple(batch_node_features[b],
                                   t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]
            out.append(h)
        out = torch.stack(out)
        return out
