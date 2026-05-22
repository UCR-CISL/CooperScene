"""F-Cooper max-fusion.

forward(x, record_len): (sum_cav, C, H, W) -> (B, C, H, W) via per-sample max-pool across CAVs.
"""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SpatialFusion(nn.Module):
    def __init__(self):
        super(SpatialFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        out = []
        for xx in split_x:
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)
