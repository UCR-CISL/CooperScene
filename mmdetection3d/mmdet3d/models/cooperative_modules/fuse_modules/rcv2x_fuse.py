"""Random-sum fusion baseline.

forward(x, record_len): (sum_cav, C, H, W) -> (B, C, H, W).
"""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RandomSumFusion(nn.Module):
    def __init__(self):
        super(RandomSumFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            if cav_num > 1:
                ego_feature = xx[0:1]
                select_indices = torch.randperm(cav_num - 1)[:1] + 1
                tx_veh = xx[select_indices]
                sum_feature = ego_feature + tx_veh
            else:
                sum_feature = xx
            out.append(sum_feature)
        return torch.cat(out, dim=0)
