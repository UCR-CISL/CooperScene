"""Self-attention fusion (AttFusion) / scaled dot-product.

AttFusion.forward(x, record_len): (sum_cav, C, H, W) -> (B, C, H, W).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


@MODELS.register_module()
class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            out.append(h)
        return torch.stack(out)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
