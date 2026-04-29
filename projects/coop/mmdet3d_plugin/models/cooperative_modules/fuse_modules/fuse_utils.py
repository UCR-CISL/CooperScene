"""Utilities: regroup (split sum_cav -> (B, L, C, H, W) with padding + mask).
"""
import torch
import numpy as np
from einops import rearrange


def regroup(dense_feature, record_len, max_len):
    cum_sum_len = list(np.cumsum(
        record_len.cpu().numpy() if torch.is_tensor(record_len)
        else record_len))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        feature_shape = split_feature.shape
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)
        padding_tensor = torch.zeros(padding_len, feature_shape[1],
                                     feature_shape[2], feature_shape[3])
        padding_tensor = padding_tensor.to(split_feature.device)
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        split_feature = split_feature.view(
            -1, feature_shape[2], feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    regroup_features = torch.cat(regroup_features, dim=0)
    regroup_features = rearrange(regroup_features,
                                 'b (l c) h w -> b l c h w', l=max_len)
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)
    return regroup_features, mask
