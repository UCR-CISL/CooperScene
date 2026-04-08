# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of R'CV2X random sum fusion.
"""
import torch
import torch.nn as nn


class RandomSumFusion(nn.Module):
    def __init__(self):
        super(RandomSumFusion, self).__init__()

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, 
                record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []

        for idx,xx in enumerate(split_x):
            # xx: (cav_num, C, W, H)
            
            # out_features = ego + one cav randomly selected
            cav_num = xx.shape[0]
            if cav_num >1:
                ego_feature = xx[0:1]
                select_num = 1
                select_indices = torch.randperm(cav_num -1)[:select_num] +1
                tx_veh = xx[select_indices]
                sum_feature = ego_feature + tx_veh
            # if only one cav, directly use it
            else:
                sum_feature = xx
            out.append(sum_feature)
        return torch.cat(out, dim=0)