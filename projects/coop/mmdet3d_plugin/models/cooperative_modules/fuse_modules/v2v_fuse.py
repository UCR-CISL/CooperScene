"""V2VNet fusion with ConvGRU message passing.

forward(x, record_len, pairwise_t_matrix): (sum_cav, C, H, W) -> (B, C, H, W).
"""
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS
from ..sub_modules.torch_transformation_utils import (
    get_discretized_transformation_matrix, get_transformation_matrix,
    warp_affine, get_rotated_roi)
from ..sub_modules.convgru import ConvGRU


@MODELS.register_module()
class V2VNetFusion(nn.Module):
    def __init__(self, in_channels, conv_gru, voxel_size, downsample_rate,
                 num_iteration, gru_flag, agg_operator):
        super(V2VNetFusion, self).__init__()

        H, W = conv_gru['H'], conv_gru['W']
        kernel_size = conv_gru['kernel_size']
        num_gru_layers = conv_gru['num_layers']

        self.discrete_ratio = voxel_size[0]
        self.downsample_rate = downsample_rate
        self.num_iteration = num_iteration
        self.gru_flag = gru_flag
        self.agg_operator = agg_operator

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = self.regroup(x, record_len)
        pairwise_t_matrix = get_discretized_transformation_matrix(
            pairwise_t_matrix.reshape(-1, L, 4, 4), self.discrete_ratio,
            self.downsample_rate).reshape(B, L, L, 2, 3)
        roi_mask = get_rotated_roi(
            (B * L, L, 1, H, W),
            pairwise_t_matrix.reshape(B * L * L, 2, 3))
        roi_mask = roi_mask.reshape(B, L, L, 1, H, W)
        batch_node_features = split_x

        for l in range(self.num_iteration):
            batch_updated_node_features = []
            for b in range(B):
                N = record_len[b]
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                updated_node_features = []
                for i in range(N):
                    mask = roi_mask[b, :N, i, ...]
                    current_t_matrix = t_matrix[:, i, :, :]
                    current_t_matrix = get_transformation_matrix(
                        current_t_matrix, (H, W))
                    neighbor_feature = warp_affine(
                        batch_node_features[b], current_t_matrix, (H, W))
                    ego_agent_feature = batch_node_features[b][i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    message = self.msg_cnn(neighbor_feature) * mask
                    if self.agg_operator == "avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator == "max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    cat_feature = torch.cat(
                        [batch_node_features[b][i, ...], agg_feature], dim=0)
                    if self.gru_flag:
                        gru_out = self.conv_gru(
                            cat_feature.unsqueeze(0).unsqueeze(0))[0][0] \
                            .squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_features[b][i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features

        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out
