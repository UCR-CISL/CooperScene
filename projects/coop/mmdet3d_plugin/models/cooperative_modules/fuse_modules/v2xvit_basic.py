"""V2X-ViT transformer (HMSA + PyramidWindow + STTF).

forward(x, mask, spatial_correction): x (B, L, H, W, C+3) -> (B, H, W, C).
"""
import math

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS
from ..sub_modules.base_transformer import PreNorm, FeedForward, CavAttention
from ..sub_modules.torch_transformation_utils import (
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask,
    get_discretized_transformation_matrix)
from .hmsa import HGTCavAttention
from .mswin import PyramidWindowAttention


class STTF(nn.Module):
    def __init__(self, voxel_size, downsample_rate):
        super(STTF, self).__init__()
        self.discrete_ratio = voxel_size[0]
        self.downsample_rate = downsample_rate

    def forward(self, x, mask, spatial_correction_matrix):
        x = x.permute(0, 1, 4, 2, 3)
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)
        B, L, C, H, W = x.shape
        T = get_transformation_matrix(
            dist_correction_matrix[:, 1:, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(
            x[:, 1:, :, :, :].reshape(-1, C, H, W), T, (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)
        x = torch.cat([x[:, 0, :, :, :].unsqueeze(1), cav_features], dim=1)
        x = x.permute(0, 1, 3, 4, 2)
        return x


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, RTE_ratio, max_len=100, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.RTE_ratio = RTE_ratio
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t * self.RTE_ratio)).unsqueeze(0).unsqueeze(1)


class RTE(nn.Module):
    def __init__(self, dim, RTE_ratio=2):
        super(RTE, self).__init__()
        self.emb = RelTemporalEncoding(dim, RTE_ratio=RTE_ratio)

    def forward(self, x, dts):
        rte_batch = []
        for b in range(x.shape[0]):
            rte_list = []
            for i in range(x.shape[1]):
                rte_list.append(
                    self.emb(x[b, i, :, :, :], dts[b, i]).unsqueeze(0))
            rte_batch.append(torch.cat(rte_list, dim=0).unsqueeze(0))
        return torch.cat(rte_batch, dim=0)


class V2XFusionBlock(nn.Module):
    def __init__(self, num_blocks, cav_att_config, pwindow_config):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_blocks):
            att = HGTCavAttention(
                cav_att_config['dim'],
                heads=cav_att_config['heads'],
                dim_head=cav_att_config['dim_head'],
                dropout=cav_att_config['dropout']) if \
                cav_att_config['use_hetero'] else CavAttention(
                cav_att_config['dim'],
                heads=cav_att_config['heads'],
                dim_head=cav_att_config['dim_head'],
                dropout=cav_att_config['dropout'])
            self.layers.append(nn.ModuleList([
                PreNorm(cav_att_config['dim'], att),
                PreNorm(cav_att_config['dim'],
                        PyramidWindowAttention(
                            pwindow_config['dim'],
                            heads=pwindow_config['heads'],
                            dim_heads=pwindow_config['dim_head'],
                            drop_out=pwindow_config['dropout'],
                            window_size=pwindow_config['window_size'],
                            relative_pos_embedding=pwindow_config[
                                'relative_pos_embedding'],
                            fuse_method=pwindow_config['fusion_method']))]))

    def forward(self, x, mask, prior_encoding):
        for cav_attn, pwindow_attn in self.layers:
            x = cav_attn(x, mask=mask, prior_encoding=prior_encoding) + x
            x = pwindow_attn(x) + x
        return x


class V2XTEncoder(nn.Module):
    def __init__(self, cav_att_config, pwindow_att_config, feed_forward,
                 num_blocks, depth, sttf, use_roi_mask):
        super().__init__()
        mlp_dim = feed_forward['mlp_dim']
        dropout = feed_forward['dropout']

        self.downsample_rate = sttf['downsample_rate']
        self.discrete_ratio = sttf['voxel_size'][0]
        self.use_roi_mask = use_roi_mask
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF(**sttf)
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3,
                                    cav_att_config['dim'])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                V2XFusionBlock(num_blocks, cav_att_config,
                               pwindow_att_config),
                PreNorm(cav_att_config['dim'],
                        FeedForward(cav_att_config['dim'], mlp_dim,
                                    dropout=dropout))
            ]))

    def forward(self, x, mask, spatial_correction_matrix):
        prior_encoding = x[..., -3:]
        x = x[..., :-3]
        if self.use_RTE:
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3) if \
            not self.use_roi_mask else get_roi_and_cav_mask(
            x.shape, mask, spatial_correction_matrix,
            self.discrete_ratio, self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x


@MODELS.register_module()
class V2XTransformer(nn.Module):
    def __init__(self, encoder):
        super(V2XTransformer, self).__init__()
        self.encoder = V2XTEncoder(**encoder)

    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output
