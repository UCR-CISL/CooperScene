"""V2VAM (Criss-Cross attention) fusion.

forward(x, record_len): (sum_cav, C, H, W) -> (B, C, H, W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Parameter

from mmdet3d.registry import MODELS


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0) \
        .unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU())
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU())
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU())
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, query, key, value):
        m_batchsize, _, height, width = query.size()
        proj_query = self.query_conv(query)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(key)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width)
        proj_value = self.value_conv(value)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(
            m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(
            m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) +
                    self.INF(m_batchsize, height, width)).view(
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous() \
            .view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(
            m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(
            m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(
            m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + value


@MODELS.register_module()
class V2VAttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(V2VAttFusion, self).__init__()
        self.cov_att = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU())
        self.CCNet = CrissCrossAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        out = []
        for xx in split_x:
            att = []
            ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1]
            for i in range(len(xx[:, 0, 0, 0])):
                att_vehicle = self.CCNet(ego_q, xx[i:i + 1], xx[i:i + 1])
                att.append(att_vehicle)
            pooling_max = torch.max(torch.cat(att, dim=0), dim=0,
                                    keepdim=True)[0]
            pooling_ave = torch.mean(torch.cat(att, dim=0), dim=0,
                                     keepdim=True)[0]
            fuse_fea = pooling_max + pooling_ave
            fuse_att = self.cov_att(fuse_fea)
            out.append(fuse_att)
        return torch.cat(out, dim=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
