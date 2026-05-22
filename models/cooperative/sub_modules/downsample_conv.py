"""DownsampleConv: BN+ReLU conv stack used as shrink_header.
"""
import torch.nn as nn
from mmdet3d.registry import MODELS


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


@MODELS.register_module()
class DownsampleConv(nn.Module):
    def __init__(self, input_dim, dim, kernal_size, stride, padding):
        super().__init__()
        self.layers = nn.ModuleList([])
        cur_dim = input_dim
        for (k, d, s, p) in zip(kernal_size, dim, stride, padding):
            self.layers.append(DoubleConv(cur_dim, d, k, s, p))
            cur_dim = d

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
