"""NaiveCompressor: channel encoder/decoder to simulate bandwidth compression.
"""
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class NaiveCompressor(nn.Module):
    def __init__(self, input_dim, compress_ratio):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim // compress_ratio, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim // compress_ratio, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        compress_x = self.encoder(x)
        decompress_x = self.decoder(compress_x)
        return decompress_x
