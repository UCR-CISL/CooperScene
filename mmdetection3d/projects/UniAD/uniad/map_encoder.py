"""BEV Map Encoder for converting rasterized map images to lane queries.

Encodes BEV map PNGs (lane markings + static environment) into a set of
lane query tokens that are consumed by MotionHead's MapInteraction layers.

This replaces UniAD's panoptic segmentation head for producing lane queries,
using a lightweight CNN encoder instead.
"""
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class BEVMapEncoder(BaseModule):
    """Encode rasterized BEV map images into lane queries.

    Takes stacked BEV map PNGs (e.g. lane + static, 6 channels for 2 RGB
    images) and produces a fixed number of lane query tokens via CNN encoding
    + adaptive pooling.

    Args:
        in_channels (int): Input channels (e.g. 6 for lane+static RGB).
        embed_dims (int): Output embedding dimensions.
        num_queries (int): Number of lane query tokens to produce.
        map_size (int): Expected input spatial size (maps resized to this).
    """

    def __init__(self,
                 in_channels=6,
                 embed_dims=256,
                 num_queries=50,
                 map_size=256,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_queries = num_queries
        self.map_size = map_size

        # CNN encoder: 256x256 -> 16x16 feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )
        # 256/16 = 16x16 = 256 spatial tokens, pool down to num_queries
        pool_h = int(num_queries ** 0.5)
        pool_w = (num_queries + pool_h - 1) // pool_h
        self.actual_queries = pool_h * pool_w
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.proj = nn.Linear(embed_dims, embed_dims)

        # Learnable positional embeddings for lane queries
        self.pos_embed = nn.Embedding(self.actual_queries, embed_dims)

    def forward(self, bev_map):
        """Encode BEV map images into lane queries.

        Args:
            bev_map: (B, C, H, W) stacked BEV map images, float [0, 1].

        Returns:
            lane_query: (B, M, D) lane query embeddings.
            lane_query_pos: (B, M, D) lane query positional embeddings.
        """
        B = bev_map.shape[0]
        feat = self.encoder(bev_map)        # (B, D, 16, 16)
        feat = self.pool(feat)              # (B, D, ph, pw)
        feat = feat.flatten(2).permute(0, 2, 1)  # (B, M, D)
        feat = self.proj(feat)

        pos = self.pos_embed.weight.unsqueeze(0).expand(B, -1, -1)
        return feat, pos
