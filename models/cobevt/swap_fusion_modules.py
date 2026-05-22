"""CoBEVT SwapFusionEncoder (local + global window attention).

forward(x, mask): x (B, L, C, H, W) with L=max_cav -> (B, C, H, W).
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from mmdet3d.registry import MODELS
from ..cooperative.sub_modules.base_transformer import FeedForward, PreNormResidual


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0., agent_size=6,
                 window_size=7):
        super().__init__()
        assert (dim % dim_head) == 0
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1), self.heads)

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        batch, agent_size, height, width, window_height, window_width, _, \
            device, h = *x.shape, x.device, self.heads
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        bias = self.relative_position_bias_table(
            self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        if mask is not None:
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))
        attn = self.attend(sim)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class SwapFusionBlockMask(nn.Module):
    def __init__(self, input_dim, mlp_dim, dim_head, window_size,
                 agent_size, drop_out):
        super(SwapFusionBlockMask, self).__init__()
        self.window_size = window_size
        self.window_attention = PreNormResidual(
            input_dim, Attention(input_dim, dim_head, drop_out,
                                 agent_size, window_size))
        self.window_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out))
        self.grid_attention = PreNormResidual(
            input_dim, Attention(input_dim, dim_head, drop_out,
                                 agent_size, window_size))
        self.grid_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out))

    def forward(self, x, mask):
        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')
        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')
        return x


class SwapFusionBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dim_head, window_size,
                 agent_size, drop_out):
        super(SwapFusionBlock, self).__init__()
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(
                input_dim, dim_head, drop_out, agent_size, window_size)),
            PreNormResidual(input_dim, FeedForward(
                input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),
            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(
                input_dim, dim_head, drop_out, agent_size, window_size)),
            PreNormResidual(input_dim, FeedForward(
                input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        x = self.block(x)
        return x


@MODELS.register_module()
class SwapFusionEncoder(nn.Module):
    def __init__(self, depth, input_dim, mlp_dim, agent_size, window_size,
                 drop_out, dim_head, mask=False):
        super(SwapFusionEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.mask = mask

        for i in range(self.depth):
            if self.mask:
                block = SwapFusionBlockMask(
                    input_dim, mlp_dim, dim_head, window_size,
                    agent_size, drop_out)
            else:
                block = SwapFusionBlock(
                    input_dim, mlp_dim, dim_head, window_size,
                    agent_size, drop_out)
            self.layers.append(block)

        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

    def forward(self, x, mask=None):
        for stage in self.layers:
            x = stage(x, mask=mask)
        return self.mlp_head(x)
