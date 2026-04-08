"""CoBEVT SwapFusion modules for multi-agent BEV feature fusion.

Implements the Swap Fusion attention mechanism from CoBEVT, which
alternates between local window attention and global grid attention
across the agent dimension to fuse BEV features from multiple CAVs.

Adapted from:
    CoBEVT/opv2v/opencood/models/fusion_modules/swap_fusion_modules.py
    OpenCOOD-modified/opencood/models/fuse_modules/swap_fusion_modules.py
"""

import torch
import torch.nn as nn
from einops import rearrange

from mmdet3d.registry import MODELS


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class PreNormResidual(nn.Module):
    """Pre-LayerNorm with residual connection."""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention(nn.Module):
    """Multi-head attention with 3D relative position bias.

    Operates over windows of shape [agent_size, window_size, window_size].

    Args:
        dim: Input feature dimension.
        dim_head: Dimension per attention head.
        dropout: Dropout rate.
        agent_size: Number of agents (CAVs).
        window_size: Spatial window size.
    """

    def __init__(self, dim, dim_head=32, dropout=0., agent_size=6,
                 window_size=7):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout),
        )

        # 3D relative position bias
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.heads)

        # Precompute relative position index
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1))
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index',
                             relative_position_index)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: (b, l, x, y, w1, w2, d) windowed features.
            mask: Optional mask.

        Returns:
            Same shape as input.
        """
        batch, agent_size, height, width, window_height, window_width, \
            _ = x.shape
        device = x.device
        h = self.heads

        # Flatten spatial dims
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')

        # Project to q, k, v
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            (q, k, v))
        q = q * self.scale

        # Attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        # Add relative position bias
        bias = self.relative_position_bias_table(
            self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # Apply mask if provided
        if mask is not None:
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))

        attn = self.attend(sim)

        # Aggregate
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(
            out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
            l=agent_size, w1=window_height, w2=window_width)

        out = self.to_out(out)
        return rearrange(
            out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
            b=batch, x=height, y=width)


class SwapFusionBlockMask(nn.Module):
    """Swap Fusion Block with mask support.

    Alternates between window attention (local) and grid attention (global)
    for multi-agent BEV feature fusion.

    Args:
        input_dim: Feature dimension.
        mlp_dim: Hidden dimension for FFN.
        dim_head: Dimension per attention head.
        window_size: Window size for partitioning.
        agent_size: Number of agents.
        drop_out: Dropout rate.
    """

    def __init__(self, input_dim, mlp_dim, dim_head, window_size,
                 agent_size, drop_out):
        super().__init__()
        self.window_size = window_size

        self.window_attention = PreNormResidual(
            input_dim,
            Attention(input_dim, dim_head, drop_out,
                      agent_size, window_size))
        self.window_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out))
        self.grid_attention = PreNormResidual(
            input_dim,
            Attention(input_dim, dim_head, drop_out,
                      agent_size, window_size))
        self.grid_ffd = PreNormResidual(
            input_dim, FeedForward(input_dim, mlp_dim, drop_out))

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: (b, l, c, h, w) multi-agent BEV features.
            mask: (b, h, w, 1, l) validity mask.

        Returns:
            (b, l, c, h, w) fused features.
        """
        ws = self.window_size

        # Window attention
        if mask is not None:
            mask_swap = rearrange(
                mask, 'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                w1=ws, w2=ws)
        else:
            mask_swap = None

        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                       w1=ws, w2=ws)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        # Grid attention
        if mask is not None:
            mask_swap = rearrange(
                mask, 'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                w1=ws, w2=ws)
        else:
            mask_swap = None

        x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                       w1=ws, w2=ws)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')

        return x


@MODELS.register_module()
class SwapFusionEncoder(nn.Module):
    """CoBEVT SwapFusion Encoder for multi-agent BEV fusion.

    Stacks multiple SwapFusionBlock layers, then mean-pools across agents
    to produce a single fused BEV feature map.

    Args:
        channels: Feature dimension (default: 256).
        n_head: Number of attention heads (default: 8).
        n_layers: Number of SwapFusion blocks (default: 3).
        window_size: Window size for attention partitioning (default: 9).
            Must divide the BEV spatial size evenly.
        agent_size: Maximum number of agents (default: 5).
        mlp_dim: Hidden dimension for FFN (default: 256).
        dim_head: Dimension per attention head (default: 32).
        dropout: Dropout rate (default: 0.1).
        use_mask: Whether to use validity mask (default: True).
    """

    def __init__(self, channels=256, n_head=8, n_layers=3, window_size=9,
                 agent_size=5, mlp_dim=256, dim_head=32, dropout=0.1,
                 use_mask=True):
        super().__init__()

        self.layers = nn.ModuleList()
        self.use_mask = use_mask

        for _ in range(n_layers):
            block = SwapFusionBlockMask(
                input_dim=channels,
                mlp_dim=mlp_dim,
                dim_head=dim_head,
                window_size=window_size,
                agent_size=agent_size,
                drop_out=dropout)
            self.layers.append(block)

        # MLP head: LayerNorm + Linear after masked mean pooling
        self.out_norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x, mask):
        """Forward pass.

        Args:
            x: (B, L, C, H, W) multi-agent BEV features.
            mask: (B, L) boolean validity mask per agent.

        Returns:
            (B, C, H, W) fused BEV features.
        """
        B, L, C, H, W = x.shape

        if self.use_mask and mask is not None:
            # Expand mask: (B, L) -> (B, H, W, 1, L) for attention
            com_mask = mask.view(B, 1, 1, 1, L).expand(
                B, H, W, 1, L).float()
        else:
            com_mask = None

        for layer in self.layers:
            x = layer(x, mask=com_mask)

        # Masked mean pool across agents
        if mask is not None:
            mask_expand = mask.float().unsqueeze(2).unsqueeze(3).unsqueeze(4)
            # (B, L, 1, 1, 1)
            x_masked = x * mask_expand
            agent_count = mask.float().sum(dim=1, keepdim=True)
            agent_count = agent_count.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            agent_count = agent_count.clamp(min=1)
            x_pooled = x_masked.sum(dim=1) / agent_count.squeeze(1)
        else:
            x_pooled = x.mean(dim=1)

        # Apply output projection (LayerNorm + Linear)
        x_pooled = rearrange(x_pooled, 'b d h w -> b h w d')
        x_pooled = self.out_norm(x_pooled)
        x_pooled = self.out_proj(x_pooled)
        x_pooled = rearrange(x_pooled, 'b h w d -> b d h w')

        return x_pooled
