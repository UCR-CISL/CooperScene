"""PlanningHead ported from UniAD for mmdet3d 1.4.

Single-mode ego-vehicle planning using BEV features + motion predictions.
"""
import torch
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange


@MODELS.register_module()
class PlanningHead(BaseModule):
    """Single-mode planning head for ego-vehicle trajectory prediction.

    Takes BEV features, SDC (self-driving car) query from motion head,
    and navigation command to predict future ego waypoints.

    Args:
        bev_h (int): BEV height.
        bev_w (int): BEV width.
        embed_dims (int): Embedding dimensions.
        planning_steps (int): Number of future planning steps.
        loss_planning (dict): Planning loss config.
        loss_collision (dict): Collision loss config.
        use_col_optim (bool): Whether to use collision optimization at inference.
        with_adapter (bool): Whether to use BEV adapter conv layers.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=dict(type='mmdet.L1Loss', loss_weight=1.0),
                 loss_collision=None,
                 use_col_optim=False,
                 col_optim_args=dict(
                     occ_filter_range=5.0,
                     sigma=1.0,
                     alpha_collision=5.0,
                 ),
                 with_adapter=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self.use_col_optim = use_col_optim
        self.col_optim_args = col_optim_args

        # Navigation command embedding (3 commands: left, right, straight)
        self.navi_embed = nn.Embedding(3, embed_dims)

        # Planning query (learnable)
        self.plan_query = nn.Embedding(1, embed_dims)

        # Cross-attention: plan_query attends to BEV features
        self.plan_attn = nn.MultiheadAttention(
            embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.plan_norm1 = nn.LayerNorm(embed_dims)

        # Self-attention with motion context
        self.motion_attn = nn.MultiheadAttention(
            embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.plan_norm2 = nn.LayerNorm(embed_dims)

        # FFN
        self.plan_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.plan_norm3 = nn.LayerNorm(embed_dims)

        # Regression head
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),  # x, y per step
        )

        # Optional BEV adapter
        self.with_adapter = with_adapter
        if with_adapter:
            self.bev_adapter = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims, embed_dims, 3, padding=1),
            )

        # Losses
        self.loss_planning = MODELS.build(loss_planning)
        if loss_collision is not None:
            self.loss_collision = MODELS.build(loss_collision)
        else:
            self.loss_collision = None

    def forward(self,
                bev_embed,
                outs_motion=None,
                command=None,
                sdc_query=None):
        """Forward pass.

        Args:
            bev_embed: (H*W, B, D) or (B, D, H, W) BEV features.
            outs_motion: dict from MotionHead (optional, for motion context).
            command: (B,) navigation command index (0=left, 1=right, 2=straight).
            sdc_query: (B, D) SDC embedding from tracker (optional).

        Returns:
            dict: 'sdc_traj' (B, planning_steps, 2) planned waypoints.
        """
        # Handle BEV format
        if bev_embed.dim() == 4:
            B, D, H, W = bev_embed.shape
            if self.with_adapter:
                bev_embed = bev_embed + self.bev_adapter(bev_embed)
            bev_feat = bev_embed.flatten(2).permute(0, 2, 1)  # (B, H*W, D)
        elif bev_embed.dim() == 3 and bev_embed.shape[0] != bev_embed.shape[1]:
            # (H*W, B, D)
            bev_feat = bev_embed.permute(1, 0, 2)  # (B, H*W, D)
            B = bev_feat.shape[0]
        else:
            bev_feat = bev_embed
            B = bev_feat.shape[0]

        # Build planning query
        plan_q = self.plan_query.weight[None, :, :].expand(B, -1, -1)  # (B, 1, D)

        # Add navigation command
        if command is not None:
            cmd_idx = command.long().clamp(0, 2)
            navi_emb = self.navi_embed(cmd_idx)  # (B, D)
            plan_q = plan_q + navi_emb[:, None, :]

        # Add SDC query from tracker/motion
        if sdc_query is not None:
            if sdc_query.dim() == 2:
                sdc_query = sdc_query.unsqueeze(1)  # (B, 1, D)
            plan_q = plan_q + sdc_query

        # Add motion context from MotionHead
        if outs_motion is not None and 'traj_query' in outs_motion:
            traj_q = outs_motion['traj_query'][-1]  # last layer: (B, A, P, D)
            B_m, A_m, P_m, D_m = traj_q.shape
            motion_ctx = traj_q.reshape(B, -1, D_m)  # (B, A*P, D)

            # Cross-attend to motion context
            residual = plan_q
            plan_q = self.plan_norm2(plan_q)
            plan_q = self.motion_attn(plan_q, motion_ctx, motion_ctx)[0]
            plan_q = residual + plan_q

        # Cross-attend to BEV features
        residual = plan_q
        plan_q = self.plan_norm1(plan_q)
        plan_q = self.plan_attn(plan_q, bev_feat, bev_feat)[0]
        plan_q = residual + plan_q

        # FFN
        residual = plan_q
        plan_q = self.plan_norm3(plan_q)
        plan_q = self.plan_ffn(plan_q)
        plan_q = residual + plan_q

        # Regress trajectory
        sdc_traj = self.reg_branch(plan_q.squeeze(1))  # (B, T*2)
        sdc_traj = sdc_traj.view(B, self.planning_steps, 2)

        return dict(sdc_traj=sdc_traj)

    def loss(self, outs_planning, sdc_planning, sdc_planning_mask,
             gt_future_boxes=None):
        """Compute planning losses.

        Args:
            outs_planning: dict from forward().
            sdc_planning: (B, T, 2) ground truth ego trajectory.
            sdc_planning_mask: (B, T, 2) validity mask.

        Returns:
            dict: Loss dict.
        """
        sdc_traj = outs_planning['sdc_traj']  # (B, T, 2)
        T = min(sdc_traj.shape[1], sdc_planning.shape[1])

        losses = dict()
        loss_plan = self.loss_planning(
            sdc_traj[:, :T] * sdc_planning_mask[:, :T],
            sdc_planning[:, :T] * sdc_planning_mask[:, :T])
        losses['loss_planning'] = loss_plan

        return losses
