"""MotionHead ported from UniAD for mmdet3d 1.4.

Predicts future trajectories for detected agents using MotionFormer decoder.
Closely matches the original UniAD MotionHead forward pass and TrajLoss.
"""
import math
import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Tuple
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from .functional import (
    bivariate_gaussian_activation,
    norm_points,
    pos2posemb2d,
    anchor_coordinate_transform,
    nonlinear_smoother,
)
from .motion_modules import MotionTransformerDecoder


# ---------------------------------------------------------------------------
# TrajLoss helper functions (ported verbatim from UniAD)
# ---------------------------------------------------------------------------

def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute average displacement error for the best trajectory in a set.

    Args:
        traj: (batch_size, num_modes, sequence_length, 5) predictions.
        traj_gt: (batch_size, sequence_length, 2) ground truth.
        masks: (batch_size, sequence_length) where 1 = INVALID.

    Returns:
        errs: (batch_size,) min ADE per sample.
        inds: (batch_size,) best mode index per sample.
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / \
        torch.clip(torch.sum((1 - masks_rpt), dim=2), min=1)
    err, inds = torch.min(err, dim=1)
    return err, inds


def min_fde(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute final displacement error for the best trajectory in a set.

    Args:
        traj: (batch_size, num_modes, sequence_length, 5) predictions.
        traj_gt: (batch_size, sequence_length, 2) ground truth.
        masks: (batch_size, sequence_length) where 1 = INVALID.

    Returns:
        errs: (batch_size,) min FDE per sample.
        inds: (batch_size,) best mode index per sample.
    """
    num_modes = traj.shape[1]
    lengths = torch.sum(1 - masks, dim=1).long()
    valid_mask = lengths > 0
    traj = traj[valid_mask]
    traj_gt = traj_gt[valid_mask]
    masks = masks[valid_mask]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    lengths = torch.sum(1 - masks, dim=1).long()
    inds = lengths.unsqueeze(1).unsqueeze(
        2).unsqueeze(3).repeat(1, num_modes, 1, 2) - 1

    traj_last = torch.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
    traj_gt_last = torch.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)

    err = traj_gt_last - traj_last[..., 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=2)
    err = torch.pow(err, exponent=0.5)
    err, inds = torch.min(err, dim=1)
    return err, inds


def miss_rate(traj: torch.Tensor, traj_gt: torch.Tensor,
              masks: torch.Tensor, dist_thresh: float = 2) -> torch.Tensor:
    """Compute miss rate for a mini batch of trajectories.

    Args:
        traj: (batch_size, num_modes, sequence_length, 5) predictions.
        traj_gt: (batch_size, sequence_length, 2) ground truth.
        masks: (batch_size, sequence_length) where 1 = INVALID.
        dist_thresh: distance threshold for miss rate.

    Returns:
        m_r: scalar miss rate.
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)
    dist, _ = torch.min(dist, dim=1)
    m_r = torch.sum(torch.as_tensor(dist > dist_thresh)) / max(len(dist), 1)
    return m_r


def traj_nll(pred_dist: torch.Tensor, traj_gt: torch.Tensor,
             masks: torch.Tensor) -> torch.Tensor:
    """Compute bivariate Gaussian NLL for best-mode trajectory.

    Note: sig_x / sig_y are PRECISION (= 1/sigma after exp activation in
    bivariate_gaussian_activation). The formula from UniAD:
        ohr = (1 - rho^2)^{-0.5}
        nll = 0.5 * ohr^2 * (sig_x^2*(x-mu_x)^2 + sig_y^2*(y-mu_y)^2
              - 2*rho*sig_x*sig_y*(x-mu_x)*(y-mu_y))
              - log(sig_x * sig_y * ohr) + 1.8379

    Args:
        pred_dist: (batch_size, sequence_length, 5) [mu_x, mu_y, sig_x, sig_y, rho].
        traj_gt: (batch_size, sequence_length, 2).
        masks: (batch_size, sequence_length) where 1 = INVALID.

    Returns:
        nll: (batch_size,) per-sample NLL averaged over valid timesteps.
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    nll = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) +
         torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2) -
         2 * rho * torch.pow(sig_x, 1) * torch.pow(sig_y, 1) *
         (x - mu_x) * (y - mu_y)) - \
        torch.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = torch.sum(nll * (1 - masks), dim=1) / \
        (torch.sum((1 - masks), dim=1) + 1e-5)
    return nll


@MODELS.register_module()
class MotionHead(BaseModule):
    """Motion prediction head using MotionFormer (transformer decoder).

    Takes BEV features + track queries from detection/tracking and predicts
    future trajectories for each agent with multiple modes.

    Closely follows the original UniAD MotionHead:
    - Classification uses LogSoftmax on mode scores
    - Regression uses cumsum trick on (x, y) before bivariate_gaussian_activation
    - Classification branch has LayerNorm
    - Loss uses the exact UniAD TrajLoss formulation

    Args:
        bev_h (int): BEV feature height.
        bev_w (int): BEV feature width.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        embed_dims (int): Embedding dimensions.
        num_anchor (int): Number of motion modes/anchors.
        predict_steps (int): Number of future prediction steps.
        num_decoder_layers (int): Number of MotionFormer decoder layers.
        num_heads (int): Number of attention heads.
        feedforward_dims (int): FFN hidden dimensions.
        num_points (int): Deformable attention sampling points.
        dropout (float): Dropout rate.
        use_bev_interaction (bool): Whether to use BEV deformable attention.
        cls_loss_weight (float): Weight for classification loss.
        nll_loss_weight (float): Weight for NLL regression loss.
        loss_weight_minade (float): Weight for min ADE loss.
        loss_weight_minfde (float): Weight for min FDE loss.
        use_variance (bool): If True, use Gaussian NLL for regression; else use ADE.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 embed_dims=256,
                 num_anchor=6,
                 predict_steps=12,
                 num_decoder_layers=3,
                 num_heads=8,
                 feedforward_dims=512,
                 num_points=4,
                 dropout=0.1,
                 use_bev_interaction=True,
                 # Legacy parameter kept for config compatibility
                 loss_traj_weight=0.5,
                 loss_traj_cls=None,
                 # UniAD TrajLoss parameters
                 cls_loss_weight=0.5,
                 nll_loss_weight=0.5,
                 loss_weight_minade=0.0,
                 loss_weight_minfde=0.25,
                 use_variance=True,
                 # K-Means anchor initialization
                 anchor_path=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.anchor_path = anchor_path

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.predict_steps = predict_steps

        # TrajLoss weights
        self.cls_loss_weight = cls_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.loss_weight_minade = loss_weight_minade
        self.loss_weight_minfde = loss_weight_minfde
        self.use_variance = use_variance

        # MotionFormer decoder
        self.motion_decoder = MotionTransformerDecoder(
            num_layers=num_decoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            num_points=num_points,
            dropout=dropout,
            use_bev_interaction=use_bev_interaction,
        )

        # Anchor embedding (learnable intention queries)
        self.anchor_embed = nn.Embedding(num_anchor, embed_dims)

        # K-Means anchor trajectory templates: used as regression bias
        # Each mode starts predicting from a different trajectory template,
        # preventing mode collapse. Shape: (P, T, 2) registered as buffer.
        # Initialize as None buffer; set in init_weights if anchor_path
        self.register_buffer('_anchor_traj_bias', None)

        # Trajectory regression branches (one per decoder layer)
        # Original: Linear(D,D) -> ReLU -> Linear(D, T*5)  (NO LayerNorm)
        reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, predict_steps * 5),
        )
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)])

        # Trajectory classification branches (one per decoder layer)
        # Original: Linear(D,D) -> LayerNorm(D) -> ReLU -> Linear(D, 1)
        cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 1),
        )
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_decoder_layers)])

        # Unflatten for cumsum trick: (B, A, P, T*5) -> (B, A, P, T, 5)
        self.unflatten_traj = nn.Unflatten(3, (predict_steps, 5))

        # LogSoftmax for classification (dim=2 = mode dimension after squeeze)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Query positional embedding from box centers (original UniAD)
        # pos2posemb2d outputs embed_dims (128*2=256 by default), then 2-layer MLP
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

        # Embedding layers for MotionTransformerDecoder (original UniAD)
        # These encode spatial information about each agent-mode pair:
        #   agent_level: WHERE each mode's anchor endpoint is (world coords)
        #   scene_level_ego: WHERE each agent is
        #   scene_level_offset: anchor displacement encoding
        self.agent_level_embedding_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.scene_level_ego_embedding_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

    def init_weights(self):
        """Initialize weights, including K-Means anchor initialization.

        Two mechanisms to prevent mode collapse:
        1. Orthogonal anchor embeddings: ensures modes start with maximally
           different representations in embedding space.
        2. Trajectory bias: each mode adds a K-Means cluster center to its
           predicted trajectory, so even with identical regression outputs,
           different modes produce different trajectories.
        """
        super().init_weights()
        if self.anchor_path is not None:
            anchors = np.load(self.anchor_path)  # (K, T, 2)
            K = anchors.shape[0]
            assert K == self.num_anchor, \
                f'Anchor modes {K} != num_anchor {self.num_anchor}'

            # 1. Orthogonal anchor embeddings
            device = self.anchor_embed.weight.device
            D = self.embed_dims
            nn.init.orthogonal_(self.anchor_embed.weight)
            # Scale to typical embedding magnitude
            self.anchor_embed.weight.data.mul_(0.1)

            # 2. Register trajectory bias from K-Means centers
            anchor_traj = torch.tensor(
                anchors, dtype=torch.float32, device=device)  # (K, T, 2)
            self._anchor_traj_bias = anchor_traj

            try:
                from mmengine.logging import MMLogger
                log = MMLogger.get_current_instance()
                log.info(
                    f'Loaded K-Means anchors from {self.anchor_path}: '
                    f'{K} modes, traj bias range '
                    f'[{anchors.min():.1f}, {anchors.max():.1f}]')
            except Exception:
                print(f'Loaded K-Means anchors from {self.anchor_path}')

    def forward(self,
                bev_embed,
                track_query,
                track_boxes,
                lane_query=None,
                lane_query_pos=None):
        """Forward pass.

        Args:
            bev_embed: (H*W, B, D) or (B, D, H, W) BEV features.
            track_query: (B, A, D) track query embeddings from detector.
            track_boxes: (B, A, 10) tracked bounding boxes
                [cx, cy, w, l, cz, h, sin, cos, vx, vy].
            lane_query: (B, M, D) lane/map query embeddings (optional).
            lane_query_pos: (B, M, D) lane query positional embeddings.

        Returns:
            dict with:
                all_traj_scores: (num_layers, B, A, P) log-softmax mode scores
                all_traj_preds: (num_layers, B, A, P, T, 5) trajectory predictions
                traj_query: (num_layers, B, A, P, D) trajectory queries
                track_query: (B, A, D) input track queries
                track_query_pos: (B, A, D) positional embedding from box centers
        """
        # Handle BEV format
        if bev_embed.dim() == 4:
            # (B, D, H, W) -> (B, H*W, D)
            B, D, H, W = bev_embed.shape
            bev_embed = bev_embed.flatten(2).permute(0, 2, 1)
        elif bev_embed.dim() == 3 and bev_embed.shape[0] == self.bev_h * self.bev_w:
            # (H*W, B, D) -> (B, H*W, D)
            bev_embed = bev_embed.permute(1, 0, 2)
            B = bev_embed.shape[0]
        else:
            B = bev_embed.shape[0]

        A = track_query.shape[1]
        P = self.num_anchor
        D = self.embed_dims
        device = track_query.device

        # Build track_query_pos from normalized box centers (original UniAD)
        ref_pts_track = track_boxes[:, :, :2]  # (B, A, 2) cx, cy
        ref_pts_norm = norm_points(ref_pts_track, self.pc_range)  # (B, A, 2) in [0,1]
        track_query_pos = self.boxes_query_embedding_layer(
            pos2posemb2d(ref_pts_norm.to(device)))  # (B, A, D)

        # Build per-agent per-mode queries
        anchor_emb = self.anchor_embed.weight  # (P, D)
        anchor_emb = anchor_emb[None, None, :, :].expand(B, A, -1, -1)  # (B, A, P, D)
        track_q_expand = track_query[:, :, None, :].expand(-1, -1, P, -1)  # (B, A, P, D)
        motion_query = anchor_emb + track_q_expand  # (B, A, P, D)
        motion_query = motion_query.reshape(B, A * P, D)

        # Initialize reference points from track box centers
        ref_pts_norm_expand = ref_pts_norm[:, :, None, None, :].expand(
            -1, -1, P, self.predict_steps, -1)  # (B, A, P, T, 2)

        # Compute spatial embeddings for MotionTransformerDecoder
        agent_level_embedding = None
        scene_level_ego_embedding = None
        scene_level_offset_embedding = None

        if self._anchor_traj_bias is not None:
            # K-Means anchor endpoints: last step displacement (P, 2)
            anchor_ep = self._anchor_traj_bias[:, -1, :]

            # Agent-mode endpoint in world coords: agent_center + anchor_disp
            agent_centers = track_boxes[:, :, :2]  # (B, A, 2)
            agent_mode_ep = (agent_centers.unsqueeze(2)
                             + anchor_ep[None, None])  # (B, A, P, 2)
            agent_mode_ep_norm = norm_points(agent_mode_ep, self.pc_range)
            agent_level_embedding = self.agent_level_embedding_layer(
                pos2posemb2d(agent_mode_ep_norm))  # (B, A, P, D)

            # Agent center positions broadcast across modes
            agent_centers_bc = ref_pts_norm.unsqueeze(2).expand(
                -1, -1, P, -1)  # (B, A, P, 2)
            scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
                pos2posemb2d(agent_centers_bc))  # (B, A, P, D)

            # Anchor displacement encoding (same for all agents per mode)
            anchor_ep_bc = anchor_ep[None, None].expand(
                B, A, -1, -1)  # (B, A, P, 2)
            offset_norm = norm_points(anchor_ep_bc, self.pc_range)
            scene_level_offset_embedding = (
                self.scene_level_offset_embedding_layer(
                    pos2posemb2d(offset_norm)))  # (B, A, P, D)

        # Spatial shapes for deformable attention
        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], device=device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=device, dtype=torch.long)

        # Run MotionFormer decoder
        intermediates, intermediate_refs = self.motion_decoder(
            query=motion_query,
            track_query=track_query,
            lane_query=lane_query,
            lane_query_pos=lane_query_pos,
            bev_embed=bev_embed,
            reference_points=ref_pts_norm_expand,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            num_agents=A,
            num_modes=P,
            reg_branches=None,  # We handle regression outside
            track_query_pos=track_query_pos,
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
        )

        # Decode trajectories from each layer
        all_traj_preds = []
        all_traj_scores = []
        all_traj_queries = []

        for lid, inter_query in enumerate(intermediates):
            # inter_query: (B, A*P, D)
            q = inter_query.reshape(B, A, P, D)

            # --- Classification branch ---
            # cls_branches output: (B, A, P, 1)
            outputs_class = self.cls_branches[lid](q)
            # Squeeze last dim -> (B, A, P), then LogSoftmax over mode dim
            outputs_class = self.log_softmax(outputs_class.squeeze(-1))
            all_traj_scores.append(outputs_class)

            # --- Regression branch with cumsum trick ---
            tmp = self.reg_branches[lid](q)  # (B, A, P, T*5)
            tmp = self.unflatten_traj(tmp)   # (B, A, P, T, 5)

            # Cumsum trick on (x, y) channels BEFORE activation
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            # Add K-Means trajectory bias: each mode starts from its
            # cluster center trajectory, preventing mode collapse.
            # anchor_traj_bias: (P, T, 2) → broadcast to (B, A, P, T, 2)
            if self._anchor_traj_bias is not None:
                bias = self._anchor_traj_bias  # (P, T, 2)
                tmp[..., :2] = tmp[..., :2] + bias[None, None, :, :, :]

            # Apply bivariate gaussian activation per batch
            for bs in range(B):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            all_traj_preds.append(tmp)

            all_traj_queries.append(q)

        all_traj_preds = torch.stack(all_traj_preds)    # (L, B, A, P, T, 5)
        all_traj_scores = torch.stack(all_traj_scores)   # (L, B, A, P)
        all_traj_queries = torch.stack(all_traj_queries)  # (L, B, A, P, D)

        return dict(
            all_traj_scores=all_traj_scores,
            all_traj_preds=all_traj_preds,
            traj_query=all_traj_queries,
            track_query=track_query,
            track_query_pos=track_query_pos,
        )

    def _match_preds_to_gt(self, pred_boxes, gt_boxes):
        """Match predicted agents to GT agents using Hungarian matching.

        Uses L2 distance between predicted detection centers and GT bbox
        centers in BEV.

        Args:
            pred_boxes: (A, 10) predicted boxes [cx, cy, w, l, cz, h, ...].
            gt_boxes: (N_gt, 7) or (N_gt, 2+) GT boxes, at least [cx, cy].

        Returns:
            pred_indices: (M,) indices into pred_boxes.
            gt_indices: (M,) indices into gt_boxes.
        """
        from scipy.optimize import linear_sum_assignment

        A = pred_boxes.shape[0]
        N = gt_boxes.shape[0]
        if A == 0 or N == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        # Cost matrix: L2 distance between centers
        pred_xy = pred_boxes[:, :2].detach().cpu().numpy()  # (A, 2)
        gt_xy = gt_boxes[:, :2].detach().cpu().numpy()      # (N, 2)
        cost = ((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(-1)  # (A, N)

        row_ind, col_ind = linear_sum_assignment(cost)
        return (torch.tensor(row_ind, dtype=torch.long),
                torch.tensor(col_ind, dtype=torch.long))

    def _compute_traj_loss(self, log_probs, traj_preds, gt_future_traj,
                           gt_future_traj_valid_mask):
        """Compute TrajLoss exactly matching UniAD's TrajLoss.forward().

        Args:
            log_probs: (N, P) log-softmax mode probabilities.
            traj_preds: (N, P, T, 5) predicted trajectories.
            gt_future_traj: (N, T, 2) ground truth future trajectories.
            gt_future_traj_valid_mask: (N, T) validity mask (1 = valid, 0 = invalid).

        Returns:
            loss_traj, l_class, l_reg, l_minade, l_minfde, l_mr
        """
        traj = traj_preds      # (N, P, T, 5)
        traj_gt = gt_future_traj  # (N, T, 2)
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # UniAD masks convention: masks=1 means INVALID
        masks = 1 - gt_future_traj_valid_mask.to(traj.dtype)

        # min FDE
        l_minfde, _ = min_fde(traj, traj_gt, masks)
        try:
            l_mr = miss_rate(traj, traj_gt, masks)
        except Exception:
            l_mr = torch.zeros_like(l_minfde) if l_minfde.numel() > 0 else \
                torch.tensor(0.0, device=traj.device)

        # min ADE -> selects best mode
        l_minade, inds = min_ade(traj, traj_gt, masks)

        # Gather best mode predictions
        inds_rep = inds.repeat(
            sequence_length, pred_params, 1, 1).permute(3, 2, 0, 1)
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1)  # (N, T, 5)

        # Regression loss
        if self.use_variance:
            l_reg = traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = l_minade

        # Classification loss: NLL of LogSoftmax
        l_class = -torch.squeeze(log_probs.gather(1, inds.unsqueeze(1)))

        # Normalize by batch size
        l_reg = torch.sum(l_reg) / (batch_size + 1e-5)
        l_class = torch.sum(l_class) / (batch_size + 1e-5)
        l_minade = torch.sum(l_minade) / (batch_size + 1e-5)
        l_minfde = torch.sum(l_minfde) / (batch_size + 1e-5)

        # Total loss
        loss = (l_class * self.cls_loss_weight +
                l_reg * self.nll_loss_weight +
                l_minade * self.loss_weight_minade +
                l_minfde * self.loss_weight_minfde)

        return loss, l_class, l_reg, l_minade, l_minfde, l_mr

    def loss(self, outs_motion, gt_fut_traj, gt_fut_traj_mask,
             track_boxes=None, gt_bboxes_3d=None):
        """Compute motion prediction losses matching UniAD's TrajLoss.

        Uses:
        - Hungarian matching to align predicted agents with GT agents
        - TrajLoss (min_ade mode selection, Gaussian NLL regression, NLL classification)

        Args:
            outs_motion: Output dict from forward().
            gt_fut_traj: (B, N_gt, T, 2) ground truth future trajectories.
            gt_fut_traj_mask: (B, N_gt, T) or (B, N_gt, T, 2) validity mask (1=valid).
            track_boxes: (B, A, 10) predicted track boxes (for matching).
            gt_bboxes_3d: list of (N_gt, 7) GT bboxes (for matching).

        Returns:
            dict: Loss dict with loss_traj, l_class, l_reg, min_ade, min_fde, mr
                  per decoder layer.
        """
        all_traj_preds = outs_motion['all_traj_preds']    # (L, B, A, P, T, 5)
        all_traj_scores = outs_motion['all_traj_scores']   # (L, B, A, P) log-probs
        num_layers = all_traj_preds.shape[0]
        B = all_traj_preds.shape[1]
        device = all_traj_preds.device

        # --- Hungarian matching per sample ---
        batch_pred_idx = []
        batch_gt_idx = []
        for b in range(B):
            if track_boxes is not None and gt_bboxes_3d is not None:
                pred_b = track_boxes[b]  # (A, 10)
                gt_b = gt_bboxes_3d[b] if isinstance(gt_bboxes_3d, list) else gt_bboxes_3d[b]
                if hasattr(gt_b, 'tensor'):
                    gt_b = gt_b.tensor
                pi, gi = self._match_preds_to_gt(pred_b, gt_b.to(device))
            else:
                # Fallback: match sequentially
                A = all_traj_preds.shape[2]
                N_gt = gt_fut_traj.shape[1]
                M = min(A, N_gt)
                pi = torch.arange(M, dtype=torch.long)
                gi = torch.arange(M, dtype=torch.long)
            batch_pred_idx.append(pi)
            batch_gt_idx.append(gi)

        # Collect per-layer losses
        losses_per_layer = []
        for lid in range(num_layers):
            traj_pred = all_traj_preds[lid]    # (B, A, P, T, 5)
            traj_score = all_traj_scores[lid]  # (B, A, P)  log-probs
            T = traj_pred.shape[3]

            # Gather matched predictions and GT across batch
            traj_prob_all = []
            traj_preds_all = []
            gt_traj_all = []
            gt_mask_all = []

            for b in range(B):
                pi, gi = batch_pred_idx[b], batch_gt_idx[b]
                M = len(pi)
                if M == 0:
                    continue

                # Matched predictions: (M, P, T, 5) and (M, P) log-probs
                batch_traj_preds = traj_pred[b, pi]
                batch_traj_prob = traj_score[b, pi]

                # Ground truth: (M, T, 2) and mask (M, T)
                gt_xy = gt_fut_traj[b, gi, :T, :2]
                mask = gt_fut_traj_mask[b, gi, :T]
                if mask.dim() == 3:
                    # (M, T, 2) -> (M, T): valid if all dims valid
                    mask = torch.all(mask > 0, dim=-1).float()
                else:
                    mask = mask.float()

                traj_preds_all.append(batch_traj_preds)
                traj_prob_all.append(batch_traj_prob)
                gt_traj_all.append(gt_xy)
                gt_mask_all.append(mask)

            if len(traj_preds_all) == 0:
                # No matched agents: return zero losses
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                losses_per_layer.append(
                    (zero, zero, zero, zero, zero, torch.tensor(0.0, device=device)))
                continue

            traj_preds_all = torch.cat(traj_preds_all, dim=0)  # (N_total, P, T, 5)
            traj_prob_all = torch.cat(traj_prob_all, dim=0)    # (N_total, P)
            gt_traj_all = torch.cat(gt_traj_all, dim=0)        # (N_total, T, 2)
            gt_mask_all = torch.cat(gt_mask_all, dim=0)         # (N_total, T)

            # Compute TrajLoss
            loss_traj, l_class, l_reg, l_minade, l_minfde, l_mr = \
                self._compute_traj_loss(
                    traj_prob_all, traj_preds_all,
                    gt_traj_all, gt_mask_all)

            losses_per_layer.append(
                (loss_traj, l_class, l_reg, l_minade, l_minfde, l_mr))

        # Format output dict (same as original UniAD)
        loss_dict = dict()

        # Last layer gets unprefixed keys
        loss_dict['loss_traj'] = losses_per_layer[-1][0]
        loss_dict['l_class'] = losses_per_layer[-1][1]
        loss_dict['l_reg'] = losses_per_layer[-1][2]
        loss_dict['min_ade'] = losses_per_layer[-1][3]
        loss_dict['min_fde'] = losses_per_layer[-1][4]
        loss_dict['mr'] = losses_per_layer[-1][5]

        # Earlier layers get d{i}. prefix
        for i, loss_tuple in enumerate(losses_per_layer[:-1]):
            loss_dict[f'd{i}.loss_traj'] = loss_tuple[0]
            loss_dict[f'd{i}.l_class'] = loss_tuple[1]
            loss_dict[f'd{i}.l_reg'] = loss_tuple[2]
            loss_dict[f'd{i}.min_ade'] = loss_tuple[3]
            loss_dict[f'd{i}.min_fde'] = loss_tuple[4]
            loss_dict[f'd{i}.mr'] = loss_tuple[5]

        return loss_dict
