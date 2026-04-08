"""DETR3D-style detection head for BEV features (ported from UniAD).

Takes flattened BEV features (H*W, B, D) and uses learnable queries with
deformable cross-attention to produce detection results and rich query
embeddings for downstream tracking and motion prediction.

Architecture matches UniAD's BEVFormerTrackHead.get_detections():
- 901 learnable queries (900 objects + 1 ego/SDC)
- 6-layer transformer decoder with deformable cross-attention
- Per-layer classification, regression, and trajectory branches
- Reference point refinement at each layer
"""

import copy
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS

try:
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
        multi_scale_deformable_attn_pytorch,
    )
    HAS_DEFORMABLE_ATTN = True
except ImportError:
    HAS_DEFORMABLE_ATTN = False
    warnings.warn('MultiScaleDeformableAttn not available, using fallback')


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ===================== Deformable Cross-Attention =====================

class DeformableCrossAttention(nn.Module):
    """Multi-scale deformable attention for BEV cross-attention.

    Samples BEV features at learned offsets around reference points.
    Ported from UniAD's CustomMSDeformableAttention.

    Args:
        embed_dims (int): Embedding dimensions.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature map levels (1 for single BEV).
        num_points (int): Sampling points per query per head per level.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1,
                 num_points=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * \
                 (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, value, query_pos=None, reference_points=None,
                spatial_shapes=None, level_start_index=None,
                key_padding_mask=None):
        """
        Args:
            query: (num_query, bs, D) or (bs, num_query, D)
            value: (H*W, bs, D) BEV features
            query_pos: same shape as query
            reference_points: (bs, num_query, num_levels, 2)
            spatial_shapes: (num_levels, 2) e.g. [[200, 200]]
            level_start_index: (num_levels,) e.g. [0]

        Returns:
            (num_query, bs, D)
        """
        identity = query
        if query_pos is not None:
            query = query + query_pos

        # Ensure (bs, num_query, D) for processing
        need_permute = query.dim() == 3 and query.shape[0] != value.shape[1]
        if need_permute:
            query = query.permute(1, 0, 2)  # (bs, nq, D)
            value = value.permute(1, 0, 2)  # (bs, HW, D)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # Compute sampling locations
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        # Apply deformable attention
        if HAS_DEFORMABLE_ATTN and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, 64)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if need_permute:
            output = output.permute(1, 0, 2)  # (nq, bs, D)

        return self.dropout(output) + identity


# ===================== Decoder Layer =====================

class DETR3DDecoderLayer(nn.Module):
    """Single DETR3D decoder layer: self-attn → cross-attn → FFN.

    Uses full cross-attention (not deformable) so each query can attend
    to ALL BEV positions via key-query similarity. This enables dynamic
    object localization even on small datasets where deformable attention
    fails to learn proper sampling patterns.

    Args:
        embed_dims (int): Embedding dimensions.
        num_heads (int): Number of attention heads.
        feedforward_dims (int): FFN hidden dimensions.
        dropout (float): Dropout rate.
        num_points (int): Unused (kept for config compatibility).
    """

    def __init__(self, embed_dims=256, num_heads=8, feedforward_dims=2048,
                 dropout=0.1, num_points=4):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (full attention with BEV)
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.dropout_cross = nn.Dropout(dropout)

        # FFN
        self.linear1 = nn.Linear(embed_dims, feedforward_dims)
        self.linear2 = nn.Linear(feedforward_dims, embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, value, query_pos=None,
                reference_points=None, spatial_shapes=None,
                level_start_index=None, key_padding_mask=None,
                bev_pos=None, **kwargs):
        """
        Args:
            query: (num_query, bs, D)
            value: (H*W, bs, D) BEV features
            query_pos: (num_query, bs, D)
            reference_points: (bs, num_query, 1, 2) — unused but kept for API
            bev_pos: (H*W, bs, D) BEV positional encoding

        Returns:
            (num_query, bs, D)
        """
        # Self-attention
        q = k = query + query_pos if query_pos is not None else query
        tgt = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout1(tgt)
        query = self.norm1(query)

        # Cross-attention with BEV (full attention)
        # query with pos → Q; BEV with pos → K; BEV → V
        q_cross = query + query_pos if query_pos is not None else query
        k_cross = value + bev_pos if bev_pos is not None else value
        tgt = self.cross_attn(q_cross, k_cross, value=value)[0]
        query = query + self.dropout_cross(tgt)
        query = self.norm2(query)

        # FFN
        tgt = self.linear2(self.dropout2(F.relu(self.linear1(query))))
        query = query + self.dropout3(tgt)
        query = self.norm3(query)

        return query


# ===================== Detection Transformer Decoder =====================

class DetectionTransformerDecoder(nn.Module):
    """Multi-layer DETR3D decoder with reference point refinement.

    Matches UniAD's DetectionTransformerDecoder exactly:
    - Iterates through decoder layers
    - Refines reference points via regression branches
    - Returns intermediate states and reference points

    Args:
        num_layers (int): Number of decoder layers.
        embed_dims (int): Embedding dimensions.
        num_heads (int): Number of attention heads.
        feedforward_dims (int): FFN hidden dimensions.
        dropout (float): Dropout rate.
        num_points (int): Deformable attention sampling points.
        return_intermediate (bool): Return intermediate outputs.
    """

    def __init__(self, num_layers=6, embed_dims=256, num_heads=8,
                 feedforward_dims=2048, dropout=0.1, num_points=4,
                 return_intermediate=True):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.layers = nn.ModuleList([
            DETR3DDecoderLayer(
                embed_dims=embed_dims, num_heads=num_heads,
                feedforward_dims=feedforward_dims, dropout=dropout,
                num_points=num_points)
            for _ in range(num_layers)
        ])

    def forward(self, query, value, query_pos=None,
                reference_points=None, reg_branches=None,
                spatial_shapes=None, level_start_index=None,
                key_padding_mask=None, bev_pos=None, **kwargs):
        """
        Args:
            query: (num_query, bs, D)
            value: (H*W, bs, D) BEV features
            query_pos: (num_query, bs, D)
            reference_points: (bs, num_query, 3) normalized [0,1]
            reg_branches: nn.ModuleList of regression heads
            bev_pos: (H*W, bs, D) BEV positional encoding

        Returns:
            inter_states: (num_layers, num_query, bs, D)
            inter_references: (num_layers, bs, num_query, 3)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)

            if self.training:
                output = torch.utils.checkpoint.checkpoint(
                    layer, output, value,
                    query_pos, reference_points_input,
                    spatial_shapes, level_start_index,
                    key_padding_mask, bev_pos,
                    use_reentrant=False)
            else:
                output = layer(
                    output, value,
                    query_pos=query_pos,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                    bev_pos=bev_pos)

            # Refine reference points
            if reg_branches is not None:
                # output is (nq, bs, D), need (bs, nq, D) for branch
                output_for_branch = output.permute(1, 0, 2)
                tmp = reg_branches[lid](output_for_branch)

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + \
                    inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + \
                    inverse_sigmoid(reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return (torch.stack(intermediate),
                    torch.stack(intermediate_reference_points))

        return output, reference_points


# ===================== DETR3D Head =====================

@MODELS.register_module()
class DETR3DHead(nn.Module):
    """DETR3D detection head matching UniAD's BEVFormerTrackHead.

    Produces detection results and rich 256-d query embeddings
    for tracking and motion prediction.

    Args:
        embed_dims (int): Embedding dimension. Default 256.
        num_query (int): Number of object queries. Default 900.
        num_classes (int): Number of classes. Default 1.
        num_decoder_layers (int): Number of decoder layers. Default 6.
        num_heads (int): Number of attention heads. Default 8.
        feedforward_dims (int): FFN hidden dimension. Default 2048.
        dropout (float): Dropout rate. Default 0.1.
        num_points (int): Deformable attention sampling points. Default 4.
        pc_range (list): Point cloud range [x_min,y_min,z_min,x_max,y_max,z_max].
        code_size (int): Bbox regression dimensions. Default 10.
        code_weights (list): Loss weights for each bbox dimension.
        bev_h (int): BEV height. Default 200.
        bev_w (int): BEV width. Default 200.
        past_steps (int): Past trajectory prediction steps. Default 4.
        fut_steps (int): Future trajectory prediction steps. Default 4.
        num_cls_fcs (int): FC layers in classification branch. Default 2.
        num_reg_fcs (int): FC layers in regression branch. Default 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_query=900,
                 num_classes=1,
                 num_decoder_layers=6,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.1,
                 num_points=4,
                 pc_range=None,
                 code_size=10,
                 code_weights=None,
                 bev_h=200,
                 bev_w=200,
                 past_steps=4,
                 fut_steps=4,
                 num_cls_fcs=2,
                 num_reg_fcs=2):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.code_size = code_size
        self.past_steps = past_steps
        self.fut_steps = fut_steps

        if code_weights is None:
            code_weights = [1.0] * 8 + [0.2, 0.2]
        self.code_weights = code_weights

        # Learnable query embeddings: pos(D) + feat(D) = 2D
        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)

        # Reference point predictor: feat(D) → 3D point
        self.reference_points_fc = nn.Linear(embed_dims, 3)

        # BEV positional encoding
        self.bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)

        # Transformer decoder
        self.decoder = DetectionTransformerDecoder(
            num_layers=num_decoder_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
            num_points=num_points,
            return_intermediate=True)

        # Per-layer prediction heads
        num_pred = num_decoder_layers
        self._build_branches(num_pred, num_cls_fcs, num_reg_fcs)

    def _build_branches(self, num_pred, num_cls_fcs, num_reg_fcs):
        """Build classification, regression, and trajectory branches."""
        # Classification branch
        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        cls_branch = nn.Sequential(*cls_branch)

        # Regression branch
        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # Past trajectory branch
        traj_dim = (self.past_steps + self.fut_steps) * 2
        traj_branch = []
        for _ in range(num_reg_fcs):
            traj_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_branch.append(nn.ReLU(inplace=True))
        traj_branch.append(nn.Linear(self.embed_dims, traj_dim))
        traj_branch = nn.Sequential(*traj_branch)

        # Clone for each decoder layer
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(cls_branch) for _ in range(num_pred)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(num_pred)])
        self.traj_branches = nn.ModuleList(
            [copy.deepcopy(traj_branch) for _ in range(num_pred)])

    def init_track_instances(self, batch_size, device):
        """Initialize fresh track instances with learnable queries.

        Returns:
            Instances with 'num_query' tracks, each having:
            - query: (num_query, 2*D) learnable embeddings
            - ref_pts: (num_query, 3) initial reference points
            - output_embedding: (num_query, D) zeros
            - obj_idxes: (num_query,) all -1
            - scores: (num_query,) all 0
            - And memory bank fields
        """
        from .track_utils import Instances

        query_embeds = self.query_embedding.weight  # (num_query, 2D)
        query_pos, query_feat = query_embeds.split(self.embed_dims, dim=1)
        ref_pts = self.reference_points_fc(query_pos).sigmoid()  # (nq, 3)

        track_instances = Instances((1, 1))
        track_instances.query = query_embeds.clone()  # (nq, 2D)
        track_instances.ref_pts = ref_pts.clone()     # (nq, 3)
        track_instances.output_embedding = torch.zeros(
            self.num_query, self.embed_dims, device=device)
        track_instances.obj_idxes = torch.full(
            (self.num_query,), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (self.num_query,), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            self.num_query, device=device)
        track_instances.scores = torch.zeros(self.num_query, device=device)
        track_instances.iou = torch.zeros(self.num_query, device=device)
        track_instances.pred_logits = torch.zeros(
            self.num_query, self.num_classes, device=device)
        track_instances.pred_boxes = torch.zeros(
            self.num_query, self.code_size, device=device)

        # Memory bank fields
        track_instances.mem_bank = torch.zeros(
            self.num_query, 4, self.embed_dims, device=device)
        track_instances.mem_padding_mask = torch.ones(
            self.num_query, 4, dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            self.num_query, dtype=torch.long, device=device)

        return track_instances

    def get_detections(self, bev_embed, object_query_embeds=None,
                       ref_points=None, img_metas=None):
        """Run DETR3D decoder to get detections.

        Matches UniAD's BEVFormerTrackHead.get_detections() interface.

        Args:
            bev_embed: (H*W, bs, D) BEV features
            object_query_embeds: (num_query, 2*D) or None
            ref_points: (num_query, 3) or None
            img_metas: batch metadata

        Returns:
            dict with:
                all_cls_scores: (num_layers, nq, bs, num_classes)
                all_bbox_preds: (num_layers, nq, bs, code_size)
                all_traj_preds: (num_layers, nq, bs, T, 2)
                last_ref_points: (nq, bs, 3)
                query_feats: (num_layers, nq, bs, D)
        """
        assert bev_embed.shape[0] == self.bev_h * self.bev_w, \
            f'BEV embed shape {bev_embed.shape[0]} != {self.bev_h * self.bev_w}'

        bs = bev_embed.shape[1]
        device = bev_embed.device

        if object_query_embeds is None:
            object_query_embeds = self.query_embedding.weight

        # Split query into pos + feat
        query_pos, query = object_query_embeds.split(self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # (bs, nq, D)
        query = query.unsqueeze(0).expand(bs, -1, -1)          # (bs, nq, D)

        if ref_points is None:
            ref_points = self.reference_points_fc(
                query_pos[:, :, :self.embed_dims]).sigmoid()
        else:
            ref_points = ref_points.unsqueeze(0).expand(bs, -1, -1)
            # ref_points already in [0,1] from init_track_instances()
            # or from previous frame's sigmoid(last_ref_points)

        init_reference = ref_points.clone()

        # (bs, nq, D) → (nq, bs, D) for decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        spatial_shapes = torch.tensor(
            [[self.bev_h, self.bev_w]], device=device)
        level_start_index = torch.tensor([0], device=device)

        # BEV positional encoding: (H*W, D) → (H*W, bs, D)
        bev_pos = self.bev_embedding.weight.unsqueeze(1).expand(
            -1, bs, -1)  # (H*W, bs, D)

        # Run decoder
        hs, inter_references = self.decoder(
            query=query,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=ref_points,
            reg_branches=self.reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_pos=bev_pos)
        # hs: (num_layers, nq, bs, D)
        # inter_references: (num_layers, bs, nq, 3)

        # Decode predictions per layer
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []

        for lvl in range(hs.shape[0]):
            hs_lvl = hs[lvl]  # (nq, bs, D)

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs_lvl)
            tmp = self.reg_branches[lvl](hs_lvl.permute(1, 0, 2))
            # tmp: (bs, nq, code_size)

            # Add reference point offsets for x, y, z
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # Last ref points for tracking velocity update
            last_ref = torch.cat(
                [tmp[..., 0:2], tmp[..., 4:5]], dim=-1)  # (bs, nq, 3)

            # Denormalize to world coordinates
            tmp[..., 0:1] = tmp[..., 0:1] * \
                (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * \
                (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * \
                (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            outputs_coords.append(tmp.permute(1, 0, 2))  # (nq, bs, code_size)

            # Trajectory predictions
            traj = self.traj_branches[lvl](hs_lvl)
            T = self.past_steps + self.fut_steps
            traj = traj.view(traj.shape[0], traj.shape[1], T, 2)
            outputs_trajs.append(traj)

            outputs_classes.append(outputs_class)

        all_cls_scores = torch.stack(outputs_classes)     # (L, nq, bs, C)
        all_bbox_preds = torch.stack(outputs_coords)       # (L, nq, bs, 10)
        all_traj_preds = torch.stack(outputs_trajs)        # (L, nq, bs, T, 2)
        last_ref_points = inverse_sigmoid(last_ref).permute(1, 0, 2)  # (nq, bs, 3)

        return {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'all_traj_preds': all_traj_preds,
            'last_ref_points': last_ref_points,
            'query_feats': hs,  # (L, nq, bs, D)
        }
