"""BEVFusion + UniAD End-to-End Model.

Full pipeline matching UniAD's architecture:
    BEVFusion lidar encoder → BEV features (B, 512, 200, 200)
        → BEV Adapter (512→256) → (B, 256, 200, 200) → (40000, B, 256)
            → DETR3D Decoder (901 queries × BEV deformable cross-attention)
                → Detection results + query features
                    → RuntimeTracker (ID assignment)
                    → MemoryBank (temporal cross-attention, 4-frame history)
                    → QIM (query interaction: self-attn + FFN + merge)
                        → Active track queries (N, 256)
                            → MotionHead (MotionFormer, 6 modes × 12 steps)
                                → Trajectory predictions + losses

Temporal processing: processes a queue of frames sequentially,
maintaining track instances across frames (matching UniAD's
forward_track_train pattern).
"""

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet3d.registry import MODELS
from mmengine.model import BaseModel

from .track_utils import Instances, RuntimeTrackerBase, \
    QueryInteractionModule, MemoryBank
from .detr3d_head import DETR3DHead, inverse_sigmoid

logger = logging.getLogger(__name__)


def normalize_bbox(bboxes, pc_range):
    """Normalize bboxes to [0, 1] using pc_range (for Hungarian matching)."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 4:5] if bboxes.shape[-1] > 4 else bboxes[..., 2:3]

    cx = (cx - pc_range[0]) / (pc_range[3] - pc_range[0])
    cy = (cy - pc_range[1]) / (pc_range[4] - pc_range[1])
    cz = (cz - pc_range[2]) / (pc_range[5] - pc_range[2])

    normalized = bboxes.clone()
    normalized[..., 0:1] = cx
    normalized[..., 1:2] = cy
    if bboxes.shape[-1] > 4:
        normalized[..., 4:5] = cz
    else:
        normalized[..., 2:3] = cz
    return normalized


@MODELS.register_module()
class BEVFusionUniADE2E(BaseModel):
    """End-to-end BEVFusion + UniAD model.

    Args:
        bevfusion (dict): BEVFusion model config.
        bev_adapter (dict): BEV feature adapter config.
        detr3d_head (dict): DETR3D detection head config.
        motion_head (dict): Motion prediction head config.
        qim (dict): Query Interaction Module config.
        memory_bank (dict): Memory Bank config.
        tracker (dict): Runtime tracker config.
        map_encoder (dict): BEV map encoder config. Optional.
        planning_head (dict): Planning head config. Optional.
        freeze_perception (bool): Freeze BEVFusion weights.
        bevfusion_checkpoint (str): Path to pretrained BEVFusion.
        num_query (int): Number of object queries.
        num_classes (int): Number of classes.
        pc_range (list): Point cloud range.
        embed_dims (int): Embedding dimensions.
        queue_length (int): Temporal queue length.
    """

    def __init__(self,
                 bevfusion: dict,
                 bev_adapter: dict,
                 detr3d_head: dict,
                 motion_head: dict,
                 qim: dict = None,
                 memory_bank: dict = None,
                 tracker: dict = None,
                 map_encoder: dict = None,
                 planning_head: dict = None,
                 freeze_perception: bool = True,
                 bevfusion_checkpoint: str = None,
                 num_query: int = 900,
                 num_classes: int = 1,
                 pc_range: list = None,
                 embed_dims: int = 256,
                 queue_length: int = 1,
                 use_tracking: bool = True,
                 det_only: bool = False,
                 use_gt_train: bool = False,
                 use_pos_embed: bool = True,
                 data_preprocessor: dict = None,
                 **kwargs):
        # Build data_preprocessor via mmdet3d registry (not mmengine's)
        if data_preprocessor is not None and isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(data_preprocessor=data_preprocessor)

        self.pc_range = pc_range
        self.use_gt_train = use_gt_train
        self.use_pos_embed = use_pos_embed
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.queue_length = queue_length
        self.bevfusion_checkpoint = bevfusion_checkpoint
        self.use_tracking = use_tracking
        self.det_only = det_only

        # ---- Perception ----
        self.bevfusion = MODELS.build(bevfusion)

        # BEV adapter: 512→256 + optional spatial resize
        in_ch = bev_adapter['in_channels']
        out_ch = bev_adapter['out_channels']
        self.bev_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.bev_in_size = bev_adapter.get('in_size', 200)
        self.bev_out_size = bev_adapter.get('out_size', 200)

        # ---- Detection ----
        self.detr3d_head = DETR3DHead(**detr3d_head)

        # ---- Query feature extraction from BEV at detection positions ----
        # Position embedding: encode (x, y, z, dx, dy, dz, sin, cos) → D
        self.pos_embed = nn.Sequential(
            nn.Linear(8, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        # ---- Tracking ----
        if qim is not None:
            self.qim = QueryInteractionModule(**qim)
        else:
            self.qim = QueryInteractionModule(
                embed_dims=embed_dims, num_heads=8,
                feedforward_dims=2048, dropout=0.0)

        if memory_bank is not None:
            self.memory_bank = MemoryBank(**memory_bank)
        else:
            self.memory_bank = MemoryBank(
                embed_dims=embed_dims, num_heads=8,
                feedforward_dims=2048, memory_bank_len=4)

        if tracker is not None:
            self.tracker = RuntimeTrackerBase(**tracker)
        else:
            self.tracker = RuntimeTrackerBase(
                score_thresh=0.5, filter_score_thresh=0.4,
                miss_tolerance=5)

        # ---- Motion prediction ----
        self.motion_head = MODELS.build(motion_head)

        # ---- Optional: Map encoder ----
        if map_encoder is not None:
            self.map_encoder = MODELS.build(map_encoder)
        else:
            self.map_encoder = None

        # ---- Optional: Planning ----
        if planning_head is not None:
            self.planning_head = MODELS.build(planning_head)
        else:
            self.planning_head = None

        # ---- Temporal tracking buffer ----
        # Maps (scenario, agent_id) → detached track_instances
        self._track_buffer = {}
        self._max_buffer_scenes = 16  # limit buffer size

        # Load and freeze perception
        self._init_perception(freeze_perception)

    def _init_perception(self, freeze: bool):
        """Load pretrained BEVFusion and optionally freeze."""
        if self.bevfusion_checkpoint:
            ckpt = torch.load(self.bevfusion_checkpoint, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # Filter to BEVFusion keys only
            bev_state = {}
            for k, v in state_dict.items():
                if k.startswith('data_preprocessor.'):
                    continue
                bev_state[k] = v

            # Use manual parameter loading to avoid spconv v2 internal
            # weight transposition issues with load_state_dict
            loaded, skipped = 0, 0
            model_state = self.bevfusion.state_dict()
            for k, v in bev_state.items():
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        model_state[k].copy_(v)
                        loaded += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            logger.info(f'Loaded BEVFusion: {loaded} params loaded, '
                        f'{skipped} skipped')

        if freeze:
            for p in self.bevfusion.parameters():
                p.requires_grad = False
            self.bevfusion.eval()
            logger.info('BEVFusion perception frozen')

    # ==================== BEV Feature Extraction ====================

    def _extract_bev_frozen(self, inputs, data_samples):
        """Run frozen BEVFusion backbone to get BEV features.

        Returns:
            bev_feat: (B, 512, H, W) raw BEV features (no grad)
        """
        batch_input_metas = [s.metainfo for s in data_samples]
        feats = self.bevfusion.extract_feat(inputs, batch_input_metas)
        return feats[0]  # (B, 512, H, W)

    def _adapt_bev(self, bev_feat):
        """Apply trainable BEV adapter: 512→256 + flatten.

        Args:
            bev_feat: (B, 512, H, W) from frozen BEVFusion (detached)

        Returns:
            bev_embed: (H*W, B, D) = (40000, B, 256)
        """
        bev_feat = self.bev_proj(bev_feat)  # (B, 256, H, W)
        if bev_feat.shape[-1] != self.bev_out_size:
            bev_feat = F.interpolate(
                bev_feat,
                size=(self.bev_out_size, self.bev_out_size),
                mode='bilinear', align_corners=False)

        B, D, H, W = bev_feat.shape
        # (B, D, H, W) → (H*W, B, D)
        bev_embed = bev_feat.flatten(2).permute(2, 0, 1)
        return bev_embed

    def extract_bev(self, inputs, data_samples):
        """Run BEVFusion and adapt BEV features (for inference).

        Returns:
            bev_embed: (H*W, B, D) = (40000, B, 256)
        """
        with torch.no_grad():
            bev_feat = self._extract_bev_frozen(inputs, data_samples)
        return self._adapt_bev(bev_feat)

    def _adapt_bev_2d(self, bev_feat):
        """Apply trainable BEV adapter: 512→256, keep 2D spatial format.

        Returns:
            bev_2d: (B, D, H, W) = (B, 256, 100, 100)
        """
        bev_feat = self.bev_proj(bev_feat)  # (B, 256, H, W)
        if bev_feat.shape[-1] != self.bev_out_size:
            bev_feat = F.interpolate(
                bev_feat,
                size=(self.bev_out_size, self.bev_out_size),
                mode='bilinear', align_corners=False)
        return bev_feat

    def _sample_bev_at_positions(self, bev_2d, positions):
        """Bilinear-sample BEV features at world coordinate positions.

        Args:
            bev_2d: (B, D, H, W) adapted BEV features
            positions: (N, 2) or (N, 3) world (x, y [, z]) coordinates

        Returns:
            features: (N, D) sampled features
        """
        B, D, H, W = bev_2d.shape
        pc_range = self.pc_range

        # Normalize (x, y) to [-1, 1] for grid_sample
        x_norm = (positions[:, 0] - pc_range[0]) / \
            (pc_range[3] - pc_range[0]) * 2.0 - 1.0
        y_norm = (positions[:, 1] - pc_range[1]) / \
            (pc_range[4] - pc_range[1]) * 2.0 - 1.0

        # grid_sample expects (B, 1, N, 2) grid
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(1)  # (1, 1, N, 2)

        # Sample: bev_2d (B, D, H, W), grid (1, 1, N, 2)
        # Use the first batch item
        sampled = F.grid_sample(
            bev_2d[:1], grid, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        # sampled: (1, D, 1, N)
        features = sampled[0, :, 0, :].T  # (N, D)
        return features

    def _get_query_features_from_boxes(self, bev_2d, bboxes_3d):
        """Create query features by sampling BEV at box centers + pos embed.

        Args:
            bev_2d: (B, D, H, W) adapted BEV features
            bboxes_3d: LiDARInstance3DBoxes or tensor (N, 7)
                Format: [x, y, z, dx, dy, dz, yaw]

        Returns:
            query_feats: (N, D) rich query features for motion head
            query_boxes: (N, 10) box params in DETR3D format
        """
        device = bev_2d.device

        if hasattr(bboxes_3d, 'tensor'):
            box_tensor = bboxes_3d.tensor.to(device)
        else:
            box_tensor = bboxes_3d.to(device)

        if box_tensor.shape[0] == 0:
            return (torch.zeros(0, self.embed_dims, device=device),
                    torch.zeros(0, 10, device=device))

        # Sample BEV features at box centers
        bev_feats = self._sample_bev_at_positions(bev_2d, box_tensor[:, :2])

        if self.use_pos_embed:
            # Position embedding: (x, y, z, dx, dy, dz, sin(yaw), cos(yaw))
            pos_input = torch.cat([
                box_tensor[:, :6],                     # x, y, z, dx, dy, dz
                box_tensor[:, 6:7].sin(),              # sin(yaw)
                box_tensor[:, 6:7].cos(),              # cos(yaw)
            ], dim=-1)  # (N, 8)
            pos_emb = self.pos_embed(pos_input)  # (N, D)
            query_feats = bev_feats + pos_emb  # (N, D)
        else:
            query_feats = bev_feats  # (N, D) pure BEV sampling

        # Convert to DETR3D format: [x, y, w, l, z, h, sin, cos, vx, vy]
        query_boxes = torch.zeros(box_tensor.shape[0], 10, device=device)
        query_boxes[:, 0] = box_tensor[:, 0]        # x
        query_boxes[:, 1] = box_tensor[:, 1]        # y
        query_boxes[:, 2] = box_tensor[:, 3]        # dx → w
        query_boxes[:, 3] = box_tensor[:, 4]        # dy → l
        query_boxes[:, 4] = box_tensor[:, 2]        # z
        query_boxes[:, 5] = box_tensor[:, 5]        # dz → h
        query_boxes[:, 6] = box_tensor[:, 6].sin()  # sin(yaw)
        query_boxes[:, 7] = box_tensor[:, 6].cos()  # cos(yaw)

        return query_feats, query_boxes

    def _run_transfusion(self, bev_feat_raw, data_samples):
        """Run frozen TransFusion head to get detections.

        Args:
            bev_feat_raw: tuple from self.bevfusion.extract_feat()
            data_samples: list of data samples

        Returns:
            list of (bboxes_3d, scores, labels) per batch item
        """
        batch_input_metas = [s.metainfo for s in data_samples]
        with torch.no_grad():
            results = self.bevfusion.bbox_head.predict(
                bev_feat_raw, batch_input_metas)
        return results

    # ==================== Detection + Tracking ====================

    def _forward_single_frame(self, bev_embed, track_instances,
                               gt_bboxes_3d=None, gt_labels_3d=None,
                               gt_inds=None):
        """Process a single frame: detect → match → track → memory.

        Args:
            bev_embed: (H*W, B, D) BEV features
            track_instances: current track instances
            gt_bboxes_3d: list of GT bboxes (for matching during training)
            gt_labels_3d: list of GT labels
            gt_inds: list of GT instance indices (for tracking)

        Returns:
            dict with updated track_instances and detection outputs
        """
        device = bev_embed.device

        # Run DETR3D detection
        det_output = self.detr3d_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts)

        # Extract last layer predictions
        num_layers = det_output['all_cls_scores'].shape[0]
        last_cls = det_output['all_cls_scores'][-1]  # (nq, bs, C)
        last_bbox = det_output['all_bbox_preds'][-1]  # (nq, bs, 10)
        last_feats = det_output['query_feats'][-1]    # (nq, bs, D)
        last_ref = det_output['last_ref_points']       # (nq, bs, 3)

        # Update track instances with detection results
        nq = last_cls.shape[0]
        # Scores: max class score after sigmoid
        scores = last_cls[:, 0].sigmoid().max(dim=-1).values  # (nq,)
        track_instances.scores = scores
        track_instances.pred_logits = last_cls[:, 0]      # (nq, C)
        track_instances.pred_boxes = last_bbox[:, 0]       # (nq, 10)
        track_instances.output_embedding = last_feats[:, 0]  # (nq, D)
        track_instances.ref_pts = last_ref[:, 0].sigmoid()  # (nq, 3)

        # Store all layer query features for downstream
        # Use last decoder layer's query features fused across layers
        all_query_feats = det_output['query_feats']  # (L, nq, bs, D)

        # Hungarian matching with GT (training only)
        matched_idxes = None
        if gt_bboxes_3d is not None and self.training:
            matched_idxes = self._match_with_gt(
                track_instances, gt_bboxes_3d, gt_labels_3d, gt_inds)

        # Update tracker (assign IDs)
        self.tracker.update(track_instances)

        # Memory bank: temporal cross-attention
        track_instances = self.memory_bank(track_instances)

        # QIM: select active + update embeddings + merge with init
        init_instances = self.detr3d_head.init_track_instances(
            1, device)
        out_instances = self.qim({
            'track_instances': track_instances,
            'init_track_instances': init_instances,
        })

        return {
            'track_instances': out_instances,
            'det_output': det_output,
            'matched_idxes': matched_idxes,
            'all_query_feats': all_query_feats,
            'bev_embed': bev_embed,
        }

    def _match_with_gt(self, track_instances, gt_bboxes_3d, gt_labels_3d,
                        gt_inds=None):
        """Hungarian matching between detections and GT.

        Updates track_instances.matched_gt_idxes and .iou.
        Returns list of matched indices per batch.
        """
        pred_boxes = track_instances.pred_boxes  # (nq, 10)
        pred_logits = track_instances.pred_logits  # (nq, C)
        pred_scores = pred_logits.sigmoid()

        # Handle gt_bboxes_3d
        if isinstance(gt_bboxes_3d, list):
            gt_bboxes = gt_bboxes_3d[0]  # batch size 1
            gt_labels = gt_labels_3d[0]
        else:
            gt_bboxes = gt_bboxes_3d
            gt_labels = gt_labels_3d

        if hasattr(gt_bboxes, 'tensor'):
            gt_tensor = gt_bboxes.tensor.to(pred_boxes.device)
        else:
            gt_tensor = gt_bboxes.to(pred_boxes.device)
        gt_labels = gt_labels.to(pred_boxes.device)

        num_gt = gt_tensor.shape[0]
        num_pred = pred_boxes.shape[0]

        if num_gt == 0:
            track_instances.matched_gt_idxes[:] = -1
            track_instances.iou[:] = 0
            return torch.full((num_pred,), -1, dtype=torch.long,
                              device=pred_boxes.device)

        # Classification cost: focal loss based
        cls_cost = -pred_scores[:, gt_labels.long()]  # (nq, num_gt)

        # L1 regression cost on normalized centers
        pred_centers = pred_boxes[:, :2]  # (nq, 2) world coords
        gt_centers = gt_tensor[:, :2]     # (num_gt, 2) world coords

        # Normalize
        pred_norm = pred_centers.clone()
        pred_norm[:, 0] = (pred_norm[:, 0] - self.pc_range[0]) / \
                          (self.pc_range[3] - self.pc_range[0])
        pred_norm[:, 1] = (pred_norm[:, 1] - self.pc_range[1]) / \
                          (self.pc_range[4] - self.pc_range[1])
        gt_norm = gt_centers.clone()
        gt_norm[:, 0] = (gt_norm[:, 0] - self.pc_range[0]) / \
                        (self.pc_range[3] - self.pc_range[0])
        gt_norm[:, 1] = (gt_norm[:, 1] - self.pc_range[1]) / \
                        (self.pc_range[4] - self.pc_range[1])

        reg_cost = torch.cdist(pred_norm, gt_norm, p=1)  # (nq, num_gt)

        # Combined cost
        cost = cls_cost * 0.15 + reg_cost * 0.25

        # Hungarian matching
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        row_ind = torch.from_numpy(row_ind).to(pred_boxes.device)
        col_ind = torch.from_numpy(col_ind).to(pred_boxes.device)

        matched = torch.full((num_pred,), -1, dtype=torch.long,
                             device=pred_boxes.device)
        matched[row_ind] = col_ind

        track_instances.matched_gt_idxes = matched

        # Compute IoU for matched pairs (simplified: use center distance)
        iou = torch.zeros(num_pred, device=pred_boxes.device)
        if len(row_ind) > 0:
            dist = torch.norm(
                pred_centers[row_ind] - gt_centers[col_ind], dim=-1)
            # Convert distance to pseudo-IoU (1.0 if dist=0, 0.0 if dist>10m)
            iou[row_ind] = (1.0 - dist / 10.0).clamp(0, 1)
        track_instances.iou = iou

        return matched

    # ==================== Extract Track Queries for Motion ====================

    def _extract_active_tracks(self, track_instances, det_output,
                                matched_idxes):
        """Extract active track query embeddings for MotionHead.

        Returns:
            track_query: (1, 1, N_active, D) - for MotionHead input
            track_boxes: list of (bboxes, scores, labels, bbox_index, mask)
            all_matched_idxes: indices mapping active tracks to GT
        """
        device = track_instances.output_embedding.device

        # Get query features from last decoder layer
        query_feats = det_output['query_feats']  # (L, nq, bs, D)
        # Use last layer
        last_feats = query_feats[-1, :, 0, :]  # (nq, D)

        if matched_idxes is None:
            # Inference: use score-based selection
            active_mask = track_instances.obj_idxes >= 0
        else:
            # Training: use matched tracks
            active_mask = matched_idxes >= 0

        n_active = active_mask.sum().item()
        if n_active == 0:
            # Return empty
            return (torch.zeros(1, 1, 0, self.embed_dims, device=device),
                    None, None)

        active_feats = last_feats[active_mask]  # (N_active, D)
        active_boxes = track_instances.pred_boxes[active_mask]  # (N_active, 10)
        active_scores = track_instances.scores[active_mask]
        active_labels = torch.zeros(n_active, dtype=torch.long, device=device)

        # Format as (1, 1, N_active, D) matching UniAD MotionHead input
        track_query = active_feats[None, None, :, :]

        # Format track_boxes as UniAD expects
        from mmdet3d.structures import LiDARInstance3DBoxes
        box_tensor = torch.zeros(n_active, 7, device=device)
        box_tensor[:, 0] = active_boxes[:, 0]   # x
        box_tensor[:, 1] = active_boxes[:, 1]   # y
        box_tensor[:, 2] = active_boxes[:, 4]   # z
        box_tensor[:, 3] = active_boxes[:, 2]   # dx/w
        box_tensor[:, 4] = active_boxes[:, 3]   # dy/l
        box_tensor[:, 5] = active_boxes[:, 5]   # dz/h
        box_tensor[:, 6] = torch.atan2(
            active_boxes[:, 6], active_boxes[:, 7])  # yaw

        bboxes_3d = LiDARInstance3DBoxes(box_tensor)
        bbox_index = torch.arange(n_active, device=device)
        mask = torch.ones(n_active, dtype=torch.bool, device=device)

        track_bbox_results = [(bboxes_3d, active_scores, active_labels,
                               bbox_index, mask)]

        # Matched GT indices for active tracks
        if matched_idxes is not None:
            active_matched = matched_idxes[active_mask]
        else:
            active_matched = torch.full(
                (n_active,), -1, dtype=torch.long, device=device)

        return track_query, track_bbox_results, [active_matched]

    # ==================== Forward Methods ====================

    def forward(self, inputs=None, data_samples=None, mode='loss',
                inputs_queue=None, data_samples_queue=None, **kwargs):
        if mode == 'loss':
            if self.det_only:
                return self.forward_train_det_only(
                    inputs, data_samples, **kwargs)
            elif inputs_queue is not None:
                # Multi-frame temporal queue training
                return self.forward_train_queue(
                    inputs_queue, data_samples_queue, **kwargs)
            elif self.use_tracking:
                return self.forward_train(inputs, data_samples, **kwargs)
            else:
                return self.forward_train_single_frame(
                    inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.forward_predict(inputs, data_samples, **kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def _store_track_buffer(self, scene_key, track_instances, timestamp):
        """Detach and store track instances for next frame carry-over.

        Args:
            scene_key: (scenario, agent_id) tuple
            track_instances: Instances to store
            timestamp: current frame timestamp
        """
        detached = Instances(track_instances._image_size)
        for k, v in track_instances._fields.items():
            if isinstance(v, torch.Tensor):
                detached.set(k, v.detach().clone())
            elif isinstance(v, list):
                detached.set(k, copy.deepcopy(v))
            else:
                detached.set(k, copy.copy(v))

        self._track_buffer[scene_key] = (detached, timestamp)

        # Evict oldest if buffer exceeds limit
        if len(self._track_buffer) > self._max_buffer_scenes:
            oldest_key = next(iter(self._track_buffer))
            del self._track_buffer[oldest_key]

    def forward_train_det_only(self, inputs, data_samples, **kwargs):
        """Detection-only training (no motion/tracking).

        Trains BEV adapter + DETR3D head only. Faster convergence for
        detection before adding motion head training.
        """
        device = next(self.parameters()).device
        B = len(data_samples)

        self.bevfusion.eval()
        with torch.no_grad():
            bev_feat = self._extract_bev_frozen(inputs, data_samples)
        bev_embed = self._adapt_bev(bev_feat)  # (H*W, B, D)

        all_losses = {}
        for bi in range(B):
            bev_i = bev_embed[:, bi:bi+1, :]
            ds = data_samples[bi]

            gt_bboxes_3d = ds.gt_instances_3d.bboxes_3d
            gt_labels_3d = ds.gt_instances_3d.labels_3d

            det_output = self.detr3d_head.get_detections(bev_i)
            det_losses = self._compute_det_losses(
                det_output, [gt_bboxes_3d], [gt_labels_3d])

            for k, v in det_losses.items():
                all_losses.setdefault(k, []).append(v)

        final_losses = {}
        for k, v_list in all_losses.items():
            final_losses[k] = sum(v_list) / len(v_list)

        return final_losses

    def forward_train(self, inputs, data_samples, **kwargs):
        """Single-frame motion training with GT boxes + BEV sampling.

        Same as forward_train_single_frame. Temporal tracking via
        forward_train_queue when inputs_queue is provided.

        Pipeline per batch item:
            1. BEVFusion (frozen) → BEV features
            2. BEV adapter (trainable) → adapted BEV
            3. GT boxes → sample BEV + pos_embed → query features
            4. MotionHead → trajectory predictions → motion losses
        """
        return self.forward_train_single_frame(inputs, data_samples, **kwargs)

    def forward_train_single_frame(self, inputs, data_samples, **kwargs):
        """Single-frame motion training.

        Supports two modes via use_gt_train:
            - use_gt_train=True:  GT boxes → BEV sampling → MotionHead (isolate motion)
            - use_gt_train=False: TransFusion det → BEV sampling → MotionHead (end-to-end)
        """
        device = next(self.parameters()).device
        B = len(data_samples)

        # 1. Extract BEV features (frozen)
        self.bevfusion.eval()
        batch_input_metas = [s.metainfo for s in data_samples]
        with torch.no_grad():
            bev_feat_raw = self.bevfusion.extract_feat(
                inputs, batch_input_metas)

        # 2. Run TransFusion if needed (frozen)
        tf_results = None
        if not self.use_gt_train:
            with torch.no_grad():
                tf_results = self.bevfusion.bbox_head.predict(
                    bev_feat_raw, batch_input_metas)

        # 3. Adapt BEV features (trainable)
        bev_2d = self._adapt_bev_2d(bev_feat_raw[0])  # (B, 256, H, W)
        bev_embed = bev_2d.flatten(2).permute(2, 0, 1)  # (H*W, B, D)

        all_losses = {}

        for bi in range(B):
            bev_i = bev_embed[:, bi:bi+1, :]  # (H*W, 1, D)
            ds = data_samples[bi]
            meta = ds.metainfo

            gt_bboxes_3d = ds.gt_instances_3d.bboxes_3d

            gt_fut_traj = torch.as_tensor(
                np.array(meta.get('gt_fut_traj', np.zeros((0, 12, 2)))),
                dtype=torch.float32, device=device)
            gt_fut_traj_mask = torch.as_tensor(
                np.array(meta.get('gt_fut_traj_mask', np.zeros((0, 12)))),
                dtype=torch.float32, device=device)

            # 4. Get boxes for query generation
            if self.use_gt_train:
                # GT mode: use GT boxes directly
                box_source = gt_bboxes_3d
                num_boxes = (box_source.tensor.shape[0] if hasattr(
                    box_source, 'tensor') else box_source.shape[0])
            else:
                # TransFusion mode: use detections (filtered by score)
                tf_result = tf_results[bi]
                pred_scores = tf_result.scores_3d
                score_mask = pred_scores > 0.1
                if score_mask.sum() > 0:
                    from mmdet3d.structures import LiDARInstance3DBoxes
                    box_source = LiDARInstance3DBoxes(
                        tf_result.bboxes_3d.tensor[score_mask])
                    num_boxes = score_mask.sum().item()
                else:
                    num_boxes = 0

            if num_boxes == 0:
                all_losses.setdefault('loss_traj', []).append(
                    torch.tensor(0.0, device=device, requires_grad=True))
                continue

            # 5. Create query features from boxes + BEV sampling
            query_feats, query_boxes = self._get_query_features_from_boxes(
                bev_2d[bi:bi+1], box_source)

            track_q = query_feats.unsqueeze(0)      # (1, N, D)
            track_boxes = query_boxes.unsqueeze(0)   # (1, N, 10)

            # Map encoder (optional)
            lane_query = None
            lane_query_pos = None
            if self.map_encoder is not None:
                bev_map = meta.get('bev_map', None)
                if bev_map is not None:
                    if isinstance(bev_map, np.ndarray):
                        bev_map = torch.from_numpy(bev_map).float().to(device)
                    if bev_map.dim() == 3:
                        bev_map = bev_map.unsqueeze(0)
                    lane_query, lane_query_pos = self.map_encoder(bev_map)

            # 6. MotionHead
            outs_motion = self.motion_head(
                bev_embed=bev_i.detach(),
                track_query=track_q,
                track_boxes=track_boxes,
                lane_query=lane_query,
                lane_query_pos=lane_query_pos)

            # 7. Motion loss (Hungarian matching handles det↔GT alignment)
            gt_fut_aligned = gt_fut_traj.unsqueeze(0)       # (1, N_gt, T, 2)
            gt_mask_aligned = gt_fut_traj_mask.unsqueeze(0)  # (1, N_gt, T)

            motion_losses = self.motion_head.loss(
                outs_motion,
                gt_fut_traj=gt_fut_aligned,
                gt_fut_traj_mask=gt_mask_aligned,
                track_boxes=track_boxes,
                gt_bboxes_3d=[gt_bboxes_3d])

            for k, v in motion_losses.items():
                all_losses.setdefault(k, []).append(v)

        # Average losses across batch items
        final_losses = {}
        for k, v_list in all_losses.items():
            final_losses[k] = sum(v_list) / len(v_list)

        return final_losses

    def forward_train_queue(self, inputs_queue, data_samples_queue, **kwargs):
        """Multi-frame temporal training with MemoryBank cross-attention.

        Processes Q consecutive frames sequentially:
        - Each frame: BEV extraction → BEV sampling → create track Instances
        - Track matching across frames using GT vehicle IDs
        - MemoryBank: temporal cross-attention enriches current queries
        - Motion loss only on the LAST frame

        Args:
            inputs_queue: list of Q input dicts, each {'points': [pts]}.
            data_samples_queue: list of Q Det3DDataSample objects.
        """
        device = next(self.parameters()).device
        Q = len(inputs_queue)
        D = self.embed_dims

        # ---- Step 1: Extract + adapt BEV for all frames (frozen) ----
        bev_2ds = []
        bev_embeds = []
        self.bevfusion.eval()
        for t in range(Q):
            with torch.no_grad():
                bev_feat = self._extract_bev_frozen(
                    inputs_queue[t], [data_samples_queue[t]])
            bev_2d = self._adapt_bev_2d(bev_feat)  # (1, 256, H, W)
            bev_embed = bev_2d.flatten(2).permute(2, 0, 1)  # (H*W, 1, D)
            bev_2ds.append(bev_2d)
            bev_embeds.append(bev_embed)

        all_losses = {}
        prev_instances = None  # Instances from previous frame (detached)

        # ---- Step 2: Sequential temporal processing ----
        for t in range(Q):
            bev_2d_t = bev_2ds[t]
            bev_i = bev_embeds[t]
            ds = data_samples_queue[t]
            meta = ds.metainfo
            is_last = (t == Q - 1)

            gt_bboxes_3d = ds.gt_instances_3d.bboxes_3d
            num_gt = (gt_bboxes_3d.tensor.shape[0] if hasattr(
                gt_bboxes_3d, 'tensor') else gt_bboxes_3d.shape[0])

            if num_gt == 0:
                prev_instances = None
                if is_last:
                    all_losses.setdefault('loss_traj', []).append(
                        torch.tensor(0.0, device=device, requires_grad=True))
                continue

            # Get GT vehicle IDs
            gt_vehicle_ids = meta.get('gt_vehicle_ids', None)
            if gt_vehicle_ids is not None:
                if isinstance(gt_vehicle_ids, np.ndarray):
                    gt_vehicle_ids = torch.from_numpy(
                        gt_vehicle_ids).long().to(device)
                elif isinstance(gt_vehicle_ids, list):
                    gt_vehicle_ids = torch.tensor(
                        gt_vehicle_ids, dtype=torch.long, device=device)
            else:
                gt_vehicle_ids = torch.arange(num_gt, device=device)

            # Create track Instances from GT boxes + BEV sampling
            track_instances = self._init_instances_from_boxes(
                bev_2d_t, gt_bboxes_3d, gt_vehicle_ids)

            # Propagate memory bank from previous frame
            if prev_instances is not None:
                self._propagate_memory(track_instances, prev_instances)

            # MemoryBank: temporal cross-attention + store to bank
            track_instances = self.memory_bank(
                track_instances, update_bank=True)

            # Store for next frame (detach to prevent graph explosion)
            prev_instances = self._detach_instances(track_instances)

            # ---- Last frame: MotionHead + loss ----
            if is_last:
                enriched_feats = track_instances.output_embedding  # (N, D)
                track_q = enriched_feats.unsqueeze(0)       # (1, N, D)
                track_boxes = track_instances.pred_boxes.unsqueeze(0)

                # Map encoder
                lane_query, lane_query_pos = None, None
                if self.map_encoder is not None:
                    bev_map = meta.get('bev_map', None)
                    if bev_map is not None:
                        if isinstance(bev_map, np.ndarray):
                            bev_map = torch.from_numpy(
                                bev_map).float().to(device)
                        if bev_map.dim() == 3:
                            bev_map = bev_map.unsqueeze(0)
                        lane_query, lane_query_pos = self.map_encoder(
                            bev_map)

                # MotionHead
                outs_motion = self.motion_head(
                    bev_embed=bev_i.detach(),
                    track_query=track_q,
                    track_boxes=track_boxes,
                    lane_query=lane_query,
                    lane_query_pos=lane_query_pos)

                # Motion loss
                gt_fut_traj = torch.as_tensor(
                    np.array(meta.get('gt_fut_traj',
                                      np.zeros((0, 12, 2)))),
                    dtype=torch.float32, device=device)
                gt_fut_traj_mask = torch.as_tensor(
                    np.array(meta.get('gt_fut_traj_mask',
                                      np.zeros((0, 12)))),
                    dtype=torch.float32, device=device)

                motion_losses = self.motion_head.loss(
                    outs_motion,
                    gt_fut_traj=gt_fut_traj.unsqueeze(0),
                    gt_fut_traj_mask=gt_fut_traj_mask.unsqueeze(0),
                    track_boxes=track_boxes,
                    gt_bboxes_3d=[gt_bboxes_3d])

                for k, v in motion_losses.items():
                    all_losses.setdefault(k, []).append(v)

        final_losses = {}
        for k, v_list in all_losses.items():
            final_losses[k] = sum(v_list) / len(v_list)
        return final_losses

    # ==================== Temporal Helpers ====================

    def _init_instances_from_boxes(self, bev_2d, bboxes_3d, vehicle_ids):
        """Create Instances from GT/detected boxes via BEV sampling.

        Args:
            bev_2d: (1, D, H, W) adapted BEV features.
            bboxes_3d: LiDARInstance3DBoxes (N, 7).
            vehicle_ids: (N,) long tensor of track IDs.

        Returns:
            Instances with fields needed by MemoryBank.
        """
        device = bev_2d.device
        D = self.embed_dims
        bank_len = self.memory_bank.memory_bank_len

        query_feats, query_boxes = self._get_query_features_from_boxes(
            bev_2d, bboxes_3d)
        N = query_feats.shape[0]

        instances = Instances((1, 1))
        instances.output_embedding = query_feats          # (N, D)
        instances.query = torch.cat([
            torch.zeros(N, D, device=device),             # query_pos placeholder
            query_feats,                                  # query_feat
        ], dim=1)                                         # (N, 2D)
        instances.pred_boxes = query_boxes                # (N, 10)
        instances.scores = torch.ones(N, device=device)   # GT = confident
        instances.obj_idxes = vehicle_ids.to(device)      # track IDs
        instances.disappear_time = torch.zeros(N, dtype=torch.long, device=device)
        instances.mem_bank = torch.zeros(N, bank_len, D, device=device)
        instances.mem_padding_mask = torch.ones(
            N, bank_len, dtype=torch.bool, device=device)
        instances.save_period = torch.zeros(N, dtype=torch.long, device=device)
        instances.iou = torch.ones(N, device=device)
        instances.matched_gt_idxes = torch.arange(N, dtype=torch.long, device=device)
        instances.ref_pts = torch.zeros(N, 3, device=device)
        return instances

    def _propagate_memory(self, cur_instances, prev_instances):
        """Transfer memory bank state from previous to current frame.

        Matches tracks by obj_idxes (vehicle IDs). For matched tracks,
        copies mem_bank and mem_padding_mask so temporal cross-attention
        has historical context.
        """
        cur_ids = cur_instances.obj_idxes
        prev_ids = prev_instances.obj_idxes

        for i in range(len(cur_ids)):
            match = (prev_ids == cur_ids[i]).nonzero(as_tuple=False)
            if match.numel() > 0:
                j = match[0, 0].item()
                cur_instances.mem_bank[i] = prev_instances.mem_bank[j]
                cur_instances.mem_padding_mask[i] = \
                    prev_instances.mem_padding_mask[j]

    def _detach_instances(self, instances):
        """Detach all tensors in Instances for next-frame carry-over."""
        det = Instances(instances._image_size)
        for k, v in instances._fields.items():
            if isinstance(v, torch.Tensor):
                det.set(k, v.detach().clone())
            else:
                det.set(k, copy.copy(v))
        return det

    def _normalize_boxes(self, pred_boxes, gt_tensor):
        """Normalize pred (10-d) and gt (7-d) to [0,1] range for loss."""
        pc_range = self.pc_range
        x_range = pc_range[3] - pc_range[0]  # 160
        y_range = pc_range[4] - pc_range[1]  # 160
        z_range = pc_range[5] - pc_range[2]  # 8

        # pred format: [x, y, w, l, z, h, sin, cos, vx, vy]
        pred_norm = pred_boxes.clone()
        pred_norm[:, 0] = (pred_norm[:, 0] - pc_range[0]) / x_range
        pred_norm[:, 1] = (pred_norm[:, 1] - pc_range[1]) / y_range
        pred_norm[:, 2] = pred_norm[:, 2] / x_range   # w
        pred_norm[:, 3] = pred_norm[:, 3] / y_range   # l
        pred_norm[:, 4] = (pred_norm[:, 4] - pc_range[2]) / z_range  # z
        pred_norm[:, 5] = pred_norm[:, 5] / z_range   # h
        # sin, cos, vx, vy stay as-is

        # gt format: [x, y, z, dx, dy, dz, yaw]
        gt_norm = gt_tensor.clone()
        gt_norm[:, 0] = (gt_norm[:, 0] - pc_range[0]) / x_range
        gt_norm[:, 1] = (gt_norm[:, 1] - pc_range[1]) / y_range
        gt_norm[:, 2] = (gt_norm[:, 2] - pc_range[2]) / z_range
        gt_norm[:, 3] = gt_norm[:, 3] / x_range  # dx
        gt_norm[:, 4] = gt_norm[:, 4] / y_range  # dy
        gt_norm[:, 5] = gt_norm[:, 5] / z_range  # dz

        return pred_norm, gt_norm

    def _compute_det_losses(self, det_output, gt_bboxes_3d, gt_labels_3d):
        """Compute detection losses (classification + regression).

        Uses normalized coordinates for stable training.
        Applies multi-layer auxiliary losses like DETR.
        """
        device = det_output['all_cls_scores'].device

        if isinstance(gt_bboxes_3d, list):
            gt_bboxes = gt_bboxes_3d[0]
            gt_labels = gt_labels_3d[0]
        else:
            gt_bboxes = gt_bboxes_3d
            gt_labels = gt_labels_3d

        if hasattr(gt_bboxes, 'tensor'):
            gt_tensor = gt_bboxes.tensor.to(device)
        else:
            gt_tensor = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device)

        num_gt = gt_tensor.shape[0]
        losses = {}

        if num_gt == 0:
            losses['loss_det_cls'] = torch.tensor(0.0, device=device,
                                                   requires_grad=True)
            losses['loss_det_bbox'] = torch.tensor(0.0, device=device,
                                                    requires_grad=True)
            return losses

        num_layers = det_output['all_cls_scores'].shape[0]
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)

        for li in range(num_layers):
            pred_logits = det_output['all_cls_scores'][li, :, 0]  # (nq, C)
            pred_boxes = det_output['all_bbox_preds'][li, :, 0]   # (nq, 10)

            # Normalize for matching and loss
            pred_norm, gt_norm = self._normalize_boxes(pred_boxes, gt_tensor)

            # Hungarian matching (on normalized coords)
            with torch.no_grad():
                pred_scores = pred_logits.sigmoid()
                cls_cost = -pred_scores[:, gt_labels.long()]
                reg_cost = torch.cdist(
                    pred_norm[:, :2], gt_norm[:, :2], p=1)
                cost = cls_cost * 2.0 + reg_cost * 5.0

                cost_np = cost.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                row_ind = torch.from_numpy(row_ind).long().to(device)
                col_ind = torch.from_numpy(col_ind).long().to(device)

            # Classification loss (focal-style)
            target = torch.zeros_like(pred_logits)
            target[row_ind, gt_labels[col_ind].long()] = 1.0
            # Focal loss: alpha=0.25, gamma=2.0
            pred_prob = pred_logits.sigmoid()
            focal_weight = target * (1 - pred_prob) ** 2 + \
                (1 - target) * pred_prob ** 2
            bce = F.binary_cross_entropy_with_logits(
                pred_logits, target, reduction='none')
            cls_loss = (focal_weight * bce).mean()

            # Regression loss (normalized coordinates)
            if len(row_ind) > 0:
                pred_m = pred_norm[row_ind]   # (M, 10)
                gt_m = gt_norm[col_ind]       # (M, 7)

                # Center: pred [0,1,4] vs gt [0,1,2]
                pred_xyz = torch.stack([
                    pred_m[:, 0], pred_m[:, 1], pred_m[:, 4]], dim=-1)
                gt_xyz = gt_m[:, :3]
                loss_center = F.l1_loss(pred_xyz, gt_xyz)

                # Dimension: pred [2,3,5] vs gt [3,4,5]
                pred_wlh = torch.stack([
                    pred_m[:, 2], pred_m[:, 3], pred_m[:, 5]], dim=-1)
                gt_wlh = gt_m[:, 3:6]
                loss_dim = F.l1_loss(pred_wlh, gt_wlh)

                # Rotation: sin/cos
                pred_sincos = pred_boxes[row_ind, 6:8]  # use unnormalized
                gt_yaw = gt_tensor[col_ind, 6]
                gt_sincos = torch.stack(
                    [gt_yaw.sin(), gt_yaw.cos()], dim=-1)
                loss_rot = F.l1_loss(pred_sincos, gt_sincos)

                reg_loss = loss_center * 10.0 + loss_dim * 5.0 + \
                    loss_rot * 1.0
            else:
                reg_loss = torch.tensor(0.0, device=device,
                                        requires_grad=True)

            total_cls_loss = total_cls_loss + cls_loss
            total_reg_loss = total_reg_loss + reg_loss

        losses['loss_det_cls'] = total_cls_loss / num_layers
        losses['loss_det_bbox'] = total_reg_loss / num_layers

        return losses

    @torch.no_grad()
    def forward_predict(self, inputs, data_samples, **kwargs):
        """Inference forward pass using TransFusion detection + BEV sampling.

        Architecture:
            1. BEVFusion (frozen) → BEV features
            2. TransFusion (frozen) → detections
            3. BEV adapter (trainable) → adapted BEV
            4. Bilinear sample at detection positions → query features
            5. Motion head → trajectory predictions
        """
        device = next(self.parameters()).device
        B = len(data_samples)

        from mmdet3d.structures import LiDARInstance3DBoxes
        from mmengine.structures import InstanceData

        self.bevfusion.eval()

        # 1. Extract BEV features (frozen)
        batch_input_metas = [s.metainfo for s in data_samples]
        with torch.no_grad():
            bev_feat_raw = self.bevfusion.extract_feat(
                inputs, batch_input_metas)

        # 2. Run TransFusion for detection (frozen)
        with torch.no_grad():
            tf_results = self.bevfusion.bbox_head.predict(
                bev_feat_raw, batch_input_metas)

        # 3. Adapt BEV features (trainable) — keep 2D for sampling
        bev_2d = self._adapt_bev_2d(bev_feat_raw[0])  # (B, 256, H, W)
        # Also get flattened for motion head
        B_bev, D_bev, H_bev, W_bev = bev_2d.shape
        bev_embed = bev_2d.flatten(2).permute(2, 0, 1)  # (H*W, B, D)

        for bi in range(B):
            bev_i = bev_embed[:, bi:bi+1, :]  # (H*W, 1, D)
            ds = data_samples[bi]
            tf_result = tf_results[bi]

            # Extract TransFusion detections
            pred_bboxes = tf_result.bboxes_3d  # LiDARInstance3DBoxes
            pred_scores = tf_result.scores_3d
            pred_labels = tf_result.labels_3d

            # Store detection results
            pred = InstanceData()
            pred.bboxes_3d = pred_bboxes
            pred.scores_3d = pred_scores
            pred.labels_3d = pred_labels
            ds.pred_instances_3d = pred

            # 4. Create query features from BEV at detection positions
            n_det = pred_scores.shape[0]
            if self.det_only or n_det == 0:
                # Detection-only mode: skip motion prediction
                continue

            query_feats, query_boxes = self._get_query_features_from_boxes(
                bev_2d[bi:bi+1], pred_bboxes)
            # query_feats: (N, D), query_boxes: (N, 10)

            track_query = query_feats.unsqueeze(0)   # (1, N, D)
            track_boxes = query_boxes.unsqueeze(0)    # (1, N, 10)

            # Map encoder (optional)
            lane_query = None
            lane_query_pos = None
            if self.map_encoder is not None:
                meta = ds.metainfo
                bev_map = meta.get('bev_map', None)
                if bev_map is not None:
                    if isinstance(bev_map, np.ndarray):
                        bev_map = torch.from_numpy(bev_map).float().to(device)
                    if bev_map.dim() == 3:
                        bev_map = bev_map.unsqueeze(0)
                    lane_query, lane_query_pos = self.map_encoder(bev_map)

            # 5. Motion prediction
            outs_motion = self.motion_head(
                bev_embed=bev_i,
                track_query=track_query,
                track_boxes=track_boxes,
                lane_query=lane_query,
                lane_query_pos=lane_query_pos)

            # Extract trajectory predictions
            traj_preds = outs_motion['all_traj_preds'][-1, 0]   # (A, P, T, 5)
            traj_scores = outs_motion['all_traj_scores'][-1, 0]  # (A, P)
            ds.pred_traj = traj_preds[..., :2]
            ds.pred_traj_score = traj_scores

        return data_samples
