"""TrackFormer: temporal track query generator for motion prediction.

Provides temporally enriched track queries by encoding past trajectories
and estimating velocity. Uses GT past trajectories for training and
evaluation (matching UniAD's evaluation protocol with GT tracking).

In original UniAD, TrackFormer is a full multi-frame tracker based on
Deformable DETR with query interaction module (QIM). Our simplified
version focuses on the key temporal signals (past trajectory encoding
+ velocity estimation) that enable MotionFormer to predict motion.
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@MODELS.register_module()
class TrackFormer(BaseModule):
    """Simplified temporal track query generator.

    Extracts track queries from BEV features at detection centers, enriches
    them with past trajectory encoding and velocity estimation.

    Args:
        embed_dims (int): Embedding dimensions.
        past_steps (int): Number of past trajectory steps.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        max_agents (int): Maximum number of tracked agents per frame.
    """

    def __init__(self,
                 embed_dims=256,
                 past_steps=4,
                 pc_range=[-72, -72, -5, 72, 72, 3],
                 max_agents=20,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.past_steps = past_steps
        self.pc_range = pc_range
        self.max_agents = max_agents

        # Past trajectory encoder: flatten (T_past, 2) -> D
        self.past_traj_encoder = nn.Sequential(
            nn.Linear(past_steps * 2, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        # Velocity encoder: (2,) -> D
        self.velocity_encoder = nn.Sequential(
            nn.Linear(2, embed_dims // 2),
            nn.ReLU(),
            nn.Linear(embed_dims // 2, embed_dims),
        )

        # Fuse base query + past_traj_feat + velocity_feat -> enriched query
        self.track_query_fuser = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims * 2),
            nn.ReLU(),
            nn.Linear(embed_dims * 2, embed_dims),
        )

    def forward(self, bev_feat, det_results, data_samples):
        """Generate temporally enriched track queries.

        Args:
            bev_feat: (B, D, H, W) adapted BEV features.
            det_results: List of Det3DDataSample with pred_instances_3d.
            data_samples: List of Det3DDataSample with GT metainfo.

        Returns:
            track_query: (B, A, D) enriched track query embeddings.
            track_boxes: (B, A, 10) tracked bounding boxes with velocity.
        """
        B, D, H, W = bev_feat.shape
        device = bev_feat.device

        # Step 1: Extract base track queries from detections via BEV sampling
        track_query, track_boxes = self._extract_track_queries(
            bev_feat, det_results)
        A = track_query.shape[1]

        # Step 2: Get past trajectories from GT (matched to detections)
        past_traj, velocity = self._get_past_traj(
            track_boxes, data_samples)

        # Step 3: Encode temporal features
        past_feat = self.past_traj_encoder(
            past_traj.reshape(B, A, -1))  # (B, A, D)
        vel_feat = self.velocity_encoder(velocity)  # (B, A, D)

        # Step 4: Fuse into enriched track query
        track_query = self.track_query_fuser(
            torch.cat([track_query, past_feat, vel_feat], dim=-1))

        # Step 5: Fill velocity in track_boxes
        track_boxes = track_boxes.clone()
        track_boxes[:, :, 8:10] = velocity

        return track_query, track_boxes

    def _extract_track_queries(self, bev_feat, det_results):
        """Extract track queries by sampling BEV features at detection centers.

        Args:
            bev_feat: (B, D, H, W) BEV features.
            det_results: List of Det3DDataSample with pred_instances_3d.

        Returns:
            track_query: (B, A, D) track query embeddings.
            track_boxes: (B, A, 10) detection boxes
                [cx, cy, w, l, cz, h, sin_yaw, cos_yaw, vx, vy].
        """
        B, D, H, W = bev_feat.shape
        device = bev_feat.device

        all_queries = []
        all_boxes = []

        for b in range(B):
            if hasattr(det_results[b], 'pred_instances_3d'):
                pred = det_results[b].pred_instances_3d
                boxes = pred.bboxes_3d.tensor  # (N, 7)
                scores = pred.scores_3d

                if len(scores) > self.max_agents:
                    topk_idx = scores.topk(self.max_agents)[1]
                    boxes = boxes[topk_idx]

                N = boxes.shape[0]
                if N > 0:
                    cx, cy = boxes[:, 0], boxes[:, 1]
                    grid_x = (2.0 * (cx - self.pc_range[0])
                              / (self.pc_range[3] - self.pc_range[0]) - 1.0)
                    grid_y = (2.0 * (cy - self.pc_range[1])
                              / (self.pc_range[4] - self.pc_range[1]) - 1.0)
                    grid = torch.stack([grid_x, grid_y], dim=-1)
                    grid = grid.view(1, N, 1, 2)

                    query = F.grid_sample(
                        bev_feat[b:b + 1], grid,
                        mode='bilinear', padding_mode='zeros',
                        align_corners=False)
                    query = query.squeeze(-1).squeeze(0).permute(1, 0)  # (N, D)

                    boxes_10 = torch.zeros(N, 10, device=device)
                    boxes_10[:, 0] = boxes[:, 0]      # cx
                    boxes_10[:, 1] = boxes[:, 1]      # cy
                    boxes_10[:, 2] = boxes[:, 3]      # dx -> w
                    boxes_10[:, 3] = boxes[:, 4]      # dy -> l
                    boxes_10[:, 4] = boxes[:, 2]      # z -> cz
                    boxes_10[:, 5] = boxes[:, 5]      # dz -> h
                    boxes_10[:, 6] = torch.sin(boxes[:, 6])  # sin(yaw)
                    boxes_10[:, 7] = torch.cos(boxes[:, 6])  # cos(yaw)

                    all_queries.append(query)
                    all_boxes.append(boxes_10)
                else:
                    all_queries.append(torch.zeros(0, D, device=device))
                    all_boxes.append(torch.zeros(0, 10, device=device))
            else:
                all_queries.append(torch.zeros(0, D, device=device))
                all_boxes.append(torch.zeros(0, 10, device=device))

        # Pad to same number of agents across batch
        max_a = max(q.shape[0] for q in all_queries)
        max_a = max(max_a, 1)

        track_query = torch.zeros(B, max_a, D, device=device)
        track_boxes = torch.zeros(B, max_a, 10, device=device)
        for b in range(B):
            N = all_queries[b].shape[0]
            if N > 0:
                track_query[b, :N] = all_queries[b]
                track_boxes[b, :N] = all_boxes[b]

        return track_query, track_boxes

    def _get_past_traj(self, track_boxes, data_samples):
        """Get past trajectories by Hungarian-matching detections to GT.

        Matches predicted detection centers to GT bbox centers, then
        retrieves the corresponding GT past trajectories and estimates
        velocity from the last two valid past steps.

        Args:
            track_boxes: (B, A, 10) detection boxes.
            data_samples: List of Det3DDataSample with GT metainfo.

        Returns:
            past_traj: (B, A, T_past, 2) past trajectory positions.
            velocity: (B, A, 2) estimated velocity (displacement per step).
        """
        B, A = track_boxes.shape[:2]
        device = track_boxes.device
        T = self.past_steps

        past_traj = torch.zeros(B, A, T, 2, device=device)
        velocity = torch.zeros(B, A, 2, device=device)

        for b in range(B):
            meta = data_samples[b].metainfo
            gt_past = meta.get('gt_past_traj', None)
            gt_past_mask = meta.get('gt_past_traj_mask', None)

            if gt_past is None:
                continue

            gt_past_t = torch.as_tensor(
                np.array(gt_past), dtype=torch.float32, device=device)
            if gt_past_t.ndim != 3 or gt_past_t.shape[0] == 0:
                continue

            # Get GT bboxes for Hungarian matching
            gt_bboxes = None
            if hasattr(data_samples[b], 'gt_instances_3d'):
                gi = data_samples[b].gt_instances_3d
                if 'bboxes_3d' in gi:
                    gt_bboxes = gi.bboxes_3d.tensor.to(device)

            if gt_bboxes is None:
                continue

            # Hungarian match: detection centers -> GT centers
            pred_xy = track_boxes[b, :, :2].detach().cpu().numpy()
            gt_xy = gt_bboxes[:, :2].detach().cpu().numpy()

            # Only match non-zero (non-padded) predictions
            valid_pred = (pred_xy ** 2).sum(-1) > 1e-6
            if not valid_pred.any():
                continue

            cost = ((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(-1)
            row_ind, col_ind = linear_sum_assignment(cost)

            # Load past trajectory mask
            if gt_past_mask is not None:
                gt_mask_t = torch.as_tensor(
                    np.array(gt_past_mask), dtype=torch.float32, device=device)
            else:
                gt_mask_t = torch.ones(
                    gt_past_t.shape[0], gt_past_t.shape[1], device=device)

            for pi, gi_idx in zip(row_ind, col_ind):
                # Skip matches with large distance (>10m)
                if cost[pi, gi_idx] > 100:
                    continue
                if gi_idx >= gt_past_t.shape[0]:
                    continue

                T_avail = min(T, gt_past_t.shape[1])
                past_traj[b, pi, :T_avail] = gt_past_t[gi_idx, :T_avail, :2]

                # Estimate velocity from last two valid past steps
                mask = gt_mask_t[gi_idx, :T_avail]
                valid_idx = mask.nonzero(as_tuple=True)[0]
                if len(valid_idx) >= 2:
                    t1, t2 = valid_idx[-2], valid_idx[-1]
                    dt = (t2 - t1).float()
                    if dt > 0:
                        velocity[b, pi] = (
                            gt_past_t[gi_idx, t2, :2]
                            - gt_past_t[gi_idx, t1, :2]) / dt

        return past_traj, velocity
