"""Cooperative BEVFusion models with CoBEVT-style multi-agent fusion.

Provides two cooperative models:

1. CoopBEVFusion: LiDAR-only cooperative fusion (proj_first).
   All cooperator points are projected to ego frame before feature
   extraction, so no STTF warp is needed.

2. CoopBEVFusionLidarCam: LiDAR+Camera cooperative fusion.
   Each vehicle performs per-vehicle cam+lidar fusion in its own
   coordinate frame, then BEV features are warped to the ego frame
   via STTF and fused across agents via SwapFusion.
"""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample

from .bevfusion import BEVFusion
from .sttf import STTF


def regroup(dense_feature, record_len, max_len):
    """Regroup flat agent features into padded (B, max_len, C, H, W).

    Args:
        dense_feature: (sum_agents, C, H, W) all agents' BEV features.
        record_len: List[int] of length B, number of agents per sample.
        max_len: Maximum number of agents to pad to.

    Returns:
        regroup_features: (B, max_len, C, H, W) padded features.
        mask: (B, max_len) boolean validity mask.
    """
    B = len(record_len)
    C, H, W = dense_feature.shape[1:]
    device = dense_feature.device
    dtype = dense_feature.dtype

    # Split by record lengths
    cum_sum = [0]
    for l in record_len:
        cum_sum.append(cum_sum[-1] + l)

    regroup_features = torch.zeros(
        B, max_len, C, H, W, device=device, dtype=dtype)
    mask = torch.zeros(B, max_len, device=device, dtype=torch.bool)

    for b in range(B):
        start = cum_sum[b]
        end = cum_sum[b + 1]
        n = end - start
        n_use = min(n, max_len)
        # Safety: clip to available features in dense_feature
        n_avail = dense_feature.shape[0] - start
        n_use = min(n_use, max(n_avail, 0))
        if n_use > 0:
            regroup_features[b, :n_use] = dense_feature[start:start + n_use]
            mask[b, :n_use] = True

    return regroup_features, mask


@MODELS.register_module()
class CoopDet3DDataPreprocessor(Det3DDataPreprocessor):
    """Data preprocessor that moves cooperative tensors to device.

    Cooperative data (coop_points, coop_camera_data) is stored in
    data_samples.metainfo to avoid pseudo_collate transposing list
    structures across batch samples. This preprocessor moves the
    coop_points tensors to the same device as ego points.
    """

    def simple_process(self, data, training=False):
        """Override to move coop_points (in metainfo) to device.

        Both coop_points and coop_camera_data are stored in
        data_samples.metainfo to prevent pseudo_collate from
        transposing list structures across batch samples.
        """
        result = super().simple_process(data, training)

        # Move coop_points tensors from metainfo to GPU
        device = result['inputs']['points'][0].device
        for ds in result['data_samples']:
            if hasattr(ds, 'metainfo') and 'coop_points' in ds.metainfo:
                coop_pts_gpu = [
                    pts.to(device) for pts in ds.metainfo['coop_points']
                ]
                ds.set_metainfo({'coop_points': coop_pts_gpu})

        return result


@MODELS.register_module()
class CoopBEVFusion(BEVFusion):
    """Cooperative BEVFusion with CoBEVT-style multi-agent fusion.

    All cooperator points are projected into the ego coordinate frame
    during data loading (proj_first), so BEV features from all agents
    are already spatially aligned. No STTF warp is needed.

    Architecture (CoBEVT-style: backbone+neck BEFORE fusion):
        For each agent: points -> voxelize -> SparseEncoder -> backbone -> neck
        Stack agents -> (B, L, C, H, W)  [C=512 after neck]
        SwapFusionEncoder cross-agent attention -> (B, C, H, W)
        Fused BEV -> TransFusionHead

    Args:
        coop_fusion: Config for SwapFusionEncoder module.
        max_cav: Maximum number of CAVs including ego (default: 5).
        fusion_channels: Channel dim for SwapFusion. If less than the
            backbone+neck output (512), a 1x1 conv compresses features
            before fusion and expands them back after (like CoBEVT's
            shrink_conv). Set to 0 or None to skip compression.
    """

    def __init__(
        self,
        coop_fusion: Optional[dict] = None,
        max_cav: int = 5,
        fusion_channels: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.max_cav = max_cav

        # CoBEVT SwapFusion encoder for cross-agent attention
        if coop_fusion is not None:
            self.coop_fusion = MODELS.build(coop_fusion)
        else:
            self.coop_fusion = None

        # Compression layer: reduce backbone+neck channels before fusion
        if fusion_channels and coop_fusion is not None:
            # Infer neck output channels from config
            neck_out = sum(self.pts_neck.out_channels) if hasattr(
                self.pts_neck, 'out_channels') else 512
            if fusion_channels < neck_out:
                self.compress = nn.Conv2d(
                    neck_out, fusion_channels, kernel_size=1)
                self.expand = nn.Conv2d(
                    fusion_channels, neck_out, kernel_size=1)
            else:
                self.compress = None
                self.expand = None
        else:
            self.compress = None
            self.expand = None

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        """Extract features with cooperative multi-agent fusion.

        CoBEVT-style: backbone+neck run per agent BEFORE fusion.
        All agents' points are already in ego frame (proj_first).

        1. Collect all agents' points into a flat list
        2. Shared sparse encoder -> backbone -> neck (all agents)
        3. Regroup into (B, max_cav, C, H, W) with zero-padding
        4. SwapFusion cross-agent attention -> (B, C, H, W)

        Args:
            batch_inputs_dict: Dict with 'points' and optionally
                'coop_points' (all already in ego frame).
            batch_input_metas: List of sample metadata dicts.

        Returns:
            Fused feature list for detection head.
        """
        B = len(batch_input_metas)

        # 1. Collect all agents' points (all in ego frame)
        all_points = []
        record_len = []

        for b in range(B):
            # Ego points
            all_points.append(batch_inputs_dict['points'][b])
            n_agents = 1

            # Cooperator points (already projected to ego frame)
            coop_pts = batch_input_metas[b].get('coop_points', [])
            for cp in coop_pts:
                if n_agents >= self.max_cav:
                    break
                if cp.numel() > 0:
                    all_points.append(cp)
                    n_agents += 1

            record_len.append(n_agents)

        # 2. Sparse encoder -> backbone -> neck for ALL agents
        all_bev = self.extract_pts_feat({'points': all_points})
        # (sum_agents, 256, H, W)
        all_bev = self.pts_backbone(all_bev)
        # tuple of multi-scale features
        all_bev = self.pts_neck(all_bev)
        # list with one concatenated tensor
        all_bev = all_bev[0]
        # (sum_agents, 512, H, W)

        # 3. Compress channels if configured (512 -> fusion_channels)
        if self.compress is not None:
            all_bev = self.compress(all_bev)

        # 4. Regroup into (B, max_cav, C, H, W)
        bev_features, masks = regroup(all_bev, record_len, self.max_cav)

        # Build validity mask from metadata
        coop_masks = []
        for b in range(B):
            meta = batch_input_metas[b]
            coop_mask = meta.get(
                'coop_mask',
                torch.zeros(self.max_cav, dtype=torch.bool).numpy())
            coop_masks.append(torch.tensor(coop_mask, dtype=torch.bool))

        mask_batch = torch.stack(coop_masks).to(bev_features.device)
        # Intersect data-based mask with metadata mask
        final_mask = masks & mask_batch

        # 5. CoBEVT SwapFusion -> (B, C, H, W)
        if self.coop_fusion is not None:
            x = self.coop_fusion(bev_features, final_mask)
        else:
            # Fallback: just use ego features
            x = bev_features[:, 0]

        # 6. Expand back to neck channels if compressed
        if self.expand is not None:
            x = self.expand(x)

        return [x]

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        losses.update(bbox_loss)

        return losses


@MODELS.register_module()
class CoopBEVFusionLidarCam(BEVFusion):
    """Cooperative BEVFusion with per-vehicle LiDAR+Camera fusion.

    Each vehicle performs cam+lidar fusion in its own coordinate frame:
        LiDAR -> Voxelize -> SparseEncoder -> lidar_BEV (256ch)
        Cameras -> Swin -> FPN -> DepthLSS -> cam_BEV (80ch)
        ConvFuser(cam_BEV, lidar_BEV) -> fused_BEV (256ch)

    Then BEV features are warped to ego frame via STTF, run through
    backbone+neck, and fused across agents via SwapFusion:
        STTF warp cooperator fused_BEV -> ego frame
        All fused_BEV -> Backbone -> Neck -> (512ch)
        compress(512->256) -> SwapFusion -> expand(256->512)
        -> TransFusionHead

    Args:
        sttf: Config dict for STTF module (discrete_ratio, downsample_rate).
        coop_fusion: Config for SwapFusionEncoder module.
        max_cav: Maximum number of CAVs including ego.
        fusion_channels: Channel dim for SwapFusion compression.
    """

    def __init__(
        self,
        sttf: Optional[dict] = None,
        coop_fusion: Optional[dict] = None,
        max_cav: int = 5,
        fusion_channels: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.max_cav = max_cav

        # STTF for BEV feature warping
        if sttf is not None:
            self.sttf = STTF(**sttf)
        else:
            self.sttf = None

        # CoBEVT SwapFusion encoder for cross-agent attention
        if coop_fusion is not None:
            self.coop_fusion = MODELS.build(coop_fusion)
        else:
            self.coop_fusion = None

        # Compression layer: reduce backbone+neck channels before fusion
        if fusion_channels and coop_fusion is not None:
            neck_out = sum(self.pts_neck.out_channels) if hasattr(
                self.pts_neck, 'out_channels') else 512
            if fusion_channels < neck_out:
                self.compress = nn.Conv2d(
                    neck_out, fusion_channels, kernel_size=1)
                self.expand = nn.Conv2d(
                    fusion_channels, neck_out, kernel_size=1)
            else:
                self.compress = None
                self.expand = None
        else:
            self.compress = None
            self.expand = None

    def _normalize_imgs(self, imgs_tensor):
        """Normalize images using data_preprocessor mean/std.

        Cooperator images arrive un-normalized; normalize them using
        the same mean/std as the ego images.

        Args:
            imgs_tensor: (N, C, H, W) float tensor in [0, 255].

        Returns:
            Normalized tensor.
        """
        mean = self.data_preprocessor.mean  # (C,)
        std = self.data_preprocessor.std    # (C,)
        # mean/std are registered buffers with shape (C, 1, 1)
        if mean.dim() == 1:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        return (imgs_tensor - mean.to(imgs_tensor.device)) / std.to(
            imgs_tensor.device)

    def _extract_all_cam_feat(self, all_points, cam_data_list):
        """Extract camera BEV features for agents that have cameras.

        Batches all agents-with-cameras together, runs shared
        Swin -> FPN -> DepthLSS, then scatters results back.

        Args:
            all_points: List of point cloud tensors, one per agent.
            cam_data_list: List of camera data dicts (or None) per agent.
                Each dict has: imgs, cam2img, lidar2cam, cam2lidar,
                lidar2img, img_aug_matrix, lidar_aug_matrix, img_shape.

        Returns:
            List of cam_BEV tensors (or None) per agent, each (1, C, H, W).
        """
        # Collect agents with cameras
        cam_indices = []
        cam_imgs_batch = []
        cam_points_batch = []
        cam_lidar2image_batch = []
        cam_intrinsics_batch = []
        cam_camera2lidar_batch = []
        cam_img_aug_batch = []
        cam_lidar_aug_batch = []
        cam_metas_batch = []

        for idx, cam_data in enumerate(cam_data_list):
            if cam_data is None:
                continue

            imgs = cam_data['imgs']  # list of (H, W, 3) np arrays
            n_views = len(imgs)

            # Stack images: (N, H, W, 3) -> (1, N, C, H, W)
            img_array = np.stack(imgs, axis=0)  # (N, H, W, 3)
            img_tensor = torch.from_numpy(img_array).float()
            img_tensor = img_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)  # (1, N, C, H, W)

            # Move to same device as points
            device = all_points[idx].device
            img_tensor = img_tensor.to(device)

            # Normalize
            B_img, N_img, C_img, H_img, W_img = img_tensor.shape
            img_flat = img_tensor.view(B_img * N_img, C_img, H_img, W_img)
            img_flat = self._normalize_imgs(img_flat)
            img_tensor = img_flat.view(B_img, N_img, C_img, H_img, W_img)

            # Calibration matrices -> tensors (1, N, 4, 4)
            lidar2img = torch.from_numpy(
                cam_data['lidar2img']).float().unsqueeze(0).to(device)
            cam2img = torch.from_numpy(
                cam_data['cam2img']).float().unsqueeze(0).to(device)
            cam2lidar = torch.from_numpy(
                cam_data['cam2lidar']).float().unsqueeze(0).to(device)

            # img_aug_matrix: list of (4,4) -> (1, N, 4, 4)
            img_aug = np.stack(cam_data['img_aug_matrix'], axis=0)
            img_aug = torch.from_numpy(img_aug).float().unsqueeze(0).to(
                device)

            # lidar_aug_matrix: identity (4,4) -> (1, 4, 4)
            lidar_aug = torch.from_numpy(
                cam_data['lidar_aug_matrix']).float().unsqueeze(0).to(
                    device)

            cam_indices.append(idx)
            cam_imgs_batch.append(img_tensor)
            cam_points_batch.append([all_points[idx]])
            cam_lidar2image_batch.append(lidar2img)
            cam_intrinsics_batch.append(cam2img)
            cam_camera2lidar_batch.append(cam2lidar)
            cam_img_aug_batch.append(img_aug)
            cam_lidar_aug_batch.append(lidar_aug)
            cam_metas_batch.append({})

        # If no agents have cameras, return all None
        results = [None] * len(cam_data_list)
        if not cam_indices:
            return results

        # Batch all camera agents together for efficiency
        batch_imgs = torch.cat(cam_imgs_batch, dim=0)
        batch_lidar2image = torch.cat(cam_lidar2image_batch, dim=0)
        batch_intrinsics = torch.cat(cam_intrinsics_batch, dim=0)
        batch_cam2lidar = torch.cat(cam_camera2lidar_batch, dim=0)
        batch_img_aug = torch.cat(cam_img_aug_batch, dim=0)
        batch_lidar_aug = torch.cat(cam_lidar_aug_batch, dim=0)

        # Flatten points for DepthLSS depth estimation
        batch_points = []
        for pts_list in cam_points_batch:
            batch_points.extend(pts_list)

        batch_metas = cam_metas_batch

        # Run through shared camera pipeline:
        # Swin backbone -> FPN neck -> DepthLSS view_transform
        cam_bev = self.extract_img_feat(
            batch_imgs,
            deepcopy(batch_points),
            batch_lidar2image,
            batch_intrinsics,
            batch_cam2lidar,
            batch_img_aug,
            batch_lidar_aug,
            batch_metas,
        )
        # cam_bev: (num_cam_agents, 80, H, W)

        # Scatter back to per-agent list
        for i, idx in enumerate(cam_indices):
            results[idx] = cam_bev[i:i + 1]  # (1, 80, H, W)

        return results

    @staticmethod
    def _transform_points_to_ego(points, T):
        """Transform point cloud from own frame to ego frame.

        Args:
            points: (N, 4+) tensor, first 3 cols are xyz.
            T: (4, 4) cav-to-ego transformation matrix (tensor).

        Returns:
            Points tensor with xyz transformed to ego frame.
        """
        pts_ego = points.clone()
        pts_3d = points[:, :3]
        ones = torch.ones(
            pts_3d.shape[0], 1, device=points.device,
            dtype=points.dtype)
        pts_homo = torch.cat([pts_3d, ones], dim=1)  # (N, 4)
        pts_ego[:, :3] = (T @ pts_homo.T).T[:, :3]
        return pts_ego

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        """Extract features: lidar in ego frame, camera warped to ego.

        Flow (Backbone/Neck only see REAL agents, no zero-padding):
        1. Collect points: own-frame (for DepthLSS) + ego-frame (for lidar)
        2. SparseEncoder on ego-frame points -> lidar_BEV (sum_agents, 256)
        3. Camera -> cam_BEV per agent (own frame)
        4. STTF warp cooperator cam_BEV -> ego frame (per agent)
        5. ConvFuser(cam_BEV_ego, lidar_BEV) per agent (flat, no padding)
        6. Backbone -> Neck on REAL agents only (no zero-padded slots)
        7. compress -> regroup (pad here) -> SwapFusion -> expand
        """
        B = len(batch_input_metas)

        # ---- 1. Collect points for lidar (ego) and camera (own) ----
        all_points_ego = []   # for lidar BEV (all in ego frame)
        all_points_own = []   # for camera DepthLSS (in own frame)
        agent_cam_data = []   # camera data per agent (or None)
        agent_t_mat = []      # per-agent 4x4 numpy transform (or None)
        record_len = []

        for b in range(B):
            ego_pts = batch_inputs_dict['points'][b]
            all_points_ego.append(ego_pts)
            all_points_own.append(ego_pts)
            n_agents = 1

            # Ego camera marker
            ego_cam_data = None
            if 'imgs' in batch_inputs_dict and \
                    batch_inputs_dict['imgs'] is not None:
                ego_cam_data = {'is_ego': True}
            agent_cam_data.append(ego_cam_data)
            agent_t_mat.append(None)  # ego: no warp needed

            # Cooperator points + camera
            coop_pts = batch_input_metas[b].get('coop_points', [])
            coop_cam_b = batch_input_metas[b].get(
                'coop_camera_data', [])
            t_matrix = batch_input_metas[b].get(
                'transformation_matrix', None)

            for i, cp in enumerate(coop_pts):
                if n_agents >= self.max_cav:
                    break
                if cp.numel() > 0:
                    all_points_own.append(cp)

                    # Transform to ego frame for lidar BEV
                    cav_idx = i + 1
                    if t_matrix is not None and cav_idx < t_matrix.shape[0]:
                        T_np = t_matrix[cav_idx]
                        T = torch.from_numpy(T_np).float().to(cp.device)
                        cp_ego = self._transform_points_to_ego(cp, T)
                        agent_t_mat.append(T_np)
                    else:
                        cp_ego = cp
                        agent_t_mat.append(None)
                    all_points_ego.append(cp_ego)

                    n_agents += 1
                    c_cam = (coop_cam_b[i]
                             if i < len(coop_cam_b) else None)
                    agent_cam_data.append(c_cam)

            record_len.append(n_agents)

        # ---- 2. Lidar BEV: ego-frame points -> SparseEncoder ----
        lidar_bev = self.extract_pts_feat({'points': all_points_ego})
        # (sum_agents, 256, H, W) — all in ego frame

        # ---- 3. Camera BEV per agent ----
        # Build cam_data_list for cooperator cameras
        cam_data_list = []
        for cam_data in agent_cam_data:
            if cam_data is None:
                cam_data_list.append(None)
            elif cam_data.get('is_ego', False):
                cam_data_list.append(None)  # ego handled separately
            else:
                cam_data_list.append(cam_data)

        # Extract cooperator camera features (own frame)
        cam_bevs = self._extract_all_cam_feat(
            all_points_own, cam_data_list)

        # Extract ego camera features (already in ego frame)
        if 'imgs' in batch_inputs_dict and \
                batch_inputs_dict['imgs'] is not None:
            imgs = batch_inputs_dict['imgs'].contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(
                    meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(
                np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(
                np.asarray(lidar_aug_matrix))

            ego_points = [batch_inputs_dict['points'][b]
                          for b in range(B)]

            ego_cam_bev = self.extract_img_feat(
                imgs, deepcopy(ego_points),
                lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix,
                lidar_aug_matrix, batch_input_metas)
            # (B, 80, H, W) — in ego frame

            # Place ego cam_bev into flat list at ego positions
            flat_idx = 0
            for b in range(B):
                cam_bevs[flat_idx] = ego_cam_bev[b:b + 1]
                flat_idx += record_len[b]

        # ---- 4. STTF warp cooperator cam_BEV -> ego frame (per agent) ----
        if self.sttf is not None:
            for idx in range(len(cam_bevs)):
                if cam_bevs[idx] is not None and \
                        agent_t_mat[idx] is not None:
                    # Cooperator cam_BEV: warp from own -> ego
                    bev = cam_bevs[idx]  # (1, C, H, W)
                    t_np = agent_t_mat[idx]  # (4, 4) numpy
                    t_tensor = torch.from_numpy(t_np).float().to(
                        bev.device)
                    # Use STTF with (1, 1, C, H, W) and (1, 1, 4, 4)
                    cam_bevs[idx] = self.sttf(
                        bev.unsqueeze(1),
                        t_tensor.unsqueeze(0).unsqueeze(0),
                    ).squeeze(1)  # (1, C, H, W)

        # ---- 5. ConvFuser per agent (flat list, NO padding) ----
        fused_list = []
        for idx in range(lidar_bev.shape[0]):
            agent_lidar = lidar_bev[idx:idx + 1]
            if cam_bevs[idx] is not None and \
                    self.fusion_layer is not None:
                agent_cam = cam_bevs[idx]
                fused = self.fusion_layer([agent_cam, agent_lidar])
            else:
                fused = agent_lidar
            fused_list.append(fused)

        all_fused = torch.cat(fused_list, dim=0)
        # (sum_agents, 256, H, W) — only real agents, no padding

        # ---- 6. Backbone -> Neck on REAL agents only ----
        all_bev = self.pts_backbone(all_fused)
        all_bev = self.pts_neck(all_bev)
        all_bev = all_bev[0]
        # (sum_agents, 512, H, W) — no zero-padded slots in BN

        # ---- 7. Compress -> Regroup -> SwapFusion -> Expand ----
        if self.compress is not None:
            all_bev = self.compress(all_bev)

        # Regroup into (B, max_cav, C, H, W) — padding happens HERE
        bev_features, masks = regroup(all_bev, record_len, self.max_cav)

        # Build validity mask
        coop_masks = []
        for b in range(B):
            coop_mask = batch_input_metas[b].get(
                'coop_mask', np.zeros(self.max_cav, dtype=bool))
            coop_masks.append(
                torch.tensor(coop_mask, dtype=torch.bool))
        mask_batch = torch.stack(coop_masks).to(bev_features.device)
        final_mask = masks & mask_batch

        if self.coop_fusion is not None:
            x = self.coop_fusion(bev_features, final_mask)
        else:
            x = bev_features[:, 0]

        if self.expand is not None:
            x = self.expand(x)

        return [x]

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)
        return res

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        losses.update(bbox_loss)

        return losses
