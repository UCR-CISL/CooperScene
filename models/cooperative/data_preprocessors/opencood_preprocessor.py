"""Cooperative data preprocessor that calls OpenCOOD's `SpVoxelPreprocessor`.

The mmcv `Voxelization` op and OpenCOOD's spconv-based voxel generator can
disagree on which points end up in dense pillars (>max_points), which
propagates into PFE features and ultimately mAP. For paper-grade
reproduction we delegate the voxelization to OpenCOOD's exact code path.

Used together with `OpenCOODCooperativeDetector`; the rest of the data flow
(per-CAV `points_per_cav`, `record_len`, `pairwise_t_matrix`) is unchanged.
"""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS
from .coop_preprocessor import CoopDet3DDataPreprocessor


@MODELS.register_module()
class OpenCOODCoopDet3DDataPreprocessor(CoopDet3DDataPreprocessor):
    """Same as `CoopDet3DDataPreprocessor` but per-CAV voxelization uses
    OpenCOOD's `SpVoxelPreprocessor` (spconv) instead of mmcv `Voxelization`.

    Extra config args:
        cav_lidar_range: list[float] (xyz_min + xyz_max) — passed to
            `SpVoxelPreprocessor` as `cav_lidar_range`.
        voxel_size: list[float] (vx, vy, vz).
        max_points_per_voxel: int — pillar cap (32 in the v2vam yaml).
        max_voxel_train / max_voxel_test: int — voxel-count caps.
    """

    def __init__(self,
                 cav_lidar_range: List[float],
                 voxel_size: List[float] = (0.4, 0.4, 4),
                 max_points_per_voxel: int = 32,
                 max_voxel_train: int = 32000,
                 max_voxel_test: int = 70000,
                 **kwargs):
        super().__init__(**kwargs)
        # Build two SpVoxelPreprocessors (train / test).  We can't tell which
        # phase we're in inside `forward`, so we instantiate both and choose
        # based on `self.training`.
        from opencood.data_utils.pre_processor.sp_voxel_preprocessor import (
            SpVoxelPreprocessor)
        preprocess_params = dict(
            core_method='SpVoxelPreprocessor',
            args=dict(
                voxel_size=list(voxel_size),
                max_points_per_voxel=max_points_per_voxel,
                max_voxel_train=max_voxel_train,
                max_voxel_test=max_voxel_test,
            ),
            cav_lidar_range=list(cav_lidar_range),
        )
        self._spvoxel_train = SpVoxelPreprocessor(preprocess_params, train=True)
        self._spvoxel_test = SpVoxelPreprocessor(preprocess_params, train=False)

    def _voxelize_per_cav(self, all_cav_points) -> Dict[str, Tensor]:
        spvoxel = self._spvoxel_train if self.training else self._spvoxel_test

        voxels, coors, num_points = [], [], []
        cav_idx = 0
        for batch_cav_list in all_cav_points:
            for cav_pts in batch_cav_list:
                if isinstance(cav_pts, Tensor):
                    pts_np = cav_pts.detach().cpu().numpy()
                else:
                    pts_np = np.asarray(cav_pts)
                if pts_np.shape[0] == 0:
                    # Insert one zero point so this cav_idx survives into
                    # coords — otherwise PointPillarScatter's batch_size =
                    # coords[:,0].max()+1 drops the last empty CAV, and
                    # v2vnet's warp_affine then sees a batch mismatch
                    # between batch_node_features and pairwise_t_matrix.
                    pts_np = np.zeros((1, 4), dtype=np.float32)

                # OpenCOOD path: voxel_features (Nvoxels, max_pts, C),
                # voxel_coords (Nvoxels, 3 = [z, y, x]),
                # voxel_num_points (Nvoxels,).
                vd = spvoxel.preprocess(pts_np.astype(np.float32))
                res_voxels = torch.from_numpy(vd['voxel_features']).float()
                res_coors = torch.from_numpy(vd['voxel_coords']).long()
                res_num = torch.from_numpy(vd['voxel_num_points']).long()

                res_voxels = res_voxels.to(self.device)
                res_coors = res_coors.to(self.device)
                res_num = res_num.to(self.device)

                # Prepend the cumulative CAV index so coords become
                # [cav_idx, z, y, x] like OpenCOOD's PointPillarScatter expects.
                res_coors = F.pad(
                    res_coors, (1, 0), mode='constant', value=cav_idx)

                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num)
                cav_idx += 1

        return dict(
            voxels=torch.cat(voxels, dim=0),
            coors=torch.cat(coors, dim=0),
            num_points=torch.cat(num_points, dim=0),
        )
