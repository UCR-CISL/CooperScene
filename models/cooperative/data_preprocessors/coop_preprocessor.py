"""Det3DDataPreprocessor subclass for cooperative perception.

Adds:
- Pass-through of `record_len`, `pairwise_t_matrix`, `coop_mask`,
  `prior_encoding` from inputs into batch_inputs.
- Per-CAV voxelization (`points_per_cav` -> per-CAV voxel grid with
  CAV-level batch index, so backbone produces (sum(record_len), C, H, W)).

Skips the merged-cloud voxelization when `points_per_cav` is present
to avoid the redundant double-voxelize that the upstream class does.
"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.models.data_preprocessors.data_preprocessor import (
    Det3DDataPreprocessor)
from mmdet3d.registry import MODELS


@MODELS.register_module()
class CoopDet3DDataPreprocessor(Det3DDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']
            if self.voxel and 'points_per_cav' not in inputs:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        for key in ('record_len', 'pairwise_t_matrix', 'coop_mask',
                    'prior_encoding'):
            if key in inputs:
                val = inputs[key]
                if isinstance(val, (list, tuple)):
                    batch_inputs[key] = torch.stack(val)
                else:
                    batch_inputs[key] = val

        if 'points_per_cav' in inputs and self.voxel:
            batch_inputs['voxels'] = self._voxelize_per_cav(
                inputs['points_per_cav'])

        if 'imgs' in inputs:
            imgs = inputs['imgs']
            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape,
                    })
                if self.boxtype2tensor:
                    from mmdet.models.utils.misc import samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)
            batch_inputs['imgs'] = imgs

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def _voxelize_per_cav(self, all_cav_points) -> Dict[str, Tensor]:
        voxels, coors, num_points, voxel_centers = [], [], [], []
        cav_idx = 0
        for batch_cav_list in all_cav_points:
            for cav_pts in batch_cav_list:
                if not isinstance(cav_pts, Tensor):
                    cav_pts = torch.from_numpy(cav_pts).float()
                cav_pts = cav_pts.to(self.device)
                if cav_pts.shape[0] == 0:
                    # Empty CAV: PointPillarScatter sets batch_size by
                    # coords[:, 0].max()+1. If the LAST CAV is empty its
                    # cav_idx vanishes from coords and scatter outputs fewer
                    # rows than record_len, breaking v2vnet's warp_affine.
                    # Insert one zero point so cav_idx survives.
                    cav_pts = torch.zeros(
                        (1, cav_pts.shape[1] if cav_pts.ndim == 2 else 4),
                        dtype=torch.float32, device=self.device)
                res_voxels, res_coors, res_num_points = \
                    self.voxel_layer(cav_pts)
                res_voxel_centers = (
                    res_coors[:, [2, 1, 0]] + 0.5
                ) * res_voxels.new_tensor(
                    self.voxel_layer.voxel_size
                ) + res_voxels.new_tensor(
                    self.voxel_layer.point_cloud_range[0:3])
                res_coors = F.pad(
                    res_coors, (1, 0), mode='constant', value=cav_idx)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
                voxel_centers.append(res_voxel_centers)
                cav_idx += 1
        return dict(
            voxels=torch.cat(voxels, dim=0),
            coors=torch.cat(coors, dim=0),
            num_points=torch.cat(num_points, dim=0),
            voxel_centers=torch.cat(voxel_centers, dim=0))
