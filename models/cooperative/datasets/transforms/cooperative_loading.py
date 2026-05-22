"""Pipeline transforms for cooperative perception.

LoadCooperativePointCloud:
    - Reads the ego .bin plus each co-vehicle .bin for the sample.
    - With proj_first=True, transforms all clouds into the ego frame.
    - Populates results with:
        points          : (N_total, C) ego-frame point cloud (concatenated)
        points_per_cav  : list[(N_i, C)], len = n_cav
        record_len      : int, number of CAVs (ego + co)
        pairwise_t_matrix: (max_cav, max_cav, 4, 4) CAV-to-CAV transforms

PackCooperative3DDetInputs:
    Packs the above into Det3DDataSample + inputs dict consumed by
    Det3DDataPreprocessor's per-CAV voxelization path.
"""
from typing import List, Optional, Union

import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.registry import FUNCTIONS

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import BasePoints, get_points_type


@FUNCTIONS.register_module()
def cooperative_collate(data_batch):
    """Collate for cooperative perception.

    Variable-size per-sample fields (``points``, ``points_per_cav``) are
    kept as a Python list across the batch. Fixed-shape fields
    (``record_len``, ``pairwise_t_matrix``, ``coop_mask``) are stacked
    into a batched tensor. ``data_samples`` is collected as a list. Any
    other field falls back to a simple stack, else list.
    """
    inputs_list = [item['inputs'] for item in data_batch]
    samples_list = [item['data_samples'] for item in data_batch]

    keys = set()
    for inp in inputs_list:
        keys.update(inp.keys())

    batched_inputs = {}
    for k in keys:
        vals = [inp.get(k) for inp in inputs_list]
        if k in ('points', 'points_per_cav'):
            batched_inputs[k] = vals
        elif k in ('record_len', 'pairwise_t_matrix', 'coop_mask'):
            tensors = [v if torch.is_tensor(v) else torch.as_tensor(v)
                       for v in vals]
            batched_inputs[k] = torch.stack(tensors)
        else:
            try:
                batched_inputs[k] = torch.stack(vals)
            except (RuntimeError, TypeError):
                batched_inputs[k] = vals

    return dict(inputs=batched_inputs, data_samples=samples_list)


@TRANSFORMS.register_module()
class LoadCooperativePointCloud(BaseTransform):

    def __init__(self,
                 coord_type: str = 'LIDAR',
                 load_dim: int = 4,
                 use_dim: Union[int, List[int]] = [0, 1, 2, 3],
                 max_cav: int = 5,
                 proj_first: bool = True,
                 point_cloud_range: Optional[List[float]] = None) -> None:
        self.coord_type = coord_type
        self.load_dim = load_dim
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.max_cav = max_cav
        self.proj_first = proj_first
        self.point_cloud_range = point_cloud_range

    def _load_points(self, pts_filename: str) -> np.ndarray:
        if pts_filename.endswith('.npy'):
            points = np.load(pts_filename)
        else:
            points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _filter_by_range(self, pts: np.ndarray) -> np.ndarray:
        if self.point_cloud_range is None:
            return pts
        r = self.point_cloud_range
        mask = ((pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
                (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
                (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5]))
        return pts[mask]

    def transform(self, results: dict) -> dict:
        cooperators = results.get('cooperators', [])
        t_matrix = results.get('transformation_matrix', None)
        coop_mask = results.get('coop_mask', None)

        ego_points = results['points']
        if isinstance(ego_points, BasePoints):
            ego_np = ego_points.tensor.numpy()
        else:
            ego_np = np.array(ego_points)
        ego_np = self._filter_by_range(ego_np)

        cav_points_list = [ego_np]
        record_len = 1

        if t_matrix is not None:
            for coop in cooperators:
                if record_len >= self.max_cav:
                    break
                lidar_path = coop['lidar_path']
                try:
                    pts = self._load_points(lidar_path)
                    valid_len = (pts.size // self.load_dim) * self.load_dim
                    pts = pts[:valid_len].reshape(-1, self.load_dim)
                    pts = pts[:, self.use_dim]

                    if self.proj_first:
                        idx = record_len
                        T = t_matrix[idx]
                        pts_xyz = pts[:, :3]
                        pts_homo = np.hstack([
                            pts_xyz,
                            np.ones((pts_xyz.shape[0], 1),
                                    dtype=np.float32)])
                        pts[:, :3] = (T @ pts_homo.T).T[:, :3]

                    pts = self._filter_by_range(pts)
                    cav_points_list.append(pts)
                    record_len += 1
                except Exception:
                    continue

        merged_points = np.concatenate(cav_points_list, axis=0)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            merged_points, points_dim=merged_points.shape[-1])

        results['points_per_cav'] = cav_points_list

        pairwise_t = np.tile(
            np.eye(4, dtype=np.float32),
            (self.max_cav, self.max_cav, 1, 1))
        if self.proj_first:
            pass
        else:
            for i in range(min(record_len, self.max_cav)):
                for j in range(min(record_len, self.max_cav)):
                    if i == j:
                        continue
                    T_i2ego = t_matrix[i]
                    T_j2ego = t_matrix[j]
                    T_j2ego_inv = np.linalg.inv(
                        T_j2ego.astype(np.float64))
                    pairwise_t[i, j] = (
                        T_j2ego_inv @ T_i2ego.astype(np.float64)
                    ).astype(np.float32)

        results['record_len'] = record_len
        results['pairwise_t_matrix'] = pairwise_t
        results['coop_mask'] = coop_mask if coop_mask is not None else \
            np.zeros(self.max_cav, dtype=bool)

        return results


@TRANSFORMS.register_module()
class PackCooperative3DDetInputs(BaseTransform):

    def __init__(
        self,
        keys: tuple = ('gt_bboxes_3d', 'gt_labels_3d'),
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape',
                            'lidar2img', 'depth2img', 'cam2img',
                            'pad_shape', 'scale_factor', 'flip',
                            'pcd_horizontal_flip', 'pcd_vertical_flip',
                            'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                            'num_pts_feats', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug'),
        coop_keys: tuple = ('record_len', 'pairwise_t_matrix', 'coop_mask',
                            'points_per_cav'),
    ) -> None:
        from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
        keys = list(keys)
        if 'points' not in keys:
            keys.append('points')
        self._base_packer = Pack3DDetInputs(keys=tuple(keys),
                                            meta_keys=meta_keys)
        self.coop_keys = coop_keys

    def transform(self, results: dict) -> dict:
        packed = self._base_packer.transform(results)

        for key in self.coop_keys:
            if key in results:
                val = results[key]
                if key == 'points_per_cav':
                    packed['inputs'][key] = [
                        torch.from_numpy(p) if isinstance(p, np.ndarray)
                        else p for p in val]
                elif isinstance(val, np.ndarray):
                    packed['inputs'][key] = torch.from_numpy(val)
                elif isinstance(val, (int, float)):
                    packed['inputs'][key] = torch.tensor(val)
                else:
                    packed['inputs'][key] = val

        return packed
