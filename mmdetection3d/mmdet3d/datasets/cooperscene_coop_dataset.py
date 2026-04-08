"""CooperScene Cooperative Dataset for 3D Object Detection.

Extends CooperSceneDataset with multi-agent cooperative perception.
The key difference from OPV2VCoopDataset is that ego_pose is stored
as a 4x4 transformation matrix instead of [x, y, z, roll, yaw, pitch].
"""

import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .cooperscene_dataset import CooperSceneDataset


def pose_to_matrix(pose):
    """Convert pose to a 4x4 matrix.

    Handles the CooperScene format where ego_pose is already a 4x4 matrix
    stored as a list of lists.

    Args:
        pose: 4x4 matrix as list of lists.

    Returns:
        4x4 numpy transformation matrix.
    """
    return np.array(pose, dtype=np.float64)


def pose_to_position(pose):
    """Extract [x, y, z] from a 4x4 pose matrix.

    Args:
        pose: 4x4 matrix as list of lists.

    Returns:
        numpy array of [x, y, z].
    """
    T = np.array(pose, dtype=np.float64)
    return T[:3, 3]


@DATASETS.register_module()
class CooperSceneCoopDataset(CooperSceneDataset):
    """CooperScene Cooperative Dataset.

    Extends CooperSceneDataset with multi-agent support.
    Loads cooperating agents' LiDAR data and computes transformation
    matrices from each agent's frame to the ego frame.

    Args:
        max_cav: Maximum number of CAVs including ego (default: 5).
        com_range: Communication range in meters (default: 70.0).
    """

    def __init__(self,
                 max_cav: int = 5,
                 com_range: float = 70.0,
                 **kwargs) -> None:
        self.max_cav = max_cav
        self.com_range = com_range
        super().__init__(**kwargs)

    def parse_data_info(self, info: dict) -> dict:
        """Parse info dict to add cooperative agent information.

        Args:
            info: Raw info dict from annotation file.

        Returns:
            Parsed data info dict with cooperative fields.
        """
        data_info = super().parse_data_info(info)

        ego_pose = info.get('ego_pose', None)
        cooperators = info.get('cooperators', [])

        valid_coops = []
        if ego_pose is not None:
            ego_pos = pose_to_position(ego_pose)
            T_ego = pose_to_matrix(ego_pose)
            T_ego_inv = np.linalg.inv(T_ego)

            for coop in cooperators:
                coop_pose = coop['ego_pose']
                coop_pos = pose_to_position(coop_pose)

                # Filter by communication range
                dist = np.linalg.norm(ego_pos[:2] - coop_pos[:2])
                if dist > self.com_range:
                    continue

                # Compute cav-to-ego transformation
                T_cav = pose_to_matrix(coop_pose)
                T_cav_to_ego = T_ego_inv @ T_cav

                coop_lidar_path = osp.join(
                    self.data_prefix.get('pts', ''),
                    coop['lidar_points']['lidar_path'])

                coop_entry = {
                    'agent_id': coop.get('agent_id', ''),
                    'lidar_path': coop_lidar_path,
                    'num_pts_feats': coop['lidar_points'].get(
                        'num_pts_feats', 4),
                    'transformation_matrix': T_cav_to_ego.astype(
                        np.float32),
                    'dist': float(dist),
                }

                if 'images' in coop and coop['images']:
                    coop_images = {}
                    for cam_name, cam_info in coop['images'].items():
                        abs_cam_info = dict(cam_info)
                        abs_cam_info['img_path'] = osp.join(
                            self.data_prefix.get('pts', ''),
                            cam_info['img_path'])
                        coop_images[cam_name] = abs_cam_info
                    coop_entry['images'] = coop_images

                valid_coops.append(coop_entry)

            valid_coops.sort(key=lambda x: x['dist'])

        # Build transformation_matrix: (max_cav, 4, 4)
        t_matrix = np.tile(
            np.eye(4, dtype=np.float32), (self.max_cav, 1, 1))

        coop_mask = np.zeros(self.max_cav, dtype=bool)
        coop_mask[0] = True  # ego is always valid

        coop_infos = []
        for i, coop in enumerate(valid_coops[:self.max_cav - 1]):
            idx = i + 1
            t_matrix[idx] = coop['transformation_matrix']
            coop_mask[idx] = True
            coop_entry = {
                'lidar_path': coop['lidar_path'],
                'num_pts_feats': coop['num_pts_feats'],
            }
            if 'images' in coop and coop['images']:
                coop_entry['images'] = coop['images']
            coop_infos.append(coop_entry)

        data_info['cooperators'] = coop_infos
        data_info['transformation_matrix'] = t_matrix
        data_info['coop_mask'] = coop_mask

        return data_info
