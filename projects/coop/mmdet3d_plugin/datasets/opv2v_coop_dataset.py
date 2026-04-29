"""OPV2V Cooperative Dataset for 3D Object Detection.

Extends the single-agent OPV2VDataset to support cooperative perception
with multiple Connected Autonomous Vehicles (CAVs). Each sample contains
ego agent data plus cooperating agents' LiDAR data and poses.
"""

import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .opv2v_dataset import OPV2VDataset


def pose_to_matrix(pose):
    """Convert [x, y, z, roll_deg, yaw_deg, pitch_deg] to a 4x4 matrix.

    Args:
        pose: List of [x, y, z, roll, yaw, pitch] in degrees.

    Returns:
        4x4 numpy transformation matrix (world frame).
    """
    x, y, z = pose[0], pose[1], pose[2]
    roll, yaw, pitch = np.radians(pose[3]), np.radians(pose[4]), \
        np.radians(pose[5])

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


@DATASETS.register_module()
class OPV2VCoopDataset(OPV2VDataset):
    """OPV2V Cooperative Dataset.

    Extends single-agent OPV2VDataset with multi-agent support.
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

        Adds cooperator LiDAR paths, transformation matrices (cav-to-ego),
        and validity masks to the data info dict.

        Args:
            info: Raw info dict from annotation file.

        Returns:
            Parsed data info dict with cooperative fields.
        """
        data_info = super().parse_data_info(info)

        ego_pose = info.get('ego_pose', None)
        cooperators = info.get('cooperators', [])

        # Compute transformation matrices and filter by range
        valid_coops = []
        if ego_pose is not None:
            ego_pos = np.array(ego_pose[:3])
            T_ego = pose_to_matrix(ego_pose)
            T_ego_inv = np.linalg.inv(T_ego)

            for coop in cooperators:
                coop_pose = coop['ego_pose']
                coop_pos = np.array(coop_pose[:3])

                # Filter by communication range
                dist = np.linalg.norm(ego_pos[:2] - coop_pos[:2])
                if dist > self.com_range:
                    continue

                # Compute cav-to-ego transformation
                T_cav = pose_to_matrix(coop_pose)
                T_cav_to_ego = T_ego_inv @ T_cav

                # Make cooperator lidar_path absolute, matching
                # Det3DDataset.parse_data_info behavior for ego path
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

                # Thread cooperator camera images if available
                if 'images' in coop and coop['images']:
                    # Make image paths absolute (like lidar_path above)
                    coop_images = {}
                    for cam_name, cam_info in coop['images'].items():
                        abs_cam_info = dict(cam_info)
                        abs_cam_info['img_path'] = osp.join(
                            self.data_prefix.get('pts', ''),
                            cam_info['img_path'])
                        coop_images[cam_name] = abs_cam_info
                    coop_entry['images'] = coop_images

                valid_coops.append(coop_entry)

            # Sort by distance so closest cooperators are selected first
            valid_coops.sort(key=lambda x: x['dist'])

        # Build transformation_matrix: (max_cav, 4, 4)
        # Index 0 = ego (identity), index 1..max_cav-1 = cooperators
        # Initialize ALL slots to identity so unused slots are still
        # invertible in STTF (zero matrices are singular and crash).
        t_matrix = np.tile(
            np.eye(4, dtype=np.float32), (self.max_cav, 1, 1))

        coop_mask = np.zeros(self.max_cav, dtype=bool)
        coop_mask[0] = True  # ego is always valid

        coop_infos = []
        for i, coop in enumerate(valid_coops[:self.max_cav - 1]):
            idx = i + 1  # 0 is ego
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
