"""CooperScene Cooperative Dataset for 3D Object Detection.

Extends the single-agent CooperSceneDataset to support cooperative perception
with multiple Connected Autonomous Vehicles (CAVs). Each sample contains
ego agent data plus cooperating agents' LiDAR data and poses.
"""

import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .cooperscene_dataset import CooperSceneDataset


def pose_to_matrix(pose):
    """Convert pose to a 4x4 transformation matrix.

    Accepts either:
      - CooperScene style: flat list [x, y, z, roll_deg, yaw_deg, pitch_deg]
      - CooperScene style: 4x4 matrix already (list of 4 lists or ndarray)

    Returns:
        4x4 numpy transformation matrix (world frame).
    """
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.ndim == 2 and pose_arr.shape == (4, 4):
        return pose_arr

    x, y, z = pose_arr[0], pose_arr[1], pose_arr[2]
    roll = np.radians(pose_arr[3])
    yaw = np.radians(pose_arr[4])
    pitch = np.radians(pose_arr[5])
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)

    # Must match OpenCOOD's `transformation_utils.x_to_world` (CARLA /
    # CooperScene left-handed convention). A naive right-handed Rz @ Ry @ Rx
    # flips the sign of the pitch/roll terms in the z-row and tilts the
    # cooperator point clouds relative to the ego frame.
    R = np.array([
        [c_p * c_y, c_y * s_p * s_r - s_y * c_r, -c_y * s_p * c_r - s_y * s_r],
        [s_y * c_p, s_y * s_p * s_r + c_y * c_r, -s_y * s_p * c_r + c_y * s_r],
        [s_p,       -c_p * s_r,                   c_p * c_r],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


@DATASETS.register_module()
class CoopDataset(CooperSceneDataset):
    """CooperScene Cooperative Dataset.

    Extends single-agent CooperSceneDataset with multi-agent support.
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
            T_ego = pose_to_matrix(ego_pose)
            ego_pos = T_ego[:3, 3]
            T_ego_inv = np.linalg.inv(T_ego)

            for coop in cooperators:
                coop_pose = coop['ego_pose']
                T_cav = pose_to_matrix(coop_pose)
                coop_pos = T_cav[:3, 3]

                # Filter by communication range
                dist = np.linalg.norm(ego_pos[:2] - coop_pos[:2])
                if dist > self.com_range:
                    continue

                # Compute cav-to-ego transformation
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

            # Match OpenCOOD's cav ordering: vehicles by cav_id ascending,
            # with the infra / roadside unit (agent '0') moved to the END
            # (OpenCOOD's RSU handling puts it last; it is never ego). V2VAM's
            # fusion is order-sensitive, so the cooperator ordering must match
            # the convention the checkpoints were trained/evaluated under -
            # otherwise the fused features (and psm/rm) differ.
            valid_coops.sort(
                key=lambda x: (1 if str(x['agent_id']) == '0' else 0,
                               int(x['agent_id'])))

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
                'agent_id': coop.get('agent_id', ''),
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
