# Copyright (c) OpenMMLab. All rights reserved.
"""OPV2V Dataset for 3D Object Detection.

OPV2V is a large-scale open simulated dataset for Vehicle-to-Vehicle
cooperative perception. This implementation treats each CAV frame as
an independent sample for single-agent training.

Reference:
    https://github.com/DerrickXuNu/OpenCOOD
    https://opencood.readthedocs.io/en/latest/md_files/data_intro.html
"""

from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset


@DATASETS.register_module()
class OPV2VDataset(Det3DDataset):
    """OPV2V Dataset.

    This class serves as the API for experiments on the OPV2V Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Defaults to 'LiDAR'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction.
            Defaults to False (OPV2V has speed but not always reliable).
        pcd_limit_range (list[float]): Point cloud range for filtering.
            Defaults to [-100, -100, -5, 100, 100, 3].
    """

    METAINFO = {
        'classes': ('vehicle',),
        'palette': [(255, 158, 0)],  # Orange for vehicles
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = False,
                 pcd_limit_range: List[float] = [-100, -100, -5, 100, 100, 3],
                 **kwargs) -> None:

        self.with_velocity = with_velocity
        self.pcd_limit_range = pcd_limit_range

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def parse_ann_info(self, info: dict) -> Optional[dict]:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)

        if ann_info is not None:
            # Filter boxes by range if needed
            gt_bboxes_3d = ann_info['gt_bboxes_3d']
            gt_labels_3d = ann_info['gt_labels_3d']

            if len(gt_bboxes_3d) > 0:
                # Filter by point cloud range
                pcd_range = np.array(self.pcd_limit_range)
                mask = (
                    (gt_bboxes_3d[:, 0] >= pcd_range[0]) &
                    (gt_bboxes_3d[:, 0] <= pcd_range[3]) &
                    (gt_bboxes_3d[:, 1] >= pcd_range[1]) &
                    (gt_bboxes_3d[:, 1] <= pcd_range[4]) &
                    (gt_bboxes_3d[:, 2] >= pcd_range[2]) &
                    (gt_bboxes_3d[:, 2] <= pcd_range[5])
                )
                gt_bboxes_3d = gt_bboxes_3d[mask]
                gt_labels_3d = gt_labels_3d[mask]

            # Handle velocity
            if self.with_velocity and 'velocities' in ann_info:
                gt_velocities = ann_info['velocities']
                if len(gt_bboxes_3d) > 0 and len(gt_velocities) > 0:
                    gt_velocities = gt_velocities[mask]
                    nan_mask = np.isnan(gt_velocities[:, 0])
                    gt_velocities[nan_mask] = [0.0, 0.0]
                    gt_bboxes_3d = np.concatenate(
                        [gt_bboxes_3d, gt_velocities], axis=-1)

            # Create LiDARInstance3DBoxes
            # OPV2V boxes: [x, y, z, dx, dy, dz, yaw]
            # Box center is at the geometric center
            box_dim = 9 if self.with_velocity else 7
            if len(gt_bboxes_3d) > 0:
                gt_bboxes_3d = LiDARInstance3DBoxes(
                    gt_bboxes_3d,
                    box_dim=gt_bboxes_3d.shape[-1],
                    origin=(0.5, 0.5, 0.5)
                ).convert_to(self.box_mode_3d)
            else:
                gt_bboxes_3d = LiDARInstance3DBoxes(
                    np.zeros((0, box_dim), dtype=np.float32),
                    box_dim=box_dim,
                    origin=(0.5, 0.5, 0.5)
                )

            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            ann_info['gt_labels_3d'] = gt_labels_3d.astype(np.int64)

        else:
            # Empty annotations
            ann_info = dict()
            box_dim = 9 if self.with_velocity else 7
            ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                np.zeros((0, box_dim), dtype=np.float32),
                box_dim=box_dim,
                origin=(0.5, 0.5, 0.5)
            )
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        # Call parent's parse_data_info
        data_info = super().parse_data_info(info)

        # Required by BEVLoadMultiViewImageFromFiles to determine
        # how to handle camera intrinsics/extrinsics
        data_info['dataset'] = 'OPV2V'

        # Add OPV2V specific fields if needed
        if 'scenario' in info:
            data_info['scenario'] = info['scenario']
        if 'agent_id' in info:
            data_info['agent_id'] = info['agent_id']
        if 'ego_pose' in info:
            data_info['ego_pose'] = info['ego_pose']

        return data_info
