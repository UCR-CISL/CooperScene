"""Cooperative perception data transforms for BEVFusion.

Provides LoadCoopPointsFromFile, LoadCoopCameraData, and
CoopPack3DDetInputs transforms for loading and packing multi-agent
point cloud and camera data.
"""

import copy

import mmcv
import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from PIL import Image

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadCoopPointsFromFile(BaseTransform):
    """Load cooperating agents' point clouds from files.

    Reads cooperator LiDAR paths from results['cooperators'] and loads
    each point cloud as a numpy array. When proj_first=True (default),
    each cooperator's points are transformed into the ego coordinate
    frame using the precomputed transformation_matrix before any
    feature extraction, following the OpenCOOD convention.

    Args:
        load_dim: Number of dimensions per point to load (default: 4).
        use_dim: Number of dimensions to keep (default: 4).
        coord_type: Coordinate type (default: 'LIDAR').
        proj_first: If True, project cooperator points to ego frame
            before feature extraction (default: True).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            used to clip transformed points to ego's perception range.
            Only effective when proj_first=True (default: None).
    """

    def __init__(self, load_dim=4, use_dim=4, coord_type='LIDAR',
                 proj_first=True, point_cloud_range=None):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.coord_type = coord_type
        self.proj_first = proj_first
        self.point_cloud_range = point_cloud_range

    def _load_points(self, pts_path):
        """Load points from a binary file."""
        try:
            points = np.fromfile(pts_path, dtype=np.float32)
        except Exception:
            return np.zeros((0, self.load_dim), dtype=np.float32)

        valid_len = (points.size // self.load_dim) * self.load_dim
        points = points[:valid_len].reshape(-1, self.load_dim)
        points = points[:, :self.use_dim]
        return points

    def transform(self, results):
        """Load cooperator point clouds and optionally project to ego frame.

        When proj_first=True, each cooperator's points are transformed
        into the ego coordinate system using transformation_matrix[i+1]
        (index 0 is ego=identity, 1..N are cooperators), then filtered
        by point_cloud_range.

        Args:
            results: Dict with 'cooperators' key containing list of dicts
                with 'lidar_path' keys, and 'transformation_matrix' of
                shape (max_cav, 4, 4) when proj_first is enabled.

        Returns:
            Updated results dict with 'coop_points' list of numpy arrays.
        """
        cooperators = results.get('cooperators', [])
        t_matrix = results.get('transformation_matrix', None)

        coop_points = []
        for i, coop in enumerate(cooperators):
            # Paths are made absolute by CoopDataset.parse_data_info
            points = self._load_points(coop['lidar_path'])

            # Project to ego coordinate frame
            if self.proj_first and t_matrix is not None and len(points) > 0:
                T = t_matrix[i + 1]  # index 0=ego(identity), 1..N=coops
                pts_3d = points[:, :3]
                ones = np.ones((pts_3d.shape[0], 1), dtype=np.float32)
                pts_homo = np.concatenate([pts_3d, ones], axis=1)  # (N, 4)
                pts_ego = (T @ pts_homo.T).T[:, :3]  # (N, 3)
                points[:, :3] = pts_ego

                # Clip to ego's perception range
                if self.point_cloud_range is not None:
                    pr = self.point_cloud_range
                    mask = (
                        (points[:, 0] >= pr[0]) & (points[:, 0] <= pr[3]) &
                        (points[:, 1] >= pr[1]) & (points[:, 1] <= pr[4]) &
                        (points[:, 2] >= pr[2]) & (points[:, 2] <= pr[5]))
                    points = points[mask]

            coop_points.append(points)

        results['coop_points'] = coop_points
        return results


@TRANSFORMS.register_module()
class LoadCoopCameraData(BaseTransform):
    """Load cooperating agents' camera images and calibration matrices.

    For each cooperator in results['cooperators'] that has an 'images'
    dict, loads camera images and computes calibration matrices
    (cam2lidar, lidar2img, etc.) following the same logic as
    BEVLoadMultiViewImageFromFiles.

    Optionally applies ImageAug3D-style augmentation per cooperator
    (train mode) with independent random parameters.

    Each cooperator's lidar_aug_matrix is set to identity (no 3D
    augmentation for cooperators — avoids frame misalignment).

    Results are stored in results['coop_camera_data']: a list of dicts
    (one per cooperator) containing:
        - imgs: list of np.ndarray images (H, W, 3)
        - cam2img: (N, 4, 4)
        - lidar2cam: (N, 4, 4)
        - cam2lidar: (N, 4, 4)
        - lidar2img: (N, 4, 4)
        - img_aug_matrix: list of (4, 4)
        - lidar_aug_matrix: (4, 4) identity
        - img_shape: (H, W)
        - ori_shape: (H, W)

    Args:
        to_float32: Whether to convert images to float32.
        color_type: Color type for mmcv.imfrombytes.
        apply_aug: Whether to apply ImageAug3D-style augmentation.
        final_dim: [H, W] target image size after augmentation.
        resize_lim: [min, max] resize ratio range.
        bot_pct_lim: [min, max] bottom crop percentage range.
        rot_lim: [min, max] rotation limit in degrees.
        rand_flip: Whether to randomly flip images.
        is_train: Whether in training mode.
    """

    def __init__(self, to_float32=True, color_type='color',
                 apply_aug=True, final_dim=(256, 512),
                 resize_lim=(0.64, 0.80), bot_pct_lim=(0.0, 0.0),
                 rot_lim=(-5.4, 5.4), rand_flip=True, is_train=True):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.apply_aug = apply_aug
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rot_lim = rot_lim
        self.rand_flip = rand_flip
        self.is_train = is_train

    def _compute_calibration(self, images_dict):
        """Compute calibration matrices from an images dict.

        Same logic as BEVLoadMultiViewImageFromFiles.

        Returns:
            cam2img, lidar2cam, cam2lidar, lidar2img as stacked arrays,
            img_paths as list of strings.
        """
        cam2img_list, lidar2cam_list = [], []
        cam2lidar_list, lidar2img_list = [], []
        img_paths = []

        for cam_name, cam_info in images_dict.items():
            img_paths.append(cam_info['img_path'])

            lidar2cam_array = np.array(
                cam_info['lidar2cam']).astype(np.float32)
            lidar2cam_4x4 = np.eye(4, dtype=np.float32)
            lidar2cam_4x4[:lidar2cam_array.shape[0],
                          :lidar2cam_array.shape[1]] = lidar2cam_array
            lidar2cam_list.append(lidar2cam_4x4)

            # cam2lidar = inv(lidar2cam)
            lidar2cam_rot = lidar2cam_4x4[:3, :3]
            lidar2cam_trans = lidar2cam_4x4[:3, 3:4]
            camera2lidar = np.eye(4, dtype=np.float32)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar_list.append(camera2lidar)

            cam2img_array = np.eye(4, dtype=np.float32)
            cam2img_raw = np.array(
                cam_info['cam2img']).astype(np.float32)
            cam2img_array[:3, :3] = cam2img_raw[:3, :3]
            cam2img_list.append(cam2img_array)

            lidar2img_list.append(cam2img_array @ lidar2cam_4x4)

        return (np.stack(cam2img_list, axis=0),
                np.stack(lidar2cam_list, axis=0),
                np.stack(cam2lidar_list, axis=0),
                np.stack(lidar2img_list, axis=0),
                img_paths)

    def _sample_augmentation(self, H, W):
        """Sample augmentation parameters (same logic as ImageAug3D)."""
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.rand_flip and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _img_transform(self, img, resize, resize_dims, crop, flip,
                       rotate):
        """Apply augmentation to a single image (ImageAug3D logic)."""
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # Compute the post-augmentation homography
        post_rot = np.eye(2) * resize
        post_tran = -np.array(crop[:2], dtype=np.float64)
        if flip:
            A = np.array([[-1, 0], [0, 1]], dtype=np.float64)
            b = np.array([crop[2] - crop[0], 0], dtype=np.float64)
            post_rot = A @ post_rot
            post_tran = A @ post_tran + b
        theta = rotate / 180 * np.pi
        A = np.array([[np.cos(theta), np.sin(theta)],
                       [-np.sin(theta), np.cos(theta)]])
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A @ (-b) + b
        post_rot = A @ post_rot
        post_tran = A @ post_tran + b

        transform = np.eye(4, dtype=np.float32)
        transform[:2, :2] = post_rot.astype(np.float32)
        transform[:2, 3] = post_tran.astype(np.float32)

        return np.array(img).astype(np.float32), transform

    def transform(self, results):
        """Load cooperator camera data and compute calibration.

        Args:
            results: Pipeline results dict with 'cooperators' key.

        Returns:
            Updated results with 'coop_camera_data' list.
        """
        cooperators = results.get('cooperators', [])
        coop_camera_data = []

        for coop in cooperators:
            images_dict = coop.get('images', {})
            if not images_dict:
                coop_camera_data.append(None)
                continue

            (cam2img, lidar2cam, cam2lidar, lidar2img,
             img_paths) = self._compute_calibration(images_dict)

            # Load images
            imgs = []
            for path in img_paths:
                img_bytes = get(path)
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type,
                    backend='pillow', channel_order='rgb')
                imgs.append(img)

            if not imgs:
                coop_camera_data.append(None)
                continue

            ori_shape = imgs[0].shape[:2]  # (H, W)

            # Apply augmentation if configured
            img_aug_matrices = []
            if self.apply_aug:
                new_imgs = []
                for img in imgs:
                    H, W = img.shape[:2]
                    (resize, resize_dims, crop, flip,
                     rotate) = self._sample_augmentation(H, W)
                    new_img, aug_mat = self._img_transform(
                        img, resize, resize_dims, crop, flip, rotate)
                    new_imgs.append(new_img)
                    img_aug_matrices.append(aug_mat)
                imgs = new_imgs
            else:
                img_aug_matrices = [
                    np.eye(4, dtype=np.float32) for _ in imgs]

            if self.to_float32:
                imgs = [img.astype(np.float32) for img in imgs]

            coop_camera_data.append({
                'imgs': imgs,
                'cam2img': cam2img,
                'lidar2cam': lidar2cam,
                'cam2lidar': cam2lidar,
                'lidar2img': lidar2img,
                'img_aug_matrix': img_aug_matrices,
                'lidar_aug_matrix': np.eye(4, dtype=np.float32),
                'img_shape': imgs[0].shape[:2],
                'ori_shape': ori_shape,
            })

        results['coop_camera_data'] = coop_camera_data
        return results


@TRANSFORMS.register_module()
class CoopPack3DDetInputs(BaseTransform):
    """Pack cooperative multi-agent data for BEVFusion.

    Extends Pack3DDetInputs to handle cooperative fields:
    - coop_points: cooperator point clouds
    - transformation_matrix: cav-to-ego transformation matrices
    - coop_mask: validity mask for each agent slot

    Args:
        keys: Keys to pack into inputs/data_samples.
        meta_keys: Keys to pack into metainfo.
    """

    def __init__(self,
                 keys=('points', 'gt_bboxes_3d', 'gt_labels_3d'),
                 meta_keys=('box_type_3d', 'sample_idx', 'lidar_path',
                            'transformation_3d_flow', 'pcd_rotation',
                            'pcd_scale_factor', 'pcd_trans')):
        from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
        self.keys = keys
        self.meta_keys = meta_keys
        self._packer = Pack3DDetInputs(
            keys=self.keys, meta_keys=self.meta_keys)

    def transform(self, results):
        """Pack results into the format expected by CoopBEVFusion.

        This delegates most packing to the standard Pack3DDetInputs,
        then adds cooperative fields to inputs and metainfo.

        Args:
            results: Pipeline results dict.

        Returns:
            Dict with 'inputs' and 'data_samples'.
        """
        from mmdet3d.structures import BasePoints

        # Use standard Pack3DDetInputs for base packing
        packed = self._packer.transform(results)

        # Add cooperative data to metainfo (NOT inputs).
        # Storing in metainfo prevents pseudo_collate from recursively
        # transposing the list structure across batch samples.
        data_sample = packed['data_samples']

        if 'coop_points' in results:
            coop_points_tensors = []
            for pts in results['coop_points']:
                if isinstance(pts, np.ndarray):
                    coop_points_tensors.append(
                        torch.from_numpy(pts).float())
                elif isinstance(pts, BasePoints):
                    coop_points_tensors.append(pts.tensor.float())
                else:
                    coop_points_tensors.append(pts.float())

            # Pad to fixed length (max_cav - 1)
            if 'coop_mask' in results:
                n_coop_slots = len(results['coop_mask']) - 1  # exclude ego
                n_feats = (coop_points_tensors[0].shape[1]
                           if coop_points_tensors else 4)
                while len(coop_points_tensors) < n_coop_slots:
                    coop_points_tensors.append(
                        torch.zeros((0, n_feats), dtype=torch.float32))

            data_sample.set_metainfo({
                'coop_points': coop_points_tensors,
            })
        if 'transformation_matrix' in results:
            data_sample.set_metainfo({
                'transformation_matrix':
                    results['transformation_matrix'].astype(np.float32),
            })
        if 'coop_mask' in results:
            data_sample.set_metainfo({
                'coop_mask': results['coop_mask'],
            })
        if 'coop_camera_data' in results:
            data_sample.set_metainfo({
                'coop_camera_data': results['coop_camera_data'],
            })

        return packed
